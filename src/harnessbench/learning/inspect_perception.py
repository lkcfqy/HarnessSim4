"""Real-image anomaly perception for InspectBot.

This module intentionally separates perception evidence from active-control
evidence.  MVTec AD supplies real cable cross-section images and pixel masks,
but it does not supply robot viewpoints or automotive-harness topology.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image, ImageDraw, ImageOps
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f

from harnessbench.inspect_public_data import audit_mvtec_cable, file_sha256


@dataclass(frozen=True)
class InspectPerceptionConfig:
    """Frozen settings for the compact spatial-Gaussian detector."""

    resize: int = 256
    crop: int = 224
    feature_size: int = 28
    selected_channels: int = 64
    fit_good_images: int = 180
    covariance_regularization: float = 0.01
    image_score_top_fraction: float = 0.01
    calibration_quantile: float = 0.99
    pixel_calibration_quantile: float = 0.995
    seed: int = 20260827
    batch_size: int = 16
    preserve_aspect_ratio: bool = False


class ResNet18SpatialFeatures(nn.Module):
    """Frozen ImageNet ResNet-18 features aligned at 28 x 28."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        size = layer2.shape[-2:]
        layer1 = torch_f.interpolate(layer1, size=size, mode="bilinear", align_corners=False)
        layer3 = torch_f.interpolate(layer3, size=size, mode="bilinear", align_corners=False)
        return torch.cat((layer1, layer2, layer3), dim=1)


def _normalization() -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return mean, std


def _deterministic_occlusion(image: Image.Image, key: str) -> Image.Image:
    output = image.copy()
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    width, height = output.size
    box_w = round(width * (0.18 + 0.04 * digest[0] / 255.0))
    box_h = round(height * (0.14 + 0.05 * digest[1] / 255.0))
    x0 = int((width - box_w) * digest[2] / 255.0)
    y0 = int((height - box_h) * digest[3] / 255.0)
    fill = tuple(int(value) for value in (38, 46, 55))
    ImageDraw.Draw(output).rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=fill)
    return output


def preprocess_image(
    path: Path,
    config: InspectPerceptionConfig,
    *,
    view_id: int = 0,
) -> torch.Tensor:
    """Load one image and apply a deterministic synthetic-view transform."""

    with Image.open(path) as raw:
        image = raw.convert("RGB")
    if config.preserve_aspect_ratio:
        thumbnail = ImageOps.contain(
            image,
            (config.crop, config.crop),
            method=Image.Resampling.BILINEAR,
        )
        sample = np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR))
        fill = tuple(int(value) for value in np.median(sample, axis=(0, 1)))
        canvas = Image.new("RGB", (config.crop, config.crop), fill)
        canvas.paste(
            thumbnail,
            ((config.crop - thumbnail.width) // 2, (config.crop - thumbnail.height) // 2),
        )
        image = canvas
    else:
        image = vision_f.resize(
            image,
            [config.resize, config.resize],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image = vision_f.center_crop(image, [config.crop, config.crop])
    if view_id == 1:
        image = vision_f.adjust_brightness(image, 0.72)
        image = vision_f.gaussian_blur(image, kernel_size=[9, 9], sigma=[1.2, 1.2])
    elif view_id == 2:
        image = vision_f.rotate(
            image,
            angle=7.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=[35, 43, 52],
        )
        image = vision_f.adjust_contrast(image, 1.18)
    elif view_id == 3:
        image = _deterministic_occlusion(image, path.as_posix())
        image = vision_f.adjust_brightness(image, 0.88)
    elif view_id != 0:
        raise ValueError(f"unknown synthetic view id: {view_id}")
    tensor = vision_f.pil_to_tensor(image).to(torch.float32) / 255.0
    mean, std = _normalization()
    return (tensor - mean) / std


def preprocess_mask(path: Path, config: InspectPerceptionConfig) -> torch.Tensor:
    with Image.open(path) as raw:
        mask = raw.convert("L")
    mask = vision_f.resize(
        mask,
        [config.resize, config.resize],
        interpolation=InterpolationMode.NEAREST,
    )
    mask = vision_f.center_crop(mask, [config.crop, config.crop])
    return vision_f.pil_to_tensor(mask).squeeze(0) > 0


def _batch_tensors(
    paths: list[Path],
    config: InspectPerceptionConfig,
    *,
    view_id: int = 0,
):
    for offset in range(0, len(paths), config.batch_size):
        chunk = paths[offset : offset + config.batch_size]
        yield chunk, torch.stack(
            [preprocess_image(path, config, view_id=view_id) for path in chunk]
        )


class SpatialGaussianDetector:
    """PaDiM-style local Gaussian distance with a compact ResNet-18 backbone."""

    def __init__(
        self,
        extractor: ResNet18SpatialFeatures,
        selected_indices: torch.Tensor,
        mean: torch.Tensor,
        covariance_cholesky: torch.Tensor,
        config: InspectPerceptionConfig,
        metadata: dict[str, Any],
    ) -> None:
        self.extractor = extractor.eval()
        self.selected_indices = selected_indices.to(torch.long).cpu()
        self.mean = mean.to(torch.float32).cpu()
        self.covariance_cholesky = covariance_cholesky.to(torch.float32).cpu()
        self.config = config
        self.metadata = metadata

    @classmethod
    def fit(
        cls,
        train_paths: list[Path],
        config: InspectPerceptionConfig,
        *,
        device_name: str,
        metadata: dict[str, Any],
    ) -> SpatialGaussianDetector:
        if len(train_paths) < 2:
            raise ValueError("at least two defect-free training images are required")
        device = torch.device(device_name)
        extractor = ResNet18SpatialFeatures(pretrained=True).to(device).eval()
        generator = torch.Generator().manual_seed(config.seed)
        all_channels = 64 + 128 + 256
        selected = torch.randperm(all_channels, generator=generator)[: config.selected_channels]
        features: list[torch.Tensor] = []
        with torch.inference_mode():
            for _, tensors in _batch_tensors(train_paths, config):
                batch = extractor(tensors.to(device))[:, selected.to(device)]
                features.append(batch.cpu())
        stacked = torch.cat(features, dim=0)
        if stacked.shape[-2:] != (config.feature_size, config.feature_size):
            raise ValueError(f"unexpected feature map size: {tuple(stacked.shape[-2:])}")
        samples = stacked.permute(2, 3, 0, 1).reshape(
            config.feature_size * config.feature_size,
            len(train_paths),
            config.selected_channels,
        )
        samples = samples.to(device)
        mean = samples.mean(dim=1)
        centered = samples - mean[:, None, :]
        covariance = torch.einsum("pnc,pnd->pcd", centered, centered)
        covariance /= float(len(train_paths) - 1)
        identity = torch.eye(config.selected_channels, device=device)[None]
        covariance = covariance + config.covariance_regularization * identity
        cholesky = torch.linalg.cholesky(covariance)
        return cls(
            extractor.cpu(),
            selected,
            mean.cpu(),
            cholesky.cpu(),
            config,
            metadata,
        )

    def _feature_batch(
        self,
        paths: list[Path],
        *,
        device_name: str,
        view_id: int,
    ) -> torch.Tensor:
        device = torch.device(device_name)
        self.extractor.to(device).eval()
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for _, tensors in _batch_tensors(paths, self.config, view_id=view_id):
                features = self.extractor(tensors.to(device))
                outputs.append(features[:, self.selected_indices.to(device)].cpu())
        self.extractor.cpu()
        return torch.cat(outputs, dim=0)

    def score_paths(
        self,
        paths: list[Path],
        *,
        device_name: str,
        view_id: int = 0,
        return_maps: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not paths:
            empty = np.empty(0, dtype=np.float64)
            return empty, np.empty((0, 0, 0), dtype=np.float32) if return_maps else None
        device = torch.device(device_name)
        features = self._feature_batch(paths, device_name=device_name, view_id=view_id)
        mean = self.mean.to(device)
        cholesky = self.covariance_cholesky.to(device)
        position_count = self.config.feature_size**2
        map_rows: list[torch.Tensor] = []
        with torch.inference_mode():
            for feature in features:
                sample = feature.permute(1, 2, 0).reshape(position_count, -1).to(device)
                delta = (sample - mean).unsqueeze(-1)
                solved = torch.linalg.solve_triangular(cholesky, delta, upper=False)
                distance = torch.sqrt(torch.sum(solved.squeeze(-1) ** 2, dim=1).clamp_min(0))
                map_rows.append(distance.reshape(self.config.feature_size, self.config.feature_size))
        maps = torch.stack(map_rows)
        flat = maps.reshape(len(paths), -1)
        top_k = max(1, math.ceil(flat.shape[1] * self.config.image_score_top_fraction))
        scores = torch.topk(flat, top_k, dim=1).values.mean(dim=1).cpu().numpy()
        if not return_maps:
            return scores.astype(np.float64), None
        upsampled = torch_f.interpolate(
            maps[:, None],
            size=(self.config.crop, self.config.crop),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return scores.astype(np.float64), upsampled.cpu().numpy().astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "name": "InspectBot spatial Gaussian anomaly detector",
                "config": asdict(self.config),
                "metadata": self.metadata,
                "selected_indices": self.selected_indices,
                "mean": self.mean,
                "covariance_cholesky": self.covariance_cholesky,
                "backbone_state_dict": self.extractor.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> SpatialGaussianDetector:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        extractor = ResNet18SpatialFeatures(pretrained=False)
        extractor.load_state_dict(payload["backbone_state_dict"])
        return cls(
            extractor,
            payload["selected_indices"],
            payload["mean"],
            payload["covariance_cholesky"],
            InspectPerceptionConfig(**payload["config"]),
            dict(payload["metadata"]),
        )


def _rgb_statistics(paths: list[Path], config: InspectPerceptionConfig) -> np.ndarray:
    rows = []
    mean, std = _normalization()
    for path in paths:
        tensor = preprocess_image(path, config) * std + mean
        channel_mean = tensor.mean(dim=(1, 2)).numpy()
        channel_std = tensor.std(dim=(1, 2)).numpy()
        horizontal = torch.abs(tensor[:, :, 1:] - tensor[:, :, :-1]).mean(dim=(1, 2)).numpy()
        vertical = torch.abs(tensor[:, 1:, :] - tensor[:, :-1, :]).mean(dim=(1, 2)).numpy()
        rows.append(np.concatenate((channel_mean, channel_std, horizontal, vertical)))
    return np.asarray(rows, dtype=np.float64)


def _global_embeddings(
    detector: SpatialGaussianDetector,
    paths: list[Path],
    *,
    device_name: str,
) -> np.ndarray:
    features = detector._feature_batch(paths, device_name=device_name, view_id=0)
    return features.mean(dim=(2, 3)).numpy().astype(np.float64)


def _mahalanobis_scores(model: LedoitWolf, features: np.ndarray) -> np.ndarray:
    delta = features - model.location_[None, :]
    values = np.einsum("ni,ij,nj->n", delta, model.precision_, delta)
    return np.sqrt(np.maximum(values, 0.0))


def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(np.mean(predictions[labels == 0])) if np.any(labels == 0) else 0.0,
    }


def _bootstrap_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    draws: int = 2_000,
) -> dict:
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    anomalous = np.flatnonzero(labels == 1)
    auc_values = np.empty(draws, dtype=np.float64)
    aupr_values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = np.concatenate(
            (
                rng.choice(normal, size=len(normal), replace=True),
                rng.choice(anomalous, size=len(anomalous), replace=True),
            )
        )
        auc_values[index] = roc_auc_score(labels[sample], scores[sample])
        aupr_values[index] = average_precision_score(labels[sample], scores[sample])
    return {
        "bootstrap_draws": draws,
        "bootstrap_unit": "held-out image, stratified by normal/anomalous label",
        "auroc_ci95": [float(value) for value in np.quantile(auc_values, [0.025, 0.975])],
        "aupr_ci95": [float(value) for value in np.quantile(aupr_values, [0.025, 0.975])],
    }


def _method_report(
    labels: np.ndarray,
    scores: np.ndarray,
    calibration_scores: np.ndarray,
    config: InspectPerceptionConfig,
    *,
    seed_offset: int,
) -> dict:
    threshold = float(np.quantile(calibration_scores, config.calibration_quantile))
    output = {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        **_threshold_metrics(labels, scores, threshold),
    }
    output.update(_bootstrap_auc_ci(labels, scores, seed=config.seed + seed_offset))
    return output


def _per_defect_report(
    defect_labels: list[str],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    """Evaluate each anomaly type against the shared held-out normal images."""

    labels_array = np.asarray(defect_labels)
    output: dict[str, dict[str, float | int]] = {}
    for defect_type in sorted(set(defect_labels) - {"good"}):
        subset = np.flatnonzero((labels_array == "good") | (labels_array == defect_type))
        binary = (labels_array[subset] == defect_type).astype(np.int64)
        subset_scores = scores[subset]
        metrics = _threshold_metrics(binary, subset_scores, threshold)
        output[defect_type] = {
            "anomaly_images": int(np.sum(binary)),
            "normal_images": int(np.sum(binary == 0)),
            "auroc": float(roc_auc_score(binary, subset_scores)),
            "aupr": float(average_precision_score(binary, subset_scores)),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
        }
    return output


def _load_index(root: Path) -> tuple[list[Path], list[Path], list[Path], list[str]]:
    train_good = sorted((root / "train" / "good").glob("*.png"))
    test_good = sorted((root / "test" / "good").glob("*.png"))
    anomaly_paths = sorted(
        path for path in (root / "test").glob("*/*.png") if path.parent.name != "good"
    )
    test_paths = [*test_good, *anomaly_paths]
    labels = ["good"] * len(test_good) + [path.parent.name for path in anomaly_paths]
    return train_good, test_good, test_paths, labels


def run_inspect_perception_experiment(
    data_root: Path,
    output_dir: Path,
    *,
    device_name: str = "cuda",
    config: InspectPerceptionConfig | None = None,
) -> dict:
    """Fit the detector on normal training images and evaluate frozen test data."""

    config = config or InspectPerceptionConfig()
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "public_data"
    audit = audit_mvtec_cable(data_root, audit_dir)
    train_good, _, test_paths, defect_labels = _load_index(data_root)
    if not 2 <= config.fit_good_images < len(train_good):
        raise ValueError("fit_good_images must leave at least one normal calibration image")
    split_rng = np.random.default_rng(config.seed)
    split_indices = split_rng.permutation(len(train_good))
    fit_paths = [train_good[int(index)] for index in split_indices[: config.fit_good_images]]
    calibration_paths = [
        train_good[int(index)] for index in split_indices[config.fit_good_images :]
    ]
    fit_manifest = "\n".join(
        f"{path.relative_to(data_root).as_posix()} {file_sha256(path)}" for path in fit_paths
    )
    fit_sha256 = hashlib.sha256(fit_manifest.encode("utf-8")).hexdigest()
    detector = SpatialGaussianDetector.fit(
        fit_paths,
        config,
        device_name=device_name,
        metadata={
            "dataset_tree_sha256": audit["tree_sha256"],
            "fit_manifest_sha256": fit_sha256,
            "fit_images": len(fit_paths),
            "calibration_images": len(calibration_paths),
            "backbone": "torchvision ResNet18_Weights.DEFAULT (ImageNet-1K V1)",
            "claim_scope": "real-image offline anomaly detection; no robot viewpoints",
        },
    )
    checkpoint = output_dir / "inspect_spatial_gaussian.pt"
    detector.save(checkpoint)

    spatial_calibration, calibration_maps = detector.score_paths(
        calibration_paths,
        device_name=device_name,
        return_maps=True,
    )
    spatial_scores, spatial_maps = detector.score_paths(
        test_paths,
        device_name=device_name,
        return_maps=True,
    )
    assert calibration_maps is not None and spatial_maps is not None

    rgb_fit = _rgb_statistics(fit_paths, config)
    rgb_calibration = _rgb_statistics(calibration_paths, config)
    rgb_test = _rgb_statistics(test_paths, config)
    rgb_model = LedoitWolf().fit(rgb_fit)
    rgb_calibration_scores = _mahalanobis_scores(rgb_model, rgb_calibration)
    rgb_scores = _mahalanobis_scores(rgb_model, rgb_test)

    global_fit = _global_embeddings(detector, fit_paths, device_name=device_name)
    global_calibration = _global_embeddings(detector, calibration_paths, device_name=device_name)
    global_test = _global_embeddings(detector, test_paths, device_name=device_name)
    global_model = LedoitWolf().fit(global_fit)
    global_calibration_scores = _mahalanobis_scores(global_model, global_calibration)
    global_scores = _mahalanobis_scores(global_model, global_test)

    labels = np.asarray([label != "good" for label in defect_labels], dtype=np.int64)
    methods = {
        "rgb_statistics": _method_report(
            labels, rgb_scores, rgb_calibration_scores, config, seed_offset=1
        ),
        "resnet_global": _method_report(
            labels, global_scores, global_calibration_scores, config, seed_offset=2
        ),
        "spatial_gaussian": _method_report(
            labels, spatial_scores, spatial_calibration, config, seed_offset=3
        ),
    }
    method_scores = {
        "rgb_statistics": rgb_scores,
        "resnet_global": global_scores,
        "spatial_gaussian": spatial_scores,
    }
    for method_name, scores in method_scores.items():
        methods[method_name]["per_defect"] = _per_defect_report(
            defect_labels,
            scores,
            float(methods[method_name]["threshold"]),
        )

    pixel_truth_rows = []
    for path, defect_type in zip(test_paths, defect_labels):
        if defect_type == "good":
            pixel_truth_rows.append(
                np.zeros((config.crop, config.crop), dtype=np.uint8)
            )
        else:
            mask_path = data_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
            pixel_truth_rows.append(preprocess_mask(mask_path, config).numpy().astype(np.uint8))
    pixel_truth = np.stack(pixel_truth_rows)
    pixel_calibration = calibration_maps.reshape(-1)
    pixel_threshold = float(np.quantile(pixel_calibration, config.pixel_calibration_quantile))
    pixel_labels = pixel_truth.reshape(-1)
    pixel_scores = spatial_maps.reshape(-1)
    pixel_predictions = pixel_scores >= pixel_threshold
    methods["spatial_gaussian"]["pixel"] = {
        "auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "aupr": float(average_precision_score(pixel_labels, pixel_scores)),
        "threshold": pixel_threshold,
        "precision": float(precision_score(pixel_labels, pixel_predictions, zero_division=0)),
        "recall": float(recall_score(pixel_labels, pixel_predictions, zero_division=0)),
        "f1": float(f1_score(pixel_labels, pixel_predictions, zero_division=0)),
    }

    predictions_path = output_dir / "inspect_perception_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "relative_path",
            "defect_type",
            "is_anomaly",
            "rgb_statistics_score",
            "resnet_global_score",
            "spatial_gaussian_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(test_paths):
            writer.writerow(
                {
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "defect_type": defect_labels[index],
                    "is_anomaly": int(labels[index]),
                    "rgb_statistics_score": float(rgb_scores[index]),
                    "resnet_global_score": float(global_scores[index]),
                    "spatial_gaussian_score": float(spatial_scores[index]),
                }
            )
    maps_path = output_dir / "inspect_spatial_maps.npz"
    np.savez_compressed(
        maps_path,
        relative_paths=np.asarray(
            [path.relative_to(data_root).as_posix() for path in test_paths]
        ),
        scores=spatial_scores,
        maps=spatial_maps,
        masks=pixel_truth,
    )
    report = {
        "schema_version": 1,
        "name": "InspectBot MVTec AD cable perception experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "dataset": audit,
        "split": {
            "fit_good": len(fit_paths),
            "calibration_good": len(calibration_paths),
            "test_good": int(np.sum(labels == 0)),
            "test_anomaly": int(np.sum(labels == 1)),
            "fit_manifest_sha256": fit_sha256,
        },
        "methods": methods,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "predictions_csv": str(predictions_path.resolve()),
        "predictions_sha256": file_sha256(predictions_path),
        "maps_npz": str(maps_path.resolve()),
        "maps_sha256": file_sha256(maps_path),
        "claim_scope": (
            "Offline anomaly detection on real MVTec cable cross-sections. Synthetic view "
            "selection and robot control are evaluated separately."
        ),
    }
    report_path = output_dir / "inspect_perception_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
