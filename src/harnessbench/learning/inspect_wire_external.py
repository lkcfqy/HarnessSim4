"""Independent real-image validation on the FAU/FAPS Stripped Wire Dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from harnessbench.inspect_public_data import file_sha256
from harnessbench.inspect_wire_data import audit_stripped_wire
from harnessbench.learning.inspect_perception import (
    InspectPerceptionConfig,
    SpatialGaussianDetector,
    _global_embeddings,
    _mahalanobis_scores,
    _method_report,
    _per_defect_report,
    _rgb_statistics,
)
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact


def _load_wire_index(root: Path) -> tuple[list[Path], list[Path], list[str]]:
    training = sorted((root / "train" / "PatchCore").glob("*.jpg"))
    test_good = sorted((root / "test" / "good").glob("*.jpg"))
    cut = sorted((root / "test" / "cut_strands").glob("*.jpg"))
    pulled = sorted((root / "test" / "pulled_strands").glob("*.jpg"))
    test = [*test_good, *cut, *pulled]
    labels = ["good"] * len(test_good) + ["cut_strands"] * len(cut) + [
        "pulled_strands"
    ] * len(pulled)
    return training, test, labels


def _spatial_nn_scores(
    reference: torch.Tensor,
    queries: torch.Tensor,
    config: InspectPerceptionConfig,
    *,
    device_name: str,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Position-wise nearest-neighbour baseline in the same frozen feature space."""

    device = torch.device(device_name)
    positions = config.feature_size**2
    reference_positions = reference.permute(2, 3, 0, 1).reshape(
        positions, reference.shape[0], reference.shape[1]
    )
    reference_positions = reference_positions.to(device)
    output_maps = []
    with torch.inference_mode():
        for offset in range(0, len(queries), batch_size):
            query = queries[offset : offset + batch_size]
            query_positions = query.permute(2, 3, 0, 1).reshape(
                positions, query.shape[0], query.shape[1]
            )
            distances = torch.cdist(query_positions.to(device), reference_positions)
            nearest = distances.min(dim=2).values.permute(1, 0)
            output_maps.append(
                nearest.reshape(-1, config.feature_size, config.feature_size).cpu()
            )
    maps = torch.cat(output_maps).numpy().astype(np.float32)
    flat = maps.reshape(len(maps), -1)
    top_k = max(1, math.ceil(flat.shape[1] * config.image_score_top_fraction))
    scores = np.partition(flat, -top_k, axis=1)[:, -top_k:].mean(axis=1)
    return scores.astype(np.float64), maps


def _paired_method_comparisons(
    labels: np.ndarray,
    methods: dict[str, dict],
    scores: dict[str, np.ndarray],
    *,
    seed: int,
    bootstrap_draws: int = 5_000,
) -> list[dict]:
    candidate_name = "spatial_gaussian"
    candidate = scores[candidate_name]
    candidate_prediction = candidate >= float(methods[candidate_name]["threshold"])
    candidate_correct = candidate_prediction == labels.astype(bool)
    normal = np.flatnonzero(labels == 0)
    anomalous = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for baseline_name in ("rgb_statistics", "resnet_global", "spatial_nn"):
        baseline = scores[baseline_name]
        baseline_prediction = baseline >= float(methods[baseline_name]["threshold"])
        baseline_correct = baseline_prediction == labels.astype(bool)
        candidate_only = int(np.sum(candidate_correct & ~baseline_correct))
        baseline_only = int(np.sum(~candidate_correct & baseline_correct))
        differences = np.empty(bootstrap_draws, dtype=np.float64)
        for draw in range(bootstrap_draws):
            sample = np.concatenate(
                (
                    rng.choice(normal, size=len(normal), replace=True),
                    rng.choice(anomalous, size=len(anomalous), replace=True),
                )
            )
            differences[draw] = roc_auc_score(labels[sample], candidate[sample]) - roc_auc_score(
                labels[sample], baseline[sample]
            )
        rows.append(
            {
                "candidate": candidate_name,
                "baseline": baseline_name,
                "paired_test_images": len(labels),
                "auroc_difference": float(
                    methods[candidate_name]["auroc"] - methods[baseline_name]["auroc"]
                ),
                "auroc_difference_ci95": [
                    float(value) for value in np.quantile(differences, [0.025, 0.975])
                ],
                "candidate_correct_baseline_wrong": candidate_only,
                "candidate_wrong_baseline_correct": baseline_only,
                "mcnemar_exact_p": _mcnemar_exact(candidate_only, baseline_only),
            }
        )
    return _holm_adjust(rows, key="mcnemar_exact_p")


def run_stripped_wire_experiment(
    data_root: Path,
    zip_path: Path,
    output_dir: Path,
    *,
    device_name: str = "cuda",
    config: InspectPerceptionConfig | None = None,
) -> dict:
    """Fit on released normal images and evaluate the untouched released test split."""

    config = config or InspectPerceptionConfig(
        fit_good_images=160,
        preserve_aspect_ratio=True,
    )
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_stripped_wire(data_root, output_dir / "public_data", zip_path=zip_path)
    training, test_paths, defect_labels = _load_wire_index(data_root)
    if not 2 <= config.fit_good_images < len(training):
        raise ValueError("fit_good_images must leave normal calibration images")
    permutation = np.random.default_rng(config.seed).permutation(len(training))
    fit_paths = [training[int(index)] for index in permutation[: config.fit_good_images]]
    calibration_paths = [training[int(index)] for index in permutation[config.fit_good_images :]]
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
            "claim_scope": "offline stripped-wire anomaly detection; no robot control",
        },
    )
    checkpoint = output_dir / "stripped_wire_spatial_gaussian.pt"
    detector.save(checkpoint)

    spatial_calibration, _ = detector.score_paths(
        calibration_paths,
        device_name=device_name,
    )
    inference_started = time.perf_counter()
    spatial_scores, spatial_maps = detector.score_paths(
        test_paths,
        device_name=device_name,
        return_maps=True,
    )
    inference_seconds = time.perf_counter() - inference_started
    assert spatial_maps is not None

    rgb_model = LedoitWolf().fit(_rgb_statistics(fit_paths, config))
    rgb_calibration = _mahalanobis_scores(
        rgb_model,
        _rgb_statistics(calibration_paths, config),
    )
    rgb_scores = _mahalanobis_scores(rgb_model, _rgb_statistics(test_paths, config))

    global_model = LedoitWolf().fit(
        _global_embeddings(detector, fit_paths, device_name=device_name)
    )
    global_calibration = _mahalanobis_scores(
        global_model,
        _global_embeddings(detector, calibration_paths, device_name=device_name),
    )
    global_scores = _mahalanobis_scores(
        global_model,
        _global_embeddings(detector, test_paths, device_name=device_name),
    )

    fit_features = detector._feature_batch(fit_paths, device_name=device_name, view_id=0)
    calibration_features = detector._feature_batch(
        calibration_paths,
        device_name=device_name,
        view_id=0,
    )
    test_features = detector._feature_batch(test_paths, device_name=device_name, view_id=0)
    nn_calibration, _ = _spatial_nn_scores(
        fit_features,
        calibration_features,
        config,
        device_name=device_name,
    )
    nn_scores, nn_maps = _spatial_nn_scores(
        fit_features,
        test_features,
        config,
        device_name=device_name,
    )

    labels = np.asarray([label != "good" for label in defect_labels], dtype=np.int64)
    score_arrays = {
        "rgb_statistics": rgb_scores,
        "resnet_global": global_scores,
        "spatial_nn": nn_scores,
        "spatial_gaussian": spatial_scores,
    }
    calibration_arrays = {
        "rgb_statistics": rgb_calibration,
        "resnet_global": global_calibration,
        "spatial_nn": nn_calibration,
        "spatial_gaussian": spatial_calibration,
    }
    methods = {
        name: _method_report(
            labels,
            score_arrays[name],
            calibration_arrays[name],
            config,
            seed_offset=index + 31,
        )
        for index, name in enumerate(score_arrays)
    }
    for name, scores in score_arrays.items():
        methods[name]["per_defect"] = _per_defect_report(
            defect_labels,
            scores,
            float(methods[name]["threshold"]),
        )

    predictions_path = output_dir / "stripped_wire_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "relative_path",
            "defect_type",
            "is_anomaly",
            *[f"{name}_score" for name in score_arrays],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(test_paths):
            writer.writerow(
                {
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "defect_type": defect_labels[index],
                    "is_anomaly": int(labels[index]),
                    **{f"{name}_score": float(scores[index]) for name, scores in score_arrays.items()},
                }
            )
    maps_path = output_dir / "stripped_wire_spatial_maps.npz"
    np.savez_compressed(
        maps_path,
        relative_paths=np.asarray(
            [path.relative_to(data_root).as_posix() for path in test_paths]
        ),
        spatial_gaussian_scores=spatial_scores,
        spatial_gaussian_maps=spatial_maps,
        spatial_nn_scores=nn_scores,
        spatial_nn_maps=nn_maps,
    )
    paired = _paired_method_comparisons(
        labels,
        methods,
        score_arrays,
        seed=config.seed + 4_001,
    )
    report = {
        "schema_version": 1,
        "name": "InspectBot independent Stripped Wire validation",
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
        "paired_comparisons": paired,
        "spatial_gaussian_test_inference": {
            "images": len(test_paths),
            "wall_seconds": inference_seconds,
            "milliseconds_per_image": 1000.0 * inference_seconds / len(test_paths),
            "scope": "end-to-end image loading, preprocessing, GPU inference, and scoring",
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "predictions_csv": str(predictions_path.resolve()),
        "predictions_sha256": file_sha256(predictions_path),
        "maps_npz": str(maps_path.resolve()),
        "maps_sha256": file_sha256(maps_path),
        "claim_scope": (
            "Offline real-image validation on controlled stripped-wire crops. The normal model is "
            "refit on this dataset; this is independent task replication, not zero-shot transfer."
        ),
    }
    report_path = output_dir / "stripped_wire_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
