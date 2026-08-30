"""Deterministic behavior-cloning trainer for TopoHarnessNet."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from harnessbench.learning.dataset import SPLIT_TO_ID, load_expert_dataset
from harnessbench.learning.model import ModelConfig, TopoHarnessNet


class GraphArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"dataset arrays have inconsistent lengths: {lengths}")

    def __len__(self) -> int:
        return len(self.arrays["task_id"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(value[index])
            if isinstance(value[index], np.ndarray)
            else torch.as_tensor(value[index])
            for key, value in self.arrays.items()
        }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    float_keys = {
        "node_features",
        "physical_adjacency",
        "semantic_adjacency",
        "node_mask",
        "global_features",
        "action",
        "action_mask",
        "binary_action_mask",
        "pointer_mask",
    }
    return {
        key: value.to(device=device, dtype=torch.float32 if key in float_keys else torch.long)
        for key, value in batch.items()
    }


def _forward(
    model: TopoHarnessNet, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.forward_with_pointers(
        batch["node_features"],
        batch["physical_adjacency"],
        batch["semantic_adjacency"],
        batch["node_mask"],
        batch["global_features"],
        batch["task_id"],
    )


def _loss(
    logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    target: torch.Tensor,
    action_mask: torch.Tensor,
    binary_mask: torch.Tensor,
    task_id: torch.Tensor,
    binary_pos_weight: torch.Tensor,
    pointer_target: torch.Tensor,
    pointer_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = torch.sigmoid(logits)
    continuous_mask = torch.clamp(action_mask - binary_mask, 0.0, 1.0)
    mse = torch.sum((prediction - target) ** 2 * continuous_mask) / continuous_mask.sum().clamp_min(
        1.0
    )
    positive_weight = binary_pos_weight[task_id]
    bce_values = -(
        positive_weight * target * nn.functional.logsigmoid(logits)
        + (1.0 - target) * nn.functional.logsigmoid(-logits)
    )
    bce = torch.sum(bce_values * binary_mask) / binary_mask.sum().clamp_min(1.0)
    pointer_losses = nn.functional.cross_entropy(
        pointer_logits.reshape(-1, pointer_logits.shape[-1]),
        pointer_target.reshape(-1),
        reduction="none",
    ).reshape(pointer_mask.shape)
    pointer_loss = torch.sum(pointer_losses * pointer_mask) / pointer_mask.sum().clamp_min(1.0)
    total = mse + 0.25 * bce + 0.12 * pointer_loss
    return total, {
        "coordinate_mse": float(mse.detach()),
        "binary_bce": float(bce.detach()),
        "pointer_cross_entropy": float(pointer_loss.detach()),
    }


def _binary_positive_weights(
    arrays: dict[str, np.ndarray], train_indices: list[int]
) -> torch.Tensor:
    weights = np.ones((4, arrays["action"].shape[1]), dtype=np.float32)
    for task_id in range(4):
        task_rows = np.asarray(
            [index for index in train_indices if arrays["task_id"][index] == task_id],
            dtype=np.int64,
        )
        if not len(task_rows):
            continue
        for dimension in range(arrays["action"].shape[1]):
            active = arrays["binary_action_mask"][task_rows, dimension] > 0.0
            values = arrays["action"][task_rows[active], dimension]
            positives = float(np.sum(values >= 0.5))
            negatives = float(np.sum(values < 0.5))
            if positives > 0.0 and negatives > 0.0:
                weights[task_id, dimension] = min(negatives / positives, 20.0)
    return torch.from_numpy(weights)


@torch.no_grad()
def evaluate_loader(
    model: TopoHarnessNet,
    loader: DataLoader,
    device: torch.device,
    binary_pos_weight: torch.Tensor,
) -> dict:
    model.eval()
    totals = {
        "samples": 0,
        "loss_sum": 0.0,
        "absolute_error_sum": 0.0,
        "active_dimensions": 0.0,
        "binary_correct": 0.0,
        "binary_dimensions": 0.0,
        "binary_true_positive": 0.0,
        "binary_positive": 0.0,
        "binary_true_negative": 0.0,
        "binary_negative": 0.0,
        "pointer_correct": 0.0,
        "pointer_events": 0.0,
    }
    per_task = {
        index: {
            "samples": 0,
            "error": 0.0,
            "dimensions": 0.0,
            "loss_sum": 0.0,
        }
        for index in range(4)
    }
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        logits, pointer_logits = _forward(model, batch)
        loss, _ = _loss(
            logits,
            pointer_logits,
            batch["action"],
            batch["action_mask"],
            batch["binary_action_mask"],
            batch["task_id"],
            binary_pos_weight,
            batch["pointer_target"],
            batch["pointer_mask"],
        )
        prediction = torch.sigmoid(logits)
        absolute = torch.abs(prediction - batch["action"]) * batch["action_mask"]
        binary_prediction = prediction >= 0.5
        binary_target = batch["action"] >= 0.5
        binary_correct = (binary_prediction == binary_target).float() * batch["binary_action_mask"]
        binary_positive = binary_target.float() * batch["binary_action_mask"]
        binary_negative = (~binary_target).float() * batch["binary_action_mask"]
        pointer_prediction = torch.argmax(pointer_logits, dim=-1)
        pointer_correct = (pointer_prediction == batch["pointer_target"]).float() * batch[
            "pointer_mask"
        ]
        batch_size = len(batch["task_id"])
        totals["samples"] += batch_size
        totals["loss_sum"] += float(loss) * batch_size
        totals["absolute_error_sum"] += float(absolute.sum())
        totals["active_dimensions"] += float(batch["action_mask"].sum())
        totals["binary_correct"] += float(binary_correct.sum())
        totals["binary_dimensions"] += float(batch["binary_action_mask"].sum())
        totals["binary_true_positive"] += float((binary_prediction.float() * binary_positive).sum())
        totals["binary_positive"] += float(binary_positive.sum())
        totals["binary_true_negative"] += float(
            ((~binary_prediction).float() * binary_negative).sum()
        )
        totals["binary_negative"] += float(binary_negative.sum())
        totals["pointer_correct"] += float(pointer_correct.sum())
        totals["pointer_events"] += float(batch["pointer_mask"].sum())
        for task_id in range(4):
            selected = batch["task_id"] == task_id
            if torch.any(selected):
                task_samples = int(selected.sum())
                task_loss, _ = _loss(
                    logits[selected],
                    pointer_logits[selected],
                    batch["action"][selected],
                    batch["action_mask"][selected],
                    batch["binary_action_mask"][selected],
                    batch["task_id"][selected],
                    binary_pos_weight,
                    batch["pointer_target"][selected],
                    batch["pointer_mask"][selected],
                )
                per_task[task_id]["samples"] += task_samples
                per_task[task_id]["error"] += float(absolute[selected].sum())
                per_task[task_id]["dimensions"] += float(batch["action_mask"][selected].sum())
                per_task[task_id]["loss_sum"] += float(task_loss) * task_samples
    per_task_loss = {
        str(task_id): values["loss_sum"] / max(values["samples"], 1)
        for task_id, values in per_task.items()
        if values["samples"]
    }
    return {
        "samples": totals["samples"],
        "loss": totals["loss_sum"] / max(totals["samples"], 1),
        "action_mae": totals["absolute_error_sum"] / max(totals["active_dimensions"], 1.0),
        "binary_accuracy": totals["binary_correct"] / max(totals["binary_dimensions"], 1.0),
        "binary_positive_recall": totals["binary_true_positive"]
        / max(totals["binary_positive"], 1.0),
        "binary_negative_recall": totals["binary_true_negative"]
        / max(totals["binary_negative"], 1.0),
        "pointer_accuracy": totals["pointer_correct"] / max(totals["pointer_events"], 1.0),
        "balanced_loss": float(np.mean(list(per_task_loss.values()))),
        "per_task_loss": per_task_loss,
        "per_task_action_mae": {
            str(task_id): values["error"] / max(values["dimensions"], 1.0)
            for task_id, values in per_task.items()
            if values["samples"]
        },
    }


@torch.no_grad()
def calibrate_binary_thresholds(
    model: TopoHarnessNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[list[float]], dict[str, dict]]:
    model.eval()
    probabilities: dict[tuple[int, int], list[float]] = {}
    targets: dict[tuple[int, int], list[int]] = {}
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        logits, _ = _forward(model, batch)
        prediction = torch.sigmoid(logits).cpu().numpy()
        task_ids = batch["task_id"].cpu().numpy()
        target = batch["action"].cpu().numpy()
        mask = batch["binary_action_mask"].cpu().numpy()
        for row, task_id in enumerate(task_ids):
            for dimension in np.flatnonzero(mask[row] > 0.0):
                key = (int(task_id), int(dimension))
                probabilities.setdefault(key, []).append(float(prediction[row, dimension]))
                targets.setdefault(key, []).append(int(target[row, dimension] >= 0.5))

    thresholds = np.full((4, 6), 0.5, dtype=np.float32)
    report: dict[str, dict] = {}
    for key, probability_values in probabilities.items():
        labels = np.asarray(targets[key], dtype=np.int64)
        values = np.asarray(probability_values, dtype=np.float64)
        best_threshold = 0.5
        best_score = -1.0
        if labels.min() != labels.max():
            for threshold in np.linspace(0.05, 0.95, 181):
                predicted = values >= threshold
                positive_recall = float(np.mean(predicted[labels == 1]))
                negative_recall = float(np.mean(~predicted[labels == 0]))
                score = 0.5 * (positive_recall + negative_recall)
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12
                    and abs(threshold - 0.5) < abs(best_threshold - 0.5)
                ):
                    best_score = score
                    best_threshold = float(threshold)
        thresholds[key] = best_threshold
        predicted = values >= best_threshold
        report[f"task_{key[0]}_dim_{key[1]}"] = {
            "threshold": best_threshold,
            "samples": len(values),
            "positive_rate": float(labels.mean()),
            "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": best_score if best_score >= 0.0 else 1.0,
        }
    return thresholds.tolist(), report


def train_policy(
    dataset_path: Path,
    output_path: Path,
    *,
    use_topology: bool,
    seed: int = 202609,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 96,
    message_passing_layers: int = 3,
    adjacency_mode: str = "combined",
    use_topology_features: bool = True,
    patience: int = 12,
    device_name: str = "cpu",
) -> dict:
    arrays = load_expert_dataset(dataset_path)
    dataset = GraphArrayDataset(arrays)
    indices = {
        name: np.flatnonzero(arrays["split"] == split_id).tolist()
        for name, split_id in SPLIT_TO_ID.items()
    }
    if any(not values for values in indices.values()):
        raise ValueError("dataset is missing at least one split")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    generator = torch.Generator().manual_seed(seed)
    train_task_ids = arrays["task_id"][indices["train"]]
    task_counts = np.bincount(train_task_ids, minlength=4)
    sample_weights = np.asarray(
        [1.0 / max(task_counts[task_id], 1) for task_id in train_task_ids],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=len(indices["train"]),
        replacement=True,
        generator=generator,
    )
    loaders = {
        "train": DataLoader(
            Subset(dataset, indices["train"]),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        ),
        "validation": DataLoader(
            Subset(dataset, indices["validation"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            Subset(dataset, indices["test"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }
    binary_pos_weight = _binary_positive_weights(arrays, indices["train"]).to(device)
    config = ModelConfig(
        hidden_dim=hidden_dim,
        message_passing_layers=message_passing_layers,
        use_topology=use_topology,
        adjacency_mode=adjacency_mode,
        use_topology_features=use_topology_features,
    )
    model = TopoHarnessNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        samples = 0
        for raw_batch in loaders["train"]:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, pointer_logits = _forward(model, batch)
            loss, components = _loss(
                logits,
                pointer_logits,
                batch["action"],
                batch["action_mask"],
                batch["binary_action_mask"],
                batch["task_id"],
                binary_pos_weight,
                batch["pointer_target"],
                batch["pointer_mask"],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_samples = len(batch["task_id"])
            loss_sum += float(loss.detach()) * batch_samples
            samples += batch_samples
        validation = evaluate_loader(model, loaders["validation"], device, binary_pos_weight)
        train_loss = loss_sum / max(samples, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_balanced_loss": validation["balanced_loss"],
                "validation_action_mae": validation["action_mae"],
                "last_batch_coordinate_mse": components["coordinate_mse"],
                "last_batch_binary_bce": components["binary_bce"],
                "last_batch_pointer_cross_entropy": components["pointer_cross_entropy"],
            }
        )
        if validation["balanced_loss"] < best_validation - 1e-6:
            best_validation = validation["balanced_loss"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_metrics = evaluate_loader(model, loaders["validation"], device, binary_pos_weight)
    test_metrics = evaluate_loader(model, loaders["test"], device, binary_pos_weight)
    binary_thresholds, calibration_report = calibrate_binary_thresholds(
        model, loaders["validation"], device
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "0.1",
        "model_config": asdict(config),
        "state_dict": best_state,
        "dataset_sha256": _file_hash(dataset_path),
        "seed": seed,
        "best_epoch": best_epoch,
        "binary_thresholds": binary_thresholds,
    }
    torch.save(checkpoint, output_path)
    if not use_topology:
        mode_name = "geometry_ablation"
    elif adjacency_mode != "combined":
        mode_name = f"{adjacency_mode}_adjacency_ablation"
    elif not use_topology_features:
        mode_name = "no_topology_features_ablation"
    else:
        mode_name = "topology"
    report = {
        "name": "TopoHarness behavior-cloning training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(output_path.resolve()),
        "mode": mode_name,
        "model_metadata": model.model_metadata(),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        "binary_positive_weights": binary_pos_weight.detach().cpu().tolist(),
        "binary_threshold_calibration": calibration_report,
        "task_balanced_sampler": True,
        "split_samples": {name: len(values) for name, values in indices.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "claim_scope": "offline imitation metrics only; closed-loop evaluation required",
        "torch_version": torch.__version__,
        "device": str(device),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "model": str(output_path.resolve()),
        "report": str(report_path.resolve()),
        "mode": report["mode"],
        "parameters": report["model_metadata"]["parameters"],
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
    }


def load_policy_model(path: Path, device_name: str = "cpu") -> TopoHarnessNet:
    device = torch.device(device_name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only was added
        checkpoint = torch.load(path, map_location=device)
    config = ModelConfig(**checkpoint["model_config"])
    model = TopoHarnessNet(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.binary_thresholds = checkpoint.get("binary_thresholds", [[0.5] * 6 for _ in range(4)])
    model.eval()
    return model
