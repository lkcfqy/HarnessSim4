"""End-to-end training and held-out evaluation for the public proxy dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harnessbench.evaluation import regression_metrics
from harnessbench.features import build_features, select_episodes, split_episode_ids
from harnessbench.lerobot import dataset_summary, load_trajectories
from harnessbench.ridge import RidgePolicy


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def train_public_baseline(
    data_dir: Path,
    artifact_dir: Path,
    train_fraction: float = 0.8,
    seed: int = 202609,
    alpha: float = 0.01,
) -> dict:
    table = load_trajectories(data_dir)
    if not np.isfinite(table.state).all() or not np.isfinite(table.action).all():
        raise ValueError("The dataset contains NaN or infinite state/action values")

    all_data = build_features(table)
    train_ids, test_ids = split_episode_ids(table.episodes, train_fraction, seed)
    train = select_episodes(all_data, train_ids)
    test = select_episodes(all_data, test_ids)
    if np.intersect1d(train.episode, test.episode).size:
        raise AssertionError("Episode leakage detected")

    policy = RidgePolicy(alpha=alpha).fit(train.x, train.y_delta)
    predicted_delta = policy.predict_delta(test.x)
    predicted_action = test.state + predicted_delta
    identity_action = test.state.copy()
    train_action_scale = np.maximum(train.action.std(axis=0), 1e-8)
    held_out_metrics = regression_metrics(test.action, predicted_action, train_action_scale)
    identity_metrics = regression_metrics(test.action, identity_action, train_action_scale)
    arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    gripper_indices = np.asarray([6, 13])

    provenance: dict = {"manifest_found": False}
    manifest_path = data_dir / "harnessbench_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        provenance = {
            "manifest_found": True,
            "dataset_id": manifest.get("dataset_id"),
            "revision": manifest.get("revision"),
            "license": manifest.get("license"),
            "portable_numpy_cache": manifest.get("portable_numpy_cache"),
            "source_file_sha256": {
                item["path"]: item["sha256"] for item in manifest.get("files", [])
            },
        }

    report = {
        "project": "HarnessBench-Insert",
        "baseline": "ridge_delta_state_v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir.resolve()),
        "provenance": provenance,
        "dataset": dataset_summary(table),
        "split": {
            "method": "seeded whole-episode holdout",
            "seed": seed,
            "train_fraction_requested": train_fraction,
            "train_episodes": [int(value) for value in train_ids],
            "test_episodes": [int(value) for value in test_ids],
            "train_frames": len(train.x),
            "test_frames": len(test.x),
            "episode_overlap": [],
        },
        "features": {
            "input_dim": int(train.x.shape[1]),
            "description": [
                "current 14-dimensional robot state",
                "within-episode 14-dimensional first difference",
                "normalized episode phase",
                "squared normalized episode phase",
            ],
            "target": "14-dimensional action minus current state",
        },
        "model": {"type": "multi-output ridge regression", "alpha": alpha},
        "held_out_metrics": held_out_metrics,
        "identity_baseline_metrics": identity_metrics,
        "held_out_metric_groups": {
            "arm_joints_12d": regression_metrics(
                test.action[:, arm_indices],
                predicted_action[:, arm_indices],
                train_action_scale[arm_indices],
            ),
            "grippers_2d": regression_metrics(
                test.action[:, gripper_indices],
                predicted_action[:, gripper_indices],
                train_action_scale[gripper_indices],
            ),
        },
        "comparison_to_identity": {
            "rmse_reduction_fraction": float(
                1.0 - held_out_metrics["rmse"] / identity_metrics["rmse"]
            ),
            "normalized_rmse_reduction_fraction": float(
                1.0 - held_out_metrics["normalized_rmse"] / identity_metrics["normalized_rmse"]
            ),
        },
        "interpretation": (
            "This is a reproducibility and data-pipeline baseline on a simulated rigid-object "
            "insertion proxy. It is not evidence of wire-terminal insertion success."
        ),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    policy.save(artifact_dir / "model.npz")
    _write_json(artifact_dir / "report.json", report)
    _write_json(
        artifact_dir / "split.json",
        {
            "seed": seed,
            "train_episodes": report["split"]["train_episodes"],
            "test_episodes": report["split"]["test_episodes"],
        },
    )
    return report
