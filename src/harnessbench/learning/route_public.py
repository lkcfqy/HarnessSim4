"""Leakage-safe offline validation on the public Berkeley cable-routing data."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harnessbench.evaluation import regression_metrics
from harnessbench.lerobot import TrajectoryTable, load_trajectories
from harnessbench.ridge import RidgePolicy
from harnessbench.route_public_data import DATASET_ID, REVISION, validate_route_table


def _episode_split(
    episode_ids: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if unique.size < 10:
        raise ValueError("At least ten episodes are required")
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test fraction is required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_end = round(len(shuffled) * train_fraction)
    validation_end = train_end + round(len(shuffled) * validation_fraction)
    return (
        np.sort(shuffled[:train_end]),
        np.sort(shuffled[train_end:validation_end]),
        np.sort(shuffled[validation_end:]),
    )


def _features(table: TrajectoryTable) -> np.ndarray:
    velocity = np.zeros_like(table.state, dtype=np.float64)
    same_episode = table.episode[1:] == table.episode[:-1]
    velocity[1:][same_episode] = table.state[1:][same_episode] - table.state[:-1][same_episode]
    phase = np.zeros((len(table.state), 1), dtype=np.float64)
    for episode_id in table.episodes:
        mask = table.episode == episode_id
        frame = table.frame[mask].astype(np.float64)
        phase[mask, 0] = (frame - frame.min()) / max(float(frame.max() - frame.min()), 1.0)
    return np.concatenate((table.state, velocity, phase, phase**2), axis=1)


def _mask(ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    return np.isin(ids, selected)


def _previous_action_baseline(
    action: np.ndarray,
    episode: np.ndarray,
    train_mean: np.ndarray,
) -> np.ndarray:
    predicted = np.repeat(train_mean[None, :], len(action), axis=0)
    same_episode = episode[1:] == episode[:-1]
    predicted[1:][same_episode] = action[:-1][same_episode]
    return predicted


def _previous_action_feature(action: np.ndarray, episode: np.ndarray) -> np.ndarray:
    previous = np.zeros_like(action, dtype=np.float64)
    same_episode = episode[1:] == episode[:-1]
    previous[1:][same_episode] = action[:-1][same_episode]
    return previous


def _episode_balanced_rmse_ci(
    true: np.ndarray,
    predicted: np.ndarray,
    episode: np.ndarray,
    active_dims: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 2_000,
) -> dict:
    unique = np.unique(episode)
    per_episode_mse = np.asarray(
        [
            np.mean((predicted[episode == value][:, active_dims] - true[episode == value][:, active_dims]) ** 2)
            for value in unique
        ],
        dtype=np.float64,
    )
    point = float(np.sqrt(np.mean(per_episode_mse)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(bootstrap_samples, len(unique)))
    distribution = np.sqrt(np.mean(per_episode_mse[draws], axis=1))
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "episode_balanced_rmse": point,
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "whole held-out episode",
    }


def _paired_episode_comparison(
    true: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    episode: np.ndarray,
    active_dims: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict:
    unique = np.unique(episode)
    reference_rmse = []
    candidate_rmse = []
    for value in unique:
        mask = episode == value
        reference_rmse.append(
            np.sqrt(np.mean((reference[mask][:, active_dims] - true[mask][:, active_dims]) ** 2))
        )
        candidate_rmse.append(
            np.sqrt(np.mean((candidate[mask][:, active_dims] - true[mask][:, active_dims]) ** 2))
        )
    reference_rmse = np.asarray(reference_rmse)
    candidate_rmse = np.asarray(candidate_rmse)
    delta = candidate_rmse - reference_rmse
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(bootstrap_samples, len(unique)))
    distribution = np.mean(delta[draws], axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    tolerance = 1e-12
    wins = int(np.sum(delta < -tolerance))
    losses = int(np.sum(delta > tolerance))
    ties = int(len(delta) - wins - losses)
    sign_trials = wins + losses
    smaller_side = min(wins, losses)
    one_tail = sum(math.comb(sign_trials, index) for index in range(smaller_side + 1)) / (
        2**sign_trials
    )
    return {
        "metric": "per-episode active-dimension RMSE",
        "candidate_minus_reference_mean": float(delta.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "relative_reduction_fraction": float(1.0 - candidate_rmse.mean() / reference_rmse.mean()),
        "candidate_wins": wins,
        "reference_wins": losses,
        "ties": ties,
        "two_sided_exact_sign_test_p": float(min(1.0, 2.0 * one_tail)),
        "bootstrap_samples": bootstrap_samples,
        "paired_unit": "whole held-out episode",
    }


def _metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
    active_dims: np.ndarray,
    episode: np.ndarray,
    *,
    seed: int,
) -> dict:
    return {
        "all_dimensions": regression_metrics(true, predicted, scale),
        "active_dimensions": regression_metrics(
            true[:, active_dims], predicted[:, active_dims], scale[active_dims]
        ),
        "active_dimension_indices": [int(value) for value in active_dims],
        **_episode_balanced_rmse_ci(
            true,
            predicted,
            episode,
            active_dims,
            seed=seed,
        ),
    }


def _write_latex_table(path: Path, report: dict) -> None:
    rows = []
    for label, key in (
        ("Zero", "zero_action"),
        ("Train mean", "train_mean_action"),
        ("Persistence", "previous_action"),
        ("State ridge", "state_ridge"),
        ("State+hist. ridge", "temporal_ridge"),
    ):
        metrics = report["test_metrics"][key]
        active = metrics["active_dimensions"]
        lower, upper = metrics["bootstrap_95_ci"]
        rows.append(
            f"{label} & {active['rmse']:.4f} & {active['normalized_rmse']:.3f} & "
            f"{metrics['episode_balanced_rmse']:.4f} "
            f"({lower:.4f},{upper:.4f}) \\\\"
        )
    table = "\n".join(
        (
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Policy & RMSE $\\downarrow$ & NRMSE $\\downarrow$ & Ep. RMSE (95\\% CI) \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        )
    )
    path.write_text(table + "\n", encoding="utf-8")


def evaluate_route_public_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int = 202_613,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> dict:
    """Fit a small auditable baseline and evaluate only on held-out real episodes."""

    table = load_trajectories(data_dir)
    dataset = validate_route_table(table)
    x = _features(table)
    temporal_x = np.concatenate(
        (x, _previous_action_feature(table.action, table.episode)), axis=1
    )
    train_ids, validation_ids, test_ids = _episode_split(table.episodes, seed=seed)
    train_mask = _mask(table.episode, train_ids)
    validation_mask = _mask(table.episode, validation_ids)
    test_mask = _mask(table.episode, test_ids)
    if np.any(train_mask & validation_mask) or np.any(train_mask & test_mask):
        raise AssertionError("Episode leakage detected")

    action_scale = np.maximum(table.action[train_mask].std(axis=0), 1e-8)
    active_dims = np.flatnonzero(action_scale > 1e-6)
    if active_dims.size == 0:
        raise ValueError("No varying action dimensions found")

    validation_scores: dict[str, dict[str, float]] = {}
    selected_alphas: dict[str, float] = {}
    for name, candidate_x in (("state_ridge", x), ("temporal_ridge", temporal_x)):
        model_scores: dict[str, float] = {}
        for alpha in alphas:
            candidate = RidgePolicy(alpha=alpha).fit(
                candidate_x[train_mask], table.action[train_mask]
            )
            prediction = candidate.predict_delta(candidate_x[validation_mask])
            score = regression_metrics(
                table.action[validation_mask][:, active_dims],
                prediction[:, active_dims],
                action_scale[active_dims],
            )["normalized_rmse"]
            model_scores[f"{alpha:g}"] = float(score)
        validation_scores[name] = model_scores
        selected_alphas[name] = float(
            min(alphas, key=lambda value: model_scores[f"{value:g}"])
        )

    fit_mask = train_mask | validation_mask
    model = RidgePolicy(alpha=selected_alphas["state_ridge"]).fit(
        x[fit_mask], table.action[fit_mask]
    )
    temporal_model = RidgePolicy(alpha=selected_alphas["temporal_ridge"]).fit(
        temporal_x[fit_mask], table.action[fit_mask]
    )
    ridge_prediction = model.predict_delta(x[test_mask])
    temporal_prediction = temporal_model.predict_delta(temporal_x[test_mask])
    train_mean = table.action[fit_mask].mean(axis=0)
    mean_prediction = np.repeat(train_mean[None, :], test_mask.sum(), axis=0)
    zero_prediction = np.zeros_like(mean_prediction)
    previous_prediction = _previous_action_baseline(
        table.action[test_mask], table.episode[test_mask], train_mean
    )
    true = table.action[test_mask]
    test_episode = table.episode[test_mask]
    test_metrics = {
        "zero_action": _metrics(
            true, zero_prediction, action_scale, active_dims, test_episode, seed=seed + 1
        ),
        "train_mean_action": _metrics(
            true, mean_prediction, action_scale, active_dims, test_episode, seed=seed + 2
        ),
        "previous_action": _metrics(
            true, previous_prediction, action_scale, active_dims, test_episode, seed=seed + 3
        ),
        "state_ridge": _metrics(
            true, ridge_prediction, action_scale, active_dims, test_episode, seed=seed + 4
        ),
        "temporal_ridge": _metrics(
            true, temporal_prediction, action_scale, active_dims, test_episode, seed=seed + 5
        ),
    }

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "RouteBot public real-robot offline action-prediction audit",
        "dataset_id": DATASET_ID,
        "revision": REVISION,
        "dataset": dataset,
        "split": {
            "unit": "whole episode",
            "seed": seed,
            "train_episodes": [int(value) for value in train_ids],
            "validation_episodes": [int(value) for value in validation_ids],
            "test_episodes": [int(value) for value in test_ids],
            "train_frames": int(train_mask.sum()),
            "validation_frames": int(validation_mask.sum()),
            "test_frames": int(test_mask.sum()),
            "episode_overlap": [],
        },
        "features": {
            "input_dim": int(x.shape[1]),
            "temporal_input_dim": int(temporal_x.shape[1]),
            "description": [
                "current 8-dimensional robot state",
                "within-episode state difference",
                "normalized episode phase and squared phase",
                "temporal model additionally receives the previously executed action",
            ],
            "target": "raw 7-dimensional Cartesian action",
        },
        "model_selection": {
            "criterion": "validation normalized RMSE on active action dimensions",
            "candidate_alphas": [float(value) for value in alphas],
            "validation_scores": validation_scores,
            "selected_alphas": selected_alphas,
        },
        "test_metrics": test_metrics,
        "paired_comparisons": {
            "temporal_ridge_vs_previous_action": _paired_episode_comparison(
                true,
                previous_prediction,
                temporal_prediction,
                test_episode,
                active_dims,
                seed=seed + 6,
            )
        },
        "claim_boundary": (
            "Held-out offline action prediction on real teleoperated cable-routing episodes. "
            "The public data has no RouteBot relation intervention or standardized closed-loop "
            "semantic-success label, so this result does not validate semantic routing, robot "
            "success rate, or production readiness."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "state_ridge.npz")
    temporal_model.save(output_dir / "temporal_ridge.npz")
    np.savez_compressed(
        output_dir / "held_out_predictions.npz",
        episode=test_episode,
        frame=table.frame[test_mask],
        true_action=true.astype(np.float32),
        ridge_action=ridge_prediction.astype(np.float32),
        temporal_ridge_action=temporal_prediction.astype(np.float32),
        previous_action=previous_prediction.astype(np.float32),
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split.json").write_text(
        json.dumps(report["split"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_latex_table(output_dir / "route_public_table.tex", report)
    return report
