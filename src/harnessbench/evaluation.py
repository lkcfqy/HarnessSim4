"""Offline metrics that avoid conflating imitation error with robot success."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    true_action: np.ndarray,
    predicted_action: np.ndarray,
    action_scale: np.ndarray,
) -> dict:
    error = predicted_action - true_action
    mae_per_dim = np.mean(np.abs(error), axis=0)
    rmse_per_dim = np.sqrt(np.mean(error**2, axis=0))
    scale = np.maximum(np.asarray(action_scale), 1e-8)
    residual_sum = np.sum(error**2, axis=0)
    centered = true_action - true_action.mean(axis=0)
    total_sum = np.sum(centered**2, axis=0)
    r2 = 1.0 - residual_sum / np.maximum(total_sum, 1e-12)

    return {
        "mae": float(np.mean(mae_per_dim)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.mean(rmse_per_dim / scale)),
        "macro_r2": float(np.mean(r2)),
        "mae_per_action_dim": [float(value) for value in mae_per_dim],
        "rmse_per_action_dim": [float(value) for value in rmse_per_dim],
        "r2_per_action_dim": [float(value) for value in r2],
        "note": (
            "Offline action regression only. These numbers are not insertion success rate and "
            "must not be used as a robot safety claim."
        ),
    }
