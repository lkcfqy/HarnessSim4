"""Dependency-light multi-output ridge regression behavior-cloning baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPSILON = 1e-8


@dataclass
class RidgePolicy:
    alpha: float = 0.01
    x_mean: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    y_scale: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgePolicy:
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x and y must be rank-2 arrays with matching rows")
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), EPSILON)
        self.y_mean = y.mean(axis=0)
        self.y_scale = np.maximum(y.std(axis=0), EPSILON)
        xn = (x - self.x_mean) / self.x_scale
        yn = (y - self.y_mean) / self.y_scale
        design = np.concatenate((xn, np.ones((len(xn), 1))), axis=1)
        penalty = np.eye(design.shape[1]) * float(self.alpha)
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ yn)
        return self

    def predict_delta(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.x_mean, self.x_scale, self.y_mean, self.y_scale)):
            raise RuntimeError("The policy has not been fitted")
        if self.weights is None:
            raise RuntimeError("The policy has not been fitted")
        xn = (x - self.x_mean) / self.x_scale
        design = np.concatenate((xn, np.ones((len(xn), 1))), axis=1)
        return (design @ self.weights) * self.y_scale + self.y_mean

    def save(self, path: Path) -> None:
        if self.weights is None:
            raise RuntimeError("Cannot save an unfitted policy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alpha=np.asarray(self.alpha),
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale,
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: Path) -> RidgePolicy:
        with np.load(path) as arrays:
            return cls(
                alpha=float(arrays["alpha"]),
                x_mean=arrays["x_mean"].copy(),
                x_scale=arrays["x_scale"].copy(),
                y_mean=arrays["y_mean"].copy(),
                y_scale=arrays["y_scale"].copy(),
                weights=arrays["weights"].copy(),
            )
