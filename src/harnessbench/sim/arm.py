"""Kinematic planar arms used for visualization and action-rate limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlanarArm:
    base: np.ndarray
    ee: np.ndarray
    link_lengths: tuple[float, float] = (0.34, 0.30)
    max_speed: float = 0.72
    elbow_up: bool = True
    name: str = "arm"

    def __post_init__(self) -> None:
        self.base = np.asarray(self.base, dtype=np.float64).copy()
        self.ee = np.asarray(self.ee, dtype=np.float64).copy()

    def move_toward(self, target: np.ndarray, dt: float) -> None:
        target = np.asarray(target, dtype=np.float64)
        delta = target - self.ee
        distance = float(np.linalg.norm(delta))
        if distance <= self.max_speed * dt or distance < 1e-12:
            self.ee = target.copy()
        else:
            self.ee += delta / distance * self.max_speed * dt

    def joints(self) -> np.ndarray:
        """Return base, elbow and reachable end-effector for rendering."""

        relative = self.ee - self.base
        distance = float(np.linalg.norm(relative))
        l1, l2 = self.link_lengths
        reachable = np.clip(distance, abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
        direction = relative / distance if distance > 1e-9 else np.asarray([1.0, 0.0])
        rendered_ee = self.base + direction * reachable
        cos_elbow = np.clip((reachable**2 - l1**2 - l2**2) / (2 * l1 * l2), -1, 1)
        elbow_angle = float(np.arccos(cos_elbow)) * (1.0 if self.elbow_up else -1.0)
        base_angle = float(np.arctan2(direction[1], direction[0])) - float(
            np.arctan2(l2 * np.sin(elbow_angle), l1 + l2 * np.cos(elbow_angle))
        )
        elbow = self.base + l1 * np.asarray([np.cos(base_angle), np.sin(base_angle)])
        return np.stack((self.base, elbow, rendered_ee))
