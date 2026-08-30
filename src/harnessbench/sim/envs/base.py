"""Minimal Gym-like interface shared by all four robot tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    robot: str
    policy: str
    seed: int
    difficulty: float
    success: bool
    total_reward: float
    steps: int
    metrics: dict[str, float | int | bool]
    frames: tuple[Image.Image, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "robot": self.robot,
            "policy": self.policy,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "success": self.success,
            "total_reward": self.total_reward,
            "steps": self.steps,
            **self.metrics,
        }


class HarnessEnv(ABC):
    task_name = "base"
    robot_name = "BaseBot"
    max_steps = 200
    dt = 0.025
    action_dim = 3

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.seed = 0
        self.difficulty = 0.5
        self.step_count = 0
        self.total_reward = 0.0
        self._terminated = False
        self._success = False

    def reset(self, seed: int = 0, difficulty: float = 0.5) -> tuple[dict, dict]:
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        self.seed = int(seed)
        self.difficulty = float(difficulty)
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.total_reward = 0.0
        self._terminated = False
        self._success = False
        self._reset_task()
        return self.observation(), self.info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("step called after episode termination; call reset")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"action must be finite with shape ({self.action_dim},)")
        reward = float(self._step_task(action))
        self.step_count += 1
        self.total_reward += reward
        self._success = bool(self.success())
        self._terminated = self._success
        truncated = self.step_count >= self.max_steps and not self._terminated
        return self.observation(), reward, self._terminated, truncated, self.info()

    def info(self) -> dict:
        return {
            "task": self.task_name,
            "robot": self.robot_name,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "step": self.step_count,
            "success": bool(self._success),
            **self.metrics(),
        }

    @abstractmethod
    def _reset_task(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _step_task(self, action: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def metrics(self) -> dict[str, float | int | bool]:
        raise NotImplementedError

    @abstractmethod
    def success(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def policy_action(self, policy_name: str, policy_rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def render(self, width: int = 640, height: int = 480) -> Image.Image:
        raise NotImplementedError
