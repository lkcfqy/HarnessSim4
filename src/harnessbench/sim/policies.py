"""Reproducible heuristic baselines for the four-task benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harnessbench.sim.envs.base import HarnessEnv


@dataclass
class Policy:
    name: str
    seed: int = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def act(self, env: HarnessEnv) -> np.ndarray:
        return env.policy_action(self.name, self.rng)


class TopologyPolicy(Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("topology", seed)
        self.reset(seed)


class GeometryPolicy(Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("geometry", seed)
        self.reset(seed)


class RandomPolicy(Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__("random", seed)
        self.reset(seed)


POLICY_REGISTRY = {
    "topology": TopologyPolicy,
    "geometry": GeometryPolicy,
    "random": RandomPolicy,
}
