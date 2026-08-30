"""Environment registry for the four simulated robots."""

from __future__ import annotations

from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.envs.branch import BranchEnv
from harnessbench.sim.envs.insert import InsertEnv
from harnessbench.sim.envs.inspect import InspectEnv
from harnessbench.sim.envs.route import RouteEnv

ENV_REGISTRY: dict[str, type[HarnessEnv]] = {
    "inspect": InspectEnv,
    "insert": InsertEnv,
    "route": RouteEnv,
    "branch": BranchEnv,
}


def make_env(name: str) -> HarnessEnv:
    try:
        return ENV_REGISTRY[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(ENV_REGISTRY))
        raise KeyError(f"Unknown task {name!r}; choose from {choices}") from exc


__all__ = [
    "ENV_REGISTRY",
    "BranchEnv",
    "HarnessEnv",
    "InsertEnv",
    "InspectEnv",
    "RouteEnv",
    "make_env",
]
