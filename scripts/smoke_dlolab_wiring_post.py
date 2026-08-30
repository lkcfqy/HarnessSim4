#!/usr/bin/env python3
"""Run a minimal, provenance-recorded smoke test in official DLO-Lab."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

WORKSPACE = Path(__file__).resolve().parents[1]
DLO_REPOSITORY = WORKSPACE / "third_party" / "DLO-Lab"
MUSHROOM_REPOSITORY = WORKSPACE / "third_party" / "mushroom-rl"
EXPERIMENTS = DLO_REPOSITORY / "experiments"
SOURCE = WORKSPACE / "src"
for source_path in (SOURCE, MUSHROOM_REPOSITORY, EXPERIMENTS):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from harnessbench.learning.dlolab_external import (
    file_sha256,
    wiring_post_curve_metrics,
)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _finite_summary(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "all_finite": bool(np.isfinite(array).all()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "artifacts" / "papers" / "routebot" / "final" / "dlolab_external",
    )
    parser.add_argument("--seed", type=int, default=63000000)
    parser.add_argument("--scene-steps-per-action", type=int, default=2)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = (
        DLO_REPOSITORY
        / "genesis"
        / "assets"
        / "dlo-lab"
        / "target_pos"
        / "wiring_post_finalpos.npy"
    )
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing official DLO-Lab target: {target_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Official DLO-Lab smoke test requires a CUDA-capable PyTorch build")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    from envs.registry import get_env_class

    config = DictConfig(
        {
            "task": "wiring_post",
            "log_dir": str(output / "official_logs"),
            "n_envs": 1,
            "n_substeps_per_step": args.scene_steps_per_action,
            "GUI": False,
            "camera": False,
            "raytracer": False,
        }
    )
    environment = get_env_class("wiring_post")(config=config)
    environment.init_rl_env(
        n_steps=2,
        pos_bound=0.01,
        angle_bound=0.1,
        n_additional_obj=2,
        steps_interval_split=1,
    )
    environment.reset()

    initial_vertices = np.asarray(environment.rope.get_all_verts()[0], dtype=np.float64)
    target = np.asarray(environment.target_pos, dtype=np.float64)
    fixtures = np.asarray([[0.28, 0.14, 0.02], [0.10, 0.275, 0.02]], dtype=np.float64)
    initial_observation = environment.compute_observation().detach().cpu().numpy()
    initial_reward = np.asarray(environment.reward(), dtype=np.float64)

    active = torch.ones((1,), dtype=torch.bool)
    zero_action = torch.zeros((1, environment._act_dim), dtype=torch.float32)
    next_observation, step_reward, absorbing, _ = environment.step_all(active, zero_action)
    final_vertices = np.asarray(environment.rope.get_all_verts()[0], dtype=np.float64)

    arrays_path = output / "wiring_post_smoke_arrays.npz"
    np.savez_compressed(
        arrays_path,
        initial_vertices=initial_vertices,
        final_vertices=final_vertices,
        target=target,
        fixtures=fixtures,
        initial_observation=initial_observation,
        next_observation=next_observation.detach().cpu().numpy(),
    )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "official DLO-Lab wiring_post reset plus one zero-action step",
        "routebot_extension_note": (
            "Ordered cable-to-fixture metrics are a RouteBot audit extension; "
            "they are not part of DLO-Lab's native reward."
        ),
        "seed": args.seed,
        "dlo_lab": {
            "repository": "https://github.com/UMass-Embodied-AGI/DLO-Lab",
            "commit": _git_head(DLO_REPOSITORY),
            "source_license": "Apache-2.0",
            "target_path": str(target_path.relative_to(WORKSPACE)),
            "target_sha256": file_sha256(target_path),
            "asset_redistribution_status": "local research use only; archive contained no license",
        },
        "mushroom_rl": {
            "repository": "https://github.com/XJay18/mushroom-rl",
            "commit": _git_head(MUSHROOM_REPOSITORY),
            "source_license": "MIT",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0),
            "packages": {
                name: _package_version(name)
                for name in (
                    "genesis-world",
                    "mushroom-rl",
                    "numpy",
                    "omegaconf",
                    "quadrants",
                )
            },
        },
        "environment": {
            "n_envs": 1,
            "scene_steps_per_action": args.scene_steps_per_action,
            "action_dimension": int(environment._act_dim),
            "observation_dimension": int(environment._obs_dim),
        },
        "initial": {
            "observation": _finite_summary(initial_observation),
            "vertices": _finite_summary(initial_vertices),
            "native_reward": float(initial_reward[0]),
            "metrics": wiring_post_curve_metrics(initial_vertices, target, fixtures),
        },
        "after_zero_action": {
            "observation": _finite_summary(next_observation.detach().cpu().numpy()),
            "vertices": _finite_summary(final_vertices),
            "native_reward": float(step_reward.detach().cpu().numpy()[0]),
            "absorbing": bool(absorbing.detach().cpu().numpy()[0]),
            "metrics": wiring_post_curve_metrics(final_vertices, target, fixtures),
        },
        "arrays": {
            "path": str(arrays_path.relative_to(WORKSPACE)),
            "sha256": file_sha256(arrays_path),
        },
    }
    report_path = output / "wiring_post_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
