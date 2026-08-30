#!/usr/bin/env python3
"""Compare matched open-loop routing strategies in official DLO-Lab."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

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
    resample_polyline_deltas,
    wiring_post_curve_metrics,
)

POLICY_NAMES = (
    "no_action",
    "endpoint_straight",
    "wrong_fixture_order",
    "ordered_fixture_route",
)


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _route_points(target: np.ndarray, *, height: float) -> np.ndarray:
    route = np.array(target[::-1][:-3], dtype=np.float64, copy=True)
    route[:, 2] = height
    return route


def _policy_paths(initial: np.ndarray, target: np.ndarray, height: float) -> dict[str, np.ndarray]:
    initial_point = np.array(initial, dtype=np.float64, copy=True)
    initial_point[2] = max(initial_point[2], height)
    route = _route_points(target, height=height)
    endpoint = route[-1]
    wrong_order_indices = [0, 21, 9, -1]
    wrong_route = route[wrong_order_indices]
    return {
        "no_action": np.stack([initial_point, initial_point]),
        "endpoint_straight": np.stack([initial_point, endpoint]),
        "wrong_fixture_order": np.vstack([initial_point, wrong_route]),
        "ordered_fixture_route": np.vstack([initial_point, route]),
    }


def _make_trajectories(
    paths: dict[str, np.ndarray], max_step_m: float
) -> tuple[np.ndarray, dict[str, int]]:
    xyz_deltas = {
        name: resample_polyline_deltas(paths[name], max_step_m=max_step_m)
        for name in POLICY_NAMES
    }
    lengths = {name: int(values.shape[0]) for name, values in xyz_deltas.items()}
    horizon = max(lengths.values())
    trajectories = np.zeros((len(POLICY_NAMES), horizon, 6), dtype=np.float32)
    for policy_index, name in enumerate(POLICY_NAMES):
        values = xyz_deltas[name]
        trajectories[policy_index, : values.shape[0], :3] = values
    return trajectories, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            WORKSPACE
            / "artifacts"
            / "papers"
            / "routebot"
            / "final"
            / "dlolab_external"
            / "matched_paths"
        ),
    )
    parser.add_argument("--seed", type=int, default=64000000)
    parser.add_argument("--scene-steps-per-action", type=int, default=20)
    parser.add_argument("--max-step-m", type=float, default=0.006)
    parser.add_argument("--grasp-height-m", type=float, default=0.03)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Official DLO-Lab evaluation requires a CUDA-capable PyTorch build")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    from envs.registry import get_env_class

    config = DictConfig(
        {
            "task": "wiring_post",
            "log_dir": str(output / "official_logs"),
            "n_envs": len(POLICY_NAMES),
            "n_substeps_per_step": args.scene_steps_per_action,
            "GUI": False,
            "camera": False,
            "raytracer": False,
        }
    )
    environment = get_env_class("wiring_post")(config=config)
    environment.reset()
    control_index = int(environment.control_idx[0])
    initial_vertices = np.asarray(environment.rope.get_all_verts(), dtype=np.float64)
    initial_grasp = initial_vertices[0, control_index]
    target = np.asarray(environment.target_pos, dtype=np.float64)
    fixtures = np.asarray([[0.28, 0.14, 0.02], [0.10, 0.275, 0.02]], dtype=np.float64)

    paths = _policy_paths(initial_grasp, target, height=args.grasp_height_m)
    trajectories, active_steps = _make_trajectories(paths, max_step_m=args.max_step_m)
    environment.init_cmaes_env(n_steps_sub=1)
    start = perf_counter()
    official_output = environment.eval_traj(trajectories)
    elapsed = perf_counter() - start
    final_vertices = np.asarray(environment.rope.get_all_verts(), dtype=np.float64)

    arrays_path = output / "matched_path_arrays.npz"
    np.savez_compressed(
        arrays_path,
        policy_names=np.asarray(POLICY_NAMES),
        trajectories=trajectories,
        initial_vertices=initial_vertices,
        final_vertices=final_vertices,
        target=target,
        fixtures=fixtures,
    )

    rows = []
    final_reward = np.asarray(official_output["final_reward"])
    cumulative_reward = np.asarray(official_output["cum_reward"])
    for index, name in enumerate(POLICY_NAMES):
        rows.append(
            {
                "policy": name,
                "active_steps": active_steps[name],
                "native_final_reward": float(final_reward[index]),
                "native_cumulative_reward": float(cumulative_reward[index]),
                "all_vertices_finite": bool(np.isfinite(final_vertices[index]).all()),
                "metrics": wiring_post_curve_metrics(final_vertices[index], target, fixtures),
            }
        )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "matched deterministic open-loop path comparison in official DLO-Lab wiring_post",
        "limitations": [
            "This is an external-simulator sanity check, not a trained-policy benchmark.",
            "DLO-Lab's native wiring_post reward is geometric and has no semantic fixture labels.",
            "RouteBot ordered relation metrics are an explicitly labeled extension.",
        ],
        "seed_requested": args.seed,
        "genesis_seed": 0,
        "dlo_lab_commit": _git_head(DLO_REPOSITORY),
        "mushroom_rl_commit": _git_head(MUSHROOM_REPOSITORY),
        "target_sha256": file_sha256(
            DLO_REPOSITORY
            / "genesis"
            / "assets"
            / "dlo-lab"
            / "target_pos"
            / "wiring_post_finalpos.npy"
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "official_forward_seconds": float(official_output["forward_time"]),
        },
        "configuration": {
            "policies": list(POLICY_NAMES),
            "horizon": int(trajectories.shape[1]),
            "scene_steps_per_action": args.scene_steps_per_action,
            "max_step_m": args.max_step_m,
            "grasp_height_m": args.grasp_height_m,
            "control_vertex_index": control_index,
            "active_steps": active_steps,
        },
        "results": rows,
        "arrays": {
            "path": str(arrays_path.relative_to(WORKSPACE)),
            "sha256": file_sha256(arrays_path),
        },
    }
    report_path = output / "matched_path_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
