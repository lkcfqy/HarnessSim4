"""Direct MuJoCo execution audit for InspectBot active scan decisions.

The policy closes its loop on frozen real-image anomaly scores and scan history.
MuJoCo supplies the current 3-D inspection-site poses and executes every scanner
motion with a UR5e visual IK twin.  The cable score evidence is not rendered by
MuJoCo, so this remains a hybrid perception/physics audit rather than hardware data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from harnessbench.inspect_public_data import file_sha256
from harnessbench.learning.inspect_active import (
    SITE_POSITIONS,
    _choose_site,
    _episode_metrics,
    _load_score_cache,
    _scenario,
    inspect_deployment_v2_config,
)
from harnessbench.learning.mujoco_transfer import (
    _advance_mocap,
    _load_mujoco,
    _mocap_id,
    _object_id,
    _render_final,
    _settle,
)
from harnessbench.sim.mujoco_backend import SCENE_SPECS, _assert_finite
from harnessbench.sim.realistic_robots import load_realistic_model

DIRECT_INSPECT_POLICIES = (
    "topology_risk",
    "geometry_coverage",
    "uncertainty",
    "random",
)


def _policy_rng(seed: int, policy: str) -> np.random.Generator:
    digest = int.from_bytes(hashlib.sha256(policy.encode()).digest()[:4], "little")
    return np.random.default_rng(seed ^ digest)


def _move_scanner(
    mujoco: Any,
    model: Any,
    data: Any,
    robot_rig: Any,
    mocap_id: int,
    target: np.ndarray,
    *,
    physics_steps: int,
    max_motion_m: float,
    tolerance_m: float,
) -> tuple[int, float, float]:
    substeps = 0
    travel = 0.0
    peak_ik_error = 0.0
    while float(np.linalg.norm(data.mocap_pos[mocap_id] - target)) > tolerance_m:
        before = data.mocap_pos[mocap_id].copy()
        _advance_mocap(
            mujoco,
            model,
            data,
            {mocap_id: target},
            physics_steps=physics_steps,
            max_motion=max_motion_m,
            robot_rig=robot_rig,
        )
        travel += float(np.linalg.norm(data.mocap_pos[mocap_id] - before))
        substeps += physics_steps
        peak_ik_error = max(peak_ik_error, max(robot_rig.errors(data), default=0.0))
        if substeps > 2_000:
            raise RuntimeError("scanner motion failed to converge")
    return substeps, travel, peak_ik_error


def _aggregate(records: list[dict]) -> list[dict]:
    output = []
    budgets = sorted({int(row["budget"]) for row in records})
    for budget in budgets:
        for policy in DIRECT_INSPECT_POLICIES:
            group = [
                row for row in records if row["budget"] == budget and row["policy"] == policy
            ]
            output.append(
                {
                    "budget": budget,
                    "policy": policy,
                    "episodes": len(group),
                    "critical_weighted_recall": float(
                        np.mean([row["critical_weighted_recall"] for row in group])
                    ),
                    "recall": float(np.mean([row["recall"] for row in group])),
                    "precision": float(np.mean([row["precision"] for row in group])),
                    "false_positive_rate": float(
                        np.mean([row["false_positive_rate"] for row in group])
                    ),
                    "coverage": float(np.mean([row["coverage"] for row in group])),
                    "world_travel_distance_m": float(
                        np.mean([row["world_travel_distance_m"] for row in group])
                    ),
                    "native_reach_success_rate": float(
                        np.mean([row["native_reach_success"] for row in group])
                    ),
                    "peak_visual_ik_error_m": float(
                        np.max([row["peak_visual_ik_error_m"] for row in group])
                    ),
                }
            )
    return output


def evaluate_inspect_mujoco_direct(
    score_cache: Path,
    output_dir: Path,
    *,
    physical_seeds: int = 20,
    base_seed: int = 78_000_000,
    budgets: tuple[int, ...] = (4, 8, 12, 16),
    physics_steps: int = 4,
    max_motion_m: float = 0.025,
    reach_tolerance_m: float = 0.001,
    render: bool = True,
) -> dict:
    """Execute active scan sequences in the independent MuJoCo scene."""

    if physical_seeds < 1:
        raise ValueError("physical_seeds must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(
        inspect_deployment_v2_config(),
        episode_count=physical_seeds,
        base_seed=base_seed,
        budgets=budgets,
    )
    scores, labels = _load_score_cache(score_cache.resolve())
    mujoco = _load_mujoco()
    scene = SCENE_SPECS["inspect"].xml_path
    model, robot_rig = load_realistic_model(mujoco, "inspect", scene)
    site_ids = np.asarray(
        [
            _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"inspection_site_{index:02d}",
            )
            for index in range(config.site_count)
        ],
        dtype=np.int32,
    )
    scanner_fov_id = _object_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "scanner_fov",
    )
    scanner_mocap = _mocap_id(mujoco, model, "scanner")
    records = []
    representative_path = output_dir / "inspect_mujoco_direct_topology_final.png"
    for episode_index in range(physical_seeds):
        seed = base_seed + episode_index
        scenario = _scenario(seed, config, scores, labels)
        for policy in DIRECT_INSPECT_POLICIES:
            data = mujoco.MjData(model)
            _settle(mujoco, model, data, steps=80, robot_rig=robot_rig)
            world_sites = data.site_xpos[site_ids].copy()
            observations: list[list[float]] = [[] for _ in range(config.site_count)]
            scans = np.zeros(config.site_count, dtype=np.int64)
            policy_position = np.asarray([0.0, 0.5], dtype=np.float64)
            rng = _policy_rng(seed, policy)
            shuffled_risk = rng.permutation(
                np.asarray([site["risk"] for site in scenario["sites"]], dtype=np.float64)
            )
            world_travel = 0.0
            motion_substeps = 0
            peak_ik_error = 0.0
            max_fov_error = 0.0
            reached_all = True
            trace = []
            for scan_number in range(1, max(config.budgets) + 1):
                site_index = _choose_site(
                    policy,
                    scenario,
                    observations,
                    scans,
                    policy_position,
                    rng,
                    shuffled_risk,
                    config,
                )
                target = world_sites[site_index] + np.asarray([0.0, -0.08, 0.08])
                substeps, distance, ik_error = _move_scanner(
                    mujoco,
                    model,
                    data,
                    robot_rig,
                    scanner_mocap,
                    target,
                    physics_steps=physics_steps,
                    max_motion_m=max_motion_m,
                    tolerance_m=reach_tolerance_m,
                )
                motion_substeps += substeps
                world_travel += distance
                peak_ik_error = max(peak_ik_error, ik_error)
                fov_error = float(
                    np.linalg.norm(data.site_xpos[scanner_fov_id] - world_sites[site_index])
                )
                max_fov_error = max(max_fov_error, fov_error)
                reached = fov_error <= 0.003
                reached_all = reached_all and reached
                view_id = config.view_order[int(scans[site_index])]
                if reached:
                    observations[site_index].append(
                        float(scenario["sites"][site_index]["view_scores"][view_id])
                    )
                    scans[site_index] += 1
                trace.append(
                    {
                        "scan": scan_number,
                        "site": site_index,
                        "view": view_id,
                        "reached": reached,
                        "fov_error_m": fov_error,
                    }
                )
                policy_position = SITE_POSITIONS[site_index].copy()
                if scan_number in config.budgets:
                    metrics = _episode_metrics(
                        scenario,
                        observations,
                        scans,
                        world_travel,
                        config,
                    )
                    metrics["world_travel_distance_m"] = metrics.pop("travel_distance")
                    records.append(
                        {
                            "seed": seed,
                            "policy": policy,
                            "budget": scan_number,
                            **metrics,
                            "native_reach_success": reached_all,
                            "max_task_space_fov_error_m": max_fov_error,
                            "peak_visual_ik_error_m": peak_ik_error,
                            "motion_physics_substeps": motion_substeps,
                            "trace_json": json.dumps(trace, separators=(",", ":")),
                        }
                    )
            _assert_finite(data, "inspect_mujoco_direct")
            if render and episode_index == 0 and policy == "topology_risk":
                _render_final(mujoco, model, data, representative_path)

    aggregate = _aggregate(records)
    episodes_path = output_dir / "inspect_mujoco_direct_episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate_path = output_dir / "inspect_mujoco_direct_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    report = {
        "schema_version": 1,
        "name": "InspectBot direct MuJoCo active-scan execution audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": f"MuJoCo {mujoco.__version__} elasticity cable plugin",
        "robot": "Universal Robots UR5e visual IK twin + dual-lens inspection head",
        "config": {
            "physical_seeds": physical_seeds,
            "base_seed": base_seed,
            "budgets": budgets,
            "physics_steps_per_motion_increment": physics_steps,
            "max_motion_increment_m": max_motion_m,
            "reach_tolerance_m": reach_tolerance_m,
        },
        "score_cache": str(score_cache.resolve()),
        "score_cache_sha256": file_sha256(score_cache.resolve()),
        "scene": str(scene.resolve()),
        "scene_sha256": file_sha256(scene),
        "episode_rows": len(records),
        "aggregate": aggregate,
        "all_native_reaches_succeeded": all(row["native_reach_success"] for row in records),
        "max_task_space_fov_error_m": max(
            row["max_task_space_fov_error_m"] for row in records
        ),
        "max_visual_ik_error_m": max(row["peak_visual_ik_error_m"] for row in records),
        "episodes_csv": str(episodes_path.resolve()),
        "episodes_sha256": file_sha256(episodes_path),
        "aggregate_csv": str(aggregate_path.resolve()),
        "aggregate_sha256": file_sha256(aggregate_path),
        "representative_render": (
            str(representative_path.resolve()) if representative_path.is_file() else None
        ),
        "representative_render_sha256": (
            file_sha256(representative_path) if representative_path.is_file() else None
        ),
        "claim_scope": (
            "Policy decisions close on frozen real-image scores while MuJoCo executes current "
            "3-D site motion. The UR5e is collision-disabled visual IK, scores are not rendered "
            "camera measurements, and this is not hardware evidence."
        ),
    }
    report_path = output_dir / "inspect_mujoco_direct_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report
