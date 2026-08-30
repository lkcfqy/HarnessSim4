"""Direct-observation BranchBot audit in the independent MuJoCo backend.

The graph policy is reconstructed from current MuJoCo cable, target, and
gripper poses at every step.  Its two task-space outputs directly command the
two MuJoCo mocap grippers; no PBD rollout or PBD event is replayed.  Detailed
UR10e/Robotiq geometry is used only for the representative rendered frame, so
the experiment remains a task-space cross-physics audit rather than a claimed
torque, force, or safety controller.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from harnessbench.learning.mujoco_transfer import (
    _advance_mocap,
    _load_mujoco,
    _mocap_id,
    _object_id,
    _render_final,
    _settle,
)
from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact
from harnessbench.sim.benchmark import aggregate_success
from harnessbench.sim.core import CableGraph
from harnessbench.sim.envs.branch import BRANCH_PHASE_BOUNDARIES, BranchEnv
from harnessbench.sim.mujoco_backend import SCENE_SPECS, _assert_finite
from harnessbench.sim.realistic_robots import load_realistic_model

DIRECT_BRANCH_POLICY_NAMES = (
    "learned_topology",
    "learned_geometry",
    "teacher_topology",
)

# A least-squares, row-vector affine calibration between the physical MuJoCo
# board and the normalized policy workspace.  It preserves the crossed branch
# task: the branch_a endpoint starts high in policy coordinates but its target
# lies low, and branch_b has the opposite assignment.
_WORLD_LANDMARKS = np.asarray(
    (
        (-0.50, 0.00),
        (-0.16, 0.00),
        (0.43, 0.30),
        (0.43, -0.30),
    ),
    dtype=np.float64,
)
_POLICY_LANDMARKS = np.asarray(
    (
        (0.08, 0.52),
        (0.42, 0.52),
        (0.86, 0.28),
        (0.86, 0.72),
    ),
    dtype=np.float64,
)


def _affine_homogeneous(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("branch affine calibration is rank deficient")
    transform = np.eye(3, dtype=np.float64)
    transform[:, :2] = coefficients
    return transform


WORLD_TO_POLICY = _affine_homogeneous(_WORLD_LANDMARKS, _POLICY_LANDMARKS)
POLICY_TO_WORLD = np.linalg.inv(WORLD_TO_POLICY)


def _transform_xy(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    one_point = points.ndim == 1
    rows = points.reshape(-1, 2)
    homogeneous = np.column_stack((rows, np.ones(len(rows), dtype=np.float64)))
    mapped = homogeneous @ transform
    result = mapped[:, :2] / mapped[:, 2:3]
    return result[0] if one_point else result


def _cable_body_names(prefix: str, count: int) -> tuple[str, ...]:
    return (
        f"{prefix}_B_first",
        *(f"{prefix}_B_{index}" for index in range(1, count - 1)),
        f"{prefix}_B_last",
    )


def _sample_ids(ids: np.ndarray, count: int, *, skip_first: bool = False) -> np.ndarray:
    start = 1 if skip_first else 0
    indices = np.rint(np.linspace(start, len(ids) - 1, count)).astype(np.int64)
    return ids[indices]


class _BranchObservationAdapter:
    task_name = "branched_harness"
    robot_name = "BranchBot"
    max_steps = 300

    def __init__(self, difficulty: float) -> None:
        self.difficulty = float(difficulty)
        self.step_count = 0
        self._observation: dict[str, Any] = {}

    def set_observation(self, observation: dict[str, Any], step_count: int) -> None:
        self._observation = observation
        self.step_count = int(step_count)

    def observation(self) -> dict[str, Any]:
        return self._observation


class _DirectTopologyTeacher:
    name = "teacher_topology"

    def __init__(self) -> None:
        self.last_debug: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        del seed
        self.last_debug = {}

    def act(self, env: _BranchObservationAdapter) -> np.ndarray:
        observation = env.observation()
        progress = env.step_count / env.max_steps
        if progress < BRANCH_PHASE_BOUNDARIES[0]:
            phase = 0
        elif progress < BRANCH_PHASE_BOUNDARIES[1]:
            phase = 1
        else:
            phase = 2
        targets = []
        for semantic in ("A", "B"):
            if phase < 2:
                targets.append(np.asarray(observation["semantic_waypoints"][semantic][phase]))
            else:
                targets.append(np.asarray(observation["semantic_targets"][semantic]))
        self.last_debug = {"phase": phase, "targets": [value.tolist() for value in targets]}
        return np.asarray((*targets[0], 1.0, *targets[1], 1.0), dtype=np.float64)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_jitter(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    seed: int,
    difficulty: float,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    scale = 0.0025 + 0.005 * float(difficulty)
    offsets: dict[str, list[float]] = {}
    for name in ("target_fixture_a", "target_fixture_b"):
        body_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
        delta = np.clip(rng.normal(0.0, scale, size=2), -2.5 * scale, 2.5 * scale)
        model.body_pos[body_id, :2] += delta
        offsets[name] = delta.tolist()
    for mocap_name in ("gripper_a", "gripper_b"):
        mocap_id = _mocap_id(mujoco, model, mocap_name)
        delta = np.clip(rng.normal(0.0, 0.5 * scale, size=2), -scale, scale)
        data.mocap_pos[mocap_id, :2] += delta
        offsets[f"{mocap_name}_initial"] = delta.tolist()
    return offsets


def _branch_scene_observation(
    data: Any,
    trunk_ids: np.ndarray,
    branch_a_ids: np.ndarray,
    branch_b_ids: np.ndarray,
    target_a_site: int,
    target_b_site: int,
    mocap_a: int,
    mocap_b: int,
    *,
    semantic_swap: bool,
) -> dict[str, Any]:
    trunk = _transform_xy(data.xpos[_sample_ids(trunk_ids, 10), :2], WORLD_TO_POLICY)
    branch_a = _transform_xy(
        data.xpos[_sample_ids(branch_a_ids, 9, skip_first=True), :2],
        WORLD_TO_POLICY,
    )
    branch_b = _transform_xy(
        data.xpos[_sample_ids(branch_b_ids, 9, skip_first=True), :2],
        WORLD_TO_POLICY,
    )
    positions = np.concatenate((trunk, branch_a, branch_b), axis=0)
    edges = np.asarray(
        [
            *(zip(range(9), range(1, 10))),
            (9, 10),
            *((left, right) for left, right in zip(range(10, 18), range(11, 19))),
            (9, 19),
            *((left, right) for left, right in zip(range(19, 27), range(20, 28))),
        ],
        dtype=np.int64,
    )
    target_physical_a = _transform_xy(data.site_xpos[target_a_site, :2], WORLD_TO_POLICY)
    target_physical_b = _transform_xy(data.site_xpos[target_b_site, :2], WORLD_TO_POLICY)
    arm_physical_a = _transform_xy(data.mocap_pos[mocap_a, :2], WORLD_TO_POLICY)
    arm_physical_b = _transform_xy(data.mocap_pos[mocap_b, :2], WORLD_TO_POLICY)
    if semantic_swap:
        endpoint_a, endpoint_b = 27, 18
        target_a, target_b = target_physical_b, target_physical_a
        arm_a, arm_b = arm_physical_b, arm_physical_a
    else:
        endpoint_a, endpoint_b = 18, 27
        target_a, target_b = target_physical_a, target_physical_b
        arm_a, arm_b = arm_physical_a, arm_physical_b
    return {
        "cable_positions": positions,
        "cable_edges": edges,
        "semantic_endpoints": {"A": endpoint_a, "B": endpoint_b},
        "semantic_targets": {"A": target_a, "B": target_b},
        "semantic_waypoints": {
            "A": BranchEnv._semantic_waypoints(target_a),
            "B": BranchEnv._semantic_waypoints(target_b),
        },
        "semantic_target_swap": bool(semantic_swap),
        "arm_a_ee": arm_a,
        "arm_b_ee": arm_b,
        "crossings": CableGraph(positions=positions, edges=edges).crossing_count(),
    }


def _make_policy(
    policy_name: str,
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    device_name: str,
) -> Any:
    if policy_name == "learned_topology":
        return LearnedGraphPolicy(
            topology_checkpoint, "branch", name=policy_name, device_name=device_name
        )
    if policy_name == "learned_geometry":
        return LearnedGraphPolicy(
            geometry_checkpoint, "branch", name=policy_name, device_name=device_name
        )
    if policy_name == "teacher_topology":
        return _DirectTopologyTeacher()
    raise KeyError(policy_name)


def _run_direct_episode(
    mujoco: Any,
    policy: Any,
    *,
    policy_name: str,
    semantic_swap: bool,
    seed: int,
    difficulty: float,
    physics_steps: int,
    max_steps: int,
    render_path: Path | None,
) -> dict[str, Any]:
    spec = SCENE_SPECS["branch"]
    if render_path is None:
        model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
        robot_rig = None
    else:
        model, robot_rig = load_realistic_model(mujoco, "branch", spec.xml_path)
    data = mujoco.MjData(model)
    fixture_offsets = _fixture_jitter(
        mujoco,
        model,
        data,
        seed=seed,
        difficulty=difficulty,
    )
    mocap_a = _mocap_id(mujoco, model, "gripper_a")
    mocap_b = _mocap_id(mujoco, model, "gripper_b")
    trunk_ids = np.asarray(
        [
            _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _cable_body_names("trunk", 18)
        ],
        dtype=np.int64,
    )
    branch_a_ids = np.asarray(
        [
            _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _cable_body_names("branch_a", 24)
        ],
        dtype=np.int64,
    )
    branch_b_ids = np.asarray(
        [
            _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _cable_body_names("branch_b", 24)
        ],
        dtype=np.int64,
    )
    target_a_site = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "target_a")
    target_b_site = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "target_b")
    _settle(mujoco, model, data, steps=100, robot_rig=robot_rig)
    adapter = _BranchObservationAdapter(difficulty)
    policy.reset(seed + 100_003)
    stable_steps = 0
    path_length_a = 0.0
    path_length_b = 0.0
    previous_a = data.mocap_pos[mocap_a, :2].copy()
    previous_b = data.mocap_pos[mocap_b, :2].copy()
    step_count = 0
    error_a = error_b = float("inf")
    for step_count in range(max_steps):
        observation = _branch_scene_observation(
            data,
            trunk_ids,
            branch_a_ids,
            branch_b_ids,
            target_a_site,
            target_b_site,
            mocap_a,
            mocap_b,
            semantic_swap=semantic_swap,
        )
        adapter.set_observation(observation, step_count)
        action = np.asarray(policy.act(adapter), dtype=np.float64)
        semantic_actions = {"A": action[:2], "B": action[3:5]}
        if semantic_swap:
            physical_a_action = semantic_actions["B"]
            physical_b_action = semantic_actions["A"]
        else:
            physical_a_action = semantic_actions["A"]
            physical_b_action = semantic_actions["B"]
        world_a = _transform_xy(physical_a_action, POLICY_TO_WORLD)
        world_b = _transform_xy(physical_b_action, POLICY_TO_WORLD)
        _advance_mocap(
            mujoco,
            model,
            data,
            {
                mocap_a: np.asarray((world_a[0], world_a[1], 0.15)),
                mocap_b: np.asarray((world_b[0], world_b[1], 0.15)),
            },
            physics_steps=physics_steps,
            max_motion=0.014,
            robot_rig=robot_rig,
        )
        current_a = data.mocap_pos[mocap_a, :2].copy()
        current_b = data.mocap_pos[mocap_b, :2].copy()
        path_length_a += float(np.linalg.norm(current_a - previous_a))
        path_length_b += float(np.linalg.norm(current_b - previous_b))
        previous_a, previous_b = current_a, current_b
        error_a = float(
            np.linalg.norm(data.xpos[branch_a_ids[-1], :2] - data.site_xpos[target_a_site, :2])
        )
        error_b = float(
            np.linalg.norm(data.xpos[branch_b_ids[-1], :2] - data.site_xpos[target_b_site, :2])
        )
        stable_steps = stable_steps + 1 if error_a <= 0.055 and error_b <= 0.055 else 0
        if stable_steps >= 4:
            break
    for _ in range(40):
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)
    _assert_finite(data, "branch_direct")
    error_a = float(
        np.linalg.norm(data.xpos[branch_a_ids[-1], :2] - data.site_xpos[target_a_site, :2])
    )
    error_b = float(
        np.linalg.norm(data.xpos[branch_b_ids[-1], :2] - data.site_xpos[target_b_site, :2])
    )
    success = bool(error_a <= 0.055 and error_b <= 0.055)
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "policy": policy_name,
        "seed": seed,
        "difficulty": difficulty,
        "semantic_target_swap": bool(semantic_swap),
        "success": success,
        "steps": step_count + 1,
        "endpoint_a_xy_error_m": error_a,
        "endpoint_b_xy_error_m": error_b,
        "mean_endpoint_xy_error_m": 0.5 * (error_a + error_b),
        "gripper_a_path_length_m": path_length_a,
        "gripper_b_path_length_m": path_length_b,
        "finite_state": True,
        "visual_ik_error_m": (
            max(robot_rig.errors(data), default=0.0) if robot_rig is not None else None
        ),
        "physics_steps": 140 + physics_steps * (step_count + 1),
        "fixture_offsets": fixture_offsets,
    }


def _paired_comparisons(records: list[dict]) -> list[dict]:
    if not any(str(row["policy"]) == "learned_topology" for row in records):
        return []
    indexed = {
        (
            int(row["seed"]),
            bool(row["semantic_target_swap"]),
            str(row["policy"]),
        ): row
        for row in records
    }
    seeds = sorted({int(row["seed"]) for row in records})
    baselines = sorted({str(row["policy"]) for row in records} - {"learned_topology"})
    output: list[dict] = []
    for semantic_swap in (False, True):
        for baseline in baselines:
            pairs = [
                (
                    bool(indexed[(seed, semantic_swap, "learned_topology")]["success"]),
                    bool(indexed[(seed, semantic_swap, baseline)]["success"]),
                )
                for seed in seeds
            ]
            topology_only = sum(left and not right for left, right in pairs)
            baseline_only = sum(right and not left for left, right in pairs)
            output.append(
                {
                    "semantic_target_swap": semantic_swap,
                    "baseline": baseline,
                    "paired_physical_seeds": len(pairs),
                    "topology_success_baseline_failure": topology_only,
                    "topology_failure_baseline_success": baseline_only,
                    "both_success": sum(left and right for left, right in pairs),
                    "both_failure": sum(not left and not right for left, right in pairs),
                    "mcnemar_exact_p": _mcnemar_exact(topology_only, baseline_only),
                }
            )
    return _holm_adjust(output)


def evaluate_branch_mujoco_direct(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    output_dir: Path,
    physical_seeds: int = 20,
    policies: Iterable[str] = DIRECT_BRANCH_POLICY_NAMES,
    difficulty: float = 0.5,
    base_seed: int = 65_000_000,
    physics_steps: int = 6,
    max_steps: int = 300,
    device_name: str = "cpu",
    render_representative: bool = True,
) -> dict[str, Any]:
    if physical_seeds < 1:
        raise ValueError("physical_seeds must be at least one")
    if physics_steps < 1 or max_steps < 1:
        raise ValueError("physics_steps and max_steps must be positive")
    policy_names = tuple(policies)
    unknown = set(policy_names) - set(DIRECT_BRANCH_POLICY_NAMES)
    if unknown:
        raise KeyError(f"unknown direct BranchBot policies: {sorted(unknown)}")
    mujoco = _load_mujoco()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_objects = {
        name: _make_policy(
            name,
            topology_checkpoint,
            geometry_checkpoint,
            device_name=device_name,
        )
        for name in policy_names
    }
    records: list[dict] = []
    representative = output_dir / "branch_mujoco_direct_topology_final.png"
    for episode in range(physical_seeds):
        seed = base_seed + episode
        for semantic_swap in (False, True):
            for policy_name, policy in policy_objects.items():
                render_path = (
                    representative
                    if render_representative
                    and episode == 0
                    and not semantic_swap
                    and policy_name == "learned_topology"
                    else None
                )
                records.append(
                    _run_direct_episode(
                        mujoco,
                        policy,
                        policy_name=policy_name,
                        semantic_swap=semantic_swap,
                        seed=seed,
                        difficulty=difficulty,
                        physics_steps=physics_steps,
                        max_steps=max_steps,
                        render_path=render_path,
                    )
                )
    checkpoints = {
        "learned_topology": str(topology_checkpoint.resolve()),
        "learned_geometry": str(geometry_checkpoint.resolve()),
    }
    return {
        "name": "BranchBot direct-observation MuJoCo cross-physics audit",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": f"MuJoCo {mujoco.__version__} elasticity cable plugin",
        "config": {
            "physical_seeds": physical_seeds,
            "semantic_assignments_per_seed": [False, True],
            "difficulty": difficulty,
            "base_seed": base_seed,
            "physics_steps_per_policy_action": physics_steps,
            "max_policy_steps": max_steps,
            "policies": list(policy_names),
            "device": device_name,
        },
        "checkpoints": checkpoints,
        "checkpoint_sha256": {
            name: _checkpoint_sha256(Path(path)) for name, path in checkpoints.items()
        },
        "claim_scope": (
            "Direct closed-loop policy observation and task-space control in independent MuJoCo; "
            "visual-arm IK is collision-disabled and this is not force-control or hardware evidence"
        ),
        "records": records,
        "aggregate_by_policy": aggregate_success(records, ("policy",)),
        "aggregate_by_policy_assignment": aggregate_success(
            records, ("policy", "semantic_target_swap")
        ),
        "paired_comparisons": _paired_comparisons(records),
        "representative_frame": str(representative.resolve()) if representative.is_file() else None,
    }


def save_branch_mujoco_direct(report: dict, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "branch_mujoco_direct_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    episodes_path = output_dir / "branch_mujoco_direct_episodes.csv"
    fields = sorted({key for row in report["records"] for key in row})
    with episodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["records"]:
            writer.writerow(
                {
                    **row,
                    "fixture_offsets": json.dumps(row["fixture_offsets"], sort_keys=True),
                }
            )
    aggregate_path = output_dir / "branch_mujoco_direct_aggregate.csv"
    rows = report["aggregate_by_policy_assignment"]
    fields = sorted({key for row in rows for key in row})
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "report": str(report_path.resolve()),
        "report_sha256": _checkpoint_sha256(report_path),
        "episodes_csv": str(episodes_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
        "episode_count": len(report["records"]),
        "representative_frame": report["representative_frame"],
    }
