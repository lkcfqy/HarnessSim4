"""Direct-observation RouteBot audit in the independent MuJoCo backend.

The learned policy receives a graph reconstructed from the current MuJoCo
cable and fixture poses at every control step. Its task-space target is then
executed by the MuJoCo mocap gripper and the native cable constraints. No PBD
rollout or PBD task event is consulted.

This is deliberately narrower than a hardware claim: the detailed UR10e is a
collision-disabled visual IK twin and the task-space gripper is not a
torque/force controller. The output is therefore a cross-physics closed-loop
audit, not real-robot or safety evidence.
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
from PIL import Image

from harnessbench.learning.mujoco_transfer import (
    _advance_mocap,
    _equality_id,
    _load_mujoco,
    _mocap_id,
    _object_id,
    _render_final,
    _set_connect_constraint,
    _settle,
)
from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact
from harnessbench.sim.benchmark import aggregate_success
from harnessbench.sim.envs.route import ROUTE_TARGET_ASSIGNMENTS
from harnessbench.sim.mujoco_backend import SCENE_SPECS, _assert_finite
from harnessbench.sim.realistic_robots import load_realistic_model

DIRECT_ROUTE_POLICY_NAMES = (
    "learned_topology",
    "learned_geometry",
    "teacher_topology",
)

# Cross-backend process calibration.  The MuJoCo fixture path consumes almost
# the full 1.35 m cable, so uniformly mapping PBD particle numbers to MuJoCo
# body numbers makes the outer members of each semantic window unreachable.
# These knots were screened with the scripted teacher only: p5/p7/p9 map to
# three adjacent bodies around clip 1, p13/p15/p17 around clip 2, and
# p21/p23/p25 around clip 3.  Intermediate graph nodes are continuous
# arc-length samples of the current MuJoCo cable, not copied PBD states.
_DIRECT_PBD_KNOTS = np.asarray(
    (0, 5, 7, 9, 13, 15, 17, 21, 23, 25, 28),
    dtype=np.float64,
)
_DIRECT_MUJOCO_KNOTS = np.asarray(
    (0, 7, 8, 9, 17, 18, 19, 24, 25, 26, 33),
    dtype=np.float64,
)
DIRECT_ROUTE_SEGMENTS = tuple(
    float(value)
    for value in np.interp(
        np.arange(29, dtype=np.float64),
        _DIRECT_PBD_KNOTS,
        _DIRECT_MUJOCO_KNOTS,
    )
)
if not np.all(np.diff(DIRECT_ROUTE_SEGMENTS) > 0):  # pragma: no cover
    raise RuntimeError("direct RouteBot arc-length samples must be strictly increasing")
if len({round(DIRECT_ROUTE_SEGMENTS[index]) for index in (5, 7, 9, 13, 15, 17, 21, 23, 25)}) != 9:
    raise RuntimeError("all semantic RouteBot targets must map to distinct MuJoCo bodies")

_CANONICAL_LANDMARKS = np.asarray(
    (
        (0.39, 0.61),
        (0.58, 0.39),
        (0.77, 0.58),
        (0.88, 0.72),
    ),
    dtype=np.float64,
)
_NOMINAL_WORLD_LANDMARKS = np.asarray(
    (
        (-0.24, -0.12),
        (-0.02, 0.08),
        (0.13, -0.02),
        (0.34, 0.14),
    ),
    dtype=np.float64,
)


def _affine_homogeneous(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a row-vector 2-D homogeneous least-squares affine transform."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("source and target must both have shape (n, 2)")
    if len(source) < 3:
        raise ValueError("at least three point pairs are required")
    design = np.column_stack((source, np.ones(len(source), dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("affine calibration points are rank deficient")
    transform = np.eye(3, dtype=np.float64)
    transform[:, :2] = coefficients
    return transform


WORLD_TO_POLICY = _affine_homogeneous(
    _NOMINAL_WORLD_LANDMARKS,
    _CANONICAL_LANDMARKS,
)
POLICY_TO_WORLD = np.linalg.inv(WORLD_TO_POLICY)


def _transform_xy(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    one_point = points.ndim == 1
    rows = points.reshape(-1, 2)
    homogeneous = np.column_stack((rows, np.ones(len(rows), dtype=np.float64)))
    mapped = homogeneous @ np.asarray(transform, dtype=np.float64)
    result = mapped[:, :2] / mapped[:, 2:3]
    return result[0] if one_point else result


def _route_segment_body_name(segment: int) -> str:
    if segment == 0:
        return "route_B_first"
    if segment == 33:
        return "route_B_last"
    return f"route_B_{segment}"


def _direct_route_body_name(particle: int) -> str:
    if particle < 0 or particle >= len(DIRECT_ROUTE_SEGMENTS):
        raise IndexError(particle)
    return _route_segment_body_name(round(DIRECT_ROUTE_SEGMENTS[particle]))


class _RouteObservationAdapter:
    """Minimal environment interface consumed by LearnedGraphPolicy."""

    task_name = "cable_routing"
    robot_name = "RouteBot"
    max_steps = 360

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
    grasp_distance = 0.032

    def __init__(self) -> None:
        self.last_debug: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        del seed
        self.last_debug = {}

    def act(self, env: _RouteObservationAdapter) -> np.ndarray:
        observation = env.observation()
        positions = np.asarray(observation["cable_positions"], dtype=np.float64)
        arm = np.asarray(observation["arm_ee"], dtype=np.float64)
        bindings = {
            int(key): int(value) for key, value in observation["bindings"].items()
        }
        grasp = observation["grasp_idx"]
        unbound = [index for index in range(3) if index not in bindings]
        if unbound:
            clip_index = unbound[0]
            particle = int(observation["target_particle_indices"][clip_index])
            destination = np.asarray(observation["clips"][clip_index], dtype=np.float64)
        else:
            particle = 28
            destination = np.asarray(observation["finish"], dtype=np.float64)
        if grasp is None:
            target = positions[particle]
            grip = float(np.linalg.norm(arm - target) <= self.grasp_distance)
            pointer_index = particle
        else:
            target = destination
            grip = 1.0
            pointer_index = 29 + unbound[0] if unbound else 32
        self.last_debug = {
            "pointer_index": pointer_index,
            "pointer_position": target.tolist(),
            "phase": "approach" if grasp is None else "transport",
        }
        return np.asarray((target[0], target[1], grip), dtype=np.float64)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_jitter(
    mujoco: Any,
    model: Any,
    *,
    seed: int,
    difficulty: float,
) -> dict[str, list[float]]:
    """Apply paired, small static-fixture perturbations to a fresh model."""

    rng = np.random.default_rng(seed)
    scale = 0.003 + 0.006 * float(difficulty)
    offsets: dict[str, list[float]] = {}
    for name in ("clip_1", "clip_2", "clip_3", "route_finish_fixture"):
        body_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
        delta = np.clip(rng.normal(0.0, scale, size=2), -2.5 * scale, 2.5 * scale)
        model.body_pos[body_id, :2] += delta
        offsets[name] = delta.tolist()
    return offsets


def _route_scene_observation(
    data: Any,
    all_cable_body_ids: np.ndarray,
    clip_body_ids: np.ndarray,
    finish_site_id: int,
    gripper_mocap: int,
    *,
    target_assignment: tuple[int, int, int],
    bindings: dict[int, int],
    grasp_idx: int | None,
    endpoint_placed: bool,
) -> dict[str, Any]:
    full_cable_world = np.asarray(
        data.xpos[all_cable_body_ids, :2],
        dtype=np.float64,
    )
    samples = np.asarray(DIRECT_ROUTE_SEGMENTS, dtype=np.float64)
    left = np.floor(samples).astype(np.int64)
    right = np.ceil(samples).astype(np.int64)
    weight = (samples - left)[:, None]
    cable_world = (
        full_cable_world[left] * (1.0 - weight)
        + full_cable_world[right] * weight
    )
    clips_world = np.asarray(data.xpos[clip_body_ids, :2], dtype=np.float64)
    finish_world = np.asarray(data.site_xpos[finish_site_id, :2], dtype=np.float64)
    arm_world = np.asarray(data.mocap_pos[gripper_mocap, :2], dtype=np.float64)
    return {
        "cable_positions": _transform_xy(cable_world, WORLD_TO_POLICY),
        "cable_edges": np.column_stack(
            (np.arange(28, dtype=np.int64), np.arange(1, 29, dtype=np.int64))
        ),
        "arm_ee": _transform_xy(arm_world, WORLD_TO_POLICY),
        "clips": _transform_xy(clips_world, WORLD_TO_POLICY),
        "finish": _transform_xy(finish_world, WORLD_TO_POLICY),
        "target_particle_indices": np.asarray(target_assignment, dtype=np.int64),
        "bindings": dict(bindings),
        "grasp_idx": grasp_idx,
        "endpoint_placed": bool(endpoint_placed),
    }


def _capture_frame(renderer: Any, data: Any) -> Image.Image:
    renderer.update_scene(data, camera="overview")
    return Image.fromarray(renderer.render()).copy()


def _run_direct_episode(
    mujoco: Any,
    policy: Any,
    *,
    policy_name: str,
    target_assignment: tuple[int, int, int],
    assignment_index: int,
    seed: int,
    difficulty: float,
    physics_steps: int,
    max_steps: int,
    render_path: Path | None,
) -> dict[str, Any]:
    spec = SCENE_SPECS["route"]
    if render_path is None:
        model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
        robot_rig = None
    else:
        model, robot_rig = load_realistic_model(mujoco, "route", spec.xml_path)
    data = mujoco.MjData(model)
    fixture_offsets = _fixture_jitter(
        mujoco,
        model,
        seed=seed,
        difficulty=difficulty,
    )
    equality_names = (
        "route_grasp",
        "route_grasp_p7",
        "route_grasp_p15",
        "route_grasp_p23",
        "route_latch_1",
        "route_latch_2",
        "route_latch_3",
        "route_finish_latch",
    )
    equality_ids = {
        name: _equality_id(mujoco, model, name) for name in equality_names
    }
    for equality_id in equality_ids.values():
        data.eq_active[equality_id] = False
    _settle(mujoco, model, data, robot_rig=robot_rig)

    gripper_mocap = _mocap_id(mujoco, model, "route_gripper")
    all_cable_body_ids = np.asarray(
        [
            _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                _route_segment_body_name(segment),
            )
            for segment in range(34)
        ],
        dtype=np.int32,
    )
    cable_body_ids = np.asarray(
        [
            _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                _direct_route_body_name(index),
            )
            for index in range(29)
        ],
        dtype=np.int32,
    )
    clip_body_ids = np.asarray(
        [
            _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"clip_{index}",
            )
            for index in range(1, 4)
        ],
        dtype=np.int32,
    )
    finish_site_id = _object_id(
        mujoco,
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "route_finish",
    )

    # Match the fixed PBD initial-arm protocol: start beside physical p7 for
    # every semantic assignment.
    start = data.xpos[cable_body_ids[7]].copy()
    start[2] = 0.14
    data.mocap_pos[gripper_mocap] = start
    if robot_rig is not None:
        robot_rig.sync(data, iterations=90)
    mujoco.mj_forward(model, data)
    renderer = None
    video_frames: list[Image.Image] = []
    video_path = render_path.with_suffix(".gif") if render_path is not None else None
    if render_path is not None:
        renderer = mujoco.Renderer(model, height=540, width=960)
        video_frames.append(_capture_frame(renderer, data))

    env = _RouteObservationAdapter(difficulty)
    policy.reset(seed + 100_003)
    bindings: dict[int, int] = {}
    active_particle: int | None = None
    endpoint_placed = False
    grasp_equality = equality_ids["route_grasp"]
    steps = 0
    wrong_latch = False

    for step in range(max_steps):
        observation = _route_scene_observation(
            data,
            all_cable_body_ids,
            clip_body_ids,
            finish_site_id,
            gripper_mocap,
            target_assignment=target_assignment,
            bindings=bindings,
            grasp_idx=active_particle,
            endpoint_placed=endpoint_placed,
        )
        env.set_observation(observation, step)
        action = np.asarray(policy.act(env), dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError(f"{policy_name} emitted invalid RouteBot action {action!r}")
        desired_xy = _transform_xy(action[:2], POLICY_TO_WORLD)
        desired = np.asarray((desired_xy[0], desired_xy[1], 0.14), dtype=np.float64)
        _advance_mocap(
            mujoco,
            model,
            data,
            {gripper_mocap: desired},
            physics_steps=physics_steps,
            max_motion=0.024,
            robot_rig=robot_rig,
        )
        steps = step + 1
        if renderer is not None and step % 4 == 0:
            video_frames.append(_capture_frame(renderer, data))

        if active_particle is None and action[2] >= 0.5:
            pointer_index = int(policy.last_debug["pointer_index"])
            if 0 <= pointer_index < 29 and pointer_index not in bindings.values():
                body_id = int(cable_body_ids[pointer_index])
                capture_error = float(
                    np.linalg.norm(
                        data.xpos[body_id, :2] - data.mocap_pos[gripper_mocap, :2]
                    )
                )
                if capture_error <= 0.060:
                    _set_connect_constraint(
                        model,
                        data,
                        grasp_equality,
                        body1_id=body_id,
                        active=True,
                    )
                    active_particle = pointer_index
                    mujoco.mj_forward(model, data)

        if active_particle is not None:
            body_id = int(cable_body_ids[active_particle])
            if len(bindings) < 3:
                # A clip closes only when it is the policy's current pointer
                # target.  Pure XY proximity is insufficient because the
                # mocap proxy has no separate vertical press/contact phase and
                # can pass above another clip during transport.
                pointer_index = int(policy.last_debug["pointer_index"])
                clip_index = pointer_index - 29
                if (
                    0 <= clip_index < 3
                    and clip_index not in bindings
                    and float(
                        np.linalg.norm(
                            data.xpos[body_id, :2]
                            - data.xpos[clip_body_ids[clip_index], :2]
                        )
                    )
                    <= 0.055
                ):
                    data.eq_active[grasp_equality] = False
                    latch_id = equality_ids[f"route_latch_{clip_index + 1}"]
                    _set_connect_constraint(
                        model,
                        data,
                        latch_id,
                        body1_id=body_id,
                        body2_anchor=np.asarray((0.0, 0.0, 0.095)),
                        active=True,
                    )
                    bindings[clip_index] = active_particle
                    wrong_latch = wrong_latch or (
                        active_particle != int(target_assignment[clip_index])
                    )
                    active_particle = None
                    mujoco.mj_forward(model, data)
            elif active_particle == 28:
                finish_error = float(
                    np.linalg.norm(
                        data.xpos[body_id, :2] - data.site_xpos[finish_site_id, :2]
                    )
                )
                if finish_error <= 0.060:
                    data.eq_active[grasp_equality] = False
                    _set_connect_constraint(
                        model,
                        data,
                        equality_ids["route_finish_latch"],
                        body1_id=body_id,
                        body2_anchor=np.asarray((0.0, 0.0, 0.095)),
                        active=True,
                    )
                    active_particle = None
                    endpoint_placed = True
                    mujoco.mj_forward(model, data)

        semantic_correct = len(bindings) == 3 and all(
            bindings.get(index) == int(target_assignment[index])
            for index in range(3)
        )
        if semantic_correct and endpoint_placed:
            break
        if wrong_latch:
            break

    for _ in range(50):
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)
    _assert_finite(data, f"route_mujoco_direct_{policy_name}")

    latch_errors: list[float] = []
    for clip_index, particle in sorted(bindings.items()):
        latch_errors.append(
            float(
                np.linalg.norm(
                    data.xpos[cable_body_ids[particle], :2]
                    - data.xpos[clip_body_ids[clip_index], :2]
                )
            )
        )
    endpoint_error = float(
        np.linalg.norm(
            data.xpos[cable_body_ids[28], :2] - data.site_xpos[finish_site_id, :2]
        )
    )
    semantic_correct = len(bindings) == 3 and all(
        bindings.get(index) == int(target_assignment[index]) for index in range(3)
    )
    success = bool(
        semantic_correct
        and endpoint_placed
        and max(latch_errors, default=float("inf")) <= 0.055
        and endpoint_error <= 0.055
    )
    if renderer is not None and video_path is not None:
        video_frames.append(_capture_frame(renderer, data))
        renderer.close()
        video_frames[0].save(
            video_path,
            save_all=True,
            append_images=video_frames[1:],
            duration=100,
            loop=0,
            optimize=False,
        )
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "task_key": "route",
        "robot": "RouteBot",
        "backend": "MuJoCo direct observation/task-space control",
        "policy": policy_name,
        "seed": int(seed),
        "difficulty": float(difficulty),
        "assignment_index": int(assignment_index),
        "target_assignment": "-".join(map(str, target_assignment)),
        "success": success,
        "semantic_complete": semantic_correct and endpoint_placed,
        "semantic_latches_correct": semantic_correct,
        "bindings": dict(bindings),
        "latched_clips": len(bindings),
        "wrong_latches": sum(
            particle != int(target_assignment[clip])
            for clip, particle in bindings.items()
        ),
        "endpoint_placed": endpoint_placed,
        "endpoint_xy_error_m": endpoint_error,
        "max_latch_xy_error_m": max(latch_errors, default=None),
        "steps": steps,
        "finite_state": True,
        "visual_ik_error_m": (
            max(robot_rig.errors(data), default=0.0) if robot_rig is not None else None
        ),
        "fixture_offsets_m": fixture_offsets,
        "render_path": str(render_path.resolve()) if render_path is not None else None,
        "video_path": str(video_path.resolve()) if video_path is not None else None,
    }


def _paired_policy_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            int(record["seed"]),
            int(record["assignment_index"]),
            str(record["policy"]),
        ): bool(record["success"])
        for record in records
    }
    baselines = sorted(
        {str(record["policy"]) for record in records} - {"learned_topology"}
    )
    rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (int(record["seed"]), int(record["assignment_index"]))
            for record in records
        }
    )
    for baseline in baselines:
        pairs = [
            (
                index[(seed, assignment, "learned_topology")],
                index[(seed, assignment, baseline)],
            )
            for seed, assignment in keys
            if (seed, assignment, "learned_topology") in index
            and (seed, assignment, baseline) in index
        ]
        if not pairs:
            continue
        topology_only = sum(left and not right for left, right in pairs)
        baseline_only = sum(right and not left for left, right in pairs)
        rows.append(
            {
                "baseline": baseline,
                "paired_episodes": len(pairs),
                "topology_success_baseline_failure": topology_only,
                "topology_failure_baseline_success": baseline_only,
                "both_success": sum(left and right for left, right in pairs),
                "both_failure": sum(not left and not right for left, right in pairs),
                "paired_success_difference": float(
                    np.mean([float(left) - float(right) for left, right in pairs])
                ),
                "mcnemar_exact_p": _mcnemar_exact(topology_only, baseline_only),
                "matched_odds_ratio": (topology_only + 0.5)
                / (baseline_only + 0.5),
            }
        )
    return _holm_adjust(rows)


def evaluate_route_mujoco_direct(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    output_dir: Path,
    physical_seeds: int = 2,
    assignment_indices: Iterable[int] = tuple(range(9)),
    policies: Iterable[str] = DIRECT_ROUTE_POLICY_NAMES,
    difficulty: float = 0.5,
    base_seed: int = 63_000_000,
    physics_steps: int = 6,
    max_steps: int = 360,
    device_name: str = "cpu",
    render_representative: bool = True,
) -> dict[str, Any]:
    if physical_seeds < 1:
        raise ValueError("physical_seeds must be at least one")
    if physics_steps < 1 or max_steps < 1:
        raise ValueError("physics_steps and max_steps must be positive")
    assignment_list = [int(index) for index in assignment_indices]
    if not assignment_list or any(
        index < 0 or index >= len(ROUTE_TARGET_ASSIGNMENTS)
        for index in assignment_list
    ):
        raise ValueError("assignment_indices must select RouteBot assignments")
    policy_names = list(dict.fromkeys(str(name) for name in policies))
    unknown = set(policy_names) - set(DIRECT_ROUTE_POLICY_NAMES)
    if unknown:
        raise ValueError(f"unknown direct RouteBot policies: {sorted(unknown)}")
    if not topology_checkpoint.is_file() or not geometry_checkpoint.is_file():
        raise FileNotFoundError("both frozen graph-policy checkpoints are required")

    mujoco = _load_mujoco()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_objects: dict[str, Any] = {}
    if "learned_topology" in policy_names:
        policy_objects["learned_topology"] = LearnedGraphPolicy(
            topology_checkpoint,
            "route",
            name="learned_topology",
            device_name=device_name,
        )
    if "learned_geometry" in policy_names:
        policy_objects["learned_geometry"] = LearnedGraphPolicy(
            geometry_checkpoint,
            "route",
            name="learned_geometry",
            device_name=device_name,
        )
    if "teacher_topology" in policy_names:
        policy_objects["teacher_topology"] = _DirectTopologyTeacher()

    records: list[dict[str, Any]] = []
    representative_render: str | None = None
    representative_video: str | None = None
    for seed_offset in range(physical_seeds):
        seed = int(base_seed + seed_offset)
        for assignment_index in assignment_list:
            target_assignment = ROUTE_TARGET_ASSIGNMENTS[assignment_index]
            for policy_name in policy_names:
                render_path = None
                if (
                    render_representative
                    and representative_render is None
                    and policy_name == "learned_topology"
                ):
                    render_path = output_dir / "route_mujoco_direct_final.png"
                    representative_render = str(render_path.resolve())
                    representative_video = str(render_path.with_suffix(".gif").resolve())
                records.append(
                    _run_direct_episode(
                        mujoco,
                        policy_objects[policy_name],
                        policy_name=policy_name,
                        target_assignment=target_assignment,
                        assignment_index=assignment_index,
                        seed=seed,
                        difficulty=difficulty,
                        physics_steps=physics_steps,
                        max_steps=max_steps,
                        render_path=render_path,
                    )
                )

    evidence_level = (
        "cross-physics-scale"
        if physical_seeds >= 20 and len(assignment_list) == len(ROUTE_TARGET_ASSIGNMENTS)
        else "development"
    )
    return {
        "name": "RouteBot direct MuJoCo closed-loop counterfactual audit",
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": f"MuJoCo {mujoco.__version__} elasticity cable plugin",
        "config": {
            "physical_seeds": physical_seeds,
            "assignment_indices": assignment_list,
            "policies": policy_names,
            "difficulty": difficulty,
            "base_seed": base_seed,
            "physics_steps_per_action": physics_steps,
            "max_policy_steps": max_steps,
            "device": device_name,
            "evidence_level": evidence_level,
        },
        "checkpoints": {
            "learned_topology": {
                "path": str(topology_checkpoint.resolve()),
                "sha256": _checkpoint_sha256(topology_checkpoint),
            },
            "learned_geometry": {
                "path": str(geometry_checkpoint.resolve()),
                "sha256": _checkpoint_sha256(geometry_checkpoint),
            },
        },
        "scene": {
            "path": str(SCENE_SPECS["route"].xml_path.resolve()),
            "sha256": _checkpoint_sha256(SCENE_SPECS["route"].xml_path),
        },
        "observation_calibration": {
            "selection_rule": (
                "Fixture path and arc-length knots were screened with the scripted "
                "teacher only, then frozen before the scaled learned-policy run."
            ),
            "pbd_particle_knots": _DIRECT_PBD_KNOTS.tolist(),
            "mujoco_segment_knots": _DIRECT_MUJOCO_KNOTS.tolist(),
            "graph_sample_segments": list(DIRECT_ROUTE_SEGMENTS),
            "nominal_world_landmarks_xy_m": _NOMINAL_WORLD_LANDMARKS.tolist(),
            "canonical_policy_landmarks_xy": _CANONICAL_LANDMARKS.tolist(),
            "world_to_policy_affine": WORLD_TO_POLICY.tolist(),
        },
        "intervention": (
            "Within each MuJoCo physical seed, fixture poses, cable state, controller, and "
            "initial gripper pose are identical. Only the observable semantic edges mapping "
            "three ordered cable particles to clips change across the nine assignments."
        ),
        "claim_scope": (
            "Closed loop from current MuJoCo object poses to graph policy to MuJoCo "
            "task-space mocap actions and native cable constraints. No PBD state or event "
            "is used. The UR10e mesh is a collision-disabled visual IK twin and control is "
            "not torque/force control; this is not hardware or production evidence."
        ),
        "records": records,
        "aggregate_by_policy": aggregate_success(records, ("policy",)),
        "aggregate_by_policy_assignment": aggregate_success(
            records,
            ("policy", "assignment_index", "target_assignment"),
        ),
        "paired_comparisons": _paired_policy_comparisons(records),
        "representative_render": representative_render,
        "representative_video": representative_video,
    }


def save_route_mujoco_direct(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.setdefault(
        "scene",
        {
            "path": str(SCENE_SPECS["route"].xml_path.resolve()),
            "sha256": _checkpoint_sha256(SCENE_SPECS["route"].xml_path),
        },
    )
    report.setdefault(
        "observation_calibration",
        {
            "selection_rule": (
                "Fixture path and arc-length knots were screened with the scripted "
                "teacher only, then frozen before the scaled learned-policy run."
            ),
            "pbd_particle_knots": _DIRECT_PBD_KNOTS.tolist(),
            "mujoco_segment_knots": _DIRECT_MUJOCO_KNOTS.tolist(),
            "graph_sample_segments": list(DIRECT_ROUTE_SEGMENTS),
            "nominal_world_landmarks_xy_m": _NOMINAL_WORLD_LANDMARKS.tolist(),
            "canonical_policy_landmarks_xy": _CANONICAL_LANDMARKS.tolist(),
            "world_to_policy_affine": WORLD_TO_POLICY.tolist(),
        },
    )
    report_path = output_dir / "route_mujoco_direct_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def write_csv(name: str, rows: list[dict[str, Any]]) -> str:
        path = output_dir / name
        if rows:
            normalized = [
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
                for row in rows
            ]
            fields = sorted({key for row in normalized for key in row})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(normalized)
        return str(path.resolve())

    episodes_path = write_csv(
        "route_mujoco_direct_episodes.csv",
        report["records"],
    )
    aggregate_path = write_csv(
        "route_mujoco_direct_aggregate.csv",
        report["aggregate_by_policy"],
    )
    assignment_path = write_csv(
        "route_mujoco_direct_assignments.csv",
        report["aggregate_by_policy_assignment"],
    )
    paired_path = write_csv(
        "route_mujoco_direct_paired.csv",
        report["paired_comparisons"],
    )

    labels = {
        "learned_topology": "TopoHarness",
        "learned_geometry": "Geometry",
        "teacher_topology": "Scripted teacher",
    }
    latex = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Policy & Episodes & Success (\\%) \\\\",
        "\\midrule",
    ]
    markdown = [
        "# RouteBot direct MuJoCo closed-loop audit",
        "",
        (
            f"> Evidence level: {report['config']['evidence_level']}. "
            "This is task-space mocap control, not torque control or hardware evidence."
        ),
        "",
        "| Policy | Episodes | Success (95% Wilson CI) |",
        "|---|---:|---:|",
    ]
    for row in report["aggregate_by_policy"]:
        label = labels.get(str(row["policy"]), str(row["policy"]))
        latex.append(
            f"{label} & {row['episodes']} & {100.0 * row['success_rate']:.1f} \\\\"
        )
        markdown.append(
            f"| {label} | {row['episodes']} | {row['success_rate']:.3f} "
            f"[{row['success_ci95_low']:.3f}, {row['success_ci95_high']:.3f}] |"
        )
    latex.extend(("\\bottomrule", "\\end{tabular}", ""))
    markdown.extend(("", report["claim_scope"], ""))
    latex_path = output_dir / "route_mujoco_direct_table.tex"
    latex_path.write_text("\n".join(latex), encoding="utf-8")
    results_path = output_dir / "ROUTE_MUJOCO_DIRECT_RESULTS.md"
    results_path.write_text("\n".join(markdown), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": episodes_path,
        "aggregate_csv": aggregate_path,
        "assignments_csv": assignment_path,
        "paired_csv": paired_path,
        "latex_table": str(latex_path.resolve()),
        "results_markdown": str(results_path.resolve()),
        "representative_render": report["representative_render"],
        "representative_video": report.get("representative_video"),
        "episode_count": len(report["records"]),
    }
