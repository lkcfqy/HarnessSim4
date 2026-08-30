"""Policy-conditioned PBD-to-MuJoCo motion-primitive replay.

This module is intentionally explicit about its evidence level. The graph
policy closes its task loop in PBD, while its selected targets and semantic
events drive MuJoCo mocap bodies and switchable equality constraints. MuJoCo
then supplies an independent 3-D dynamics execution check. This is stronger
than a fixed scripted animation, but it is not an end-to-end MuJoCo-observation
policy and must not be reported as such.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.sim.benchmark import aggregate_success
from harnessbench.sim.envs import make_env
from harnessbench.sim.mujoco_backend import SCENE_SPECS, _assert_finite, _load_mujoco
from harnessbench.sim.policies import TopologyPolicy
from harnessbench.sim.realistic_robots import RealisticRobotRig, load_realistic_model


@dataclass
class PBDTransition:
    action: np.ndarray
    before: dict
    after: dict


@dataclass
class PBDRollout:
    task: str
    policy: str
    seed: int
    difficulty: float
    success: bool
    steps: int
    metrics: dict
    transitions: list[PBDTransition]


def _collect_pbd_rollout(task: str, policy, seed: int, difficulty: float) -> PBDRollout:
    env = make_env(task)
    env.reset(seed=seed, difficulty=difficulty)
    policy.reset(seed + 100_003)
    transitions: list[PBDTransition] = []
    terminated = truncated = False
    while not (terminated or truncated):
        before = deepcopy(env.observation())
        action = np.asarray(policy.act(env), dtype=np.float64)
        _, _, terminated, truncated, _ = env.step(action)
        transitions.append(
            PBDTransition(
                action=action.copy(),
                before=before,
                after=deepcopy(env.observation()),
            )
        )
    return PBDRollout(
        task=task,
        policy=policy.name,
        seed=seed,
        difficulty=difficulty,
        success=bool(terminated and env.success()),
        steps=env.step_count,
        metrics=env.metrics(),
        transitions=transitions,
    )


def _object_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise KeyError(name)
    return object_id


def _mocap_id(mujoco: Any, model: Any, body_name: str) -> int:
    body_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"{body_name} is not a mocap body")
    return mocap_id


def _advance_mocap(
    mujoco: Any,
    model: Any,
    data: Any,
    targets: dict[int, np.ndarray],
    *,
    physics_steps: int,
    max_motion: float | None = 0.018,
    robot_rig: RealisticRobotRig | None = None,
) -> None:
    starts = {mocap_id: data.mocap_pos[mocap_id].copy() for mocap_id in targets}
    ends: dict[int, np.ndarray] = {}
    for mocap_id, desired in targets.items():
        delta = np.asarray(desired, dtype=float) - starts[mocap_id]
        distance = float(np.linalg.norm(delta))
        if max_motion is not None and distance > max_motion:
            delta *= max_motion / distance
        ends[mocap_id] = starts[mocap_id] + delta
    for step in range(physics_steps):
        weight = (step + 1) / physics_steps
        weight = weight * weight * (3.0 - 2.0 * weight)
        for mocap_id in targets:
            data.mocap_pos[mocap_id] = starts[mocap_id] * (1.0 - weight) + ends[mocap_id] * weight
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)
    if robot_rig is not None:
        robot_rig.sync(data, iterations=18)


def _settle(
    mujoco: Any,
    model: Any,
    data: Any,
    steps: int = 80,
    *,
    robot_rig: RealisticRobotRig | None = None,
) -> None:
    if robot_rig is None:
        mujoco.mj_forward(model, data)
    else:
        robot_rig.initialize(data)
    for _ in range(steps):
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.asarray(source, dtype=float), np.ones(len(source))))
    coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(target, dtype=float), rcond=None)
    return coefficients


def _map_affine(point: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray([point[0], point[1], 1.0], dtype=float) @ coefficients


def _render_final(mujoco: Any, model: Any, data: Any, path: Path) -> None:
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        renderer.update_scene(data, camera="overview")
        Image.fromarray(renderer.render()).save(path, optimize=True)
    finally:
        renderer.close()


def _replay_inspect(
    mujoco: Any,
    rollout: PBDRollout,
    *,
    physics_steps: int,
    render_path: Path | None,
) -> dict:
    spec = SCENE_SPECS["inspect"]
    model, robot_rig = load_realistic_model(mujoco, "inspect", spec.xml_path)
    data = mujoco.MjData(model)
    scanner_mocap = _mocap_id(mujoco, model, "scanner")
    _settle(mujoco, model, data, robot_rig=robot_rig)
    initial = rollout.transitions[0].before
    sites = np.asarray(initial["sites"], dtype=float)
    x_low, x_high = float(sites[:, 0].min()), float(sites[:, 0].max())
    defect_ids = [
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, f"defect_{index}")
        for index in range(1, 4)
    ]
    fov_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "scanner_fov")
    visited: set[int] = set()
    peak_speed = 0.0
    previous = data.mocap_pos[scanner_mocap].copy()
    for transition in rollout.transitions:
        x_fraction = (float(transition.action[0]) - x_low) / max(x_high - x_low, 1e-9)
        target = np.asarray([-0.40 + 0.82 * x_fraction, -0.09, 0.24])
        _advance_mocap(
            mujoco,
            model,
            data,
            {scanner_mocap: target},
            physics_steps=physics_steps,
            robot_rig=robot_rig,
        )
        current = data.mocap_pos[scanner_mocap].copy()
        peak_speed = max(peak_speed, float(np.linalg.norm(current - previous)))
        previous = current
        if transition.action[2] >= 0.5:
            fov_position = data.site_xpos[fov_id]
            for defect_index, site_id in enumerate(defect_ids):
                if np.linalg.norm(data.site_xpos[site_id] - fov_position) <= 0.13:
                    visited.add(defect_index)
    _assert_finite(data, "inspect_transfer")
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "mujoco_replay_success": len(visited) == len(defect_ids),
        "native_coverage": len(visited) / len(defect_ids),
        "visited_native_sites": len(visited),
        "peak_mocap_step_m": peak_speed,
        "finite_state": True,
        "visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
        "physics_steps": 80 + physics_steps * rollout.steps,
    }


def _replay_insert(
    mujoco: Any,
    rollout: PBDRollout,
    *,
    physics_steps: int,
    render_path: Path | None,
) -> dict:
    spec = SCENE_SPECS["insert"]
    model, robot_rig = load_realistic_model(mujoco, "insert", spec.xml_path)
    data = mujoco.MjData(model)
    gripper_mocap = _mocap_id(mujoco, model, "gripper")
    _settle(mujoco, model, data, robot_rig=robot_rig)
    initial = rollout.transitions[0].before
    socket = np.asarray(initial["socket"], dtype=float)
    axis = np.asarray(initial["socket_axis"], dtype=float)
    normal = np.asarray([-axis[1], axis[0]])
    preinsert = socket - axis * (0.13 + 0.035 * rollout.difficulty)
    travel = float(np.dot(socket + axis * 0.008 - preinsert, axis))
    for transition in rollout.transitions:
        delta = transition.action[:2] - preinsert
        along = float(np.dot(delta, axis)) / max(travel, 1e-9)
        lateral = float(np.dot(delta, normal))
        target = np.asarray([0.20 + 0.03 * along, 0.5 * lateral, 0.13])
        _advance_mocap(
            mujoco,
            model,
            data,
            {gripper_mocap: target},
            physics_steps=physics_steps,
            robot_rig=robot_rig,
        )
    _assert_finite(data, "insert_transfer")
    endpoint_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "insert_B_last")
    goal_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "insertion_goal")
    goal_error = float(np.linalg.norm(data.xpos[endpoint_id, :2] - data.site_xpos[goal_id, :2]))
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "mujoco_replay_success": goal_error <= 0.020,
        "native_goal_xy_error_m": goal_error,
        "finite_state": True,
        "visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
        "physics_steps": 80 + physics_steps * rollout.steps,
    }


def _route_body_name(pbd_index: int) -> str:
    # Calibrated arc-length correspondence between the 29-node PBD chain and
    # the 34 moving bodies in the MuJoCo cable.  The knots are the three
    # typed clip particles plus the endpoint; interpolation also gives a
    # deterministic mapping for geometry-only policy selections.
    segment = round(
        float(
            np.interp(
                np.clip(pbd_index, 0, 28),
                np.asarray([0, 7, 15, 23, 28]),
                np.asarray([0, 8, 18, 25, 33]),
            )
        )
    )
    if segment == 0:
        return "route_B_first"
    if segment == 33:
        return "route_B_last"
    return f"route_B_{segment}"


def _equality_id(mujoco: Any, model: Any, name: str) -> int:
    return _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_EQUALITY, name)


def _set_connect_constraint(
    model: Any,
    data: Any,
    equality_id: int,
    *,
    body1_id: int | None = None,
    body2_anchor: np.ndarray | None = None,
    active: bool,
) -> None:
    if body1_id is not None:
        model.eq_obj1id[equality_id] = body1_id
    model.eq_data[equality_id, :3] = 0.0
    model.eq_data[equality_id, 3:6] = np.zeros(3) if body2_anchor is None else body2_anchor
    data.eq_active[equality_id] = active


def _replay_route(
    mujoco: Any,
    rollout: PBDRollout,
    *,
    physics_steps: int,
    render_path: Path | None,
) -> dict:
    spec = SCENE_SPECS["route"]
    model, robot_rig = load_realistic_model(mujoco, "route", spec.xml_path)
    data = mujoco.MjData(model)
    gripper_mocap = _mocap_id(mujoco, model, "route_gripper")
    equality_names = [
        "route_grasp",
        "route_grasp_p7",
        "route_grasp_p15",
        "route_grasp_p23",
        "route_latch_1",
        "route_latch_2",
        "route_latch_3",
        "route_finish_latch",
    ]
    equality_ids = {name: _equality_id(mujoco, model, name) for name in equality_names}
    for equality_id in equality_ids.values():
        data.eq_active[equality_id] = False
    _settle(mujoco, model, data, robot_rig=robot_rig)

    initial = rollout.transitions[0].before
    pbd_landmarks = np.vstack((initial["clips"], initial["finish"]))
    clip_body_ids = [
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, f"clip_{index}")
        for index in range(1, 4)
    ]
    finish_site_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "route_finish")
    mujoco_landmarks = np.vstack(
        ([data.xpos[body_id, :2] for body_id in clip_body_ids], data.site_xpos[finish_site_id, :2])
    )
    affine = _fit_affine(pbd_landmarks, mujoco_landmarks)

    def map_route_action(action_xy: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(pbd_landmarks - np.asarray(action_xy), axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= 0.08:
            return mujoco_landmarks[nearest].copy()
        return _map_affine(action_xy, affine)

    grasp_constraints = {
        7: equality_ids["route_grasp_p7"],
        15: equality_ids["route_grasp_p15"],
        23: equality_ids["route_grasp_p23"],
        28: equality_ids["route_grasp"],
    }
    active_grasp_id: int | None = None
    active_particle: int | None = None
    bound_particles: dict[int, int] = {}
    finish_capture_error: float | None = None
    finish_latched = False

    for transition in rollout.transitions:
        before_grasp = transition.before["grasp_idx"]
        after_grasp = transition.after["grasp_idx"]
        if before_grasp is None:
            positions = np.asarray(transition.before["cable_positions"])
            selected_particle = int(
                np.argmin(np.linalg.norm(positions - transition.action[:2], axis=1))
            )
            selected_body = _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                _route_body_name(selected_particle),
            )
            desired = data.xpos[selected_body].copy()
            desired[2] = 0.14
        else:
            mapped_xy = map_route_action(transition.action[:2])
            desired = np.asarray([mapped_xy[0], mapped_xy[1], 0.14])
        _advance_mocap(
            mujoco,
            model,
            data,
            {gripper_mocap: desired},
            physics_steps=physics_steps,
            robot_rig=robot_rig,
        )

        if before_grasp is None and after_grasp is not None:
            active_particle = int(after_grasp)
            body_id = _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                _route_body_name(active_particle),
            )
            capture = data.xpos[body_id].copy()
            capture[2] = 0.14
            _advance_mocap(
                mujoco,
                model,
                data,
                {gripper_mocap: capture},
                physics_steps=max(physics_steps * 2, 10),
                max_motion=None,
                robot_rig=robot_rig,
            )
            active_grasp_id = grasp_constraints.get(active_particle)
            if active_grasp_id is not None:
                _set_connect_constraint(
                    model,
                    data,
                    active_grasp_id,
                    active=True,
                )
                mujoco.mj_forward(model, data)

        before_bindings = transition.before["bindings"]
        after_bindings = transition.after["bindings"]
        new_clips = set(after_bindings) - set(before_bindings)
        for clip_index in new_clips:
            particle = int(after_bindings[clip_index])
            latch_id = equality_ids[f"route_latch_{int(clip_index) + 1}"]
            if particle == int(initial["target_particle_indices"][int(clip_index)]):
                _set_connect_constraint(
                    model,
                    data,
                    latch_id,
                    body2_anchor=np.asarray([0.0, 0.0, 0.095]),
                    active=True,
                )
            if active_grasp_id is not None:
                data.eq_active[active_grasp_id] = False
                active_grasp_id = None
            bound_particles[int(clip_index)] = particle
            active_particle = None
            mujoco.mj_forward(model, data)
        if before_grasp is not None and after_grasp is None and not new_clips:
            if int(before_grasp) == 28:
                endpoint_id = _object_id(
                    mujoco,
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "route_B_last",
                )
                finish_capture_error = float(
                    np.linalg.norm(data.xpos[endpoint_id, :2] - data.site_xpos[finish_site_id, :2])
                )
                if finish_capture_error <= 0.06:
                    _set_connect_constraint(
                        model,
                        data,
                        equality_ids["route_finish_latch"],
                        body2_anchor=np.asarray([0.0, 0.0, 0.095]),
                        active=True,
                    )
                    finish_latched = True
            if active_grasp_id is not None:
                data.eq_active[active_grasp_id] = False
                active_grasp_id = None
            active_particle = None

    if rollout.transitions:
        final_xy = map_route_action(rollout.transitions[-1].action[:2])
        _advance_mocap(
            mujoco,
            model,
            data,
            {gripper_mocap: np.asarray([final_xy[0], final_xy[1], 0.14])},
            physics_steps=120,
            max_motion=None,
            robot_rig=robot_rig,
        )
        if active_particle == 28 and not finish_latched:
            endpoint_id = _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "route_B_last",
            )
            finish_capture_error = float(
                np.linalg.norm(data.xpos[endpoint_id, :2] - data.site_xpos[finish_site_id, :2])
            )
            if finish_capture_error <= 0.06:
                _set_connect_constraint(
                    model,
                    data,
                    equality_ids["route_finish_latch"],
                    body2_anchor=np.asarray([0.0, 0.0, 0.095]),
                    active=True,
                )
                finish_latched = True
                if active_grasp_id is not None:
                    data.eq_active[active_grasp_id] = False
                    active_grasp_id = None
                active_particle = None
                mujoco.mj_forward(model, data)
        for _ in range(60):
            robot_rig.hold(data)
            mujoco.mj_step(model, data)

    _assert_finite(data, "route_transfer")
    latch_errors = []
    for clip_index, particle in sorted(bound_particles.items()):
        body_id = _object_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            _route_body_name(particle),
        )
        latch_errors.append(
            float(np.linalg.norm(data.xpos[body_id, :2] - data.xpos[clip_body_ids[clip_index], :2]))
        )
    endpoint_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "route_B_last")
    finish_error = float(
        np.linalg.norm(data.xpos[endpoint_id, :2] - data.site_xpos[finish_site_id, :2])
    )
    target_indices = [int(value) for value in initial["target_particle_indices"]]
    semantic_correct = len(bound_particles) == 3 and all(
        bound_particles.get(index) == target_indices[index] for index in range(3)
    )
    native_success = bool(
        semantic_correct
        and len(latch_errors) == 3
        and max(latch_errors, default=float("inf")) <= 0.05
        and finish_latched
        and finish_error <= 0.05
    )
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "mujoco_replay_success": native_success,
        "native_semantic_latches_correct": semantic_correct,
        "native_latched_clips": len(bound_particles),
        "native_max_latch_xy_error_m": max(latch_errors, default=None),
        "native_finish_captured": finish_latched,
        "native_finish_capture_xy_error_m": finish_capture_error,
        "native_finish_xy_error_m": finish_error,
        "finite_state": True,
        "visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
        "physics_steps": 260 + physics_steps * rollout.steps,
        "active_particle_at_end": active_particle,
    }


def _branch_candidate(action_xy: np.ndarray, initial: dict) -> tuple[str, int]:
    candidates: list[tuple[np.ndarray, str, int]] = []
    for semantic in ("A", "B"):
        for phase, waypoint in enumerate(initial["semantic_waypoints"][semantic]):
            side = "lower" if float(waypoint[1]) < 0.5 else "upper"
            candidates.append((np.asarray(waypoint), side, phase))
        target = np.asarray(initial["semantic_targets"][semantic])
        side = "lower" if float(target[1]) < 0.5 else "upper"
        candidates.append((target, side, 2))
    distances = [float(np.linalg.norm(point - action_xy)) for point, _, _ in candidates]
    _, side, phase = candidates[int(np.argmin(distances))]
    return side, phase


def _replay_branch(
    mujoco: Any,
    rollout: PBDRollout,
    *,
    physics_steps: int,
    render_path: Path | None,
) -> dict:
    spec = SCENE_SPECS["branch"]
    model, robot_rig = load_realistic_model(mujoco, "branch", spec.xml_path)
    data = mujoco.MjData(model)
    mocap_a = _mocap_id(mujoco, model, "gripper_a")
    mocap_b = _mocap_id(mujoco, model, "gripper_b")
    _settle(mujoco, model, data, robot_rig=robot_rig)
    initial = rollout.transitions[0].before
    semantic_a_is_physical_a = int(initial["semantic_endpoints"]["A"]) == 18
    primitive_targets = {
        "lower": (
            np.asarray([0.08, -0.38, 0.18]),
            np.asarray([0.34, 0.34, 0.20]),
            np.asarray([0.43, 0.30, 0.15]),
        ),
        "upper": (
            np.asarray([0.08, 0.38, 0.19]),
            np.asarray([0.04, -0.36, 0.23]),
            np.asarray([0.43, -0.30, 0.15]),
        ),
    }
    last_targets: dict[int, np.ndarray] = {}
    for transition in rollout.transitions:
        semantic_actions = {
            "A": transition.action[:2],
            "B": transition.action[3:5],
        }
        physical_a_action = semantic_actions["A" if semantic_a_is_physical_a else "B"]
        physical_b_action = semantic_actions["B" if semantic_a_is_physical_a else "A"]
        side_a, phase_a = _branch_candidate(physical_a_action, initial)
        side_b, phase_b = _branch_candidate(physical_b_action, initial)
        last_targets = {
            mocap_a: primitive_targets[side_a][phase_a],
            mocap_b: primitive_targets[side_b][phase_b],
        }
        _advance_mocap(
            mujoco,
            model,
            data,
            last_targets,
            physics_steps=physics_steps,
            robot_rig=robot_rig,
        )
    _advance_mocap(
        mujoco,
        model,
        data,
        last_targets,
        physics_steps=120,
        max_motion=None,
        robot_rig=robot_rig,
    )
    for _ in range(60):
        robot_rig.hold(data)
        mujoco.mj_step(model, data)
    _assert_finite(data, "branch_transfer")
    endpoint_a = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "branch_a_B_last")
    endpoint_b = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "branch_b_B_last")
    target_a = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "target_a")
    target_b = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "target_b")
    error_a = float(np.linalg.norm(data.xpos[endpoint_a, :2] - data.site_xpos[target_a, :2]))
    error_b = float(np.linalg.norm(data.xpos[endpoint_b, :2] - data.site_xpos[target_b, :2]))
    native_success = error_a <= 0.055 and error_b <= 0.055
    if render_path is not None:
        _render_final(mujoco, model, data, render_path)
    return {
        "mujoco_replay_success": native_success,
        "native_endpoint_a_xy_error_m": error_a,
        "native_endpoint_b_xy_error_m": error_b,
        "native_mean_endpoint_xy_error_m": 0.5 * (error_a + error_b),
        "finite_state": True,
        "visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
        "physics_steps": 260 + physics_steps * rollout.steps,
        "semantic_target_swap": bool(initial["semantic_target_swap"]),
    }


def _replay_rollout(
    mujoco: Any,
    rollout: PBDRollout,
    *,
    physics_steps: int,
    render_path: Path | None,
) -> dict:
    handlers = {
        "inspect": _replay_inspect,
        "insert": _replay_insert,
        "route": _replay_route,
        "branch": _replay_branch,
    }
    return handlers[rollout.task](
        mujoco,
        rollout,
        physics_steps=physics_steps,
        render_path=render_path,
    )


def _rank(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _ranking_agreement(records: list[dict]) -> list[dict]:
    output = []
    for task in sorted({record["task_key"] for record in records}):
        task_records = [record for record in records if record["task_key"] == task]
        policies = sorted({record["policy"] for record in task_records})
        pbd_rates = [
            float(
                np.mean(
                    [record["pbd_success"] for record in task_records if record["policy"] == policy]
                )
            )
            for policy in policies
        ]
        mujoco_rates = [
            float(
                np.mean(
                    [
                        record["mujoco_replay_success"]
                        for record in task_records
                        if record["policy"] == policy
                    ]
                )
            )
            for policy in policies
        ]
        pbd_ranks, mujoco_ranks = _rank(pbd_rates), _rank(mujoco_rates)
        if np.std(pbd_ranks) > 0 and np.std(mujoco_ranks) > 0:
            spearman = float(np.corrcoef(pbd_ranks, mujoco_ranks)[0, 1])
        else:
            spearman = None
        output.append(
            {
                "task_key": task,
                "policies": policies,
                "pbd_success_rates": pbd_rates,
                "mujoco_replay_success_rates": mujoco_rates,
                "spearman_policy_ranking": spearman,
            }
        )
    return output


def evaluate_mujoco_transfer(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    output_dir: Path,
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    episodes: int = 2,
    difficulty: float = 0.5,
    base_seed: int = 51_000_000,
    physics_steps_per_action: int = 6,
    device_name: str = "cpu",
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if physics_steps_per_action < 1:
        raise ValueError("physics_steps_per_action must be at least one")
    mujoco = _load_mujoco()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_list = list(tasks)
    records: list[dict] = []
    final_frames: list[str] = []
    for task_index, task in enumerate(task_list):
        policies = {
            "learned_topology": LearnedGraphPolicy(
                topology_checkpoint,
                task,
                name="learned_topology",
                device_name=device_name,
            ),
            "learned_geometry": LearnedGraphPolicy(
                geometry_checkpoint,
                task,
                name="learned_geometry",
                device_name=device_name,
            ),
            "teacher_topology": TopologyPolicy(base_seed),
        }
        for episode in range(episodes):
            seed = base_seed + task_index * 100_000 + episode
            for policy_name, policy in policies.items():
                rollout = _collect_pbd_rollout(task, policy, seed, difficulty)
                render_path = None
                if episode == 0 and policy_name == "learned_topology":
                    render_path = output_dir / f"{task}_learned_topology_mujoco_final.png"
                    final_frames.append(render_path.name)
                native = _replay_rollout(
                    mujoco,
                    rollout,
                    physics_steps=physics_steps_per_action,
                    render_path=render_path,
                )
                records.append(
                    {
                        "task_key": task,
                        "policy": policy_name,
                        "seed": seed,
                        "difficulty": difficulty,
                        "pbd_success": rollout.success,
                        "pbd_steps": rollout.steps,
                        **native,
                    }
                )

    pbd_records = [{**record, "success": record["pbd_success"]} for record in records]
    mujoco_records = [{**record, "success": record["mujoco_replay_success"]} for record in records]
    return {
        "name": "HarnessSim4 policy-conditioned PBD-to-MuJoCo transfer audit",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": f"MuJoCo {mujoco.__version__} elasticity cable plugin",
        "config": {
            "tasks": task_list,
            "episodes_per_task_policy": episodes,
            "difficulty": difficulty,
            "base_seed": base_seed,
            "physics_steps_per_action": physics_steps_per_action,
            "device": device_name,
        },
        "checkpoints": {
            "learned_topology": str(topology_checkpoint.resolve()),
            "learned_geometry": str(geometry_checkpoint.resolve()),
        },
        "claim_scope": (
            "Policy-conditioned hierarchical transfer: PBD closes task events and selects typed "
            "motion primitives; MuJoCo executes 3-D dynamics and native goal checks. This is not "
            "an end-to-end policy observing MuJoCo state and is not real-world evidence."
        ),
        "records": records,
        "pbd_aggregate": aggregate_success(pbd_records, ("task_key", "policy")),
        "mujoco_replay_aggregate": aggregate_success(mujoco_records, ("task_key", "policy")),
        "ranking_agreement": _ranking_agreement(records),
        "final_frames": final_frames,
    }


def save_mujoco_transfer(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mujoco_transfer_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    records_path = output_dir / "mujoco_transfer_episodes.csv"
    fieldnames = sorted({key for record in report["records"] for key in record})
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["records"])

    aggregate_path = output_dir / "mujoco_transfer_aggregate.csv"
    aggregate_rows = []
    for backend_key in ("pbd_aggregate", "mujoco_replay_aggregate"):
        for row in report[backend_key]:
            aggregate_rows.append({"evaluation": backend_key.removesuffix("_aggregate"), **row})
    aggregate_fields = sorted({key for row in aggregate_rows for key in row})
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    lines = [
        "# PBD → MuJoCo policy-conditioned transfer audit",
        "",
        "> Hierarchical development evidence only; not end-to-end MuJoCo observation control.",
        "",
        "| Task | Policy | PBD success | MuJoCo replay success |",
        "|---|---|---:|---:|",
    ]
    pbd_index = {(row["task_key"], row["policy"]): row for row in report["pbd_aggregate"]}
    for row in report["mujoco_replay_aggregate"]:
        pbd = pbd_index[(row["task_key"], row["policy"])]
        lines.append(
            f"| {row['task_key']} | {row['policy']} | {pbd['success_rate']:.3f} | "
            f"{row['success_rate']:.3f} |"
        )
    results_path = output_dir / "MUJOCO_TRANSFER_RESULTS.md"
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": str(records_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
        "results_markdown": str(results_path.resolve()),
        "final_frames": [str((output_dir / name).resolve()) for name in report["final_frames"]],
        "episode_count": len(report["records"]),
    }
