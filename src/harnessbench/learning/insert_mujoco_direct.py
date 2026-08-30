"""Direct closed-loop MuJoCo contact audit for InsertBot.

Each controller observes the current MuJoCo terminal pose and native contact
force, then commands a compliant mocap target.  No event or trajectory from the
2.5-D benchmark is replayed.  The audit is intentionally task-space: the KUKA
model is visual-only and the experiment does not claim actuator, torque, or
hardware calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from harnessbench.learning.mujoco_transfer import _mocap_id, _object_id
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact
from harnessbench.sim.mujoco_backend import SCENE_SPECS, _assert_finite, _load_mujoco
from harnessbench.sim.realistic_robots import load_realistic_model

DIRECT_INSERT_POLICIES = (
    "contact_belief",
    "guarded_admittance",
    "direct_insertion",
    "contact_no_orientation",
    "contact_no_retract",
)


@dataclass(frozen=True)
class DirectInsertConfig:
    difficulties: tuple[float, ...] = (0.2, 0.5, 0.8)
    physical_seeds_per_difficulty: int = 20
    base_seed: int = 91_000_000
    policies: tuple[str, ...] = DIRECT_INSERT_POLICIES
    max_control_steps: int = 220
    physics_steps_per_control: int = 6
    verification_steps: int = 5


@dataclass(frozen=True)
class DirectScenario:
    seed: int
    difficulty: float
    start_y_m: float
    start_z_m: float
    start_yaw_rad: float
    start_pitch_rad: float
    vision_bias_y_m: float
    vision_bias_z_m: float
    vision_bias_yaw_rad: float
    vision_bias_pitch_rad: float
    force_bias_n: float
    opening_half_y_m: float
    opening_half_z_m: float
    friction: float
    damage_force_n: float
    noise_position_m: tuple[float, ...]
    noise_angle_rad: tuple[float, ...]
    noise_force_n: tuple[float, ...]


@dataclass
class DirectMemory:
    target_y_obs_m: float
    target_z_obs_m: float
    target_yaw_obs_rad: float = 0.0
    target_pitch_obs_rad: float = 0.0
    retract_steps: int = 0
    cooldown_steps: int = 0
    retries: int = 0
    contact_updates: int = 0
    stage: str = "prealign"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_insert_scenario(
    seed: int,
    difficulty: float,
    config: DirectInsertConfig,
) -> DirectScenario:
    rng = np.random.default_rng(seed + round(10_000 * difficulty))
    position_scale = 0.0035 + 0.0075 * difficulty
    angle_scale = 0.035 + 0.105 * difficulty
    bias_scale = 0.0008 + 0.0038 * difficulty
    angle_bias_scale = 0.008 + 0.050 * difficulty
    half_y = float(rng.uniform(0.0093, 0.0107) - 0.0013 * difficulty)
    half_z = float(rng.uniform(0.0085, 0.0098) - 0.0010 * difficulty)
    return DirectScenario(
        seed=int(seed),
        difficulty=float(difficulty),
        start_y_m=float(rng.uniform(-position_scale, position_scale)),
        start_z_m=float(0.055 + rng.uniform(-0.75 * position_scale, 0.75 * position_scale)),
        start_yaw_rad=float(rng.uniform(-angle_scale, angle_scale)),
        start_pitch_rad=float(rng.uniform(-0.8 * angle_scale, 0.8 * angle_scale)),
        vision_bias_y_m=float(rng.normal(0.0, bias_scale)),
        vision_bias_z_m=float(rng.normal(0.0, 0.8 * bias_scale)),
        vision_bias_yaw_rad=float(rng.normal(0.0, angle_bias_scale)),
        vision_bias_pitch_rad=float(rng.normal(0.0, 0.8 * angle_bias_scale)),
        force_bias_n=float(rng.normal(0.0, 0.20 + 0.22 * difficulty)),
        opening_half_y_m=half_y,
        opening_half_z_m=half_z,
        friction=float(rng.uniform(0.35, 0.52) + 0.20 * difficulty),
        damage_force_n=float(rng.uniform(90.0, 115.0) - 12.0 * difficulty),
        noise_position_m=tuple(
            float(value)
            for value in rng.normal(
                0.0,
                0.00018 + 0.00022 * difficulty,
                config.max_control_steps,
            )
        ),
        noise_angle_rad=tuple(
            float(value)
            for value in rng.normal(
                0.0,
                0.0025 + 0.0045 * difficulty,
                config.max_control_steps,
            )
        ),
        noise_force_n=tuple(
            float(value)
            for value in rng.normal(
                0.0,
                0.12 + 0.14 * difficulty,
                config.max_control_steps,
            )
        ),
    )


def _quat_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    half_pitch = 0.5 * pitch
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    quaternion = np.asarray((cy * cp, -sy * sp, cy * sp, sy * cp), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _yaw_pitch(rotation_flat: np.ndarray) -> tuple[float, float]:
    rotation = np.asarray(rotation_flat, dtype=np.float64).reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    return float(yaw), float(pitch)


def _configure_model(
    model: Any,
    mujoco: Any,
    scenario: DirectScenario,
) -> tuple[set[int], int]:
    guide_names = (
        "socket_guide_y_neg",
        "socket_guide_y_pos",
        "socket_guide_z_neg",
        "socket_guide_z_pos",
        "socket_backstop",
    )
    guide_ids = {_object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in guide_names}
    y_neg = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "socket_guide_y_neg")
    y_pos = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "socket_guide_y_pos")
    z_neg = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "socket_guide_z_neg")
    z_pos = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "socket_guide_z_pos")
    model.geom_pos[y_neg, 1] = -(scenario.opening_half_y_m + model.geom_size[y_neg, 1])
    model.geom_pos[y_pos, 1] = scenario.opening_half_y_m + model.geom_size[y_pos, 1]
    model.geom_pos[z_neg, 2] = -(scenario.opening_half_z_m + model.geom_size[z_neg, 2])
    model.geom_pos[z_pos, 2] = scenario.opening_half_z_m + model.geom_size[z_pos, 2]
    for geom_id in guide_ids:
        model.geom_friction[geom_id, 0] = scenario.friction
    terminal_geom = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "terminal_contact")
    return guide_ids, terminal_geom


def _initialize_state(
    mujoco: Any,
    model: Any,
    data: Any,
    scenario: DirectScenario,
    *,
    robot_rig: Any | None,
) -> int:
    mocap = _mocap_id(mujoco, model, "gripper")
    position = np.asarray((0.140, scenario.start_y_m, scenario.start_z_m), dtype=np.float64)
    quaternion = _quat_from_yaw_pitch(scenario.start_yaw_rad, scenario.start_pitch_rad)
    data.mocap_pos[mocap] = position
    data.mocap_quat[mocap] = quaternion
    joint_id = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "terminal_free")
    address = int(model.jnt_qposadr[joint_id])
    data.qpos[address : address + 3] = position
    data.qpos[address + 3 : address + 7] = quaternion
    if robot_rig is None:
        mujoco.mj_forward(model, data)
    else:
        robot_rig.initialize(data)
    for _ in range(60):
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)
    return mocap


def _native_contact(
    mujoco: Any,
    model: Any,
    data: Any,
    guide_ids: set[int],
    terminal_geom: int,
) -> tuple[float, float, float, int]:
    total_force = 0.0
    weighted_y = 0.0
    weighted_z = 0.0
    count = 0
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if terminal_geom not in pair or not pair.intersection(guide_ids):
            continue
        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, index, wrench)
        magnitude = float(np.linalg.norm(wrench[:3]))
        total_force += magnitude
        weighted_y += magnitude * float(contact.pos[1])
        weighted_z += magnitude * float(contact.pos[2] - 0.055)
        count += 1
    sign_y = float(np.sign(weighted_y)) if total_force > 1e-9 else 0.0
    sign_z = float(np.sign(weighted_z)) if total_force > 1e-9 else 0.0
    return total_force, sign_y, sign_z, count


def _advance_target(
    mujoco: Any,
    model: Any,
    data: Any,
    mocap: int,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    guide_ids: set[int],
    terminal_geom: int,
    *,
    physics_steps: int,
    force_stop_n: float | None,
    robot_rig: Any | None,
) -> tuple[float, float, float, int]:
    start_position = data.mocap_pos[mocap].copy()
    start_quaternion = data.mocap_quat[mocap].copy()
    if float(np.dot(start_quaternion, target_quaternion)) < 0.0:
        target_quaternion = -target_quaternion
    peak_force = 0.0
    peak_sign_y = 0.0
    peak_sign_z = 0.0
    native_contacts = 0
    for substep in range(physics_steps):
        weight = (substep + 1) / physics_steps
        weight = weight * weight * (3.0 - 2.0 * weight)
        data.mocap_pos[mocap] = (1.0 - weight) * start_position + weight * target_position
        quaternion = (1.0 - weight) * start_quaternion + weight * target_quaternion
        data.mocap_quat[mocap] = quaternion / np.linalg.norm(quaternion)
        if robot_rig is not None:
            robot_rig.hold(data)
        mujoco.mj_step(model, data)
        force, sign_y, sign_z, count = _native_contact(
            mujoco,
            model,
            data,
            guide_ids,
            terminal_geom,
        )
        native_contacts += count
        if force > peak_force:
            peak_force = force
            peak_sign_y = sign_y
            peak_sign_z = sign_z
        if force_stop_n is not None and force >= force_stop_n:
            break
    return peak_force, peak_sign_y, peak_sign_z, native_contacts


def _controller_target(
    policy: str,
    scenario: DirectScenario,
    memory: DirectMemory,
    observation: dict[str, float],
    current_mocap: np.ndarray,
    current_command_yaw: float,
    current_command_pitch: float,
) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    orientation_enabled = policy != "contact_no_orientation"
    threshold = 22.0 if policy == "guarded_admittance" else 6.5
    contact_detected = observation["force_n"] >= threshold
    if memory.cooldown_steps > 0:
        memory.cooldown_steps -= 1
    if (
        policy != "direct_insertion"
        and contact_detected
        and memory.cooldown_steps == 0
        and memory.retract_steps == 0
    ):
        memory.contact_updates += 1
        sign_y = observation["contact_sign_y"] or float(
            np.sign(observation["y_obs_m"] - memory.target_y_obs_m)
        )
        sign_z = observation["contact_sign_z"] or float(
            np.sign(observation["z_obs_m"] - memory.target_z_obs_m)
        )
        correction = 0.0015 if policy == "guarded_admittance" else 0.0022
        memory.target_y_obs_m -= sign_y * correction
        memory.target_z_obs_m -= sign_z * correction
        if orientation_enabled and policy != "guarded_admittance":
            memory.target_yaw_obs_rad -= float(np.sign(observation["yaw_obs_rad"])) * 0.030
            memory.target_pitch_obs_rad -= float(np.sign(observation["pitch_obs_rad"])) * 0.025
        memory.cooldown_steps = 6
        if policy != "contact_no_retract":
            memory.retract_steps = 5
            memory.retries += 1
            memory.stage = "retract"
        else:
            memory.stage = "contact_slide"

    target = current_mocap.copy()
    yaw_target = current_command_yaw
    pitch_target = current_command_pitch
    if memory.retract_steps > 0:
        target[0] -= 0.00125
        memory.retract_steps -= 1
        if memory.retract_steps == 0:
            memory.stage = "realign"
        stage = "retract"
    else:
        y_error = memory.target_y_obs_m - observation["y_obs_m"]
        z_error = memory.target_z_obs_m - observation["z_obs_m"]
        target[1] += float(np.clip(0.34 * y_error, -0.0015, 0.0015))
        target[2] += float(np.clip(0.34 * z_error, -0.0015, 0.0015))
        if orientation_enabled:
            yaw_target += float(
                np.clip(
                    0.28 * (memory.target_yaw_obs_rad - observation["yaw_obs_rad"]),
                    -0.015,
                    0.015,
                )
            )
            pitch_target += float(
                np.clip(
                    0.28 * (memory.target_pitch_obs_rad - observation["pitch_obs_rad"]),
                    -0.015,
                    0.015,
                )
            )
        aligned = (
            abs(y_error) < 0.0012
            and abs(z_error) < 0.0012
            and (
                not orientation_enabled
                or (
                    abs(observation["yaw_obs_rad"] - memory.target_yaw_obs_rad) < 0.024
                    and abs(observation["pitch_obs_rad"] - memory.target_pitch_obs_rad) < 0.024
                )
            )
        )
        if not aligned:
            if observation["tip_x_m"] < 0.195:
                target[0] += 0.00010
            stage = "prealign"
        else:
            speeds = {
                "contact_belief": 0.00068,
                "guarded_admittance": 0.00056,
                "direct_insertion": 0.00088,
                "contact_no_orientation": 0.00068,
                "contact_no_retract": 0.00068,
            }
            target[0] += speeds[policy]
            stage = "contact_slide" if memory.stage == "contact_slide" else "insert"
    if not orientation_enabled:
        yaw_target = memory.target_yaw_obs_rad
        pitch_target = memory.target_pitch_obs_rad
    return target, _quat_from_yaw_pitch(yaw_target, pitch_target), stage, yaw_target, pitch_target


def run_direct_insert_episode(
    scenario: DirectScenario,
    policy: str,
    config: DirectInsertConfig,
    *,
    render_path: Path | None = None,
    record_trace: bool = False,
) -> dict[str, Any]:
    if policy not in DIRECT_INSERT_POLICIES:
        raise KeyError(policy)
    mujoco = _load_mujoco()
    spec = SCENE_SPECS["insert"]
    if render_path is None:
        model = mujoco.MjModel.from_xml_path(str(spec.xml_path))
        robot_rig = None
    else:
        model, robot_rig = load_realistic_model(mujoco, "insert", spec.xml_path)
    data = mujoco.MjData(model)
    guide_ids, terminal_geom = _configure_model(model, mujoco, scenario)
    mocap = _initialize_state(
        mujoco,
        model,
        data,
        scenario,
        robot_rig=robot_rig,
    )
    terminal_body = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "terminal_tool")
    tip_site = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "terminal_tip")
    seat_site = _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "seat_goal")
    seat = data.site_xpos[seat_site].copy()
    memory = DirectMemory(
        target_y_obs_m=float(seat[1]),
        target_z_obs_m=float(seat[2]),
        target_yaw_obs_rad=(scenario.start_yaw_rad if policy == "contact_no_orientation" else 0.0),
        target_pitch_obs_rad=(
            scenario.start_pitch_rad if policy == "contact_no_orientation" else 0.0
        ),
    )
    peak_force = 0.0
    native_contact_samples = 0
    last_force = 0.0
    last_sign_y = 0.0
    last_sign_z = 0.0
    verification = 0
    success = False
    damage = False
    command_yaw = scenario.start_yaw_rad
    command_pitch = scenario.start_pitch_rad
    trace: list[dict[str, Any]] = []
    terminal_step = config.max_control_steps
    for step in range(config.max_control_steps):
        yaw, pitch = _yaw_pitch(data.xmat[terminal_body])
        noise_position = scenario.noise_position_m[step]
        noise_angle = scenario.noise_angle_rad[step]
        observation = {
            "tip_x_m": float(data.site_xpos[tip_site, 0]),
            "y_obs_m": float(
                data.xpos[terminal_body, 1] + scenario.vision_bias_y_m + noise_position
            ),
            "z_obs_m": float(
                data.xpos[terminal_body, 2] + scenario.vision_bias_z_m - 0.7 * noise_position
            ),
            "yaw_obs_rad": yaw + scenario.vision_bias_yaw_rad + noise_angle,
            "pitch_obs_rad": pitch + scenario.vision_bias_pitch_rad - 0.8 * noise_angle,
            "yaw_rad": yaw,
            "pitch_rad": pitch,
            "force_n": max(0.0, last_force + scenario.force_bias_n + scenario.noise_force_n[step]),
            "contact_sign_y": last_sign_y,
            "contact_sign_z": last_sign_z,
        }
        target_position, target_quaternion, stage, command_yaw, command_pitch = _controller_target(
            policy,
            scenario,
            memory,
            observation,
            data.mocap_pos[mocap].copy(),
            command_yaw,
            command_pitch,
        )
        force, sign_y, sign_z, contacts = _advance_target(
            mujoco,
            model,
            data,
            mocap,
            target_position,
            target_quaternion,
            guide_ids,
            terminal_geom,
            physics_steps=config.physics_steps_per_control,
            force_stop_n=(
                None
                if policy == "direct_insertion"
                else 22.0
                if policy == "guarded_admittance"
                else 6.5
            ),
            robot_rig=robot_rig,
        )
        last_force, last_sign_y, last_sign_z = force, sign_y, sign_z
        peak_force = max(peak_force, force)
        native_contact_samples += contacts
        damage = peak_force >= scenario.damage_force_n
        yaw, pitch = _yaw_pitch(data.xmat[terminal_body])
        tip_error = data.site_xpos[tip_site] - seat
        seated = (
            float(data.site_xpos[tip_site, 0]) >= float(seat[0]) - 0.0020
            and abs(float(tip_error[1])) <= max(0.001, scenario.opening_half_y_m - 0.0054)
            and abs(float(tip_error[2])) <= max(0.001, scenario.opening_half_z_m - 0.0054)
            and abs(yaw) <= 0.18
            and abs(pitch) <= 0.18
            and not damage
        )
        verification = verification + 1 if seated else 0
        if record_trace:
            trace.append(
                {
                    "step": step,
                    "stage": stage,
                    "tip_x_m": float(data.site_xpos[tip_site, 0]),
                    "tip_y_error_m": float(tip_error[1]),
                    "tip_z_error_m": float(tip_error[2]),
                    "yaw_rad": yaw,
                    "pitch_rad": pitch,
                    "native_contact_force_n": force,
                    "peak_force_n": peak_force,
                    "native_contacts": contacts,
                }
            )
        if damage or verification >= config.verification_steps:
            terminal_step = step + 1
            success = verification >= config.verification_steps and not damage
            break
    _assert_finite(data, "insert_mujoco_direct")
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        if robot_rig is not None:
            robot_rig.sync(data, iterations=24)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        try:
            renderer.update_scene(data, camera="contact_view")
            Image.fromarray(renderer.render()).save(render_path, optimize=True)
        finally:
            renderer.close()
    final_yaw, final_pitch = _yaw_pitch(data.xmat[terminal_body])
    final_error = data.site_xpos[tip_site] - seat
    result: dict[str, Any] = {
        "seed": scenario.seed,
        "difficulty": scenario.difficulty,
        "policy": policy,
        "success": bool(success),
        "damage": bool(damage),
        "steps": terminal_step,
        "cycle_time_s": (
            terminal_step * config.physics_steps_per_control * float(model.opt.timestep)
        ),
        "peak_native_contact_force_n": peak_force,
        "native_contact_samples": native_contact_samples,
        "contact_updates": memory.contact_updates,
        "retries": memory.retries,
        "final_tip_x_m": float(data.site_xpos[tip_site, 0]),
        "final_lateral_error_m": float(math.hypot(final_error[1], final_error[2])),
        "final_angle_error_rad": float(math.hypot(final_yaw, final_pitch)),
        "opening_half_y_m": scenario.opening_half_y_m,
        "opening_half_z_m": scenario.opening_half_z_m,
        "friction": scenario.friction,
        "damage_force_n": scenario.damage_force_n,
        "finite_state": True,
    }
    if record_trace:
        result["trace"] = trace
    return result


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(float(row["difficulty"]), str(row["policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for (difficulty, policy), rows in sorted(grouped.items()):
        output.append(
            {
                "difficulty": difficulty,
                "policy": policy,
                "episodes": len(rows),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "damage_rate": float(np.mean([row["damage"] for row in rows])),
                "peak_force_mean_n": float(
                    np.mean([row["peak_native_contact_force_n"] for row in rows])
                ),
                "peak_force_p95_n": float(
                    np.quantile([row["peak_native_contact_force_n"] for row in rows], 0.95)
                ),
                "cycle_time_mean_s": float(np.mean([row["cycle_time_s"] for row in rows])),
                "contact_updates_mean": float(np.mean([row["contact_updates"] for row in rows])),
            }
        )
    return output


def _paired(records: list[dict[str, Any]], config: DirectInsertConfig) -> list[dict[str, Any]]:
    indexed = {
        (float(row["difficulty"]), int(row["seed"]), str(row["policy"])): row for row in records
    }
    output: list[dict[str, Any]] = []
    for difficulty in config.difficulties:
        seeds = sorted(
            row["seed"]
            for row in records
            if row["difficulty"] == difficulty and row["policy"] == "contact_belief"
        )
        for baseline in config.policies:
            if baseline == "contact_belief":
                continue
            pairs = [
                (
                    indexed[(difficulty, seed, "contact_belief")],
                    indexed[(difficulty, seed, baseline)],
                )
                for seed in seeds
            ]
            left_only = sum(left["success"] and not right["success"] for left, right in pairs)
            right_only = sum(right["success"] and not left["success"] for left, right in pairs)
            output.append(
                {
                    "difficulty": difficulty,
                    "baseline": baseline,
                    "paired_episodes": len(pairs),
                    "contact_success_baseline_failure": left_only,
                    "contact_failure_baseline_success": right_only,
                    "success_rate_difference": float(
                        np.mean(
                            [
                                float(left["success"]) - float(right["success"])
                                for left, right in pairs
                            ]
                        )
                    ),
                    "mcnemar_exact_p": _mcnemar_exact(left_only, right_only),
                    "peak_force_difference_n": float(
                        np.mean(
                            [
                                left["peak_native_contact_force_n"]
                                - right["peak_native_contact_force_n"]
                                for left, right in pairs
                            ]
                        )
                    ),
                }
            )
    _holm_adjust(output)
    return output


def run_insert_mujoco_direct(
    output_dir: Path,
    *,
    config: DirectInsertConfig | None = None,
    render_representative: bool = True,
) -> dict[str, Any]:
    config = config or DirectInsertConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for difficulty_index, difficulty in enumerate(config.difficulties):
        for offset in range(config.physical_seeds_per_difficulty):
            seed = config.base_seed + difficulty_index * 1_000_000 + offset
            scenario = direct_insert_scenario(seed, difficulty, config)
            for policy in config.policies:
                records.append(run_direct_insert_episode(scenario, policy, config))

    episodes_path = output_dir / "insert_mujoco_direct_episodes.csv"
    with episodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate = _aggregate(records)
    aggregate_path = output_dir / "insert_mujoco_direct_aggregate.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    paired = _paired(records, config)

    representative: dict[str, Any] | None = None
    frame_path = output_dir / "insert_mujoco_contact_frame.png"
    trace_path = output_dir / "insert_mujoco_contact_trace.json"
    if render_representative:
        hard_rows = [
            row
            for row in records
            if row["difficulty"] == max(config.difficulties)
            and row["policy"] == "contact_belief"
            and row["success"]
            and row["contact_updates"] > 0
        ]
        chosen = max(hard_rows, key=lambda row: row["peak_native_contact_force_n"])
        scenario = direct_insert_scenario(int(chosen["seed"]), float(chosen["difficulty"]), config)
        representative = run_direct_insert_episode(
            scenario,
            "contact_belief",
            config,
            render_path=frame_path,
            record_trace=True,
        )
        trace_path.write_text(json.dumps(representative, indent=2) + "\n", encoding="utf-8")

    workspace = Path(__file__).resolve().parents[3]
    provenance = {
        "direct_source": Path(__file__).resolve(),
        "scene": SCENE_SPECS["insert"].xml_path.resolve(),
        "runner": workspace / "scripts" / "run_insertbot_mujoco_direct.py",
        "tests": workspace / "tests" / "test_insert_mujoco_direct.py",
        "protocol": workspace / "docs" / "INSERTBOT_MUJOCO_PROTOCOL.md",
    }
    report = {
        "schema_version": 1,
        "name": "InsertBot direct MuJoCo contact audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "MuJoCo native collision/contact dynamics",
        "config": asdict(config),
        "episode_rows": len(records),
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "episodes_csv": str(episodes_path.resolve()),
        "episodes_sha256": _sha256(episodes_path),
        "aggregate_csv": str(aggregate_path.resolve()),
        "aggregate_sha256": _sha256(aggregate_path),
        "representative_trace": str(trace_path.resolve()) if representative else None,
        "representative_trace_sha256": _sha256(trace_path) if representative else None,
        "representative_frame": str(frame_path.resolve()) if representative else None,
        "representative_frame_sha256": _sha256(frame_path) if representative else None,
        "provenance_files": {name: str(path.resolve()) for name, path in provenance.items()},
        "provenance_sha256": {name: _sha256(path) for name, path in provenance.items()},
        "claim_scope": (
            "Direct closed-loop MuJoCo-observation task-space contact audit with a compliant "
            "terminal and collision-enabled socket guides. It is independent of the 2.5-D "
            "rollout, but is not a calibrated connector, robot torque controller, safety "
            "validation, or hardware result."
        ),
    }
    report_path = output_dir / "insert_mujoco_direct_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


__all__ = [
    "DIRECT_INSERT_POLICIES",
    "DirectInsertConfig",
    "DirectScenario",
    "direct_insert_scenario",
    "run_direct_insert_episode",
    "run_insert_mujoco_direct",
]
