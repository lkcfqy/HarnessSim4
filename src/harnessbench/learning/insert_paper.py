"""Paired paper-scale contact benchmark for InsertBot.

The model is deliberately compact: it is a reproducible 2.5-D insertion testbed
for controller comparisons, not a calibrated connector finite-element model.
It represents axial depth, lateral offset, yaw/pitch error, clearance, friction,
cable drag, vision bias, contact wrench, jamming, damage, retry, and lock
verification.  Every policy receives the same frozen physical scenario and
sensor-noise sequence for a given seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact

INSERT_POLICY_NAMES = (
    "contact_belief",
    "guarded_admittance",
    "vision_staged_no_force",
    "spiral_search",
    "direct_insertion",
    "contact_no_orientation",
    "contact_no_retract",
    "oracle_teacher",
)
INSERT_POLICY_LABELS = {
    "contact_belief": "ContactBelief (ours)",
    "guarded_admittance": "Guarded admittance",
    "vision_staged_no_force": "Vision staged, no force",
    "spiral_search": "Spiral/search baseline",
    "direct_insertion": "Direct insertion",
    "contact_no_orientation": "No orientation update",
    "contact_no_retract": "No retract state",
    "oracle_teacher": "State-aware oracle",
}


@dataclass(frozen=True)
class InsertPaperConfig:
    difficulties: tuple[float, ...] = (0.2, 0.5, 0.8)
    physical_seeds_per_difficulty: int = 200
    base_seed: int = 83_000_000
    max_steps: int = 320
    dt: float = 0.02
    insertion_depth_m: float = 0.040
    bootstrap_draws: int = 10_000
    policies: tuple[str, ...] = INSERT_POLICY_NAMES


@dataclass(frozen=True)
class InsertScenario:
    seed: int
    difficulty: float
    clearance_m: float
    friction: float
    contact_stiffness_npm: float
    jam_force_n: float
    damage_force_n: float
    damage_impulse_ns: float
    cable_drag_y: float
    cable_drag_yaw: float
    cable_drag_pitch: float
    vision_bias_y_m: float
    vision_bias_yaw_rad: float
    vision_bias_pitch_rad: float
    force_bias_n: float
    initial_y_m: float
    initial_yaw_rad: float
    initial_pitch_rad: float
    noise_y_m: tuple[float, ...]
    noise_angle_rad: tuple[float, ...]
    noise_force_n: tuple[float, ...]


@dataclass
class InsertState:
    x_m: float
    y_m: float
    yaw_rad: float
    pitch_rad: float
    axial_force_n: float = 0.0
    lateral_force_n: float = 0.0
    contact_torque_nm: float = 0.0
    peak_force_n: float = 0.0
    force_impulse_ns: float = 0.0
    contact_steps: int = 0
    jam_events: int = 0
    jammed: bool = False
    damage: bool = False
    locked: bool = False
    verification_steps: int = 0
    verified: bool = False


@dataclass(frozen=True)
class InsertAction:
    dx_m: float
    dy_m: float
    dyaw_rad: float
    dpitch_rad: float
    stage: str


@dataclass
class PolicyMemory:
    stage: str = "approach"
    target_y_obs_m: float = 0.0
    target_yaw_obs_rad: float = 0.0
    target_pitch_obs_rad: float = 0.0
    retract_steps: int = 0
    retries: int = 0
    phase: float = 0.0
    contact_count: int = 0
    last_contact_sign: float = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def insert_scenario(seed: int, difficulty: float, config: InsertPaperConfig) -> InsertScenario:
    """Construct one immutable physical/sensor scenario."""

    rng = np.random.default_rng(seed + round(10_000 * difficulty))
    difficulty = float(difficulty)
    clearance = float(rng.uniform(0.0075, 0.0105) * (1.0 - 0.43 * difficulty))
    friction = float(rng.uniform(0.14, 0.28) + difficulty * rng.uniform(0.20, 0.42))
    stiffness = float(rng.uniform(1_500.0, 2_100.0) * (1.0 + 0.35 * difficulty))
    jam_force = float(rng.uniform(10.0, 13.0) - 2.0 * difficulty)
    damage_force = float(rng.uniform(24.0, 29.0) - 2.5 * difficulty)
    damage_impulse = float(rng.uniform(1.2, 1.7) - 0.25 * difficulty)
    bias_scale = 0.0015 + 0.0045 * difficulty
    angle_bias_scale = 0.018 + 0.070 * difficulty
    initial_scale = 0.013 + 0.015 * difficulty
    angle_scale = 0.10 + 0.20 * difficulty
    return InsertScenario(
        seed=int(seed),
        difficulty=difficulty,
        clearance_m=clearance,
        friction=friction,
        contact_stiffness_npm=stiffness,
        jam_force_n=jam_force,
        damage_force_n=damage_force,
        damage_impulse_ns=damage_impulse,
        cable_drag_y=float(rng.normal(0.0, 0.07 + 0.06 * difficulty)),
        cable_drag_yaw=float(rng.normal(0.0, 0.09 + 0.08 * difficulty)),
        cable_drag_pitch=float(rng.normal(0.0, 0.08 + 0.07 * difficulty)),
        vision_bias_y_m=float(rng.normal(0.0, bias_scale)),
        vision_bias_yaw_rad=float(rng.normal(0.0, angle_bias_scale)),
        vision_bias_pitch_rad=float(rng.normal(0.0, 0.8 * angle_bias_scale)),
        force_bias_n=float(rng.normal(0.0, 0.18 + 0.20 * difficulty)),
        initial_y_m=float(rng.uniform(-initial_scale, initial_scale)),
        initial_yaw_rad=float(rng.uniform(-angle_scale, angle_scale)),
        initial_pitch_rad=float(rng.uniform(-0.8 * angle_scale, 0.8 * angle_scale)),
        noise_y_m=tuple(
            float(value)
            for value in rng.normal(0.0, 0.00025 + 0.00025 * difficulty, config.max_steps)
        ),
        noise_angle_rad=tuple(
            float(value) for value in rng.normal(0.0, 0.004 + 0.006 * difficulty, config.max_steps)
        ),
        noise_force_n=tuple(
            float(value) for value in rng.normal(0.0, 0.12 + 0.12 * difficulty, config.max_steps)
        ),
    )


def _tip_offset(state: InsertState) -> float:
    tip_length = 0.030
    return float(state.y_m + tip_length * state.yaw_rad + 0.45 * tip_length * state.pitch_rad)


def _alignment_envelope(state: InsertState) -> float:
    tip_length = 0.030
    return float(
        abs(_tip_offset(state)) + 0.38 * tip_length * (abs(state.yaw_rad) + abs(state.pitch_rad))
    )


def _observation(
    scenario: InsertScenario,
    state: InsertState,
    step: int,
    *,
    force_enabled: bool = True,
) -> dict[str, float]:
    noise_y = scenario.noise_y_m[min(step, len(scenario.noise_y_m) - 1)]
    noise_angle = scenario.noise_angle_rad[min(step, len(scenario.noise_angle_rad) - 1)]
    noise_force = scenario.noise_force_n[min(step, len(scenario.noise_force_n) - 1)]
    return {
        "x_m": state.x_m,
        "y_obs_m": state.y_m + scenario.vision_bias_y_m + noise_y,
        "yaw_obs_rad": state.yaw_rad + scenario.vision_bias_yaw_rad + noise_angle,
        "pitch_obs_rad": (state.pitch_rad + scenario.vision_bias_pitch_rad - 0.8 * noise_angle),
        "axial_force_n": (
            max(0.0, state.axial_force_n + scenario.force_bias_n + noise_force)
            if force_enabled
            else 0.0
        ),
        "lateral_force_n": (state.lateral_force_n + 0.45 * noise_force if force_enabled else 0.0),
        "contact_torque_nm": (
            state.contact_torque_nm + 0.004 * noise_force if force_enabled else 0.0
        ),
    }


def _clip_action(
    dx: float,
    dy: float,
    dyaw: float,
    dpitch: float,
    stage: str,
) -> InsertAction:
    return InsertAction(
        dx_m=float(np.clip(dx, -0.0018, 0.0013)),
        dy_m=float(np.clip(dy, -0.0018, 0.0018)),
        dyaw_rad=float(np.clip(dyaw, -0.018, 0.018)),
        dpitch_rad=float(np.clip(dpitch, -0.018, 0.018)),
        stage=stage,
    )


def _vision_alignment_action(
    obs: dict[str, float],
    memory: PolicyMemory,
    *,
    dx: float,
    orientation: bool = True,
    stage: str,
) -> InsertAction:
    dy = 0.35 * (memory.target_y_obs_m - obs["y_obs_m"])
    dyaw = 0.24 * (memory.target_yaw_obs_rad - obs["yaw_obs_rad"]) if orientation else 0.0
    dpitch = 0.24 * (memory.target_pitch_obs_rad - obs["pitch_obs_rad"]) if orientation else 0.0
    return _clip_action(dx, dy, dyaw, dpitch, stage)


def _contact_update(
    obs: dict[str, float],
    memory: PolicyMemory,
    scenario: InsertScenario,
    *,
    orientation: bool,
    allow_retract: bool,
) -> None:
    force = float(obs["axial_force_n"])
    contact_load = max(abs(float(obs["lateral_force_n"])), force - 5.0)
    memory.contact_count += 1
    if allow_retract:
        memory.retries += 1
    lateral_sign = float(np.sign(obs["lateral_force_n"]))
    if lateral_sign == 0.0:
        lateral_sign = memory.last_contact_sign or 1.0
    memory.last_contact_sign = lateral_sign
    gain = float(np.clip(0.00055 * contact_load, 0.0010, 0.0042))
    memory.target_y_obs_m -= lateral_sign * gain
    if orientation:
        torque_sign = float(np.sign(obs["contact_torque_nm"]))
        memory.target_yaw_obs_rad -= torque_sign * min(0.035, 0.006 + 0.0015 * force)
        memory.target_pitch_obs_rad -= torque_sign * min(0.024, 0.004 + 0.0010 * force)
    if allow_retract:
        memory.retract_steps = 8 + int(3 * scenario.difficulty)
        memory.stage = "retract"
    else:
        # Without a retract state, the same correction is attempted while the
        # terminal remains loaded against the guide.  The physics model treats
        # this as contact sliding rather than a free-space realignment.
        memory.stage = "contact_slide"


def _policy_action(
    policy: str,
    scenario: InsertScenario,
    state: InsertState,
    obs: dict[str, float],
    memory: PolicyMemory,
    step: int,
) -> InsertAction:
    if policy == "oracle_teacher":
        dx = 0.00115 if state.x_m >= -0.010 else 0.0013
        return _clip_action(
            dx,
            -0.45 * state.y_m,
            -0.35 * state.yaw_rad,
            -0.35 * state.pitch_rad,
            "oracle_insert",
        )

    if policy == "direct_insertion":
        return _vision_alignment_action(
            obs,
            memory,
            dx=0.00125,
            orientation=True,
            stage="direct",
        )

    if policy == "spiral_search":
        memory.phase += 0.23
        amplitude = 0.0035 + 0.0030 * scenario.difficulty
        memory.target_y_obs_m = amplitude * math.sin(memory.phase)
        memory.target_yaw_obs_rad = 0.045 * math.sin(0.61 * memory.phase + 0.8)
        return _vision_alignment_action(
            obs,
            memory,
            dx=0.00072,
            orientation=True,
            stage="spiral",
        )

    force_enabled = policy != "vision_staged_no_force"
    orientation = policy != "contact_no_orientation"
    allow_retract = policy != "contact_no_retract"
    if force_enabled:
        lateral_contact = abs(float(obs["lateral_force_n"]))
        if policy == "guarded_admittance":
            contact_detected = obs["axial_force_n"] >= 9.5 or lateral_contact >= 2.0
        else:
            contact_detected = (
                obs["axial_force_n"] >= scenario.jam_force_n - 0.5 or lateral_contact >= 0.9
            )
        if contact_detected and memory.retract_steps == 0:
            if policy == "guarded_admittance":
                memory.retries += 1
                lateral_sign = float(np.sign(obs["lateral_force_n"])) or 1.0
                memory.target_y_obs_m -= lateral_sign * min(
                    0.0028,
                    0.00030 * obs["axial_force_n"],
                )
                memory.retract_steps = 7
                memory.stage = "retract"
            else:
                _contact_update(
                    obs,
                    memory,
                    scenario,
                    orientation=orientation,
                    allow_retract=allow_retract,
                )

    if memory.retract_steps > 0:
        memory.retract_steps -= 1
        if memory.retract_steps == 0:
            memory.stage = "realign"
        return _vision_alignment_action(
            obs,
            memory,
            dx=-0.00155,
            orientation=orientation,
            stage="retract",
        )

    aligned = abs(obs["y_obs_m"] - memory.target_y_obs_m) < 0.0015 and (
        not orientation
        or (
            abs(obs["yaw_obs_rad"] - memory.target_yaw_obs_rad) < 0.028
            and abs(obs["pitch_obs_rad"] - memory.target_pitch_obs_rad) < 0.028
        )
    )
    if state.x_m < -0.010 and not aligned:
        return _vision_alignment_action(
            obs,
            memory,
            dx=0.00035,
            orientation=orientation,
            stage="prealign",
        )
    speed = 0.00068 if policy == "guarded_admittance" else 0.00082
    if policy == "vision_staged_no_force":
        speed = 0.00092
    if policy == "contact_no_retract":
        speed = 0.00078
    action_stage = (
        "contact_slide"
        if policy == "contact_no_retract" and memory.stage == "contact_slide"
        else "probe_insert"
    )
    return _vision_alignment_action(
        obs,
        memory,
        dx=speed,
        orientation=orientation,
        stage=action_stage,
    )


def _physics_step(
    scenario: InsertScenario,
    state: InsertState,
    action: InsertAction,
    config: InsertPaperConfig,
) -> None:
    contact_active = state.x_m > -0.002 or action.dx_m > 0.0 and state.x_m > -0.004
    lateral_scale = 0.42 if contact_active and state.axial_force_n > 2.0 else 1.0
    state.y_m += lateral_scale * action.dy_m
    state.yaw_rad += lateral_scale * action.dyaw_rad
    state.pitch_rad += lateral_scale * action.dpitch_rad

    if action.dx_m > 0.0:
        state.y_m += scenario.cable_drag_y * action.dx_m * 0.12
        state.yaw_rad += scenario.cable_drag_yaw * action.dx_m * 0.35
        state.pitch_rad += scenario.cable_drag_pitch * action.dx_m * 0.35

    proposed_x = state.x_m + action.dx_m
    envelope = _alignment_envelope(state)
    violation = max(0.0, envelope - scenario.clearance_m)
    penetration = max(0.0, proposed_x + 0.0015)
    base_force = 0.0
    contact_force = 0.0
    if proposed_x > -0.002:
        base_force = (
            0.55
            + 7.5 * scenario.friction
            + 2.0 * abs(scenario.cable_drag_y)
            + 4.5 * max(0.0, action.dx_m) / 0.0013
        )
        contact_force = (
            scenario.contact_stiffness_npm * violation * (1.0 + 10.0 * min(penetration, 0.025))
        )
    total_force = base_force + contact_force
    if action.stage == "contact_slide" and contact_force > 0.0:
        # Tangential correction under load raises effective friction and local
        # edge pressure.  A retract-and-realign controller avoids this term.
        total_force *= 1.45

    if action.dx_m < 0.0:
        state.x_m = max(-0.055, proposed_x)
        if state.x_m < -0.006:
            state.jammed = False
        total_force *= 0.20
    elif proposed_x > -0.002 and violation > 0.0:
        if total_force >= scenario.jam_force_n and not state.jammed:
            state.jammed = True
            state.jam_events += 1
        scale = 0.015 if state.jammed else float(np.exp(-max(0.0, total_force - 2.0) / 4.8))
        state.x_m += action.dx_m * scale
    else:
        state.x_m = min(config.insertion_depth_m, proposed_x)

    sign = float(np.sign(_tip_offset(state)))
    state.axial_force_n = float(max(0.0, total_force))
    state.lateral_force_n = float(sign * contact_force)
    state.contact_torque_nm = float(
        np.sign(state.yaw_rad + 0.45 * state.pitch_rad) * contact_force * 0.024
    )
    if total_force > 0.5:
        state.contact_steps += 1
    state.peak_force_n = max(state.peak_force_n, state.axial_force_n)
    impulse_floor_n = 8.0 if action.stage == "contact_slide" else 10.0
    state.force_impulse_ns += max(0.0, state.axial_force_n - impulse_floor_n) * config.dt
    if (
        state.peak_force_n >= scenario.damage_force_n
        or state.force_impulse_ns >= scenario.damage_impulse_ns
    ):
        state.damage = True

    if (
        state.x_m >= config.insertion_depth_m - 0.0005
        and envelope <= 0.90 * scenario.clearance_m
        and not state.damage
    ):
        state.locked = True
    if state.locked:
        state.x_m = config.insertion_depth_m
        state.verification_steps += 1
        if state.verification_steps >= 6:
            state.verified = True


def _failure_type(state: InsertState, config: InsertPaperConfig) -> str:
    if state.verified and not state.damage:
        return "success"
    if state.damage:
        return "damage_limit"
    if state.jammed or state.jam_events > 0:
        return "persistent_jam"
    if _alignment_envelope(state) > 0.012:
        return "pose_misalignment"
    if state.x_m < config.insertion_depth_m - 0.006:
        return "incomplete_depth"
    if state.locked and not state.verified:
        return "lock_not_verified"
    return "timeout"


def simulate_insert_episode(
    scenario: InsertScenario,
    policy: str,
    config: InsertPaperConfig,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    if policy not in INSERT_POLICY_NAMES:
        raise KeyError(policy)
    state = InsertState(
        x_m=-0.050,
        y_m=scenario.initial_y_m,
        yaw_rad=scenario.initial_yaw_rad,
        pitch_rad=scenario.initial_pitch_rad,
    )
    memory = PolicyMemory()
    trace: list[dict[str, Any]] = []
    terminal_step = config.max_steps
    for step in range(config.max_steps):
        force_enabled = policy != "vision_staged_no_force"
        obs = _observation(scenario, state, step, force_enabled=force_enabled)
        action = _policy_action(policy, scenario, state, obs, memory, step)
        _physics_step(scenario, state, action, config)
        if record_trace:
            trace.append(
                {
                    "step": step,
                    "stage": action.stage,
                    "x_m": state.x_m,
                    "y_m": state.y_m,
                    "yaw_rad": state.yaw_rad,
                    "pitch_rad": state.pitch_rad,
                    "axial_force_n": state.axial_force_n,
                    "peak_force_n": state.peak_force_n,
                    "jammed": state.jammed,
                    "locked": state.locked,
                }
            )
        if state.verified or state.damage:
            terminal_step = step + 1
            break

    success = bool(state.verified and not state.damage)
    result: dict[str, Any] = {
        "seed": scenario.seed,
        "difficulty": scenario.difficulty,
        "policy": policy,
        "success": success,
        "first_pass_success": bool(success and memory.retries == 0),
        "locked": state.locked,
        "lock_verified": state.verified,
        "damage": state.damage,
        "failure_type": _failure_type(state, config),
        "clearance_m": scenario.clearance_m,
        "friction": scenario.friction,
        "vision_bias_y_m": scenario.vision_bias_y_m,
        "vision_bias_angle_rad": math.hypot(
            scenario.vision_bias_yaw_rad,
            scenario.vision_bias_pitch_rad,
        ),
        "steps": terminal_step,
        "cycle_time_s": terminal_step * config.dt,
        "retries": memory.retries,
        "contact_updates": memory.contact_count,
        "jam_events": state.jam_events,
        "peak_force_n": state.peak_force_n,
        "force_impulse_ns": state.force_impulse_ns,
        "final_depth_m": state.x_m,
        "final_lateral_error_m": abs(state.y_m),
        "final_angle_error_rad": math.hypot(state.yaw_rad, state.pitch_rad),
    }
    if record_trace:
        result["trace"] = trace
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        means[draw] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _aggregate(records: list[dict[str, Any]], config: InsertPaperConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(float(row["difficulty"]), str(row["policy"]))].append(row)
    output: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(grouped)):
        difficulty, policy = key
        rows = grouped[key]
        success = np.asarray([float(row["success"]) for row in rows])
        first_pass = np.asarray([float(row["first_pass_success"]) for row in rows])
        peak_force = np.asarray([float(row["peak_force_n"]) for row in rows])
        cycle = np.asarray([float(row["cycle_time_s"]) for row in rows])
        damage = np.asarray([float(row["damage"]) for row in rows])
        success_ci = _bootstrap_mean_ci(
            success,
            seed=config.base_seed + 10_000 + index,
            draws=config.bootstrap_draws,
        )
        force_ci = _bootstrap_mean_ci(
            peak_force,
            seed=config.base_seed + 20_000 + index,
            draws=config.bootstrap_draws,
        )
        output.append(
            {
                "difficulty": difficulty,
                "policy": policy,
                "episodes": len(rows),
                "success_rate": float(np.mean(success)),
                "success_ci95": list(success_ci),
                "first_pass_success_rate": float(np.mean(first_pass)),
                "damage_rate": float(np.mean(damage)),
                "peak_force_mean_n": float(np.mean(peak_force)),
                "peak_force_ci95": list(force_ci),
                "peak_force_p95_n": float(np.quantile(peak_force, 0.95)),
                "cycle_time_mean_s": float(np.mean(cycle)),
                "retries_mean": float(np.mean([float(row["retries"]) for row in rows])),
                "jam_rate": float(np.mean([float(row["jam_events"] > 0) for row in rows])),
            }
        )
    return output


def _paired_comparisons(
    records: list[dict[str, Any]],
    config: InsertPaperConfig,
) -> list[dict[str, Any]]:
    indexed = {
        (float(row["difficulty"]), int(row["seed"]), str(row["policy"])): row for row in records
    }
    output: list[dict[str, Any]] = []
    for difficulty in config.difficulties:
        seeds = sorted(
            int(row["seed"])
            for row in records
            if float(row["difficulty"]) == difficulty and str(row["policy"]) == "contact_belief"
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
            left_only = sum(
                bool(left["success"]) and not bool(right["success"]) for left, right in pairs
            )
            right_only = sum(
                bool(right["success"]) and not bool(left["success"]) for left, right in pairs
            )
            success_diff = np.asarray(
                [float(left["success"]) - float(right["success"]) for left, right in pairs]
            )
            force_diff = np.asarray(
                [
                    float(left["peak_force_n"]) - float(right["peak_force_n"])
                    for left, right in pairs
                ]
            )
            cycle_diff = np.asarray(
                [
                    float(left["cycle_time_s"]) - float(right["cycle_time_s"])
                    for left, right in pairs
                ]
            )
            success_ci = _bootstrap_mean_ci(
                success_diff,
                seed=config.base_seed + 30_000 + len(output),
                draws=config.bootstrap_draws,
            )
            force_ci = _bootstrap_mean_ci(
                force_diff,
                seed=config.base_seed + 40_000 + len(output),
                draws=config.bootstrap_draws,
            )
            nonzero_force = force_diff[np.abs(force_diff) > 1e-12]
            force_p = (
                float(wilcoxon(nonzero_force, alternative="less").pvalue)
                if len(nonzero_force)
                else 1.0
            )
            output.append(
                {
                    "difficulty": difficulty,
                    "baseline": baseline,
                    "paired_episodes": len(pairs),
                    "contact_success_baseline_failure": left_only,
                    "contact_failure_baseline_success": right_only,
                    "success_rate_difference": float(np.mean(success_diff)),
                    "success_difference_ci95": list(success_ci),
                    "mcnemar_exact_p": _mcnemar_exact(left_only, right_only),
                    "peak_force_difference_n": float(np.mean(force_diff)),
                    "peak_force_difference_ci95": list(force_ci),
                    "peak_force_wilcoxon_less_p": force_p,
                    "cycle_time_difference_s": float(np.mean(cycle_diff)),
                }
            )
    _holm_adjust(output, key="mcnemar_exact_p")
    _holm_adjust(output, key="peak_force_wilcoxon_less_p")
    return output


def run_insert_paper_benchmark(
    output_dir: Path,
    *,
    config: InsertPaperConfig | None = None,
) -> dict[str, Any]:
    config = config or InsertPaperConfig()
    unknown = set(config.policies) - set(INSERT_POLICY_NAMES)
    if unknown:
        raise KeyError(f"unknown policies: {sorted(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for difficulty_index, difficulty in enumerate(config.difficulties):
        for offset in range(config.physical_seeds_per_difficulty):
            seed = config.base_seed + difficulty_index * 1_000_000 + offset
            scenario = insert_scenario(seed, difficulty, config)
            for policy in config.policies:
                records.append(simulate_insert_episode(scenario, policy, config))

    episodes_path = output_dir / "insert_paper_episodes.csv"
    fieldnames = list(records[0])
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    aggregate = _aggregate(records, config)
    aggregate_path = output_dir / "insert_paper_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    paired = _paired_comparisons(records, config)
    paired_path = output_dir / "insert_paper_paired.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    failures = Counter(str(row["failure_type"]) for row in records)
    workspace = Path(__file__).resolve().parents[3]
    provenance_files = {
        "benchmark_source": Path(__file__).resolve(),
        "runner": workspace / "scripts" / "run_insertbot_formal.py",
        "protocol": workspace / "docs" / "INSERTBOT_CONTACT_PROTOCOL.md",
        "tests": workspace / "tests" / "test_insert_paper.py",
    }
    report = {
        "schema_version": 2,
        "name": "InsertBot paired contact benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "policy_labels": INSERT_POLICY_LABELS,
        "episode_rows": len(records),
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "failure_counts": dict(sorted(failures.items())),
        "episodes_csv": str(episodes_path.resolve()),
        "episodes_sha256": _sha256(episodes_path),
        "aggregate_csv": str(aggregate_path.resolve()),
        "aggregate_sha256": _sha256(aggregate_path),
        "paired_csv": str(paired_path.resolve()),
        "paired_sha256": _sha256(paired_path),
        "provenance_files": {name: str(path.resolve()) for name, path in provenance_files.items()},
        "provenance_sha256": {name: _sha256(path) for name, path in provenance_files.items()},
        "claim_scope": (
            "Paired synthetic 2.5-D contact benchmark with engineering force and damage proxies. "
            "It is not a calibrated connector, finite-element, camera, force-sensor, or hardware result."
        ),
    }
    report_path = output_dir / "insert_paper_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


__all__ = [
    "INSERT_POLICY_LABELS",
    "INSERT_POLICY_NAMES",
    "InsertPaperConfig",
    "InsertScenario",
    "insert_scenario",
    "run_insert_paper_benchmark",
    "simulate_insert_episode",
]
