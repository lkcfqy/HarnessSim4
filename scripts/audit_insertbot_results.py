#!/usr/bin/env python3
"""Independently audit InsertBot public-data, contact, and MuJoCo evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(numerator, 1 << discordant)))


def _holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


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


def _finite_json(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _close(first: float, second: float, *, atol: float = 1e-11) -> bool:
    return math.isclose(float(first), float(second), rel_tol=1e-11, abs_tol=atol)


def _ci_close(first: list[float], second: tuple[float, float]) -> bool:
    return len(first) == 2 and all(_close(left, right) for left, right in zip(first, second))


def _regression_metrics(
    true_action: np.ndarray,
    predicted_action: np.ndarray,
    action_scale: np.ndarray,
) -> dict[str, object]:
    error = predicted_action - true_action
    mae_per_dim = np.mean(np.abs(error), axis=0)
    rmse_per_dim = np.sqrt(np.mean(error**2, axis=0))
    scale = np.maximum(np.asarray(action_scale), 1e-8)
    residual_sum = np.sum(error**2, axis=0)
    centered = true_action - true_action.mean(axis=0)
    total_sum = np.sum(centered**2, axis=0)
    r2 = 1.0 - residual_sum / np.maximum(total_sum, 1e-12)
    return {
        "mae": float(np.mean(mae_per_dim)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.mean(rmse_per_dim / scale)),
        "macro_r2": float(np.mean(r2)),
        "mae_per_action_dim": [float(value) for value in mae_per_dim],
        "rmse_per_action_dim": [float(value) for value in rmse_per_dim],
        "r2_per_action_dim": [float(value) for value in r2],
    }


def _metric_block_close(reported: dict[str, Any], expected: dict[str, object]) -> bool:
    for key, value in expected.items():
        actual = reported[key]
        if isinstance(value, list):
            if len(actual) != len(value) or not all(
                _close(left, right) for left, right in zip(actual, value)
            ):
                return False
        elif not _close(actual, value):
            return False
    return True


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, passed: bool, name: str, observed: object = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "observed": observed})

    @property
    def passed(self) -> bool:
        return all(bool(row["passed"]) for row in self.checks)


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "difficulty": float(raw["difficulty"]),
                    "policy": raw["policy"],
                    "success": _boolean(raw["success"]),
                    "first_pass_success": _boolean(raw["first_pass_success"]),
                    "locked": _boolean(raw["locked"]),
                    "lock_verified": _boolean(raw["lock_verified"]),
                    "damage": _boolean(raw["damage"]),
                    "failure_type": raw["failure_type"],
                    "clearance_m": float(raw["clearance_m"]),
                    "friction": float(raw["friction"]),
                    "vision_bias_y_m": float(raw["vision_bias_y_m"]),
                    "vision_bias_angle_rad": float(raw["vision_bias_angle_rad"]),
                    "steps": int(raw["steps"]),
                    "cycle_time_s": float(raw["cycle_time_s"]),
                    "retries": int(raw["retries"]),
                    "contact_updates": int(raw["contact_updates"]),
                    "jam_events": int(raw["jam_events"]),
                    "peak_force_n": float(raw["peak_force_n"]),
                    "force_impulse_ns": float(raw["force_impulse_ns"]),
                    "final_depth_m": float(raw["final_depth_m"]),
                    "final_lateral_error_m": float(raw["final_lateral_error_m"]),
                    "final_angle_error_rad": float(raw["final_angle_error_rad"]),
                }
            )
    return rows


def _audit_structure(audit: Audit, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    config = report["config"]
    policies = tuple(config["policies"])
    difficulties = tuple(float(value) for value in config["difficulties"])
    physical_seeds = int(config["physical_seeds_per_difficulty"])
    expected = len(policies) * len(difficulties) * physical_seeds
    audit.require(
        len(rows) == expected == int(report["episode_rows"]), "episode row count", len(rows)
    )
    audit.require(physical_seeds >= 200, "formal physical-seed count", physical_seeds)
    audit.require(
        int(config["bootstrap_draws"]) == 10_000,
        "formal bootstrap draws",
        config["bootstrap_draws"],
    )
    audit.require(
        tuple(difficulties) == (0.2, 0.5, 0.8),
        "frozen difficulty strata",
        difficulties,
    )
    keys = [(row["difficulty"], row["seed"], row["policy"]) for row in rows]
    audit.require(len(keys) == len(set(keys)), "unique episode keys", len(set(keys)))

    paired: dict[tuple[float, int], set[str]] = defaultdict(set)
    scenario_values: dict[tuple[float, int], set[tuple[float, ...]]] = defaultdict(set)
    for row in rows:
        cell = (row["difficulty"], row["seed"])
        paired[cell].add(row["policy"])
        scenario_values[cell].add(
            (
                row["clearance_m"],
                row["friction"],
                row["vision_bias_y_m"],
                row["vision_bias_angle_rad"],
            )
        )
    audit.require(
        len(paired) == len(difficulties) * physical_seeds
        and all(names == set(policies) for names in paired.values()),
        "complete policy pairing",
        len(paired),
    )
    audit.require(
        all(len(values) == 1 for values in scenario_values.values()),
        "scenario fields invariant across paired policies",
    )

    base_seed = int(config["base_seed"])
    expected_cells = {
        (difficulty, base_seed + index * 1_000_000 + offset)
        for index, difficulty in enumerate(difficulties)
        for offset in range(physical_seeds)
    }
    audit.require(set(paired) == expected_cells, "frozen seed schedule", len(paired))


def _audit_episode_invariants(
    audit: Audit,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    config = report["config"]
    dt = float(config["dt"])
    max_steps = int(config["max_steps"])
    finite_fields = (
        "clearance_m",
        "friction",
        "vision_bias_y_m",
        "vision_bias_angle_rad",
        "cycle_time_s",
        "peak_force_n",
        "force_impulse_ns",
        "final_depth_m",
        "final_lateral_error_m",
        "final_angle_error_rad",
    )
    audit.require(
        all(math.isfinite(float(row[field])) for row in rows for field in finite_fields),
        "all episode values finite",
    )
    audit.require(
        all(
            1 <= row["steps"] <= max_steps and _close(row["cycle_time_s"], row["steps"] * dt)
            for row in rows
        ),
        "step/time invariants",
    )
    audit.require(
        all(row["peak_force_n"] >= 0.0 and row["force_impulse_ns"] >= 0.0 for row in rows),
        "nonnegative force metrics",
    )
    audit.require(
        all(
            row["success"] == (row["locked"] and row["lock_verified"] and not row["damage"])
            for row in rows
        ),
        "verified-undamaged success invariant",
    )
    audit.require(
        all((row["failure_type"] == "success") == row["success"] for row in rows),
        "failure-label invariant",
    )
    audit.require(
        all(
            not row["first_pass_success"] or row["success"] and row["retries"] == 0 for row in rows
        ),
        "first-pass invariant",
    )
    audit.require(
        all(not row["lock_verified"] or row["locked"] for row in rows),
        "lock-verification invariant",
    )
    audit.require(
        {row["failure_type"] for row in rows}
        <= {
            "success",
            "damage_limit",
            "persistent_jam",
            "pose_misalignment",
            "incomplete_depth",
            "lock_not_verified",
            "timeout",
        },
        "known failure taxonomy",
    )


def _audit_aggregate(
    audit: Audit,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    config = report["config"]
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["difficulty"], row["policy"])].append(row)
    reported = {(float(row["difficulty"]), str(row["policy"])): row for row in report["aggregate"]}
    audit.require(set(reported) == set(grouped), "aggregate cell set", len(reported))
    for index, key in enumerate(sorted(grouped)):
        cell = grouped[key]
        item = reported[key]
        success = np.asarray([float(row["success"]) for row in cell])
        first = np.asarray([float(row["first_pass_success"]) for row in cell])
        damage = np.asarray([float(row["damage"]) for row in cell])
        force = np.asarray([float(row["peak_force_n"]) for row in cell])
        cycle = np.asarray([float(row["cycle_time_s"]) for row in cell])
        success_ci = _bootstrap_mean_ci(
            success,
            seed=int(config["base_seed"]) + 10_000 + index,
            draws=int(config["bootstrap_draws"]),
        )
        force_ci = _bootstrap_mean_ci(
            force,
            seed=int(config["base_seed"]) + 20_000 + index,
            draws=int(config["bootstrap_draws"]),
        )
        checks = (
            int(item["episodes"]) == len(cell),
            _close(item["success_rate"], np.mean(success)),
            _ci_close(item["success_ci95"], success_ci),
            _close(item["first_pass_success_rate"], np.mean(first)),
            _close(item["damage_rate"], np.mean(damage)),
            _close(item["peak_force_mean_n"], np.mean(force)),
            _ci_close(item["peak_force_ci95"], force_ci),
            _close(item["peak_force_p95_n"], np.quantile(force, 0.95)),
            _close(item["cycle_time_mean_s"], np.mean(cycle)),
            _close(item["retries_mean"], np.mean([row["retries"] for row in cell])),
            _close(item["jam_rate"], np.mean([row["jam_events"] > 0 for row in cell])),
        )
        audit.require(all(checks), f"aggregate recomputation {key}")


def _audit_pairs(
    audit: Audit,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    config = report["config"]
    indexed = {(row["difficulty"], row["seed"], row["policy"]): row for row in rows}
    reported = report["paired_comparisons"]
    recalculated: list[dict[str, Any]] = []
    for difficulty in (float(value) for value in config["difficulties"]):
        seeds = sorted(
            row["seed"]
            for row in rows
            if row["difficulty"] == difficulty and row["policy"] == "contact_belief"
        )
        for baseline in config["policies"]:
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
            success_diff = np.asarray(
                [float(left["success"]) - float(right["success"]) for left, right in pairs]
            )
            force_diff = np.asarray(
                [left["peak_force_n"] - right["peak_force_n"] for left, right in pairs]
            )
            cycle_diff = np.asarray(
                [left["cycle_time_s"] - right["cycle_time_s"] for left, right in pairs]
            )
            index = len(recalculated)
            nonzero_force = force_diff[np.abs(force_diff) > 1e-12]
            recalculated.append(
                {
                    "difficulty": difficulty,
                    "baseline": baseline,
                    "paired_episodes": len(pairs),
                    "contact_success_baseline_failure": left_only,
                    "contact_failure_baseline_success": right_only,
                    "success_rate_difference": float(np.mean(success_diff)),
                    "success_difference_ci95": _bootstrap_mean_ci(
                        success_diff,
                        seed=int(config["base_seed"]) + 30_000 + index,
                        draws=int(config["bootstrap_draws"]),
                    ),
                    "mcnemar_exact_p": _mcnemar_exact(left_only, right_only),
                    "peak_force_difference_n": float(np.mean(force_diff)),
                    "peak_force_difference_ci95": _bootstrap_mean_ci(
                        force_diff,
                        seed=int(config["base_seed"]) + 40_000 + index,
                        draws=int(config["bootstrap_draws"]),
                    ),
                    "peak_force_wilcoxon_less_p": (
                        float(wilcoxon(nonzero_force, alternative="less").pvalue)
                        if len(nonzero_force)
                        else 1.0
                    ),
                    "cycle_time_difference_s": float(np.mean(cycle_diff)),
                }
            )
    success_holm = _holm([row["mcnemar_exact_p"] for row in recalculated])
    force_holm = _holm([row["peak_force_wilcoxon_less_p"] for row in recalculated])
    audit.require(
        len(reported) == len(recalculated) == 21, "paired comparison count", len(reported)
    )
    for index, (expected, actual) in enumerate(zip(recalculated, reported)):
        checks = (
            float(actual["difficulty"]) == expected["difficulty"],
            actual["baseline"] == expected["baseline"],
            int(actual["paired_episodes"]) == expected["paired_episodes"],
            int(actual["contact_success_baseline_failure"])
            == expected["contact_success_baseline_failure"],
            int(actual["contact_failure_baseline_success"])
            == expected["contact_failure_baseline_success"],
            _close(actual["success_rate_difference"], expected["success_rate_difference"]),
            _ci_close(actual["success_difference_ci95"], expected["success_difference_ci95"]),
            _close(actual["mcnemar_exact_p"], expected["mcnemar_exact_p"]),
            _close(actual["mcnemar_exact_p_holm"], success_holm[index]),
            _close(actual["peak_force_difference_n"], expected["peak_force_difference_n"]),
            _ci_close(
                actual["peak_force_difference_ci95"],
                expected["peak_force_difference_ci95"],
            ),
            _close(
                actual["peak_force_wilcoxon_less_p"],
                expected["peak_force_wilcoxon_less_p"],
            ),
            _close(actual["peak_force_wilcoxon_less_p_holm"], force_holm[index]),
            _close(actual["cycle_time_difference_s"], expected["cycle_time_difference_s"]),
        )
        audit.require(
            all(checks),
            f"paired recomputation {(expected['difficulty'], expected['baseline'])}",
        )


def _read_direct_episodes(path: Path) -> list[dict[str, Any]]:
    boolean_fields = ("success", "damage", "finite_state")
    integer_fields = ("seed", "steps", "native_contact_samples", "contact_updates", "retries")
    float_fields = (
        "difficulty",
        "cycle_time_s",
        "peak_native_contact_force_n",
        "final_tip_x_m",
        "final_lateral_error_m",
        "final_angle_error_rad",
        "opening_half_y_m",
        "opening_half_z_m",
        "friction",
        "damage_force_n",
    )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in boolean_fields:
                row[field] = _boolean(raw[field])
            for field in integer_fields:
                row[field] = int(raw[field])
            for field in float_fields:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def _audit_direct_mujoco(
    audit: Audit,
    report_path: Path,
) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes_path = Path(report["episodes_csv"])
    aggregate_path = Path(report["aggregate_csv"])
    trace_path = Path(report["representative_trace"])
    frame_path = Path(report["representative_frame"])
    rows = _read_direct_episodes(episodes_path)
    config = report["config"]
    policies = tuple(config["policies"])
    difficulties = tuple(float(value) for value in config["difficulties"])
    physical_seeds = int(config["physical_seeds_per_difficulty"])
    expected_rows = len(policies) * len(difficulties) * physical_seeds

    audit.require(_finite_json(report), "mujoco: all report numbers finite")
    audit.require(
        len(rows) == expected_rows == int(report["episode_rows"]),
        "mujoco: episode row count",
        len(rows),
    )
    audit.require(physical_seeds >= 20, "mujoco: formal physical-seed count", physical_seeds)
    audit.require(
        _sha256(episodes_path) == report["episodes_sha256"],
        "mujoco: episode CSV hash",
    )
    audit.require(
        _sha256(aggregate_path) == report["aggregate_sha256"],
        "mujoco: aggregate CSV hash",
    )
    audit.require(
        _sha256(trace_path) == report["representative_trace_sha256"],
        "mujoco: representative trace hash",
    )
    audit.require(
        _sha256(frame_path) == report["representative_frame_sha256"],
        "mujoco: representative frame hash",
    )
    audit.require(frame_path.stat().st_size >= 50_000, "mujoco: rendered frame is nontrivial")
    for name, source in report["provenance_files"].items():
        audit.require(
            _sha256(Path(source)) == report["provenance_sha256"][name],
            f"mujoco: frozen provenance hash {name}",
        )
    audit.require(
        "Direct closed-loop MuJoCo-observation" in report["claim_scope"]
        and "not a calibrated connector" in report["claim_scope"],
        "mujoco: cross-physics claim boundary",
        report["claim_scope"],
    )

    keys = [(row["difficulty"], row["seed"], row["policy"]) for row in rows]
    audit.require(len(keys) == len(set(keys)), "mujoco: unique episode keys", len(set(keys)))
    paired: dict[tuple[float, int], set[str]] = defaultdict(set)
    scenarios: dict[tuple[float, int], set[tuple[float, ...]]] = defaultdict(set)
    indexed: dict[tuple[float, int, str], dict[str, Any]] = {}
    for row in rows:
        cell = (row["difficulty"], row["seed"])
        paired[cell].add(row["policy"])
        scenarios[cell].add(
            (
                row["opening_half_y_m"],
                row["opening_half_z_m"],
                row["friction"],
                row["damage_force_n"],
            )
        )
        indexed[(*cell, row["policy"])] = row
    audit.require(
        len(paired) == len(difficulties) * physical_seeds
        and all(names == set(policies) for names in paired.values()),
        "mujoco: complete policy pairing",
        len(paired),
    )
    audit.require(
        all(len(values) == 1 for values in scenarios.values()),
        "mujoco: scenario fields invariant across policies",
    )
    expected_cells = {
        (difficulty, int(config["base_seed"]) + index * 1_000_000 + offset)
        for index, difficulty in enumerate(difficulties)
        for offset in range(physical_seeds)
    }
    audit.require(set(paired) == expected_cells, "mujoco: frozen seed schedule")
    numeric_fields = (
        "cycle_time_s",
        "peak_native_contact_force_n",
        "final_tip_x_m",
        "final_lateral_error_m",
        "final_angle_error_rad",
        "opening_half_y_m",
        "opening_half_z_m",
        "friction",
        "damage_force_n",
    )
    audit.require(
        all(
            row["finite_state"] and all(math.isfinite(row[field]) for field in numeric_fields)
            for row in rows
        ),
        "mujoco: all native states and metrics finite",
    )
    audit.require(
        all(not row["success"] or not row["damage"] for row in rows),
        "mujoco: success/damage invariant",
    )
    audit.require(
        all(
            row["damage"] == (row["peak_native_contact_force_n"] >= row["damage_force_n"])
            for row in rows
        ),
        "mujoco: damage threshold invariant",
    )
    expected_dt = int(config["physics_steps_per_control"]) * 0.002
    audit.require(
        all(_close(row["cycle_time_s"], row["steps"] * expected_dt) for row in rows),
        "mujoco: native step/time invariant",
    )

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["difficulty"], row["policy"])].append(row)
    aggregate = {(float(row["difficulty"]), str(row["policy"])): row for row in report["aggregate"]}
    audit.require(set(grouped) == set(aggregate), "mujoco: aggregate cell set")
    for key, cell in grouped.items():
        item = aggregate[key]
        forces = [row["peak_native_contact_force_n"] for row in cell]
        checks = (
            int(item["episodes"]) == len(cell),
            _close(item["success_rate"], np.mean([row["success"] for row in cell])),
            _close(item["damage_rate"], np.mean([row["damage"] for row in cell])),
            _close(item["peak_force_mean_n"], np.mean(forces)),
            _close(item["peak_force_p95_n"], np.quantile(forces, 0.95)),
            _close(item["cycle_time_mean_s"], np.mean([row["cycle_time_s"] for row in cell])),
            _close(
                item["contact_updates_mean"],
                np.mean([row["contact_updates"] for row in cell]),
            ),
        )
        audit.require(all(checks), f"mujoco: aggregate recomputation {key}")

    recalculated_pairs: list[dict[str, Any]] = []
    for difficulty in difficulties:
        seeds = sorted(seed for cell_difficulty, seed in paired if cell_difficulty == difficulty)
        for baseline in policies:
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
            recalculated_pairs.append(
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
    adjusted = _holm([row["mcnemar_exact_p"] for row in recalculated_pairs])
    reported_pairs = report["paired_comparisons"]
    audit.require(
        len(reported_pairs) == len(recalculated_pairs) == 12,
        "mujoco: paired comparison count",
    )
    for index, (expected, actual) in enumerate(zip(recalculated_pairs, reported_pairs)):
        checks = (
            float(actual["difficulty"]) == expected["difficulty"],
            actual["baseline"] == expected["baseline"],
            int(actual["paired_episodes"]) == expected["paired_episodes"],
            int(actual["contact_success_baseline_failure"])
            == expected["contact_success_baseline_failure"],
            int(actual["contact_failure_baseline_success"])
            == expected["contact_failure_baseline_success"],
            _close(actual["success_rate_difference"], expected["success_rate_difference"]),
            _close(actual["mcnemar_exact_p"], expected["mcnemar_exact_p"]),
            _close(actual["mcnemar_exact_p_holm"], adjusted[index]),
            _close(actual["peak_force_difference_n"], expected["peak_force_difference_n"]),
        )
        audit.require(
            all(checks),
            f"mujoco: paired recomputation {(expected['difficulty'], expected['baseline'])}",
        )

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    audit.require(
        _finite_json(trace)
        and trace["success"]
        and trace["contact_updates"] > 0
        and len(trace["trace"]) > 0,
        "mujoco: representative successful native-contact trace",
    )
    return {
        "mujoco_report_sha256": _sha256(report_path),
        "mujoco_episodes_sha256": _sha256(episodes_path),
        "mujoco_aggregate_sha256": _sha256(aggregate_path),
        "mujoco_trace_sha256": _sha256(trace_path),
        "mujoco_frame_sha256": _sha256(frame_path),
    }


def _audit_public_data(
    audit: Audit,
    report_path: Path,
    manifest_path: Path,
) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = report_path.parent / "split.json"
    model_path = report_path.parent / "model.npz"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    cache_path = manifest_path.parent / manifest["portable_numpy_cache"]["path"]

    audit.require(_finite_json(report), "public: all report numbers finite")
    audit.require(
        report["project"] == "HarnessBench-Insert"
        and report["baseline"] == "ridge_delta_state_v0",
        "public: frozen baseline identity",
    )
    audit.require(
        manifest["dataset_id"] == "lerobot/aloha_sim_insertion_human"
        and manifest["license"] == "MIT",
        "public: dataset identity and license",
    )
    audit.require(
        report["provenance"]["dataset_id"] == manifest["dataset_id"]
        and report["provenance"]["license"] == manifest["license"],
        "public: report/manifest provenance agreement",
    )

    source_hashes = report["provenance"]["source_file_sha256"]
    for item in manifest["files"]:
        source_path = manifest_path.parent / item["path"]
        passed = (
            source_path.is_file()
            and source_path.stat().st_size == int(item["size"])
            and _sha256(source_path) == item["sha256"] == source_hashes[item["path"]]
        )
        audit.require(passed, f"public: frozen source hash {item['path']}")

    cache_hash = _sha256(cache_path)
    audit.require(
        cache_hash
        == manifest["portable_numpy_cache"]["sha256"]
        == report["provenance"]["portable_numpy_cache"]["sha256"],
        "public: portable cache hash",
        cache_hash,
    )
    with np.load(cache_path) as arrays:
        state = np.asarray(arrays["state"], dtype=np.float64)
        action = np.asarray(arrays["action"], dtype=np.float64)
        episode = np.asarray(arrays["episode"], dtype=np.int64)
        frame = np.asarray(arrays["frame"], dtype=np.int64)
        timestamp = np.asarray(arrays["timestamp"], dtype=np.float64)
    audit.require(
        state.shape == action.shape == (25_000, 14)
        and episode.shape == frame.shape == timestamp.shape == (25_000,),
        "public: cache array shapes",
    )
    audit.require(
        all(np.isfinite(array).all() for array in (state, action, timestamp)),
        "public: cache values finite",
    )
    order = np.lexsort((frame, episode))
    audit.require(
        np.array_equal(order, np.arange(len(order))),
        "public: episode/frame ordering",
    )
    episode_ids, counts = np.unique(episode, return_counts=True)
    audit.require(
        len(episode_ids) == 50 and np.all(counts == 500),
        "public: 50 complete 500-frame episodes",
    )
    dataset = report["dataset"]
    audit.require(
        int(dataset["frames"]) == 25_000
        and int(dataset["episodes"]) == 50
        and int(dataset["state_dim"]) == int(dataset["action_dim"]) == 14
        and int(dataset["episode_frame_min"]) == int(dataset["episode_frame_max"]) == 500
        and bool(dataset["finite"]),
        "public: report dataset summary",
    )

    train_ids = np.asarray(report["split"]["train_episodes"], dtype=np.int64)
    test_ids = np.asarray(report["split"]["test_episodes"], dtype=np.int64)
    train_mask = np.isin(episode, train_ids)
    test_mask = np.isin(episode, test_ids)
    audit.require(
        len(train_ids) == 40
        and len(test_ids) == 10
        and np.intersect1d(train_ids, test_ids).size == 0
        and np.array_equal(np.union1d(train_ids, test_ids), episode_ids),
        "public: whole-episode split without leakage",
    )
    audit.require(
        int(train_mask.sum()) == int(report["split"]["train_frames"]) == 20_000
        and int(test_mask.sum()) == int(report["split"]["test_frames"]) == 5_000,
        "public: train/test frame counts",
    )
    audit.require(
        split["seed"] == report["split"]["seed"] == 202609
        and split["train_episodes"] == report["split"]["train_episodes"]
        and split["test_episodes"] == report["split"]["test_episodes"],
        "public: split artifact agreement",
    )

    velocity = np.zeros_like(state)
    same_episode = episode[1:] == episode[:-1]
    velocity[1:][same_episode] = state[1:][same_episode] - state[:-1][same_episode]
    phase = np.zeros((len(state), 1), dtype=np.float64)
    for episode_id in episode_ids:
        mask = episode == episode_id
        episode_frames = frame[mask].astype(np.float64)
        denominator = max(float(episode_frames.max() - episode_frames.min()), 1.0)
        phase[mask, 0] = (episode_frames - episode_frames.min()) / denominator
    features = np.concatenate((state, velocity, phase, phase**2), axis=1)
    audit.require(
        features.shape == (25_000, 30) and int(report["features"]["input_dim"]) == 30,
        "public: independently rebuilt feature matrix",
    )

    with np.load(model_path) as model:
        alpha = float(model["alpha"])
        x_mean = np.asarray(model["x_mean"], dtype=np.float64)
        x_scale = np.asarray(model["x_scale"], dtype=np.float64)
        y_mean = np.asarray(model["y_mean"], dtype=np.float64)
        y_scale = np.asarray(model["y_scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
    audit.require(
        _close(alpha, report["model"]["alpha"])
        and x_mean.shape == x_scale.shape == (30,)
        and y_mean.shape == y_scale.shape == (14,)
        and weights.shape == (31, 14),
        "public: saved ridge-model structure",
    )
    normalized = (features[test_mask] - x_mean) / x_scale
    design = np.concatenate((normalized, np.ones((int(test_mask.sum()), 1))), axis=1)
    predicted_delta = (design @ weights) * y_scale + y_mean
    predicted_action = state[test_mask] + predicted_delta
    action_scale = np.maximum(action[train_mask].std(axis=0), 1e-8)
    held_out = _regression_metrics(action[test_mask], predicted_action, action_scale)
    identity = _regression_metrics(action[test_mask], state[test_mask], action_scale)
    audit.require(
        _metric_block_close(report["held_out_metrics"], held_out),
        "public: held-out ridge metrics recomputed",
    )
    audit.require(
        _metric_block_close(report["identity_baseline_metrics"], identity),
        "public: identity metrics recomputed",
    )
    arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    gripper_indices = np.asarray([6, 13])
    audit.require(
        _metric_block_close(
            report["held_out_metric_groups"]["arm_joints_12d"],
            _regression_metrics(
                action[test_mask][:, arm_indices],
                predicted_action[:, arm_indices],
                action_scale[arm_indices],
            ),
        )
        and _metric_block_close(
            report["held_out_metric_groups"]["grippers_2d"],
            _regression_metrics(
                action[test_mask][:, gripper_indices],
                predicted_action[:, gripper_indices],
                action_scale[gripper_indices],
            ),
        ),
        "public: arm/gripper metric groups recomputed",
    )
    audit.require(
        _close(
            report["comparison_to_identity"]["rmse_reduction_fraction"],
            1.0 - float(held_out["rmse"]) / float(identity["rmse"]),
        )
        and _close(
            report["comparison_to_identity"]["normalized_rmse_reduction_fraction"],
            1.0
            - float(held_out["normalized_rmse"]) / float(identity["normalized_rmse"]),
        ),
        "public: baseline reductions recomputed",
    )
    audit.require(
        "not evidence of wire-terminal insertion success" in report["interpretation"],
        "public: task-mismatch claim boundary",
    )
    return {
        "public_report_sha256": _sha256(report_path),
        "public_manifest_sha256": _sha256(manifest_path),
        "public_cache_sha256": cache_hash,
        "public_model_sha256": _sha256(model_path),
        "public_split_sha256": _sha256(split_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/papers/insertbot/formal/contact_v1/insert_paper_report.json"),
    )
    parser.add_argument(
        "--mujoco-report",
        type=Path,
        default=Path(
            "artifacts/papers/insertbot/formal/mujoco_direct_v1/insert_mujoco_direct_report.json"
        ),
    )
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("artifacts/aloha_ridge_v0/report.json"),
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("data/public/aloha_sim_insertion_human/harnessbench_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/papers/insertbot/formal/audit/insertbot_result_audit.json"),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    episodes_path = Path(report["episodes_csv"])
    aggregate_path = Path(report["aggregate_csv"])
    paired_path = Path(report["paired_csv"])
    rows = _read_episodes(episodes_path)
    audit = Audit()
    audit.require(_finite_json(report), "all report numbers finite")
    audit.require(_sha256(episodes_path) == report["episodes_sha256"], "episode CSV hash")
    audit.require(_sha256(aggregate_path) == report["aggregate_sha256"], "aggregate CSV hash")
    audit.require(_sha256(paired_path) == report["paired_sha256"], "paired CSV hash")
    for name, source in report["provenance_files"].items():
        audit.require(
            _sha256(Path(source)) == report["provenance_sha256"][name],
            f"frozen provenance hash: {name}",
        )
    audit.require(
        "not a calibrated connector" in report["claim_scope"],
        "synthetic claim boundary",
        report["claim_scope"],
    )
    _audit_structure(audit, report, rows)
    _audit_episode_invariants(audit, report, rows)
    _audit_aggregate(audit, report, rows)
    _audit_pairs(audit, report, rows)
    public_inputs = _audit_public_data(audit, args.public_report, args.public_manifest)
    mujoco_inputs = _audit_direct_mujoco(audit, args.mujoco_report)

    result = {
        "name": "InsertBot frozen-result independent audit",
        "passed": audit.passed,
        "checks": audit.checks,
        "inputs": {
            "report_sha256": _sha256(args.report),
            "episodes_sha256": _sha256(episodes_path),
            "aggregate_sha256": _sha256(aggregate_path),
            "paired_sha256": _sha256(paired_path),
            **public_inputs,
            **mujoco_inputs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(audit.checks),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
