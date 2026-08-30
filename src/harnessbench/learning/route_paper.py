"""Paired, paper-scale RouteBot evaluation with an ACT-style baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from harnessbench.learning.act_baseline import ACT_PAPER_URL, ActChunkPolicy
from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.sim.benchmark import aggregate_success, run_episode, summarize
from harnessbench.sim.envs import make_env
from harnessbench.sim.envs.base import EpisodeResult
from harnessbench.sim.envs.route import (
    ROUTE_SCREENED_TARGET_ASSIGNMENTS,
    ROUTE_TARGET_ASSIGNMENTS,
)
from harnessbench.sim.policies import RandomPolicy, TopologyPolicy

_ROUTE_WORKER_POLICIES: dict[str, object] | None = None
ROUTE_POLICY_NAMES = (
    "learned_topology",
    "learned_geometry",
    "learned_physical",
    "learned_semantic",
    "learned_no_features",
    "learned_data20",
    "learned_data50",
    "act_chunk",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
    "random",
)
ROUTE_POLICY_LABELS = {
    "learned_topology": "TopoHarness",
    "learned_geometry": "Geometry",
    "learned_physical": "Physical-only",
    "learned_semantic": "Semantic-only",
    "learned_no_features": "No ID/order feats.",
    "learned_data20": "TopoHarness (20\\% data)",
    "learned_data50": "TopoHarness (50\\% data)",
    "act_chunk": "ACT direct",
    "act_chunk_typed": "ACT-Pointer",
    "act_relational_typed": "ACT-RelPool",
    "teacher_topology": "Scripted teacher",
    "random": "Random",
}
ROUTE_POLICY_DISPLAY_ORDER = (
    "learned_topology",
    "act_relational_typed",
    "learned_geometry",
    "act_chunk_typed",
    "act_chunk",
    "learned_semantic",
    "learned_physical",
    "learned_no_features",
    "learned_data20",
    "learned_data50",
    "teacher_topology",
    "random",
)


def _order_policy_rows(rows: list[dict]) -> list[dict]:
    rank = {name: index for index, name in enumerate(ROUTE_POLICY_DISPLAY_ORDER)}
    return sorted(
        rows,
        key=lambda row: (rank.get(str(row.get("policy")), len(rank)), str(row.get("policy"))),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    """Convert NumPy report scalars without weakening JSON type checking."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _attach_checkpoint_hashes(report: dict) -> None:
    report.setdefault(
        "checkpoint_sha256",
        {
            name: _file_sha256(Path(path))
            for name, path in sorted(report["checkpoints"].items())
        },
    )


def _mcnemar_exact(topology_only: int, baseline_only: int) -> float:
    discordant = topology_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(topology_only, baseline_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _holm_adjust(rows: list[dict], key: str = "mcnemar_exact_p") -> list[dict]:
    """Attach step-down Holm family-wise adjusted p-values."""

    if not rows:
        return rows
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][key]))
    running = 0.0
    count = len(rows)
    adjusted = [1.0] * count
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(rows[index][key]))
        adjusted[index] = min(1.0, running)
    for index, value in enumerate(adjusted):
        rows[index][f"{key}_holm"] = value
    return rows


def _route_failure_type(record: dict) -> str:
    if bool(record["success"]):
        return "success"
    if int(record.get("wrong_latches", 0)) > 0:
        return "wrong_segment_or_order"
    if float(record.get("clip_coverage", 0.0)) < 1.0:
        return "incomplete_clip_routing"
    if not bool(record.get("endpoint_placed", False)):
        return "endpoint_not_placed"
    if int(record.get("crossings", 0)) > 0:
        return "residual_crossing"
    return "other"


def _paired_comparisons(records: list[dict]) -> list[dict]:
    indexed = {
        (float(row["difficulty"]), int(row["seed"]), str(row["policy"])): row for row in records
    }
    baselines = sorted({str(row["policy"]) for row in records} - {"learned_topology"})
    output: list[dict] = []
    for difficulty in sorted({float(row["difficulty"]) for row in records}):
        seeds = sorted(
            {int(row["seed"]) for row in records if float(row["difficulty"]) == difficulty}
        )
        for baseline in baselines:
            pairs = [
                (
                    bool(indexed[(difficulty, seed, "learned_topology")]["success"]),
                    bool(indexed[(difficulty, seed, baseline)]["success"]),
                )
                for seed in seeds
                if (difficulty, seed, "learned_topology") in indexed
                and (difficulty, seed, baseline) in indexed
            ]
            if not pairs:
                continue
            topology_only = sum(topology and not other for topology, other in pairs)
            baseline_only = sum(other and not topology for topology, other in pairs)
            both_success = sum(topology and other for topology, other in pairs)
            both_failure = sum(not topology and not other for topology, other in pairs)
            differences = np.asarray(
                [float(topology) - float(other) for topology, other in pairs],
                dtype=np.float64,
            )
            mean = float(differences.mean())
            margin = (
                1.96 * float(differences.std(ddof=1)) / math.sqrt(len(differences))
                if len(differences) > 1
                else 0.0
            )
            output.append(
                {
                    "difficulty": difficulty,
                    "baseline": baseline,
                    "paired_episodes": len(pairs),
                    "topology_success_baseline_failure": topology_only,
                    "topology_failure_baseline_success": baseline_only,
                    "both_success": both_success,
                    "both_failure": both_failure,
                    "paired_success_difference": mean,
                    "difference_ci95_low": max(-1.0, mean - margin),
                    "difference_ci95_high": min(1.0, mean + margin),
                    "mcnemar_exact_p": _mcnemar_exact(topology_only, baseline_only),
                    "matched_odds_ratio": (topology_only + 0.5) / (baseline_only + 0.5),
                }
            )
    return _holm_adjust(output)


def _paired_counterfactual_comparisons(records: list[dict]) -> list[dict]:
    indexed = {
        (
            float(row["difficulty"]),
            str(row["target_assignment"]),
            int(row["seed"]),
            str(row["policy"]),
        ): row
        for row in records
    }
    baselines = sorted({str(row["policy"]) for row in records} - {"learned_topology"})
    output: list[dict] = []
    for difficulty in sorted({float(row["difficulty"]) for row in records}):
        assignments = sorted(
            {
                str(row["target_assignment"])
                for row in records
                if float(row["difficulty"]) == difficulty
            }
        )
        for assignment in assignments:
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in records
                    if float(row["difficulty"]) == difficulty
                    and str(row["target_assignment"]) == assignment
                }
            )
            for baseline in baselines:
                pairs = [
                    (
                        bool(indexed[(difficulty, assignment, seed, "learned_topology")]["success"]),
                        bool(indexed[(difficulty, assignment, seed, baseline)]["success"]),
                    )
                    for seed in seeds
                    if (difficulty, assignment, seed, "learned_topology") in indexed
                    and (difficulty, assignment, seed, baseline) in indexed
                ]
                if not pairs:
                    continue
                topology_only = sum(left and not right for left, right in pairs)
                baseline_only = sum(right and not left for left, right in pairs)
                output.append(
                    {
                        "difficulty": difficulty,
                        "target_assignment": assignment,
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
    return _holm_adjust(output)


def _failure_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], Counter] = {}
    for record in records:
        key = (str(record["policy"]), float(record["difficulty"]))
        grouped.setdefault(key, Counter())[_route_failure_type(record)] += 1
    return [
        {
            "policy": policy,
            "difficulty": difficulty,
            "outcome": outcome,
            "episodes": count,
        }
        for (policy, difficulty), counts in sorted(grouped.items())
        for outcome, count in sorted(counts.items())
    ]


def _aggregate_boolean_metric(
    records: list[dict],
    metric: str,
    keys: tuple[str, ...],
) -> list[dict]:
    remapped = [{**record, "success": bool(record[metric])} for record in records]
    return [
        {**row, "metric": metric}
        for row in aggregate_success(remapped, keys)
    ]


def _make_route_policies(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    act_checkpoint: Path,
    *,
    act_relational_checkpoint: Path | None = None,
    extra_checkpoints: dict[str, Path] | None = None,
    policy_names: Iterable[str] | None = None,
    base_seed: int,
    device_name: str,
) -> dict[str, object]:
    extras = extra_checkpoints or {}
    optional_graph_names = {
        "learned_physical",
        "learned_semantic",
        "learned_no_features",
        "learned_data20",
        "learned_data50",
    }
    default_names = [
        name
        for name in ROUTE_POLICY_NAMES
        if (name != "act_relational_typed" or act_relational_checkpoint is not None)
        and (name not in optional_graph_names or name in extras)
    ]
    requested = list(policy_names) if policy_names is not None else default_names
    unknown = set(requested) - set(ROUTE_POLICY_NAMES)
    if unknown:
        raise KeyError(f"unknown RouteBot paper policies: {sorted(unknown)}")
    if "act_relational_typed" in requested and act_relational_checkpoint is None:
        raise ValueError("act_relational_typed requires an ACT relational checkpoint")
    missing_extras = (set(requested) & optional_graph_names) - set(extras)
    if missing_extras:
        raise ValueError(f"missing graph-ablation checkpoints: {sorted(missing_extras)}")

    policies: dict[str, object] = {}
    for name in requested:
        if name == "learned_topology":
            policies[name] = LearnedGraphPolicy(
                topology_checkpoint, "route", name=name, device_name=device_name
            )
        elif name == "learned_geometry":
            policies[name] = LearnedGraphPolicy(
                geometry_checkpoint, "route", name=name, device_name=device_name
            )
        elif name in optional_graph_names:
            policies[name] = LearnedGraphPolicy(
                extras[name], "route", name=name, device_name=device_name
            )
        elif name in {"act_chunk", "act_chunk_typed"}:
            policies[name] = ActChunkPolicy(
                act_checkpoint,
                "route",
                name=name,
                device_name=device_name,
                typed_grounding=name == "act_chunk_typed",
            )
        elif name == "act_relational_typed":
            policies[name] = ActChunkPolicy(
                act_relational_checkpoint,
                "route",
                name=name,
                device_name=device_name,
                typed_grounding=True,
            )
        elif name == "teacher_topology":
            policies[name] = TopologyPolicy(base_seed)
        elif name == "random":
            policies[name] = RandomPolicy(base_seed)
    return policies


def _initialize_route_worker(
    topology_checkpoint: str,
    geometry_checkpoint: str,
    act_checkpoint: str,
    act_relational_checkpoint: str,
    extra_checkpoint_items: tuple[tuple[str, str], ...],
    policy_names: tuple[str, ...],
    base_seed: int,
    device_name: str,
) -> None:
    global _ROUTE_WORKER_POLICIES
    if device_name == "cpu":
        torch.set_num_threads(1)
    _ROUTE_WORKER_POLICIES = _make_route_policies(
        Path(topology_checkpoint),
        Path(geometry_checkpoint),
        Path(act_checkpoint),
        act_relational_checkpoint=(
            Path(act_relational_checkpoint) if act_relational_checkpoint else None
        ),
        extra_checkpoints={name: Path(path) for name, path in extra_checkpoint_items},
        policy_names=policy_names,
        base_seed=base_seed,
        device_name=device_name,
    )


def _route_main_seed_records(
    policies: dict[str, object],
    *,
    difficulty: float,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    for policy_name, policy in policies.items():
        result = run_episode("route", policy, seed=seed, difficulty=difficulty)
        record = {"task_key": "route", **result.to_record()}
        record["policy"] = policy_name
        record["failure_type"] = _route_failure_type(record)
        records.append(record)
    return records


def _route_main_worker(job: tuple[float, int]) -> list[dict]:
    if _ROUTE_WORKER_POLICIES is None:
        raise RuntimeError("RouteBot process worker was not initialized")
    difficulty, seed = job
    return _route_main_seed_records(
        _ROUTE_WORKER_POLICIES,
        difficulty=difficulty,
        seed=seed,
    )


def evaluate_route_paper_table(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    act_checkpoint: Path,
    *,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    episodes: int = 100,
    base_seed: int = 61_000_000,
    device_name: str = "cpu",
    workers: int = 1,
    act_relational_checkpoint: Path | None = None,
    extra_checkpoints: dict[str, Path] | None = None,
    policy_names: Iterable[str] | None = None,
) -> dict:
    """Evaluate every policy on exactly the same frozen RouteBot seeds."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    difficulty_list = [float(value) for value in difficulties]
    extras = extra_checkpoints or {}
    optional_graph_names = {
        "learned_physical",
        "learned_semantic",
        "learned_no_features",
        "learned_data20",
        "learned_data50",
    }
    requested_policy_names = tuple(policy_names) if policy_names is not None else tuple(
        name
        for name in ROUTE_POLICY_NAMES
        if (name != "act_relational_typed" or act_relational_checkpoint is not None)
        and (name not in optional_graph_names or name in extras)
    )
    jobs = [
        (difficulty, base_seed + difficulty_index * 10_000 + episode)
        for difficulty_index, difficulty in enumerate(difficulty_list)
        for episode in range(episodes)
    ]
    started = perf_counter()
    if workers == 1:
        policy_specs = _make_route_policies(
            topology_checkpoint,
            geometry_checkpoint,
            act_checkpoint,
            act_relational_checkpoint=act_relational_checkpoint,
            extra_checkpoints=extras,
            policy_names=requested_policy_names,
            base_seed=base_seed,
            device_name=device_name,
        )
        grouped_records = [
            _route_main_seed_records(policy_specs, difficulty=difficulty, seed=seed)
            for difficulty, seed in jobs
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_route_worker,
            initargs=(
                str(topology_checkpoint),
                str(geometry_checkpoint),
                str(act_checkpoint),
                str(act_relational_checkpoint) if act_relational_checkpoint else "",
                tuple((name, str(path)) for name, path in sorted(extras.items())),
                requested_policy_names,
                base_seed,
                device_name,
            ),
        ) as executor:
            grouped_records = list(executor.map(_route_main_worker, jobs, chunksize=1))
    records = [record for group in grouped_records for record in group]
    runtime_seconds = perf_counter() - started
    return {
        "name": "RouteBot paired paper-scale policy evaluation",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "difficulties": difficulty_list,
            "episodes_per_difficulty": episodes,
            "base_seed": base_seed,
            "paired_seed_design": True,
            "device": device_name,
            "workers": workers,
            "policies": list(requested_policy_names),
            "evidence_level": "paper-scale" if episodes >= 100 else "development",
        },
        "checkpoints": {
            "learned_topology": str(topology_checkpoint.resolve()),
            "learned_geometry": str(geometry_checkpoint.resolve()),
            "act_chunk": str(act_checkpoint.resolve()),
            **(
                {"act_relational": str(act_relational_checkpoint.resolve())}
                if act_relational_checkpoint is not None
                else {}
            ),
            **{name: str(path.resolve()) for name, path in sorted(extras.items())},
        },
        "software": {
            "torch": torch.__version__,
            "runtime_seconds": runtime_seconds,
            "episode_throughput_per_second": len(records) / max(runtime_seconds, 1e-9),
        },
        "baseline_reference": {
            "act_paper": ACT_PAPER_URL,
            "scope": "independent state-based ACT-style adaptation",
        },
        "claim_scope": (
            "paired 2-D PBD simulation evidence; not end-to-end MuJoCo or real-robot evidence"
        ),
        "records": records,
        "summary": summarize(records),
        "aggregate_by_policy": aggregate_success(records, ("task_key", "robot", "policy")),
        "strict_crossing_free_aggregate": _aggregate_boolean_metric(
            records,
            "strict_crossing_free_success",
            ("task_key", "robot", "policy"),
        ),
        "paired_comparisons": _paired_comparisons(records),
        "failure_summary": _failure_summary(records),
    }


def _run_route_counterfactual_episode(
    policy,
    *,
    seed: int,
    difficulty: float,
    target_assignment: tuple[int, int, int],
) -> EpisodeResult:
    env = make_env("route")
    env.target_indices_override = target_assignment
    env.reset(seed=seed, difficulty=difficulty)
    policy.reset(seed + 100_003)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(policy.act(env))
    return EpisodeResult(
        task=env.task_name,
        robot=env.robot_name,
        policy=policy.name,
        seed=seed,
        difficulty=difficulty,
        success=bool(terminated and env.success()),
        total_reward=float(env.total_reward),
        steps=env.step_count,
        metrics=env.metrics(),
    )


def _route_counterfactual_seed_records(
    policies: dict[str, object],
    *,
    difficulty: float,
    seed: int,
    target_assignments: tuple[tuple[int, int, int], ...],
) -> list[dict]:
    records: list[dict] = []
    for target_assignment in target_assignments:
        assignment_label = "-".join(str(value) for value in target_assignment)
        for policy_name, policy in policies.items():
            result = _run_route_counterfactual_episode(
                policy,
                seed=seed,
                difficulty=difficulty,
                target_assignment=target_assignment,
            )
            record = {"task_key": "route", **result.to_record()}
            record["policy"] = policy_name
            record["target_assignment"] = assignment_label
            record["failure_type"] = _route_failure_type(record)
            records.append(record)
    return records


def _route_counterfactual_worker(
    job: tuple[float, int, tuple[tuple[int, int, int], ...]],
) -> list[dict]:
    if _ROUTE_WORKER_POLICIES is None:
        raise RuntimeError("RouteBot process worker was not initialized")
    difficulty, seed, target_assignments = job
    return _route_counterfactual_seed_records(
        _ROUTE_WORKER_POLICIES,
        difficulty=difficulty,
        seed=seed,
        target_assignments=target_assignments,
    )


def audit_route_assignment_feasibility(
    *,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    physical_seeds: int = 10,
    base_seed: int = 60_500_000,
    target_assignments: Iterable[tuple[int, int, int]] = ROUTE_TARGET_ASSIGNMENTS,
) -> dict:
    """Screen the predeclared assignment set with the privileged heuristic teacher.

    This is a reproducible controller-feasibility audit, not a proof that every
    physical cable realization is solvable.  It is intentionally run before
    learned-policy comparisons so assignment inclusion does not depend on the
    proposed method's outcome.
    """

    if physical_seeds < 1:
        raise ValueError("physical_seeds must be at least one")
    difficulty_list = [float(value) for value in difficulties]
    assignment_list = [tuple(int(index) for index in value) for value in target_assignments]
    if not assignment_list:
        raise ValueError("target_assignments must be non-empty")
    records: list[dict] = []
    started = perf_counter()
    teacher = TopologyPolicy(base_seed)
    for difficulty_index, difficulty in enumerate(difficulty_list):
        for episode in range(physical_seeds):
            seed = base_seed + difficulty_index * 10_000 + episode
            for target_assignment in assignment_list:
                result = _run_route_counterfactual_episode(
                    teacher,
                    seed=seed,
                    difficulty=difficulty,
                    target_assignment=target_assignment,
                )
                record = {"task_key": "route", **result.to_record()}
                record["policy"] = "privileged_topology_teacher"
                record["target_assignment"] = "-".join(
                    str(value) for value in target_assignment
                )
                record["failure_type"] = _route_failure_type(record)
                records.append(record)
    cells = aggregate_success(records, ("difficulty", "target_assignment"))
    runtime_seconds = perf_counter() - started
    return {
        "name": "RouteBot predeclared assignment teacher-feasibility audit",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "difficulties": difficulty_list,
            "physical_seeds_per_difficulty": physical_seeds,
            "target_assignments": [
                "-".join(str(value) for value in assignment)
                for assignment in assignment_list
            ],
            "base_seed": base_seed,
            "performed_before_learned_policy_selection": True,
        },
        "interpretation": (
            "Privileged heuristic-controller screen only; a failed rollout may indicate "
            "controller or horizon failure rather than physical infeasibility."
        ),
        "software": {
            "runtime_seconds": runtime_seconds,
            "episode_throughput_per_second": len(records) / max(runtime_seconds, 1e-9),
        },
        "records": records,
        "cells": cells,
        "all_cells_nonzero": all(float(cell["success_rate"]) > 0.0 for cell in cells),
        "minimum_cell_success_rate": min(float(cell["success_rate"]) for cell in cells),
        "failure_summary": _failure_summary(records),
    }


def _assignment_invariance(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, str], list[dict]] = {}
    for record in records:
        key = (
            str(record["policy"]),
            float(record["difficulty"]),
            str(record["target_assignment"]),
        )
        grouped.setdefault(key, []).append(record)

    output: list[dict] = []
    for policy in sorted({key[0] for key in grouped}):
        for difficulty in sorted({key[1] for key in grouped if key[0] == policy}):
            rates = {
                assignment: float(np.mean([row["success"] for row in rows]))
                for (item_policy, item_difficulty, assignment), rows in grouped.items()
                if item_policy == policy and item_difficulty == difficulty
            }
            output.append(
                {
                    "policy": policy,
                    "difficulty": difficulty,
                    "assignments": len(rates),
                    "mean_success_rate": float(np.mean(list(rates.values()))),
                    "min_success_rate": float(np.min(list(rates.values()))),
                    "max_success_rate": float(np.max(list(rates.values()))),
                    "assignment_success_range": float(
                        np.max(list(rates.values())) - np.min(list(rates.values()))
                    ),
                }
            )
    return output


def evaluate_route_counterfactual(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    act_checkpoint: Path,
    *,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    physical_seeds: int = 10,
    base_seed: int = 62_000_000,
    device_name: str = "cpu",
    target_assignments: Iterable[
        tuple[int, int, int]
    ] = ROUTE_SCREENED_TARGET_ASSIGNMENTS,
    workers: int = 1,
    act_relational_checkpoint: Path | None = None,
    extra_checkpoints: dict[str, Path] | None = None,
    policy_names: Iterable[str] | None = None,
) -> dict:
    """Change feasible segment-to-clip assignments on identical physics."""

    if physical_seeds < 1:
        raise ValueError("physical_seeds must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    difficulty_list = [float(value) for value in difficulties]
    assignment_list = [tuple(int(index) for index in value) for value in target_assignments]
    if not assignment_list:
        raise ValueError("target_assignments must be non-empty")
    extras = extra_checkpoints or {}
    optional_graph_names = {
        "learned_physical",
        "learned_semantic",
        "learned_no_features",
        "learned_data20",
        "learned_data50",
    }
    requested_policy_names = tuple(policy_names) if policy_names is not None else tuple(
        name
        for name in ROUTE_POLICY_NAMES
        if (name != "act_relational_typed" or act_relational_checkpoint is not None)
        and (name not in optional_graph_names or name in extras)
    )
    jobs = [
        (
            difficulty,
            base_seed + difficulty_index * 10_000 + episode,
            tuple(assignment_list),
        )
        for difficulty_index, difficulty in enumerate(difficulty_list)
        for episode in range(physical_seeds)
    ]
    started = perf_counter()
    if workers == 1:
        policies = _make_route_policies(
            topology_checkpoint,
            geometry_checkpoint,
            act_checkpoint,
            act_relational_checkpoint=act_relational_checkpoint,
            extra_checkpoints=extras,
            policy_names=requested_policy_names,
            base_seed=base_seed,
            device_name=device_name,
        )
        grouped_records = [
            _route_counterfactual_seed_records(
                policies,
                difficulty=difficulty,
                seed=seed,
                target_assignments=assignments,
            )
            for difficulty, seed, assignments in jobs
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_route_worker,
            initargs=(
                str(topology_checkpoint),
                str(geometry_checkpoint),
                str(act_checkpoint),
                str(act_relational_checkpoint) if act_relational_checkpoint else "",
                tuple((name, str(path)) for name, path in sorted(extras.items())),
                requested_policy_names,
                base_seed,
                device_name,
            ),
        ) as executor:
            grouped_records = list(
                executor.map(_route_counterfactual_worker, jobs, chunksize=1)
            )
    records = [record for group in grouped_records for record in group]
    runtime_seconds = perf_counter() - started
    return {
        "name": "RouteBot exact feasible segment-to-clip assignment counterfactual",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "difficulties": difficulty_list,
            "physical_seeds_per_difficulty": physical_seeds,
            "target_assignments": [
                "-".join(str(value) for value in assignment) for assignment in assignment_list
            ],
            "base_seed": base_seed,
            "paired_physics": True,
            "device": device_name,
            "workers": workers,
            "policies": list(requested_policy_names),
            "evidence_level": "paper-scale" if physical_seeds >= 100 else "development",
        },
        "checkpoints": {
            "learned_topology": str(topology_checkpoint.resolve()),
            "learned_geometry": str(geometry_checkpoint.resolve()),
            "act_chunk": str(act_checkpoint.resolve()),
            **(
                {"act_relational": str(act_relational_checkpoint.resolve())}
                if act_relational_checkpoint is not None
                else {}
            ),
            **{name: str(path.resolve()) for name, path in sorted(extras.items())},
        },
        "software": {
            "torch": torch.__version__,
            "runtime_seconds": runtime_seconds,
            "episode_throughput_per_second": len(records) / max(runtime_seconds, 1e-9),
        },
        "intervention": (
            "Only the three increasing cable-particle assignments to ordered clips change "
            "within local arc-length windows; cable geometry, fixtures, material parameters, "
            "random seed and initial arm state are fixed. The mapping is observable only "
            "through semantic graph edges."
        ),
        "claim_scope": "paired 2-D PBD counterfactual; not causal real-world evidence",
        "records": records,
        "aggregate_by_policy_assignment": aggregate_success(
            records,
            ("task_key", "robot", "policy", "difficulty", "target_assignment"),
        ),
        "aggregate_by_policy": aggregate_success(records, ("policy",)),
        "strict_crossing_free_aggregate": _aggregate_boolean_metric(
            records,
            "strict_crossing_free_success",
            ("task_key", "robot", "policy", "difficulty", "target_assignment"),
        ),
        "assignment_invariance": _assignment_invariance(records),
        "paired_comparisons": _paired_counterfactual_comparisons(records),
        "failure_summary": _failure_summary(records),
    }


def save_route_paper_table(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_checkpoint_hashes(report)
    report_path = output_dir / "route_paper_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    def write_csv(name: str, rows: list[dict]) -> str:
        path = output_dir / name
        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return str(path.resolve())

    episodes_csv = write_csv("route_paper_episodes.csv", report["records"])
    summary_csv = write_csv("route_paper_summary.csv", report["summary"])
    aggregate_csv = write_csv("route_paper_aggregate.csv", report["aggregate_by_policy"])
    strict_csv = write_csv(
        "route_paper_strict_crossing_free.csv",
        report["strict_crossing_free_aggregate"],
    )
    paired_csv = write_csv("route_paper_paired_tests.csv", report["paired_comparisons"])
    failures_csv = write_csv("route_paper_failures.csv", report["failure_summary"])

    warning = (
        "This run meets the configured 100-seed-per-cell scale gate, but remains PBD-only."
        if report["config"]["evidence_level"] == "paper-scale"
        else "Development run: do not use these confidence intervals as final paper evidence."
    )
    lines = [
        "# RouteBot paired policy table",
        "",
        f"> {warning}",
        "",
        "| Policy | Episodes | Semantic success (95% Wilson CI) | Crossing-free success |",
        "|---|---:|---:|---:|",
    ]
    strict_by_policy = {
        str(row["policy"]): row for row in report["strict_crossing_free_aggregate"]
    }
    for row in _order_policy_rows(report["aggregate_by_policy"]):
        strict = strict_by_policy[str(row["policy"])]
        lines.append(
            f"| {row['policy']} | {row['episodes']} | {row['success_rate']:.3f} "
            f"[{row['success_ci95_low']:.3f}, {row['success_ci95_high']:.3f}] | "
            f"{strict['success_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Paired tests use the same physical seed for every policy. McNemar exact p-values ",
            "are reported in `route_paper_paired_tests.csv`. This is 2-D simulation evidence, ",
            "not a real-production success claim.",
            "",
        ]
    )
    results_path = output_dir / "ROUTE_PAPER_RESULTS.md"
    results_path.write_text("\n".join(lines), encoding="utf-8")

    latex_lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Policy & Sem. success (\\%) & Cross.-free (\\%) \\\\",
        "\\midrule",
    ]
    for row in _order_policy_rows(report["aggregate_by_policy"]):
        strict = strict_by_policy[str(row["policy"])]
        label = ROUTE_POLICY_LABELS.get(str(row["policy"]), str(row["policy"]))
        latex_lines.append(
            f"{label} & {100.0 * row['success_rate']:.1f} & "
            f"{100.0 * strict['success_rate']:.1f} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    latex_path = output_dir / "route_main_table.tex"
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": episodes_csv,
        "summary_csv": summary_csv,
        "aggregate_csv": aggregate_csv,
        "strict_crossing_free_csv": strict_csv,
        "paired_tests_csv": paired_csv,
        "failures_csv": failures_csv,
        "results_markdown": str(results_path.resolve()),
        "latex_table": str(latex_path.resolve()),
        "episode_count": len(report["records"]),
    }


def save_route_counterfactual(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_checkpoint_hashes(report)
    report_path = output_dir / "route_counterfactual_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    def write_csv(name: str, rows: list[dict]) -> str:
        path = output_dir / name
        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return str(path.resolve())

    episodes_csv = write_csv("route_counterfactual_episodes.csv", report["records"])
    aggregate_csv = write_csv(
        "route_counterfactual_aggregate.csv",
        report["aggregate_by_policy_assignment"],
    )
    strict_csv = write_csv(
        "route_counterfactual_strict_crossing_free.csv",
        report["strict_crossing_free_aggregate"],
    )
    invariance_csv = write_csv(
        "route_counterfactual_invariance.csv",
        report["assignment_invariance"],
    )
    failures_csv = write_csv("route_counterfactual_failures.csv", report["failure_summary"])
    paired_csv = write_csv(
        "route_counterfactual_paired_tests.csv",
        report["paired_comparisons"],
    )
    policy_csv = write_csv(
        "route_counterfactual_policy_aggregate.csv",
        report["aggregate_by_policy"],
    )
    evidence_note = (
        "> Paper-scale paired evidence: every assignment contains at least "
        "100 physical seeds."
        if int(report["config"]["physical_seeds_per_difficulty"]) >= 100
        else "> Development evidence: fewer than 100 physical seeds per assignment."
    )
    lines = [
        "# RouteBot segment-to-clip counterfactual",
        "",
        evidence_note,
        "",
        report["intervention"],
        "",
        "| Policy | Difficulty | Mean success | Min | Max | Assignment range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["assignment_invariance"]:
        lines.append(
            f"| {row['policy']} | {row['difficulty']:.1f} | "
            f"{row['mean_success_rate']:.3f} | {row['min_success_rate']:.3f} | "
            f"{row['max_success_rate']:.3f} | {row['assignment_success_range']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All screened semantic assignments reuse identical cable geometry, fixture layout, ",
            "material parameters and random seeds. The frozen set was selected by an independent ",
            "teacher-feasibility audit before learned-policy evaluation. Results remain 2-D ",
            "simulation evidence.",
            "",
        ]
    )
    results_path = output_dir / "ROUTE_COUNTERFACTUAL_RESULTS.md"
    results_path.write_text("\n".join(lines), encoding="utf-8")

    strict_policy_rows = _aggregate_boolean_metric(
        report["records"],
        "strict_crossing_free_success",
        ("policy",),
    )
    strict_by_policy = {str(row["policy"]): row for row in strict_policy_rows}
    latex_lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Policy & Sem. success (\\%) & Cross.-free (\\%) \\\\",
        "\\midrule",
    ]
    for row in _order_policy_rows(report["aggregate_by_policy"]):
        policy = str(row["policy"])
        strict = strict_by_policy[policy]
        label = ROUTE_POLICY_LABELS.get(policy, policy)
        latex_lines.append(
            f"{label} & {100.0 * row['success_rate']:.1f} & "
            f"{100.0 * strict['success_rate']:.1f} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    latex_path = output_dir / "route_counterfactual_table.tex"
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": episodes_csv,
        "aggregate_csv": aggregate_csv,
        "strict_crossing_free_csv": strict_csv,
        "invariance_csv": invariance_csv,
        "failures_csv": failures_csv,
        "paired_tests_csv": paired_csv,
        "policy_aggregate_csv": policy_csv,
        "results_markdown": str(results_path.resolve()),
        "latex_table": str(latex_path.resolve()),
        "episode_count": len(report["records"]),
    }


def save_route_assignment_audit(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "route_assignment_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    cells_path = output_dir / "route_assignment_cells.csv"
    with cells_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in report["cells"] for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["cells"])

    lines = [
        "# RouteBot predeclared assignment audit",
        "",
        f"> {report['interpretation']}",
        "",
        "| Difficulty | Target particles | Episodes | Teacher success (95% Wilson CI) |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        lines.append(
            f"| {row['difficulty']:.1f} | {row['target_assignment']} | "
            f"{row['episodes']} | {row['success_rate']:.3f} "
            f"[{row['success_ci95_low']:.3f}, {row['success_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            (
                f"All cells nonzero: **{report['all_cells_nonzero']}**. Minimum cell success: "
                f"**{report['minimum_cell_success_rate']:.3f}**."
            ),
            "",
        ]
    )
    markdown_path = output_dir / "ROUTE_ASSIGNMENT_AUDIT.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "cells_csv": str(cells_path.resolve()),
        "results_markdown": str(markdown_path.resolve()),
        "episode_count": len(report["records"]),
        "all_cells_nonzero": report["all_cells_nonzero"],
        "minimum_cell_success_rate": report["minimum_cell_success_rate"],
    }
