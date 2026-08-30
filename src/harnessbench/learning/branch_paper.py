"""Paper-scale paired evaluation for semantic dual-arm BranchBot.

Every physical seed is replayed under both observable A/B label assignments.
The intervention changes semantic names and arm/target bindings while keeping
the cable geometry fixed.  This makes the comparison resistant to memorizing
screen position, branch colour, or a single canonical target ordering.
"""

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
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact
from harnessbench.sim.benchmark import aggregate_success, summarize
from harnessbench.sim.envs import make_env
from harnessbench.sim.envs.base import EpisodeResult
from harnessbench.sim.policies import RandomPolicy, TopologyPolicy

BRANCH_POLICY_NAMES = (
    "learned_topology",
    "learned_geometry",
    "learned_physical",
    "learned_semantic",
    "learned_no_features",
    "act_chunk",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
    "random",
)
BRANCH_POLICY_LABELS = {
    "learned_topology": "TopoHarness",
    "learned_geometry": "Geometry",
    "learned_physical": "No semantic edges",
    "learned_semantic": "No physical edges",
    "learned_no_features": "Relations, no ID/order feats.",
    "act_chunk": "ACT direct",
    "act_chunk_typed": "ACT-Pointer",
    "act_relational_typed": "ACT-RelPool",
    "teacher_topology": "Scripted teacher",
    "random": "Random",
}
BRANCH_POLICY_DISPLAY_ORDER = BRANCH_POLICY_NAMES

_BRANCH_WORKER_POLICIES: dict[str, object] | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _branch_failure_type(record: dict) -> str:
    if bool(record["success"]):
        return "success"
    error_a = float(record.get("endpoint_a_error", math.inf))
    error_b = float(record.get("endpoint_b_error", math.inf))
    if error_a > 0.08 and error_b > 0.08:
        return "both_endpoints_misplaced"
    if error_a > 0.08 or error_b > 0.08:
        return "single_endpoint_misplaced"
    if int(record.get("crossings", 0)) > 0:
        return "residual_crossing"
    return "unstable_or_timeout"


def _make_branch_policies(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    act_checkpoint: Path,
    *,
    act_relational_checkpoint: Path | None,
    extra_checkpoints: dict[str, Path] | None,
    policy_names: Iterable[str],
    base_seed: int,
    device_name: str,
) -> dict[str, object]:
    requested = tuple(policy_names)
    extras = extra_checkpoints or {}
    optional_graph_names = {
        "learned_physical",
        "learned_semantic",
        "learned_no_features",
    }
    unknown = set(requested) - set(BRANCH_POLICY_NAMES)
    if unknown:
        raise KeyError(f"unknown BranchBot paper policies: {sorted(unknown)}")
    if "act_relational_typed" in requested and act_relational_checkpoint is None:
        raise ValueError("act_relational_typed requires a relational ACT checkpoint")
    missing_extras = (set(requested) & optional_graph_names) - set(extras)
    if missing_extras:
        raise ValueError(f"missing BranchBot graph-ablation checkpoints: {sorted(missing_extras)}")
    policies: dict[str, object] = {}
    for name in requested:
        if name == "learned_topology":
            policies[name] = LearnedGraphPolicy(
                topology_checkpoint, "branch", name=name, device_name=device_name
            )
        elif name == "learned_geometry":
            policies[name] = LearnedGraphPolicy(
                geometry_checkpoint, "branch", name=name, device_name=device_name
            )
        elif name in optional_graph_names:
            policies[name] = LearnedGraphPolicy(
                extras[name], "branch", name=name, device_name=device_name
            )
        elif name in {"act_chunk", "act_chunk_typed"}:
            policies[name] = ActChunkPolicy(
                act_checkpoint,
                "branch",
                name=name,
                device_name=device_name,
                typed_grounding=name == "act_chunk_typed",
            )
        elif name == "act_relational_typed":
            assert act_relational_checkpoint is not None
            policies[name] = ActChunkPolicy(
                act_relational_checkpoint,
                "branch",
                name=name,
                device_name=device_name,
                typed_grounding=True,
            )
        elif name == "teacher_topology":
            policies[name] = TopologyPolicy(base_seed)
        elif name == "random":
            policies[name] = RandomPolicy(base_seed)
    return policies


def _initialize_branch_worker(
    topology_checkpoint: str,
    geometry_checkpoint: str,
    act_checkpoint: str,
    act_relational_checkpoint: str,
    extra_checkpoint_items: tuple[tuple[str, str], ...],
    policy_names: tuple[str, ...],
    base_seed: int,
    device_name: str,
) -> None:
    global _BRANCH_WORKER_POLICIES
    if device_name == "cpu":
        torch.set_num_threads(1)
    _BRANCH_WORKER_POLICIES = _make_branch_policies(
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


def _run_branch_episode(
    policy,
    *,
    seed: int,
    difficulty: float,
    semantic_swap: bool,
) -> EpisodeResult:
    env = make_env("branch")
    env.semantic_swap_override = bool(semantic_swap)
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


def _branch_seed_records(
    policies: dict[str, object],
    *,
    difficulty: float,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    for semantic_swap in (False, True):
        for policy_name, policy in policies.items():
            result = _run_branch_episode(
                policy,
                seed=seed,
                difficulty=difficulty,
                semantic_swap=semantic_swap,
            )
            record = {"task_key": "branch", **result.to_record()}
            record["policy"] = policy_name
            record["semantic_target_swap"] = bool(semantic_swap)
            record["failure_type"] = _branch_failure_type(record)
            records.append(record)
    return records


def _branch_worker(job: tuple[float, int]) -> list[dict]:
    if _BRANCH_WORKER_POLICIES is None:
        raise RuntimeError("BranchBot process worker was not initialized")
    difficulty, seed = job
    return _branch_seed_records(
        _BRANCH_WORKER_POLICIES,
        difficulty=difficulty,
        seed=seed,
    )


def _paired_comparisons(records: list[dict]) -> list[dict]:
    indexed = {
        (
            float(row["difficulty"]),
            bool(row["semantic_target_swap"]),
            int(row["seed"]),
            str(row["policy"]),
        ): row
        for row in records
    }
    baselines = sorted({str(row["policy"]) for row in records} - {"learned_topology"})
    output: list[dict] = []
    for difficulty in sorted({float(row["difficulty"]) for row in records}):
        for semantic_swap in (False, True):
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in records
                    if float(row["difficulty"]) == difficulty
                    and bool(row["semantic_target_swap"]) == semantic_swap
                }
            )
            for baseline in baselines:
                pairs = [
                    (
                        bool(indexed[(difficulty, semantic_swap, seed, "learned_topology")]["success"]),
                        bool(indexed[(difficulty, semantic_swap, seed, baseline)]["success"]),
                    )
                    for seed in seeds
                    if (difficulty, semantic_swap, seed, "learned_topology") in indexed
                    and (difficulty, semantic_swap, seed, baseline) in indexed
                ]
                if not pairs:
                    continue
                topology_only = sum(left and not right for left, right in pairs)
                baseline_only = sum(right and not left for left, right in pairs)
                differences = np.asarray(
                    [float(left) - float(right) for left, right in pairs],
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
                        "semantic_target_swap": semantic_swap,
                        "baseline": baseline,
                        "paired_physical_seeds": len(pairs),
                        "topology_success_baseline_failure": topology_only,
                        "topology_failure_baseline_success": baseline_only,
                        "both_success": sum(left and right for left, right in pairs),
                        "both_failure": sum(not left and not right for left, right in pairs),
                        "paired_success_difference": mean,
                        "difference_ci95_low": max(-1.0, mean - margin),
                        "difference_ci95_high": min(1.0, mean + margin),
                        "mcnemar_exact_p": _mcnemar_exact(topology_only, baseline_only),
                        "matched_odds_ratio": (topology_only + 0.5) / (baseline_only + 0.5),
                    }
                )
    return _holm_adjust(output)


def _semantic_invariance(records: list[dict]) -> list[dict]:
    indexed = {
        (
            float(row["difficulty"]),
            int(row["seed"]),
            str(row["policy"]),
            bool(row["semantic_target_swap"]),
        ): row
        for row in records
    }
    output: list[dict] = []
    for policy in sorted({str(row["policy"]) for row in records}):
        for difficulty in sorted({float(row["difficulty"]) for row in records}):
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in records
                    if float(row["difficulty"]) == difficulty
                    and str(row["policy"]) == policy
                }
            )
            pairs = [
                (
                    bool(indexed[(difficulty, seed, policy, False)]["success"]),
                    bool(indexed[(difficulty, seed, policy, True)]["success"]),
                )
                for seed in seeds
            ]
            false_only = sum(left and not right for left, right in pairs)
            true_only = sum(right and not left for left, right in pairs)
            output.append(
                {
                    "policy": policy,
                    "difficulty": difficulty,
                    "paired_physical_seeds": len(pairs),
                    "outcome_agreement": float(np.mean([left == right for left, right in pairs])),
                    "canonical_success_swapped_failure": false_only,
                    "canonical_failure_swapped_success": true_only,
                    "mcnemar_exact_p": _mcnemar_exact(false_only, true_only),
                }
            )
    return output


def _failure_summary(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float, bool], Counter] = {}
    for row in records:
        key = (
            str(row["policy"]),
            float(row["difficulty"]),
            bool(row["semantic_target_swap"]),
        )
        grouped.setdefault(key, Counter())[str(row["failure_type"])] += 1
    return [
        {
            "policy": policy,
            "difficulty": difficulty,
            "semantic_target_swap": swap,
            "outcome": outcome,
            "episodes": count,
        }
        for (policy, difficulty, swap), counts in sorted(grouped.items())
        for outcome, count in sorted(counts.items())
    ]


def evaluate_branch_paper_table(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    act_checkpoint: Path,
    *,
    act_relational_checkpoint: Path | None = None,
    extra_checkpoints: dict[str, Path] | None = None,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    physical_seeds: int = 100,
    base_seed: int = 66_000_000,
    device_name: str = "cpu",
    workers: int = 1,
    policy_names: Iterable[str] | None = None,
) -> dict:
    """Run paired semantic interventions on frozen physical BranchBot seeds."""

    if physical_seeds < 1:
        raise ValueError("physical_seeds must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    difficulties_list = [float(value) for value in difficulties]
    extras = extra_checkpoints or {}
    optional_graph_names = {
        "learned_physical",
        "learned_semantic",
        "learned_no_features",
    }
    requested = tuple(policy_names) if policy_names is not None else tuple(
        name
        for name in BRANCH_POLICY_NAMES
        if name != "random"
        and (name != "act_relational_typed" or act_relational_checkpoint is not None)
        and (name not in optional_graph_names or name in extras)
    )
    jobs = [
        (difficulty, base_seed + difficulty_index * 10_000 + episode)
        for difficulty_index, difficulty in enumerate(difficulties_list)
        for episode in range(physical_seeds)
    ]
    started = perf_counter()
    if workers == 1:
        policies = _make_branch_policies(
            topology_checkpoint,
            geometry_checkpoint,
            act_checkpoint,
            act_relational_checkpoint=act_relational_checkpoint,
            extra_checkpoints=extras,
            policy_names=requested,
            base_seed=base_seed,
            device_name=device_name,
        )
        grouped = [
            _branch_seed_records(policies, difficulty=difficulty, seed=seed)
            for difficulty, seed in jobs
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_branch_worker,
            initargs=(
                str(topology_checkpoint),
                str(geometry_checkpoint),
                str(act_checkpoint),
                str(act_relational_checkpoint) if act_relational_checkpoint else "",
                tuple((name, str(path)) for name, path in sorted(extras.items())),
                requested,
                base_seed,
                device_name,
            ),
        ) as executor:
            grouped = list(executor.map(_branch_worker, jobs, chunksize=1))
    records = [row for group in grouped for row in group]
    runtime_seconds = perf_counter() - started
    checkpoints = {
        "learned_topology": str(topology_checkpoint.resolve()),
        "learned_geometry": str(geometry_checkpoint.resolve()),
        "act_chunk": str(act_checkpoint.resolve()),
        **(
            {"act_relational": str(act_relational_checkpoint.resolve())}
            if act_relational_checkpoint is not None
            else {}
        ),
        **{name: str(path.resolve()) for name, path in sorted(extras.items())},
    }
    return {
        "name": "BranchBot paired paper-scale semantic counterfactual evaluation",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "difficulties": difficulties_list,
            "physical_seeds_per_difficulty": physical_seeds,
            "semantic_assignments_per_physical_seed": [False, True],
            "base_seed": base_seed,
            "paired_seed_design": True,
            "device": device_name,
            "workers": workers,
            "policies": list(requested),
            "evidence_level": "paper-scale" if physical_seeds >= 100 else "development",
        },
        "checkpoints": checkpoints,
        "checkpoint_sha256": {
            name: _file_sha256(Path(path)) for name, path in sorted(checkpoints.items())
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
            "paired 2-D PBD simulation with a fixed two-branch dual-arm harness; "
            "not yet variable-branch, direct-MuJoCo, or real-robot evidence"
        ),
        "records": records,
        "summary": summarize(records),
        "aggregate_by_policy": aggregate_success(records, ("task_key", "robot", "policy")),
        "aggregate_by_policy_assignment": aggregate_success(
            records,
            ("task_key", "robot", "policy", "semantic_target_swap"),
        ),
        "aggregate_by_policy_difficulty_assignment": aggregate_success(
            records,
            ("policy", "difficulty", "semantic_target_swap"),
        ),
        "paired_comparisons": _paired_comparisons(records),
        "semantic_invariance": _semantic_invariance(records),
        "failure_summary": _failure_summary(records),
    }


def save_branch_paper_table(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "branch_paper_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )

    def write_csv(name: str, rows: list[dict]) -> str:
        path = output_dir / name
        if rows:
            fields = sorted({key for row in rows for key in row})
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return str(path.resolve())

    episodes_path = write_csv("branch_paper_episodes.csv", report["records"])
    aggregate_path = write_csv("branch_paper_aggregate.csv", report["aggregate_by_policy"])
    cells_path = write_csv(
        "branch_paper_cells.csv",
        report["aggregate_by_policy_difficulty_assignment"],
    )
    paired_path = write_csv("branch_paper_paired.csv", report["paired_comparisons"])
    invariance_path = write_csv("branch_semantic_invariance.csv", report["semantic_invariance"])
    failure_path = write_csv("branch_failure_summary.csv", report["failure_summary"])

    order = {name: index for index, name in enumerate(BRANCH_POLICY_DISPLAY_ORDER)}
    aggregate = sorted(
        report["aggregate_by_policy"],
        key=lambda row: order.get(str(row["policy"]), len(order)),
    )
    lines = [
        "# BranchBot paired paper-scale results",
        "",
        "> Two semantic assignments are paired on every frozen physical seed.",
        "",
        "| Policy | Episodes | Success (95% Wilson CI) |",
        "|---|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {BRANCH_POLICY_LABELS.get(row['policy'], row['policy'])} | "
            f"{row['episodes']} | {row['success_rate']:.3f} "
            f"[{row['success_ci95_low']:.3f}, {row['success_ci95_high']:.3f}] |"
        )
    lines.extend(["", f"Report SHA-256: `{_file_sha256(report_path)}`", ""])
    results_path = output_dir / "BRANCH_PAPER_RESULTS.md"
    results_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "report_sha256": _file_sha256(report_path),
        "episodes_csv": episodes_path,
        "aggregate_csv": aggregate_path,
        "cells_csv": cells_path,
        "paired_csv": paired_path,
        "invariance_csv": invariance_path,
        "failure_csv": failure_path,
        "results_markdown": str(results_path.resolve()),
        "episode_count": len(report["records"]),
    }
