"""Paired closed-loop evaluation for learned topology and geometry policies."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from harnessbench.learning.graph import TASK_ORDER
from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.sim.benchmark import aggregate_success, run_episode, summarize
from harnessbench.sim.envs import make_env
from harnessbench.sim.envs.base import EpisodeResult
from harnessbench.sim.policies import TopologyPolicy


def _paired_advantage(records: list[dict]) -> list[dict]:
    indexed = {
        (record["task_key"], record["difficulty"], record["seed"], record["policy"]): record
        for record in records
    }
    output = []
    for task in sorted({record["task_key"] for record in records}):
        for difficulty in sorted({record["difficulty"] for record in records}):
            differences = []
            for seed in sorted({record["seed"] for record in records}):
                topology = indexed.get((task, difficulty, seed, "learned_topology"))
                geometry = indexed.get((task, difficulty, seed, "learned_geometry"))
                if topology is not None and geometry is not None:
                    differences.append(float(topology["success"]) - float(geometry["success"]))
            if differences:
                values = np.asarray(differences, dtype=float)
                margin = (
                    1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0.0
                )
                output.append(
                    {
                        "task_key": task,
                        "difficulty": difficulty,
                        "paired_episodes": len(values),
                        "success_advantage": float(values.mean()),
                        "ci95_low": max(-1.0, float(values.mean()) - margin),
                        "ci95_high": min(1.0, float(values.mean()) + margin),
                    }
                )
    return output


def evaluate_learned_policies(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    tasks: Iterable[str] = TASK_ORDER,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    episodes: int = 5,
    base_seed: int = 31_000_000,
    device_name: str = "cpu",
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    task_list = list(tasks)
    difficulty_list = [float(value) for value in difficulties]
    records: list[dict] = []
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
        }
        for difficulty_index, difficulty in enumerate(difficulty_list):
            for episode in range(episodes):
                seed = base_seed + task_index * 100_000 + difficulty_index * 10_000 + episode
                for policy in policies.values():
                    result = run_episode(
                        task,
                        policy,
                        seed=seed,
                        difficulty=difficulty,
                    )
                    records.append({"task_key": task, **result.to_record()})
                teacher = TopologyPolicy(seed)
                result = run_episode(
                    task,
                    teacher,
                    seed=seed,
                    difficulty=difficulty,
                )
                teacher_record = {"task_key": task, **result.to_record()}
                teacher_record["policy"] = "teacher_topology"
                records.append(teacher_record)
    return {
        "name": "HarnessSim4 learned-policy closed-loop evaluation",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tasks": task_list,
            "difficulties": difficulty_list,
            "episodes_per_cell": episodes,
            "base_seed": base_seed,
            "device": device_name,
        },
        "checkpoints": {
            "learned_topology": str(topology_checkpoint.resolve()),
            "learned_geometry": str(geometry_checkpoint.resolve()),
        },
        "software": {"torch": torch.__version__},
        "claim_scope": (
            "closed-loop PBD development evaluation on held-out seeds; not MuJoCo or real-world evidence"
        ),
        "records": records,
        "summary": summarize(records),
        "aggregate_by_task_policy": aggregate_success(records, ("task_key", "robot", "policy")),
        "paired_learned_topology_advantage": _paired_advantage(records),
    }


def _run_branch_counterfactual_episode(
    policy,
    *,
    seed: int,
    difficulty: float,
    semantic_swap: bool,
) -> EpisodeResult:
    env = make_env("branch")
    env.semantic_swap_override = semantic_swap
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


def _counterfactual_paired_advantage(records: list[dict]) -> list[dict]:
    indexed = {
        (
            record["difficulty"],
            record["seed"],
            bool(record["semantic_target_swap"]),
            record["policy"],
        ): record
        for record in records
    }
    output = []
    for difficulty in sorted({record["difficulty"] for record in records}):
        for semantic_swap in (False, True):
            differences = []
            for seed in sorted({record["seed"] for record in records}):
                topology = indexed.get((difficulty, seed, semantic_swap, "learned_topology"))
                geometry = indexed.get((difficulty, seed, semantic_swap, "learned_geometry"))
                if topology is not None and geometry is not None:
                    differences.append(float(topology["success"]) - float(geometry["success"]))
            if differences:
                values = np.asarray(differences, dtype=float)
                margin = (
                    1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0.0
                )
                output.append(
                    {
                        "difficulty": difficulty,
                        "semantic_target_swap": semantic_swap,
                        "paired_physical_seeds": len(values),
                        "success_advantage": float(values.mean()),
                        "ci95_low": max(-1.0, float(values.mean()) - margin),
                        "ci95_high": min(1.0, float(values.mean()) + margin),
                    }
                )
    return output


def evaluate_branch_counterfactual(
    topology_checkpoint: Path,
    geometry_checkpoint: Path,
    *,
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    episodes: int = 10,
    base_seed: int = 41_000_000,
    device_name: str = "cpu",
) -> dict:
    """Pair both semantic assignments on exactly the same physical seeds."""

    if episodes < 1:
        raise ValueError("episodes must be at least one")
    policies = {
        "learned_topology": LearnedGraphPolicy(
            topology_checkpoint,
            "branch",
            name="learned_topology",
            device_name=device_name,
        ),
        "learned_geometry": LearnedGraphPolicy(
            geometry_checkpoint,
            "branch",
            name="learned_geometry",
            device_name=device_name,
        ),
        "teacher_topology": TopologyPolicy(base_seed),
    }
    records: list[dict] = []
    difficulty_list = [float(value) for value in difficulties]
    for difficulty_index, difficulty in enumerate(difficulty_list):
        for episode in range(episodes):
            seed = base_seed + difficulty_index * 10_000 + episode
            for semantic_swap in (False, True):
                for policy_name, policy in policies.items():
                    result = _run_branch_counterfactual_episode(
                        policy,
                        seed=seed,
                        difficulty=difficulty,
                        semantic_swap=semantic_swap,
                    )
                    record = {"task_key": "branch", **result.to_record()}
                    record["policy"] = policy_name
                    records.append(record)
    return {
        "name": "HarnessSim4 paired branch-semantic counterfactual evaluation",
        "schema_version": "0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "difficulties": difficulty_list,
            "physical_seeds_per_difficulty": episodes,
            "semantic_assignments_per_seed": [False, True],
            "base_seed": base_seed,
            "device": device_name,
        },
        "checkpoints": {
            "learned_topology": str(topology_checkpoint.resolve()),
            "learned_geometry": str(geometry_checkpoint.resolve()),
        },
        "software": {"torch": torch.__version__},
        "claim_scope": (
            "paired PBD semantic intervention with identical physical seeds; "
            "not MuJoCo or real-world evidence"
        ),
        "records": records,
        "summary": summarize(records),
        "aggregate_by_task_policy": aggregate_success(
            records,
            ("task_key", "robot", "policy", "semantic_target_swap"),
        ),
        "paired_learned_topology_advantage": _counterfactual_paired_advantage(records),
    }


def save_learning_evaluation(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "learned_policy_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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

    episodes_csv = write_csv("learned_episodes.csv", report["records"])
    summary_csv = write_csv("learned_summary.csv", report["summary"])
    aggregate_csv = write_csv("learned_aggregate.csv", report["aggregate_by_task_policy"])
    paired_csv = write_csv(
        "learned_paired_advantage.csv", report["paired_learned_topology_advantage"]
    )
    lines = [
        "# TopoHarness learned-policy results",
        "",
        "> Development evidence only: held-out PBD seeds, not paper or real-world results.",
        "",
        "| Robot | Policy | Semantic swap | Episodes | Success (95% Wilson CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["aggregate_by_task_policy"]:
        lines.append(
            f"| {row['robot']} | {row['policy']} | "
            f"{row.get('semantic_target_swap', 'all')} | {row['episodes']} | "
            f"{row['success_rate']:.3f} [{row['success_ci95_low']:.3f}, "
            f"{row['success_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "The teacher is a scripted upper bound. A learned policy must be judged by closed-loop ",
            "success, not offline action error alone.",
            "",
        ]
    )
    results_path = output_dir / "LEARNED_RESULTS.md"
    results_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": episodes_csv,
        "summary_csv": summary_csv,
        "aggregate_csv": aggregate_csv,
        "paired_csv": paired_csv,
        "results_markdown": str(results_path.resolve()),
        "episode_count": len(report["records"]),
    }
