"""Episode runner, statistical summaries and reproducible benchmark exports."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from harnessbench import __version__
from harnessbench.sim.envs import ENV_REGISTRY, make_env
from harnessbench.sim.envs.base import EpisodeResult
from harnessbench.sim.policies import POLICY_REGISTRY, Policy

PRIMARY_METRIC = {
    "inspect": "f1",
    "insert": "terminal_error",
    "route": "topology_accuracy",
    "branch": "mean_endpoint_error",
}
PRIMARY_METRIC_BOUNDS = {
    "inspect": (0.0, 1.0),
    "insert": (0.0, None),
    "route": (0.0, 1.0),
    "branch": (0.0, None),
}


def run_episode(
    task: str,
    policy: Policy,
    *,
    seed: int,
    difficulty: float,
    record_every: int = 0,
    render_size: tuple[int, int] = (640, 480),
) -> EpisodeResult:
    env = make_env(task)
    env.reset(seed=seed, difficulty=difficulty)
    policy.reset(seed + 100_003)
    frames = []
    if record_every > 0:
        frames.append(env.render(*render_size))
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(env)
        _, _, terminated, truncated, _ = env.step(action)
        if record_every > 0 and (env.step_count % record_every == 0 or terminated or truncated):
            frames.append(env.render(*render_size))
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
        frames=tuple(frames),
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_interval(
    values: list[float], bounds: tuple[float | None, float | None] | None = None
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean
    margin = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    low, high = mean - margin, mean + margin
    if bounds is not None:
        lower_bound, upper_bound = bounds
        low = max(low, lower_bound) if lower_bound is not None else low
        high = min(high, upper_bound) if upper_bound is not None else high
    return mean, low, high


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["task_key"], record["policy"], record["difficulty"])].append(record)

    summaries = []
    for (task_key, policy, difficulty), group in sorted(grouped.items()):
        successes = sum(bool(record["success"]) for record in group)
        low, high = _wilson_interval(successes, len(group))
        reward_mean, reward_low, reward_high = _mean_interval(
            [float(record["total_reward"]) for record in group]
        )
        metric_name = PRIMARY_METRIC[task_key]
        metric_mean, metric_low, metric_high = _mean_interval(
            [float(record[metric_name]) for record in group],
            PRIMARY_METRIC_BOUNDS[task_key],
        )
        summaries.append(
            {
                "task_key": task_key,
                "task": group[0]["task"],
                "robot": group[0]["robot"],
                "policy": policy,
                "difficulty": difficulty,
                "episodes": len(group),
                "success_rate": successes / len(group),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "steps_mean": float(np.mean([record["steps"] for record in group])),
                "reward_mean": reward_mean,
                "reward_ci95_low": reward_low,
                "reward_ci95_high": reward_high,
                "primary_metric": metric_name,
                "primary_metric_mean": metric_mean,
                "primary_metric_ci95_low": metric_low,
                "primary_metric_ci95_high": metric_high,
            }
        )
    return summaries


def aggregate_success(records: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    output = []
    for group_key, group in sorted(grouped.items()):
        successes = sum(bool(record["success"]) for record in group)
        low, high = _wilson_interval(successes, len(group))
        output.append(
            {
                **dict(zip(keys, group_key)),
                "episodes": len(group),
                "success_rate": successes / len(group),
                "success_ci95_low": low,
                "success_ci95_high": high,
            }
        )
    return output


def paired_topology_advantage(records: list[dict]) -> list[dict]:
    indexed = {
        (record["task_key"], record["difficulty"], record["seed"], record["policy"]): record
        for record in records
    }
    output = []
    for task in sorted({record["task_key"] for record in records}):
        for difficulty in sorted({record["difficulty"] for record in records}):
            deltas = []
            for seed in sorted({record["seed"] for record in records}):
                topology = indexed.get((task, difficulty, seed, "topology"))
                geometry = indexed.get((task, difficulty, seed, "geometry"))
                if topology is not None and geometry is not None:
                    deltas.append(float(topology["success"]) - float(geometry["success"]))
            if deltas:
                mean, low, high = _mean_interval(deltas, (-1.0, 1.0))
                output.append(
                    {
                        "task_key": task,
                        "difficulty": difficulty,
                        "paired_episodes": len(deltas),
                        "success_advantage": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return output


def _run_benchmark_job(job: tuple[str, str, int, float]) -> dict:
    """Run one deterministic task/policy/seed cell; safe for process pools."""

    task, policy_name, seed, difficulty = job
    policy = POLICY_REGISTRY[policy_name](seed)
    result = run_episode(task, policy, seed=seed, difficulty=difficulty)
    return {"task_key": task, **result.to_record()}


def run_benchmark(
    *,
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    policies: Iterable[str] = ("topology", "geometry", "random"),
    difficulties: Iterable[float] = (0.2, 0.5, 0.8),
    episodes: int = 10,
    base_seed: int = 202609,
    workers: int = 1,
) -> dict:
    task_list = list(tasks)
    policy_list = list(policies)
    difficulty_list = [float(value) for value in difficulties]
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    unknown_tasks = set(task_list) - set(ENV_REGISTRY)
    unknown_policies = set(policy_list) - set(POLICY_REGISTRY)
    if unknown_tasks or unknown_policies:
        raise KeyError(
            f"unknown tasks={sorted(unknown_tasks)}, policies={sorted(unknown_policies)}"
        )

    jobs: list[tuple[str, str, int, float]] = []
    for task_index, task in enumerate(task_list):
        for difficulty_index, difficulty in enumerate(difficulty_list):
            for episode in range(episodes):
                seed = base_seed + task_index * 100_000 + difficulty_index * 10_000 + episode
                for policy_name in policy_list:
                    jobs.append((task, policy_name, seed, difficulty))

    started = perf_counter()
    if workers == 1:
        records = [_run_benchmark_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # map preserves input ordering, so CSV/JSON ordering stays stable.
            records = list(executor.map(_run_benchmark_job, jobs, chunksize=4))
    runtime_seconds = perf_counter() - started

    return {
        "benchmark": "HarnessSim4",
        "schema_version": "0.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "harnessbench_version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "backend": "2-D position-based dynamics fast backend",
            "runtime_seconds": runtime_seconds,
            "episode_throughput_per_second": len(records) / max(runtime_seconds, 1e-9),
        },
        "config": {
            "tasks": task_list,
            "policies": policy_list,
            "difficulties": difficulty_list,
            "episodes_per_cell": episodes,
            "base_seed": base_seed,
            "workers": workers,
            "evidence_level": "paper" if episodes >= 100 else "development",
        },
        "limitations": [
            "Fast backend is 2-D and is not calibrated to a physical cable.",
            "Heuristic baselines are diagnostic controls, not learned policies.",
            "Simulation success must not be reported as real-robot success.",
        ],
        "records": records,
        "summary": summarize(records),
        "aggregate_by_task_policy": aggregate_success(records, ("task_key", "robot", "policy")),
        "aggregate_by_policy": aggregate_success(records, ("policy",)),
        "paired_topology_advantage": paired_topology_advantage(records),
    }


def save_benchmark(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Backfill aggregate views when re-exporting an older report schema.
    report["aggregate_by_task_policy"] = aggregate_success(
        report["records"], ("task_key", "robot", "policy")
    )
    report["aggregate_by_policy"] = aggregate_success(report["records"], ("policy",))
    report_path = output_dir / "benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "episodes.csv", report["records"])
    write_csv(output_dir / "summary.csv", report["summary"])
    write_csv(output_dir / "paired_topology_advantage.csv", report["paired_topology_advantage"])
    results_path = output_dir / "RESULTS.md"
    development_warning = (
        "This is a development run; confidence intervals are too wide for paper claims."
        if report["config"]["evidence_level"] == "development"
        else "This run meets the configured episode-count gate for the paper table."
    )
    lines = [
        "# HarnessSim4 benchmark results",
        "",
        f"> {development_warning}",
        "",
        f"Generated: {report['created_at_utc']}",
        "",
        "## Aggregate success",
        "",
        "| Robot | Policy | Episodes | Success (95% Wilson CI) |",
        "|---|---|---:|---:|",
    ]
    for row in report["aggregate_by_task_policy"]:
        lines.append(
            f"| {row['robot']} | {row['policy']} | {row['episodes']} | "
            f"{row['success_rate']:.3f} [{row['success_ci95_low']:.3f}, "
            f"{row['success_ci95_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Per-difficulty results",
            "",
            "| Robot | Policy | Difficulty | Success (95% Wilson CI) | Primary metric |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["summary"]:
        lines.append(
            f"| {row['robot']} | {row['policy']} | {row['difficulty']:.1f} | "
            f"{row['success_rate']:.3f} [{row['success_ci95_low']:.3f}, "
            f"{row['success_ci95_high']:.3f}] | {row['primary_metric']}="
            f"{row['primary_metric_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The fast backend is 2-D and uncalibrated. These values compare simulation policies; ",
            "they are not real manufacturing success rates.",
            "",
        ]
    )
    results_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": str(report_path.resolve()),
        "episodes_csv": str((output_dir / "episodes.csv").resolve()),
        "summary_csv": str((output_dir / "summary.csv").resolve()),
        "results_markdown": str(results_path.resolve()),
        "episode_count": len(report["records"]),
        "summary_cells": len(report["summary"]),
    }
