#!/usr/bin/env python3
"""Independent consistency audit for the frozen BranchBot research package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2**discordant))


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, condition: bool, name: str, detail: Any = None) -> None:
        row = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        if not condition:
            raise AssertionError(f"{name}: {detail}")


def _audit_dataset(audit: Audit, metadata_path: Path, dataset_path: Path) -> tuple[dict, set[int]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit.require(dataset_path.is_file(), "successful dataset exists", str(dataset_path))
    digest = _sha256(dataset_path)
    audit.require(digest == metadata["sha256"], "successful dataset hash", digest)
    audit.require(
        metadata["selected_episode_count"] == 280,
        "selected episode count",
        metadata["selected_episode_count"],
    )
    audit.require(
        metadata["split_samples"] == {"train": 14712, "validation": 3676, "test": 7352},
        "frozen split sample counts",
        metadata["split_samples"],
    )
    audit.require(
        all(float(value) == 1.0 for value in metadata["episode_success_rate"].values()),
        "successful-only episode manifest",
        metadata["episode_success_rate"],
    )
    split_seed_sets = {name: set(values) for name, values in metadata["split_seeds"].items()}
    split_physical_sets = {
        name: set(values) for name, values in metadata["split_physical_seeds"].items()
    }
    names = tuple(split_seed_sets)
    audit.require(
        all(
            split_seed_sets[names[left]].isdisjoint(split_seed_sets[names[right]])
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ),
        "episode seeds disjoint across splits",
    )
    audit.require(
        all(
            split_physical_sets[names[left]].isdisjoint(split_physical_sets[names[right]])
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ),
        "physical seeds disjoint across splits",
    )
    for split, strata in metadata["selection_counts"]["branch"].items():
        values = list(strata.values())
        audit.require(
            max(values) - min(values) <= 1,
            f"balanced difficulty/semantic strata: {split}",
            strata,
        )
    corpus_physical = set().union(*split_physical_sets.values())
    return metadata, corpus_physical


def _audit_training_reports(
    audit: Audit,
    report: dict,
    dataset_sha256: str,
) -> None:
    for name, raw_checkpoint in report["checkpoints"].items():
        checkpoint = Path(raw_checkpoint)
        audit.require(checkpoint.is_file(), f"checkpoint exists: {name}", str(checkpoint))
        audit.require(
            _sha256(checkpoint) == report["checkpoint_sha256"][name],
            f"checkpoint hash: {name}",
        )
        training_report = checkpoint.with_suffix(".report.json")
        if name in {"learned_topology", "learned_geometry", "act_chunk", "act_relational"} or name.startswith(
            "learned_"
        ):
            audit.require(
                training_report.is_file(),
                f"training report exists: {name}",
                str(training_report),
            )
            metadata = json.loads(training_report.read_text(encoding="utf-8"))
            audit.require(
                metadata["dataset_sha256"] == dataset_sha256,
                f"training dataset hash matches: {name}",
                metadata["dataset_sha256"],
            )
            tasks = metadata.get("tasks")
            if tasks is not None:
                audit.require(tasks == ["branch"], f"training task scope: {name}", tasks)


def _audit_pbd(
    audit: Audit,
    report: dict,
    corpus_physical_seeds: set[int],
) -> None:
    config = report["config"]
    difficulties = tuple(float(value) for value in config["difficulties"])
    policies = tuple(config["policies"])
    physical_seeds = int(config["physical_seeds_per_difficulty"])
    audit.require(physical_seeds >= 100, "paper-scale PBD seeds", physical_seeds)
    expected = len(difficulties) * physical_seeds * 2 * len(policies)
    records = report["records"]
    audit.require(len(records) == expected, "PBD episode count", {"expected": expected, "actual": len(records)})
    keys = [
        (
            float(row["difficulty"]),
            int(row["seed"]),
            bool(row["semantic_target_swap"]),
            str(row["policy"]),
        )
        for row in records
    ]
    audit.require(len(keys) == len(set(keys)), "unique PBD episode keys")
    evaluation_seeds = {int(row["seed"]) for row in records}
    audit.require(
        evaluation_seeds.isdisjoint(corpus_physical_seeds),
        "PBD evaluation seeds disjoint from corpus",
        sorted(evaluation_seeds & corpus_physical_seeds),
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[str(row["policy"])].append(row)
    aggregate = {str(row["policy"]): row for row in report["aggregate_by_policy"]}
    for policy in policies:
        successes = sum(bool(row["success"]) for row in grouped[policy])
        episodes = len(grouped[policy])
        audit.require(
            episodes == len(difficulties) * physical_seeds * 2,
            f"PBD episodes per policy: {policy}",
            episodes,
        )
        audit.require(
            int(aggregate[policy]["episodes"]) == episodes
            and math.isclose(float(aggregate[policy]["success_rate"]), successes / episodes),
            f"PBD aggregate recomputation: {policy}",
            {"successes": successes, "episodes": episodes},
        )

    indexed = {key: row for key, row in zip(keys, records)}
    reported_pairs = {
        (
            float(row["difficulty"]),
            bool(row["semantic_target_swap"]),
            str(row["baseline"]),
        ): row
        for row in report["paired_comparisons"]
    }
    raw_rows: list[tuple[tuple[float, bool, str], float, int, int]] = []
    for difficulty in difficulties:
        seeds = sorted(
            {
                int(row["seed"])
                for row in records
                if float(row["difficulty"]) == difficulty
            }
        )
        audit.require(len(seeds) == physical_seeds, f"physical seeds at difficulty {difficulty}")
        for swap in (False, True):
            for baseline in sorted(set(policies) - {"learned_topology"}):
                pairs = [
                    (
                        bool(indexed[(difficulty, seed, swap, "learned_topology")]["success"]),
                        bool(indexed[(difficulty, seed, swap, baseline)]["success"]),
                    )
                    for seed in seeds
                ]
                left_only = sum(left and not right for left, right in pairs)
                right_only = sum(right and not left for left, right in pairs)
                raw_rows.append(
                    ((difficulty, swap, baseline), _mcnemar_exact(left_only, right_only), left_only, right_only)
                )
    adjusted = _holm([row[1] for row in raw_rows])
    for (key, raw_p, left_only, right_only), adjusted_p in zip(raw_rows, adjusted):
        row = reported_pairs[key]
        audit.require(
            int(row["topology_success_baseline_failure"]) == left_only
            and int(row["topology_failure_baseline_success"]) == right_only,
            f"discordant-pair recomputation: {key}",
        )
        audit.require(
            math.isclose(float(row["mcnemar_exact_p"]), raw_p, rel_tol=1e-12, abs_tol=0.0)
            and math.isclose(
                float(row["mcnemar_exact_p_holm"]), adjusted_p, rel_tol=1e-12, abs_tol=0.0
            ),
            f"exact/Holm recomputation: {key}",
            {"raw": raw_p, "adjusted": adjusted_p},
        )


def _audit_mujoco(audit: Audit, report: dict, pbd_report: dict) -> None:
    config = report["config"]
    physical_seeds = int(config["physical_seeds"])
    policies = tuple(config["policies"])
    audit.require(physical_seeds >= 20, "paper-scale direct MuJoCo seeds", physical_seeds)
    records = report["records"]
    expected = physical_seeds * 2 * len(policies)
    audit.require(
        len(records) == expected,
        "direct MuJoCo episode count",
        {"expected": expected, "actual": len(records)},
    )
    keys = [
        (int(row["seed"]), bool(row["semantic_target_swap"]), str(row["policy"]))
        for row in records
    ]
    audit.require(len(keys) == len(set(keys)), "unique direct MuJoCo episode keys")
    audit.require(all(bool(row["finite_state"]) for row in records), "all MuJoCo states finite")
    audit.require(
        "Direct closed-loop" in report["claim_scope"],
        "direct-MuJoCo claim scope is explicit",
        report["claim_scope"],
    )
    for name in ("learned_topology", "learned_geometry"):
        audit.require(
            report["checkpoint_sha256"][name] == pbd_report["checkpoint_sha256"][name],
            f"same PBD/MuJoCo checkpoint: {name}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/papers/branchbot/final/data/branch_successful_demos.npz"),
    )
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("artifacts/papers/branchbot/final/data/branch_successful_demos.json"),
    )
    parser.add_argument(
        "--pbd-report",
        type=Path,
        default=Path("artifacts/papers/branchbot/final/counterfactual/branch_paper_report.json"),
    )
    parser.add_argument(
        "--mujoco-report",
        type=Path,
        default=Path("artifacts/papers/branchbot/final/mujoco_direct/branch_mujoco_direct_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/papers/branchbot/final/audit/branchbot_result_audit.json"),
    )
    args = parser.parse_args()

    audit = Audit()
    dataset_metadata, corpus_physical = _audit_dataset(
        audit,
        args.dataset_metadata,
        args.dataset,
    )
    pbd_report = json.loads(args.pbd_report.read_text(encoding="utf-8"))
    _audit_training_reports(audit, pbd_report, dataset_metadata["sha256"])
    _audit_pbd(audit, pbd_report, corpus_physical)
    mujoco_report = json.loads(args.mujoco_report.read_text(encoding="utf-8"))
    _audit_mujoco(audit, mujoco_report, pbd_report)
    result = {
        "name": "BranchBot frozen-result independent audit",
        "passed": all(row["passed"] for row in audit.checks),
        "checks": audit.checks,
        "inputs": {
            "dataset_sha256": _sha256(args.dataset),
            "pbd_report_sha256": _sha256(args.pbd_report),
            "mujoco_report_sha256": _sha256(args.mujoco_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(audit.checks), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
