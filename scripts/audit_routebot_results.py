#!/usr/bin/env python3
"""Independently audit frozen RouteBot reports and their paired design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DLO_LAB_COMMIT = "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796"
MUSHROOM_RL_COMMIT = "ec3364740627da945b8bab6e01d8151edb0f83f1"
DLO_TARGET_SHA256 = "4e852122d01b4d0e0e1aa59cd286f749031f22a962ef375a8ab1c3a4a9953072"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, float(Fraction(2 * probability, 1 << discordant)))


def _mcnemar_exact_scientific(first_only: int, second_only: int) -> str:
    discordant = first_only + second_only
    if discordant == 0:
        return "1.000000E+0"
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    fraction = min(Fraction(1, 1), Fraction(2 * probability, 1 << discordant))
    with localcontext() as context:
        context.prec = 50
        value = Decimal(fraction.numerator) / Decimal(fraction.denominator)
    return f"{value:.6E}"


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _json_numbers_are_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []
        self.summaries: dict[str, object] = {}

    def check(self, name: str, passed: bool, observed: object) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "observed": observed})

    @property
    def passed(self) -> bool:
        return all(bool(row["passed"]) for row in self.checks)


def _audit_checkpoint_hashes(audit: Audit, name: str, report: dict) -> None:
    expected = report.get("checkpoint_sha256", {})
    observed = {}
    for checkpoint_name, digest in expected.items():
        checkpoint_path = Path(report["checkpoints"][checkpoint_name])
        observed[checkpoint_name] = _sha256(checkpoint_path)
        audit.check(
            f"{name}.checkpoint.{checkpoint_name}",
            observed[checkpoint_name] == digest,
            observed[checkpoint_name],
        )


def audit_main(audit: Audit, directory: Path) -> None:
    report_path = directory / "route_paper_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    policies = tuple(report["config"]["policies"])
    difficulties = tuple(float(value) for value in report["config"]["difficulties"])
    episodes = int(report["config"]["episodes_per_difficulty"])
    expected_records = len(policies) * len(difficulties) * episodes
    audit.check("main.record_count", len(records) == expected_records, len(records))
    audit.check(
        "main.paper_scale",
        report["config"]["evidence_level"] == "paper-scale" and episodes >= 100,
        report["config"]["evidence_level"],
    )

    grouped: dict[tuple[float, int], set[str]] = defaultdict(set)
    by_cell_policy: dict[tuple[float, int, str], bool] = {}
    for row in records:
        key = (float(row["difficulty"]), int(row["seed"]))
        grouped[key].add(str(row["policy"]))
        by_cell_policy[(*key, str(row["policy"]))] = bool(row["success"])
    complete_pairing = len(grouped) == len(difficulties) * episodes and all(
        names == set(policies) for names in grouped.values()
    )
    audit.check("main.complete_policy_pairing", complete_pairing, len(grouped))

    aggregate_observed = {
        policy: sum(bool(row["success"]) for row in records if row["policy"] == policy)
        for policy in policies
    }
    aggregate_reported = {
        str(row["policy"]): round(float(row["success_rate"]) * int(row["episodes"]))
        for row in report["aggregate_by_policy"]
    }
    audit.check(
        "main.aggregate_success_recalculation",
        aggregate_observed == aggregate_reported,
        aggregate_observed,
    )

    pairs = [
        (
            by_cell_policy[(*key, "learned_topology")],
            by_cell_policy[(*key, "learned_geometry")],
        )
        for key in grouped
    ]
    topology_only = sum(first and not second for first, second in pairs)
    geometry_only = sum(second and not first for first, second in pairs)
    paired_summary = {
        "paired_cells": len(pairs),
        "topology_only": topology_only,
        "geometry_only": geometry_only,
        "both_success": sum(first and second for first, second in pairs),
        "both_failure": sum(not first and not second for first, second in pairs),
        "mcnemar_exact_p_unadjusted": _mcnemar_exact(topology_only, geometry_only),
    }
    audit.summaries["main_topology_vs_geometry"] = paired_summary
    audit.check(
        "main.episode_csv_rows",
        _csv_data_rows(directory / "route_paper_episodes.csv") == len(records),
        _csv_data_rows(directory / "route_paper_episodes.csv"),
    )
    _audit_checkpoint_hashes(audit, "main", report)
    audit.summaries["main_report_sha256"] = _sha256(report_path)


def audit_counterfactual(audit: Audit, directory: Path) -> None:
    report_path = directory / "route_counterfactual_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    policies = tuple(report["config"]["policies"])
    difficulties = tuple(float(value) for value in report["config"]["difficulties"])
    assignments = tuple(report["config"]["target_assignments"])
    physical_seeds = int(report["config"]["physical_seeds_per_difficulty"])
    expected_records = len(policies) * len(difficulties) * len(assignments) * physical_seeds
    audit.check("counterfactual.record_count", len(records) == expected_records, len(records))
    audit.check(
        "counterfactual.paper_scale",
        report["config"]["evidence_level"] == "paper-scale" and physical_seeds >= 100,
        report["config"]["evidence_level"],
    )

    grouped: dict[tuple[float, int, str], set[str]] = defaultdict(set)
    by_cell_policy: dict[tuple[float, int, str, str], bool] = {}
    for row in records:
        cell = (float(row["difficulty"]), int(row["seed"]), row["target_assignment"])
        policy = str(row["policy"])
        grouped[cell].add(policy)
        by_cell_policy[(*cell, policy)] = bool(row["success"])
    expected_cells = len(difficulties) * len(assignments) * physical_seeds
    complete_pairing = len(grouped) == expected_cells and all(
        names == set(policies) for names in grouped.values()
    )
    audit.check("counterfactual.complete_policy_pairing", complete_pairing, len(grouped))
    audit.check(
        "counterfactual.assignment_set",
        {key[2] for key in grouped} == set(assignments),
        sorted({key[2] for key in grouped}),
    )
    audit.check(
        "counterfactual.episode_csv_rows",
        _csv_data_rows(directory / "route_counterfactual_episodes.csv") == len(records),
        _csv_data_rows(directory / "route_counterfactual_episodes.csv"),
    )

    aggregate_observed = {
        policy: sum(bool(row["success"]) for row in records if row["policy"] == policy)
        for policy in policies
    }
    aggregate_reported = {
        str(row["policy"]): round(float(row["success_rate"]) * int(row["episodes"]))
        for row in report["aggregate_by_policy"]
    }
    audit.check(
        "counterfactual.aggregate_success_recalculation",
        aggregate_observed == aggregate_reported,
        aggregate_observed,
    )
    audit.check(
        "counterfactual.relation_variant_partition_equivalence",
        all(
            by_cell_policy[(*cell, "learned_topology")]
            == by_cell_policy[(*cell, "act_relational_typed")]
            for cell in grouped
        ),
        len(grouped),
    )

    paired_recalculated = []
    paired_reported = report["paired_comparisons"]
    for row in paired_reported:
        difficulty = float(row["difficulty"])
        assignment = str(row["target_assignment"])
        baseline = str(row["baseline"])
        cells = [
            cell
            for cell in grouped
            if cell[0] == difficulty and cell[2] == assignment
        ]
        pairs = [
            (
                by_cell_policy[(*cell, "learned_topology")],
                by_cell_policy[(*cell, baseline)],
            )
            for cell in cells
        ]
        first_only = sum(first and not second for first, second in pairs)
        second_only = sum(second and not first for first, second in pairs)
        both_success = sum(first and second for first, second in pairs)
        both_failure = sum(not first and not second for first, second in pairs)
        paired_recalculated.append(
            {
                "paired_episodes": len(pairs),
                "topology_success_baseline_failure": first_only,
                "topology_failure_baseline_success": second_only,
                "both_success": both_success,
                "both_failure": both_failure,
                "mcnemar_exact_p": _mcnemar_exact(first_only, second_only),
            }
        )
    paired_counts_and_raw_p_match = all(
        all(
            math.isclose(float(recalculated[key]), float(reported[key]), rel_tol=1e-12)
            for key in recalculated
        )
        for recalculated, reported in zip(paired_recalculated, paired_reported, strict=True)
    )
    audit.check(
        "counterfactual.paired_counts_and_raw_p",
        paired_counts_and_raw_p_match,
        len(paired_recalculated),
    )
    adjusted = _holm_adjust(
        [float(row["mcnemar_exact_p"]) for row in paired_recalculated]
    )
    reported_adjusted = [float(row["mcnemar_exact_p_holm"]) for row in paired_reported]
    audit.check(
        "counterfactual.holm_recalculation",
        all(
            math.isclose(observed, reported, rel_tol=1e-12, abs_tol=1e-300)
            for observed, reported in zip(adjusted, reported_adjusted, strict=True)
        ),
        len(adjusted),
    )

    aggregate_pairs = {}
    for baseline in policies:
        if baseline == "learned_topology":
            continue
        pairs = [
            (
                by_cell_policy[(*cell, "learned_topology")],
                by_cell_policy[(*cell, baseline)],
            )
            for cell in grouped
        ]
        first_only = sum(first and not second for first, second in pairs)
        second_only = sum(second and not first for first, second in pairs)
        aggregate_pairs[baseline] = {
            "paired_cells": len(pairs),
            "topology_only": first_only,
            "baseline_only": second_only,
            "both_success": sum(first and second for first, second in pairs),
            "both_failure": sum(not first and not second for first, second in pairs),
            "paired_success_difference": (first_only - second_only) / len(pairs),
            "mcnemar_exact_p_unadjusted": _mcnemar_exact(first_only, second_only),
            "mcnemar_exact_p_scientific": _mcnemar_exact_scientific(
                first_only, second_only
            ),
        }
    audit.summaries["counterfactual_aggregate_pairs"] = aggregate_pairs
    _audit_checkpoint_hashes(audit, "counterfactual", report)
    audit.summaries["counterfactual_report_sha256"] = _sha256(report_path)


def audit_public(audit: Audit, directory: Path) -> None:
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split = report["split"]
    episode_sets = {
        name: {int(value) for value in split[f"{name}_episodes"]}
        for name in ("train", "validation", "test")
    }
    disjoint = all(
        not episode_sets[left] & episode_sets[right]
        for left in episode_sets
        for right in episode_sets
        if left < right
    )
    audit.check("public.episode_disjoint", disjoint, split["episode_overlap"])
    audit.check(
        "public.episode_counts",
        [len(episode_sets[name]) for name in ("train", "validation", "test")]
        == [1153, 247, 247],
        {name: len(values) for name, values in episode_sets.items()},
    )
    comparison = report["paired_comparisons"]["temporal_ridge_vs_previous_action"]
    audit.check(
        "public.paired_unit",
        comparison["paired_unit"] == "whole held-out episode",
        comparison["paired_unit"],
    )
    audit.summaries["public_report_sha256"] = _sha256(report_path)


def audit_mujoco(audit: Audit, directory: Path) -> None:
    report_path = directory / "route_mujoco_direct_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    config = report["config"]
    expected = (
        int(config["physical_seeds"])
        * len(config["assignment_indices"])
        * len(config["policies"])
    )
    audit.check("mujoco.record_count", len(records) == expected, len(records))
    grouped: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in records:
        grouped[(int(row["seed"]), int(row["assignment_index"]))].add(str(row["policy"]))
    audit.check(
        "mujoco.complete_policy_pairing",
        all(names == set(config["policies"]) for names in grouped.values()),
        len(grouped),
    )
    audit.check(
        "mujoco.all_finite",
        all(bool(row["finite_state"]) for row in records),
        sum(bool(row["finite_state"]) for row in records),
    )
    audit.summaries["mujoco_report_sha256"] = _sha256(report_path)


def audit_dlolab(audit: Audit, directory: Path) -> None:
    smoke_path = directory / "wiring_post_smoke_report.json"
    matched_path = directory / "matched_paths" / "matched_path_report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    target_path = WORKSPACE / smoke["dlo_lab"]["target_path"]
    target_hash = _sha256(target_path)

    provenance = {
        "smoke_dlo_lab": smoke["dlo_lab"]["commit"],
        "smoke_mushroom_rl": smoke["mushroom_rl"]["commit"],
        "matched_dlo_lab": matched["dlo_lab_commit"],
        "matched_mushroom_rl": matched["mushroom_rl_commit"],
        "target_sha256": target_hash,
    }
    audit.check(
        "dlolab.provenance",
        provenance
        == {
            "smoke_dlo_lab": DLO_LAB_COMMIT,
            "smoke_mushroom_rl": MUSHROOM_RL_COMMIT,
            "matched_dlo_lab": DLO_LAB_COMMIT,
            "matched_mushroom_rl": MUSHROOM_RL_COMMIT,
            "target_sha256": DLO_TARGET_SHA256,
        },
        provenance,
    )
    audit.check(
        "dlolab.target_hash_in_reports",
        smoke["dlo_lab"]["target_sha256"] == matched["target_sha256"] == target_hash,
        {
            "smoke": smoke["dlo_lab"]["target_sha256"],
            "matched": matched["target_sha256"],
            "observed": target_hash,
        },
    )
    smoke_finite = (
        bool(smoke["initial"]["observation"]["all_finite"])
        and bool(smoke["initial"]["vertices"]["all_finite"])
        and bool(smoke["after_zero_action"]["observation"]["all_finite"])
        and bool(smoke["after_zero_action"]["vertices"]["all_finite"])
        and not bool(smoke["after_zero_action"]["absorbing"])
        and _json_numbers_are_finite(smoke["initial"])
        and _json_numbers_are_finite(smoke["after_zero_action"])
    )
    audit.check("dlolab.smoke_finite_and_active", smoke_finite, smoke_finite)

    smoke_arrays = WORKSPACE / smoke["arrays"]["path"]
    matched_arrays = WORKSPACE / matched["arrays"]["path"]
    array_hashes = {
        "smoke": _sha256(smoke_arrays),
        "matched": _sha256(matched_arrays),
    }
    audit.check(
        "dlolab.array_hashes",
        array_hashes
        == {
            "smoke": smoke["arrays"]["sha256"],
            "matched": matched["arrays"]["sha256"],
        },
        array_hashes,
    )

    expected_policies = {
        "no_action",
        "endpoint_straight",
        "wrong_fixture_order",
        "ordered_fixture_route",
    }
    result_policies = {str(row["policy"]) for row in matched["results"]}
    matched_finite = all(
        bool(row["all_vertices_finite"])
        and _json_numbers_are_finite(row)
        for row in matched["results"]
    )
    audit.check(
        "dlolab.matched_policy_set",
        result_policies == expected_policies and len(matched["results"]) == 4,
        sorted(result_policies),
    )
    audit.check("dlolab.matched_results_finite", matched_finite, matched_finite)
    audit.summaries["dlolab"] = {
        "smoke_report_sha256": _sha256(smoke_path),
        "matched_report_sha256": _sha256(matched_path),
        "policies": sorted(result_policies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "artifacts"
        / "papers"
        / "routebot"
        / "final"
        / "audit"
        / "routebot_result_audit.json",
    )
    parser.add_argument(
        "--allow-pending-counterfactual",
        action="store_true",
        help="Audit completed evidence while the full counterfactual run is still in progress.",
    )
    parser.add_argument(
        "--allow-pending-dlolab",
        action="store_true",
        help="Audit completed evidence while the external DLO-Lab run is still in progress.",
    )
    args = parser.parse_args()

    final = WORKSPACE / "artifacts" / "papers" / "routebot" / "final"
    audit = Audit()
    audit_main(audit, final / "main_table")
    counterfactual = final / "counterfactual"
    if counterfactual.is_dir():
        audit_counterfactual(audit, counterfactual)
    else:
        audit.check(
            "counterfactual.available",
            args.allow_pending_counterfactual,
            "pending" if args.allow_pending_counterfactual else "missing",
        )
    audit_public(audit, WORKSPACE / "artifacts" / "papers" / "routebot" / "public_real")
    audit_mujoco(audit, final / "mujoco_direct")
    dlolab = final / "dlolab_external"
    if (
        (dlolab / "wiring_post_smoke_report.json").is_file()
        and (dlolab / "matched_paths" / "matched_path_report.json").is_file()
    ):
        audit_dlolab(audit, dlolab)
    else:
        audit.check(
            "dlolab.available",
            args.allow_pending_dlolab,
            "pending" if args.allow_pending_dlolab else "missing",
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": audit.passed,
        "checks": audit.checks,
        "summaries": audit.summaries,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
