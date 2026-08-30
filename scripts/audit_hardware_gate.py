#!/usr/bin/env python3
"""Audit real-robot evidence without treating simulation or a visual twin as hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed


def _safe_evidence_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"evidence path escapes the study directory: {relative}")
    return candidate


def _binary(value: str) -> int:
    parsed = int(value)
    if parsed not in (0, 1):
        raise ValueError(value)
    return parsed


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total**2))
    margin /= denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _mcnemar_exact(primary_only: int, comparator_only: int) -> float:
    discordant = primary_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = min(primary_only, comparator_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trials(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _summarize_metrics(
    rows: list[dict[str, str]], policies: list[str], metrics: list[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for policy in policies:
        selected = [row for row in rows if row.get("policy") == policy]
        successes = sum(_binary(row["success"]) for row in selected)
        metric_summary: dict[str, Any] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            metric_summary[metric] = {
                "count": len(values),
                "mean": fmean(values),
                "p95": _percentile(values, 0.95),
                "minimum": min(values),
                "maximum": max(values),
            }
        report[policy] = {
            "trials": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected) if selected else math.nan,
            "success_wilson_95": _wilson(successes, len(selected)),
            "metrics": metric_summary,
        }
    return report


def _paired_statistics(
    rows: list[dict[str, str]], primary: str, comparators: list[str]
) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["policy"]] = row
    comparisons: list[dict[str, Any]] = []
    for comparator in comparators:
        complete = [
            pair for pair in by_pair.values() if primary in pair and comparator in pair
        ]
        primary_only = sum(
            _binary(pair[primary]["success"]) == 1
            and _binary(pair[comparator]["success"]) == 0
            for pair in complete
        )
        comparator_only = sum(
            _binary(pair[primary]["success"]) == 0
            and _binary(pair[comparator]["success"]) == 1
            for pair in complete
        )
        primary_successes = sum(_binary(pair[primary]["success"]) for pair in complete)
        comparator_successes = sum(_binary(pair[comparator]["success"]) for pair in complete)
        total = len(complete)
        comparisons.append(
            {
                "primary": primary,
                "comparator": comparator,
                "complete_pairs": total,
                "primary_success_rate": primary_successes / total if total else math.nan,
                "comparator_success_rate": comparator_successes / total if total else math.nan,
                "paired_rate_difference": (
                    (primary_successes - comparator_successes) / total if total else math.nan
                ),
                "primary_only_successes": primary_only,
                "comparator_only_successes": comparator_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(primary_only, comparator_only),
            }
        )
    adjusted = _holm([item["mcnemar_exact_two_sided_p"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["holm_adjusted_p"] = value
    return comparisons


def audit_hardware_gate(
    config_path: Path, evidence_root: Path, output_dir: Path
) -> dict[str, Any]:
    config = _load_json(config_path)
    protocol_hash = _sha256(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_files = {
        "trials": evidence_root / "trials.csv",
        "provenance": evidence_root / "provenance.json",
        "calibration": evidence_root / "calibration.json",
        "media": evidence_root / "media_manifest.json",
        "protocol_snapshot": evidence_root / "protocol_snapshot.json",
    }
    for name, path in expected_files.items():
        record(f"{name}_file_present", path.is_file(), str(path))

    rows: list[dict[str, str]] = []
    columns: list[str] = []
    if expected_files["trials"].is_file():
        rows, columns = _read_trials(expected_files["trials"])
    required_columns = config["required_columns"]
    record(
        "required_trial_columns",
        set(required_columns).issubset(columns),
        sorted(set(required_columns) - set(columns)),
    )
    record("nonempty_trials", bool(rows), len(rows))

    value_required_columns = [
        column for column in required_columns if column not in config.get("optional_value_columns", [])
    ]
    missing_cells = [
        [index + 2, column]
        for index, row in enumerate(rows)
        for column in value_required_columns
        if row.get(column, "").strip() == ""
    ]
    record("no_missing_required_values", not missing_cells, missing_cells[:20])

    trial_ids = [row.get("trial_id", "") for row in rows]
    record("unique_trial_ids", len(trial_ids) == len(set(trial_ids)), len(trial_ids))
    pair_policy = [(row.get("pair_id", ""), row.get("policy", "")) for row in rows]
    record("unique_pair_policy_rows", len(pair_policy) == len(set(pair_policy)), len(pair_policy))

    expected_policies = [config["primary_policy"], *config["comparators"]]
    policy_counts = Counter(row.get("policy", "") for row in rows)
    unexpected_policies = sorted(set(policy_counts) - set(expected_policies))
    record("policy_vocabulary", not unexpected_policies, unexpected_policies)
    minimum_trials = int(config["minimum_trials_per_policy"])
    record(
        "minimum_trials_per_policy",
        all(policy_counts[policy] >= minimum_trials for policy in expected_policies),
        {policy: policy_counts[policy] for policy in expected_policies},
    )

    observed_conditions = {row.get("condition", "") for row in rows}
    missing_conditions = sorted(set(config["required_conditions"]) - observed_conditions)
    record("required_conditions", not missing_conditions, missing_conditions)

    binary_errors: list[list[Any]] = []
    numeric_errors: list[list[Any]] = []
    for row_index, row in enumerate(rows, start=2):
        for column in config["binary_columns"]:
            try:
                _binary(row[column])
            except (KeyError, TypeError, ValueError):
                binary_errors.append([row_index, column, row.get(column)])
        for column in config["numeric_columns"]:
            try:
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(value)
                if column in config["nonnegative_columns"] and value < 0.0:
                    raise ValueError(value)
            except (KeyError, TypeError, ValueError):
                numeric_errors.append([row_index, column, row.get(column)])
    record("binary_values_valid", not binary_errors, binary_errors[:20])
    record("numeric_values_valid", not numeric_errors, numeric_errors[:20])

    timestamp_errors: list[list[Any]] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            _parse_utc(row["timestamp_utc"])
        except (KeyError, TypeError, ValueError):
            timestamp_errors.append([row_index, row.get("timestamp_utc")])
    record("trial_timestamps_valid", not timestamp_errors, timestamp_errors[:20])

    distinct_report = {
        column: len({row.get(column, "") for row in rows})
        for column in config["minimum_distinct"]
    }
    record(
        "minimum_distinct_coverage",
        all(
            distinct_report[column] >= int(minimum)
            for column, minimum in config["minimum_distinct"].items()
        ),
        distinct_report,
    )

    consistency_errors: list[list[Any]] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            if _binary(row["success"]) != 1:
                continue
            for column in config["success_rules"]["must_be_one"]:
                if _binary(row[column]) != 1:
                    consistency_errors.append([row_index, column, row[column]])
            for column in config["success_rules"]["must_be_zero"]:
                if _binary(row[column]) != 0:
                    consistency_errors.append([row_index, column, row[column]])
        except (KeyError, TypeError, ValueError):
            consistency_errors.append([row_index, "success_rule", "unparseable"])
    record("success_invariants", not consistency_errors, consistency_errors[:20])

    complete_pair_counts: dict[str, int] = {}
    by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_pair[row.get("pair_id", "")].add(row.get("policy", ""))
    for comparator in config["comparators"]:
        complete_pair_counts[comparator] = sum(
            config["primary_policy"] in policies and comparator in policies
            for policies in by_pair.values()
        )
    minimum_pairs = int(config["minimum_complete_pairs_per_comparison"])
    record(
        "minimum_complete_pairs",
        all(count >= minimum_pairs for count in complete_pair_counts.values()),
        complete_pair_counts,
    )

    provenance: dict[str, Any] = {}
    if expected_files["provenance"].is_file():
        provenance = _load_json(expected_files["provenance"])
    provenance_fields = config["required_provenance_fields"]
    missing_provenance = [field for field in provenance_fields if not provenance.get(field)]
    record("provenance_fields", not missing_provenance, missing_provenance)
    record(
        "protocol_hash_frozen",
        provenance.get("protocol_sha256") == protocol_hash
        and provenance.get("protocol_frozen_before_collection") is True,
        {"expected": protocol_hash, "observed": provenance.get("protocol_sha256")},
    )
    snapshot_hash = (
        _sha256(expected_files["protocol_snapshot"])
        if expected_files["protocol_snapshot"].is_file()
        else None
    )
    record(
        "protocol_snapshot_hash",
        snapshot_hash == protocol_hash == provenance.get("protocol_sha256"),
        {"configuration": protocol_hash, "snapshot": snapshot_hash},
    )
    chronology_valid = False
    try:
        frozen_at = _parse_utc(provenance["protocol_frozen_at_utc"])
        started_at = _parse_utc(provenance["collection_started_at_utc"])
        completed_at = _parse_utc(provenance["collection_completed_at_utc"])
        chronology_valid = frozen_at < started_at <= completed_at
    except (KeyError, TypeError, ValueError):
        pass
    record("study_chronology", chronology_valid, "freeze < start <= completion")
    record(
        "independent_review_and_safety",
        bool(provenance.get("independent_reviewer_id"))
        and provenance.get("safety_review_approved") is True,
        {
            "reviewer": provenance.get("independent_reviewer_id"),
            "safety_review_approved": provenance.get("safety_review_approved"),
        },
    )
    record(
        "study_state_completed",
        provenance.get("state") == "COMPLETED",
        provenance.get("state"),
    )
    trial_window_errors: list[list[Any]] = []
    try:
        started = _parse_utc(provenance["collection_started_at_utc"])
        completed = _parse_utc(provenance["collection_completed_at_utc"])
        for row_index, row in enumerate(rows, start=2):
            timestamp = _parse_utc(row["timestamp_utc"])
            if not started <= timestamp <= completed:
                trial_window_errors.append([row_index, row["timestamp_utc"]])
    except (KeyError, TypeError, ValueError):
        trial_window_errors.append(["study", "unparseable collection window"])
    record("trials_within_collection_window", not trial_window_errors, trial_window_errors[:20])

    calibration: dict[str, Any] = {}
    if expected_files["calibration"].is_file():
        calibration = _load_json(expected_files["calibration"])
    calibration_records = calibration.get("records", [])
    required_kinds = set(config["required_calibration_kinds"])
    passing_kinds = {
        record_item.get("kind") for record_item in calibration_records if record_item.get("passed")
    }
    record("required_calibrations", required_kinds.issubset(passing_kinds), sorted(passing_kinds))
    calibration_file_errors: list[str] = []
    calibration_window_errors: list[str] = []
    for item in calibration_records:
        try:
            path = _safe_evidence_path(evidence_root, item["artifact_path"])
            if not path.is_file() or _sha256(path) != item["sha256"]:
                calibration_file_errors.append(item.get("artifact_path", ""))
        except (KeyError, OSError, ValueError):
            calibration_file_errors.append(item.get("artifact_path", ""))
        try:
            performed = _parse_utc(item["performed_at_utc"])
            valid_through = _parse_utc(item["valid_through_utc"])
            started = _parse_utc(provenance["collection_started_at_utc"])
            completed = _parse_utc(provenance["collection_completed_at_utc"])
            if performed > started or valid_through < completed:
                calibration_window_errors.append(item.get("kind", ""))
        except (KeyError, TypeError, ValueError):
            calibration_window_errors.append(item.get("kind", ""))
    record("calibration_artifact_hashes", not calibration_file_errors, calibration_file_errors)
    record("calibration_validity_window", not calibration_window_errors, calibration_window_errors)

    media: dict[str, Any] = {}
    if expected_files["media"].is_file():
        media = _load_json(expected_files["media"])
    media_items = media.get("items", [])
    media_kinds = {item.get("kind") for item in media_items}
    media_errors: list[str] = []
    known_trials = set(trial_ids)
    for item in media_items:
        trial_id = item.get("trial_id")
        if trial_id and trial_id not in known_trials:
            media_errors.append(f"unknown trial: {trial_id}")
        try:
            path = _safe_evidence_path(evidence_root, item["path"])
            if not path.is_file() or _sha256(path) != item["sha256"]:
                media_errors.append(item.get("path", ""))
        except (KeyError, OSError, ValueError):
            media_errors.append(item.get("path", ""))
    record(
        "minimum_media_evidence",
        len(media_items) >= int(config["minimum_media_items"]),
        len(media_items),
    )
    missing_media_kinds = sorted(set(config["required_media_kinds"]) - media_kinds)
    record("required_media_kinds", not missing_media_kinds, missing_media_kinds)
    record("media_hashes_and_trial_links", not media_errors, media_errors[:20])

    can_compute = (
        bool(rows)
        and set(required_columns).issubset(columns)
        and not binary_errors
        and not numeric_errors
        and all(policy_counts[policy] > 0 for policy in expected_policies)
    )
    summaries: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    if can_compute:
        summaries = _summarize_metrics(rows, expected_policies, config["reported_metrics"])
        comparisons = _paired_statistics(
            rows, config["primary_policy"], config["comparators"]
        )

    report = {
        "schema_version": 1,
        "robot": config["robot"],
        "evidence_kind": "real_robot_hardware",
        "claim_boundary": config["claim_boundary"],
        "protocol": {
            "path": str(config_path),
            "version": config["protocol_version"],
            "sha256": protocol_hash,
        },
        "evidence_root": str(evidence_root.resolve()),
        "trial_file_sha256": _sha256(expected_files["trials"])
        if expected_files["trials"].is_file()
        else None,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "policy_summaries": summaries,
        "paired_comparisons": comparisons,
        "interpretation": (
            "PASS certifies traceable, calibrated, paired real-robot evidence. It does not "
            "certify a positive effect, production reliability, commercial ROI, or safety approval."
        ),
    }
    output_path = output_dir / "hardware_gate_audit.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("inspectbot", "insertbot", "routebot", "branchbot"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or Path(f"configs/hardware/{args.robot}_gate.json")
    output_dir = args.output or Path(f"artifacts/hardware/{args.robot}")
    report = audit_hardware_gate(config_path, args.evidence_root, output_dir)
    summary = {
        "robot": report["robot"],
        "passed": report["passed"],
        "checks_passed": sum(item["passed"] for item in report["checks"]),
        "checks_total": len(report["checks"]),
        "output": str((output_dir / "hardware_gate_audit.json").resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
