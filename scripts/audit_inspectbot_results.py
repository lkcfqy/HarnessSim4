#!/usr/bin/env python3
"""Independent consistency audit for the frozen InspectBot research package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _finite_rows(rows: Iterable[dict[str, str]], ignored: set[str] | None = None) -> bool:
    ignored = ignored or set()
    for row in rows:
        for key, value in row.items():
            if key in ignored or value == "":
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                return False
    return True


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, condition: bool, name: str, detail: Any = None) -> None:
        row: dict[str, Any] = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        if not condition:
            raise AssertionError(f"{name}: {detail}")


def _audit_public_data(
    audit: Audit,
    mvtec_path: Path,
    wire_path: Path,
    wire_archive: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mvtec = _read_json(mvtec_path)
    wire = _read_json(wire_path)
    audit.require(
        mvtec["counts"]
        == {
            "train_good": 224,
            "test_good": 58,
            "test_anomaly": 92,
            "masks": 92,
            "indexed_images": 374,
            "all_files": 468,
        },
        "MVTec cable frozen counts",
        mvtec["counts"],
    )
    audit.require(
        sum(int(value) for value in mvtec["defect_counts"].values()) == 92,
        "MVTec anomaly subtype total",
        mvtec["defect_counts"],
    )
    audit.require(
        mvtec["transport_mirror_revision"] == "c75b39616f84db43677bcc8228caaafaf5096d7f",
        "MVTec transport revision pinned",
    )
    audit.require(
        mvtec["tree_sha256"] == "5ecf3b6b9feeec54eae602cbeb3e1c1241fd3110c9722166c4255a7a725a37c1",
        "MVTec tree hash frozen",
    )
    audit.require(wire_archive.is_file(), "stripped-wire source archive exists", str(wire_archive))
    archive_hash = _sha256(wire_archive)
    audit.require(
        archive_hash == wire["archive"]["sha256"] == wire["archive"]["expected_sha256"],
        "stripped-wire archive hash",
        archive_hash,
    )
    audit.require(wire_archive.stat().st_size == 18_485_958, "stripped-wire archive bytes")
    audit.require(
        wire["counts"]
        == {
            "test/cut_strands": 68,
            "test/good": 133,
            "test/pulled_strands": 99,
            "train/PatchCore": 200,
            "train/VLM": 3,
        },
        "stripped-wire frozen counts",
        wire["counts"],
    )
    audit.require(wire["total_images"] == 503, "stripped-wire image total")
    audit.require(
        wire["train_test_exact_duplicate_count"] == 0,
        "stripped-wire train/test exact-duplicate isolation",
    )
    audit.require(
        wire["tree_sha256"] == "19d7aafc80aba1d41f851e70d6f794127f5b69b4b0f90b8b5e97e637d6640e2f",
        "stripped-wire tree hash frozen",
    )
    return mvtec, wire


def _audit_perception_report(
    audit: Audit,
    report_path: Path,
    predictions_path: Path,
    checkpoint_path: Path,
    maps_path: Path,
    expected_rows: int,
    expected_normal: int,
    expected_anomaly: int,
    expected_split: dict[str, int],
    method_names: tuple[str, ...],
) -> dict[str, Any]:
    report = _read_json(report_path)
    rows = _read_csv(predictions_path)
    audit.require(len(rows) == expected_rows, f"{report['name']}: held-out row count", len(rows))
    audit.require(
        sum(int(row["is_anomaly"]) == 0 for row in rows) == expected_normal
        and sum(int(row["is_anomaly"]) == 1 for row in rows) == expected_anomaly,
        f"{report['name']}: held-out class counts",
    )
    audit.require(
        len({row["relative_path"] for row in rows}) == expected_rows,
        f"{report['name']}: unique held-out paths",
    )
    audit.require(
        all(int(report["split"][key]) == value for key, value in expected_split.items()),
        f"{report['name']}: frozen split",
        report["split"],
    )
    for path, expected, label in (
        (predictions_path, report["predictions_sha256"], "predictions"),
        (checkpoint_path, report["checkpoint_sha256"], "checkpoint"),
        (maps_path, report["maps_sha256"], "maps"),
    ):
        audit.require(path.is_file(), f"{report['name']}: {label} exists", str(path))
        audit.require(_sha256(path) == expected, f"{report['name']}: {label} hash")

    labels = np.asarray([int(row["is_anomaly"]) for row in rows], dtype=np.int64)
    for method in method_names:
        audit.require(method in report["methods"], f"{report['name']}: method present: {method}")
        scores = np.asarray([float(row[f"{method}_score"]) for row in rows], dtype=np.float64)
        audit.require(np.isfinite(scores).all(), f"{report['name']}: finite scores: {method}")
        reported = report["methods"][method]
        audit.require(
            _close(roc_auc_score(labels, scores), reported["auroc"]),
            f"{report['name']}: AUROC recomputation: {method}",
        )
        audit.require(
            _close(average_precision_score(labels, scores), reported["aupr"]),
            f"{report['name']}: AUPR recomputation: {method}",
        )
        predictions = scores >= float(reported["threshold"])
        true_positive = int(np.sum(predictions & (labels == 1)))
        false_positive = int(np.sum(predictions & (labels == 0)))
        false_negative = int(np.sum(~predictions & (labels == 1)))
        true_negative = int(np.sum(~predictions & (labels == 0)))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        fpr = false_positive / max(1, false_positive + true_negative)
        audit.require(
            all(
                _close(actual, reported[key])
                for actual, key in (
                    (precision, "precision"),
                    (recall, "recall"),
                    (f1, "f1"),
                    (fpr, "false_positive_rate"),
                )
            ),
            f"{report['name']}: threshold metrics recomputation: {method}",
        )
    return report


def _audit_active_run(
    audit: Audit, report_path: Path, expected_protocol: str, expected_episodes: int
) -> dict[str, Any]:
    report = _read_json(report_path)
    config = report["config"]
    if expected_protocol:
        audit.require(
            config.get("protocol_id") == expected_protocol, f"active protocol: {expected_protocol}"
        )
    policies = sorted({str(row["policy"]) for row in report["aggregate"]})
    budgets = [int(value) for value in config["budgets"]]
    audit.require(len(policies) == 7, f"{expected_protocol or 'stress_v1'} policy count", policies)
    audit.require(
        config["episode_count"] == expected_episodes,
        f"{expected_protocol or 'stress_v1'} scenario count",
    )
    episodes_path = report_path.parent / "inspect_active_episodes.csv"
    aggregate_path = report_path.parent / "inspect_active_aggregate.csv"
    paired_path = report_path.parent / "inspect_active_paired.csv"
    rows = _read_csv(episodes_path)
    expected_rows = expected_episodes * len(policies) * len(budgets)
    audit.require(
        len(rows) == expected_rows, f"{expected_protocol or 'stress_v1'} episode rows", len(rows)
    )
    keys = {(int(row["seed"]), row["policy"], int(row["budget"])) for row in rows}
    audit.require(
        len(keys) == expected_rows, f"{expected_protocol or 'stress_v1'} unique episode keys"
    )
    audit.require(_finite_rows(rows), f"{expected_protocol or 'stress_v1'} finite episode metrics")
    for path, field, label in (
        (episodes_path, "episodes_sha256", "episodes"),
        (aggregate_path, "aggregate_sha256", "aggregate"),
        (paired_path, "paired_sha256", "paired"),
    ):
        audit.require(
            _sha256(path) == report[field], f"{expected_protocol or 'stress_v1'} {label} hash"
        )

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget"]), row["policy"])].append(row)
    for summary in report["aggregate"]:
        key = (int(summary["budget"]), str(summary["policy"]))
        group = grouped[key]
        audit.require(len(group) == expected_episodes, f"active aggregate cell size: {key}")
        for metric in (
            "critical_weighted_recall",
            "recall",
            "precision",
            "false_positive_rate",
            "coverage",
            "travel_distance",
            "critical_success_rate",
        ):
            source_key = "critical_success" if metric == "critical_success_rate" else metric
            values = [
                float(_as_bool(row[source_key]))
                if source_key == "critical_success"
                else float(row[source_key])
                for row in group
            ]
            audit.require(
                _close(float(np.mean(values)), float(summary[metric])),
                f"active aggregate recomputation: {key}/{metric}",
            )
    return report


def _audit_oracle_correction(
    audit: Audit,
    corrected_path: Path,
    initial_path: Path,
    corrected_report: dict[str, Any],
) -> None:
    corrected = _read_csv(corrected_path)
    initial = _read_csv(initial_path)
    non_oracle_corrected = {
        (row["seed"], row["policy"], row["budget"]): row
        for row in corrected
        if row["policy"] != "oracle"
    }
    non_oracle_initial = {
        (row["seed"], row["policy"], row["budget"]): row
        for row in initial
        if row["policy"] != "oracle"
    }
    audit.require(len(non_oracle_corrected) == 72_000, "oracle correction non-oracle row count")
    audit.require(
        non_oracle_corrected.keys() == non_oracle_initial.keys(),
        "oracle correction non-oracle keys",
    )
    shared_fields = tuple(
        field
        for field in next(iter(non_oracle_initial.values()))
        if field in next(iter(non_oracle_corrected.values()))
    )
    audit.require(len(shared_fields) == 17, "oracle correction shared field count", shared_fields)
    mismatches = 0
    for key in non_oracle_corrected:
        if any(
            non_oracle_corrected[key][field] != non_oracle_initial[key][field]
            for field in shared_fields
        ):
            mismatches += 1
    audit.require(
        mismatches == 0, "oracle correction leaves all non-oracle fields unchanged", mismatches
    )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in corrected:
        grouped[(row["seed"], row["budget"])].append(row)
    violations = 0
    for rows in grouped.values():
        oracle = next(row for row in rows if row["policy"] == "oracle")
        best = max(
            float(row["critical_weighted_recall"]) for row in rows if row["policy"] != "oracle"
        )
        if float(oracle["critical_weighted_recall"]) + 1e-12 < best:
            violations += 1
    audit.require(
        violations == 0, "corrected exact oracle is a per-scenario upper bound", violations
    )
    audit.require(
        corrected_report["paired_sha256"]
        == "4e9a48764f3f64e49d35aa8659fe4cc99655ec74c6efb02678680d35d017a340",
        "deployment-v2 paired table frozen",
    )


def _audit_mujoco(
    audit: Audit,
    report_path: Path,
    active_episodes_path: Path,
) -> dict[str, Any]:
    report = _read_json(report_path)
    episodes_path = report_path.parent / "inspect_mujoco_direct_episodes.csv"
    aggregate_path = report_path.parent / "inspect_mujoco_direct_aggregate.csv"
    render_path = report_path.parent / "inspect_mujoco_direct_topology_final.png"
    rows = _read_csv(episodes_path)
    audit.require(len(rows) == 320, "direct MuJoCo episode rows", len(rows))
    audit.require(
        len({(row["seed"], row["policy"], row["budget"]) for row in rows}) == 320,
        "direct MuJoCo unique episode keys",
    )
    audit.require(_finite_rows(rows, {"trace_json"}), "direct MuJoCo finite metrics")
    audit.require(
        all(_as_bool(row["native_reach_success"]) for row in rows),
        "all native MuJoCo reaches succeeded",
    )
    audit.require(bool(report["all_native_reaches_succeeded"]), "MuJoCo report reach flag")
    audit.require(
        float(report["max_task_space_fov_error_m"]) <= 0.003, "MuJoCo FOV error within 3 mm"
    )
    audit.require(
        float(report["max_visual_ik_error_m"]) <= 0.005, "MuJoCo visual IK error within 5 mm"
    )
    for path, field, label in (
        (episodes_path, "episodes_sha256", "episodes"),
        (aggregate_path, "aggregate_sha256", "aggregate"),
        (render_path, "representative_render_sha256", "representative render"),
    ):
        audit.require(path.is_file(), f"MuJoCo {label} exists", str(path))
        audit.require(_sha256(path) == report[field], f"MuJoCo {label} hash")
    scene_path = Path("projects/01_inspectbot/mujoco/scene.xml")
    audit.require(_sha256(scene_path) == report["scene_sha256"], "MuJoCo scene hash")

    active_rows = {
        (row["seed"], row["policy"], row["budget"]): row for row in _read_csv(active_episodes_path)
    }
    common_fields = (
        "true_anomalies",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "critical_weighted_recall",
        "critical_miss_rate",
        "false_positive_rate",
        "coverage",
        "revisit_fraction",
        "scans_used",
        "critical_success",
    )
    mismatches = 0
    for row in rows:
        key = (row["seed"], row["policy"], row["budget"])
        reference = active_rows[key]
        for field in common_fields:
            if field == "critical_success":
                equal = _as_bool(row[field]) == _as_bool(reference[field])
            else:
                equal = _close(float(row[field]), float(reference[field]))
            mismatches += int(not equal)
    audit.require(
        mismatches == 0, "MuJoCo and deployment-v2 shared outcomes match exactly", mismatches
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path("artifacts/papers/inspectbot/final")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "audit" / "inspectbot_result_audit.json",
    )
    args = parser.parse_args()
    root = args.root
    audit = Audit()

    _audit_public_data(
        audit,
        root / "public_data/mvtec_cable_audit.json",
        root / "public_data/stripped_wire/stripped_wire_audit.json",
        Path("data/public/stripped_wire_zenodo_16686806/Insulated_wire_dataset.zip"),
    )
    mvtec_report = _audit_perception_report(
        audit,
        root / "perception/inspect_perception_report.json",
        root / "perception/inspect_perception_predictions.csv",
        root / "perception/inspect_spatial_gaussian.pt",
        root / "perception/inspect_spatial_maps.npz",
        expected_rows=150,
        expected_normal=58,
        expected_anomaly=92,
        expected_split={
            "fit_good": 180,
            "calibration_good": 44,
            "test_good": 58,
            "test_anomaly": 92,
        },
        method_names=("rgb_statistics", "resnet_global", "spatial_gaussian"),
    )
    wire_report = _audit_perception_report(
        audit,
        root / "stripped_wire/stripped_wire_report.json",
        root / "stripped_wire/stripped_wire_predictions.csv",
        root / "stripped_wire/stripped_wire_spatial_gaussian.pt",
        root / "stripped_wire/stripped_wire_spatial_maps.npz",
        expected_rows=300,
        expected_normal=133,
        expected_anomaly=167,
        expected_split={
            "fit_good": 160,
            "calibration_good": 40,
            "test_good": 133,
            "test_anomaly": 167,
        },
        method_names=("rgb_statistics", "resnet_global", "spatial_nn", "spatial_gaussian"),
    )
    audit.require(
        "separately" in mvtec_report["claim_scope"].lower(),
        "MVTec claim boundary explicit",
        mvtec_report["claim_scope"],
    )
    audit.require(
        "independent" in wire_report["claim_scope"].lower(),
        "stripped-wire independent-refit claim explicit",
        wire_report["claim_scope"],
    )

    stress = _audit_active_run(
        audit,
        root / "active/stress_v1/inspect_active_report.json",
        expected_protocol="",
        expected_episodes=1000,
    )
    deployment = _audit_active_run(
        audit,
        root / "active/deployment_v2_corrected/inspect_active_report.json",
        expected_protocol="deployment_v2",
        expected_episodes=2000,
    )
    audit.require(
        stress["config"]["view_order"] == [3, 1, 2, 0], "stress-v1 adverse-first view order"
    )
    audit.require(
        deployment["config"]["view_order"] == [0, 1, 2, 3], "deployment-v2 clean-first view order"
    )
    audit.require(
        stress["score_cache_sha256"] == deployment["score_cache_sha256"],
        "same frozen score cache across active protocols",
    )
    _audit_oracle_correction(
        audit,
        root / "active/deployment_v2_corrected/inspect_active_episodes.csv",
        root / "active/deployment_v2_initial_oracle_bug/inspect_active_episodes.csv",
        deployment,
    )

    deployment_index = {(int(row["budget"]), row["policy"]): row for row in deployment["aggregate"]}
    expected_deltas = {4: 0.09433331642820258, 6: 0.07808711657429203, 8: 0.0773, 10: 0.0471}
    for budget, expected in expected_deltas.items():
        delta = float(
            deployment_index[(budget, "topology_risk")]["critical_weighted_recall"]
        ) - float(deployment_index[(budget, "geometry_coverage")]["critical_weighted_recall"])
        tolerance = 1e-12 if budget in {4, 6} else 5e-5
        audit.require(
            _close(delta, expected, tolerance),
            f"deployment-v2 topology/geometry delta at budget {budget}",
            delta,
        )
    audit.require(
        _close(
            deployment_index[(12, "topology_risk")]["critical_weighted_recall"],
            deployment_index[(12, "geometry_coverage")]["critical_weighted_recall"],
        ),
        "deployment-v2 full-coverage tie at budget 12",
    )
    audit.require(
        float(deployment_index[(16, "topology_risk")]["critical_weighted_recall"])
        < float(deployment_index[(16, "geometry_coverage")]["critical_weighted_recall"]),
        "deployment-v2 late-revisit topology reversal retained",
    )

    mujoco_report = _audit_mujoco(
        audit,
        root / "mujoco_direct/inspect_mujoco_direct_report.json",
        root / "active/deployment_v2_corrected/inspect_active_episodes.csv",
    )
    result = {
        "name": "InspectBot frozen-result independent audit",
        "passed": all(row["passed"] for row in audit.checks),
        "checks": audit.checks,
        "inputs": {
            "mvtec_report_sha256": _sha256(root / "perception/inspect_perception_report.json"),
            "stripped_wire_report_sha256": _sha256(
                root / "stripped_wire/stripped_wire_report.json"
            ),
            "stress_v1_report_sha256": _sha256(
                root / "active/stress_v1/inspect_active_report.json"
            ),
            "deployment_v2_report_sha256": _sha256(
                root / "active/deployment_v2_corrected/inspect_active_report.json"
            ),
            "mujoco_report_sha256": _sha256(
                root / "mujoco_direct/inspect_mujoco_direct_report.json"
            ),
            "mujoco_scene_sha256": mujoco_report["scene_sha256"],
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


if __name__ == "__main__":
    main()
