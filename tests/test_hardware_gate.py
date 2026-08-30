from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.audit_hardware_gate import audit_hardware_gate

CONFIG_PATH = Path("configs/hardware/insertbot_gate.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_evidence(root: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    (root / "protocol_snapshot.json").write_bytes(CONFIG_PATH.read_bytes())
    policies = [config["primary_policy"], *config["comparators"]]
    cutoffs = {"contact_belief": 90, "guarded_admittance": 80, "direct_insertion": 60}
    with (root / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=config["required_columns"])
        writer.writeheader()
        for pair_index in range(100):
            for policy in policies:
                success = int(pair_index < cutoffs[policy])
                writer.writerow(
                    {
                        "trial_id": f"trial-{pair_index:03d}-{policy}",
                        "pair_id": f"pair-{pair_index:03d}",
                        "timestamp_utc": "2026-02-15T10:00:00+00:00",
                        "specimen_id": f"specimen-{pair_index % 30:02d}",
                        "setup_id": f"setup-{pair_index % 2}",
                        "connector_family": f"family-{pair_index % 3}",
                        "condition": config["required_conditions"][pair_index % 3],
                        "policy": policy,
                        "success": success,
                        "locked": success,
                        "pull_test_pass": success,
                        "damage": 0,
                        "safety_stop": 0,
                        "operator_intervention": 0,
                        "peak_force_n": 8.0 + pair_index / 100,
                        "peak_torque_nm": 0.4,
                        "cycle_time_s": 12.0,
                        "retries": int(not success),
                        "insertion_depth_mm": 40.0 if success else 30.0,
                        "failure_type": "none" if success else "not_locked",
                        "notes": "",
                    }
                )

    provenance = {
        "schema_version": 1,
        "robot": "insertbot",
        "state": "COMPLETED",
        "study_id": "test-study",
        "site_id": "test-site",
        "protocol_sha256": _sha256(CONFIG_PATH),
        "protocol_frozen_before_collection": True,
        "protocol_frozen_at_utc": "2026-01-15T00:00:00+00:00",
        "collection_started_at_utc": "2026-02-01T00:00:00+00:00",
        "collection_completed_at_utc": "2026-03-01T00:00:00+00:00",
        "operator_ids": ["operator-a"],
        "independent_reviewer_id": "reviewer-b",
        "safety_review_approved": True,
        "robot_serial": "robot-test-001",
        "software_revision": "test-revision",
    }
    (root / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    calibration_dir = root / "calibration"
    calibration_dir.mkdir()
    calibration_records = []
    for kind in config["required_calibration_kinds"]:
        artifact = calibration_dir / f"{kind}.txt"
        artifact.write_text(f"traceable calibration record for {kind}\n", encoding="utf-8")
        calibration_records.append(
            {
                "kind": kind,
                "device_id": f"device-{kind}",
                "performed_at_utc": "2026-01-20T00:00:00+00:00",
                "valid_through_utc": "2027-01-20T00:00:00+00:00",
                "passed": True,
                "artifact_path": str(artifact.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(artifact),
            }
        )
    (root / "calibration.json").write_text(
        json.dumps({"schema_version": 1, "records": calibration_records}, indent=2) + "\n",
        encoding="utf-8",
    )

    media_dir = root / "media"
    media_dir.mkdir()
    media_items = []
    kinds = config["required_media_kinds"]
    for index in range(config["minimum_media_items"]):
        artifact = media_dir / f"item-{index:02d}.bin"
        artifact.write_bytes(f"evidence-{index}".encode())
        media_items.append(
            {
                "trial_id": f"trial-{index:03d}-contact_belief",
                "kind": kinds[index % len(kinds)],
                "path": str(artifact.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(artifact),
            }
        )
    (root / "media_manifest.json").write_text(
        json.dumps({"schema_version": 1, "items": media_items}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_valid_insertbot_hardware_evidence_passes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    _write_valid_evidence(evidence)
    report = audit_hardware_gate(CONFIG_PATH, evidence, output)
    failed = [item for item in report["checks"] if not item["passed"]]
    assert report["passed"] is True, failed
    assert report["evidence_kind"] == "real_robot_hardware"
    assert len(report["paired_comparisons"]) == 2
    assert report["paired_comparisons"][0]["complete_pairs"] == 100
    assert (output / "hardware_gate_audit.json").is_file()


def test_protocol_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    _write_valid_evidence(evidence)
    provenance_path = evidence / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["protocol_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    report = audit_hardware_gate(CONFIG_PATH, evidence, output)
    assert report["passed"] is False
    hash_check = next(item for item in report["checks"] if item["name"] == "protocol_hash_frozen")
    assert hash_check["passed"] is False


def test_empty_study_skeleton_fails_closed_without_crashing(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    evidence.mkdir()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    (evidence / "protocol_snapshot.json").write_bytes(CONFIG_PATH.read_bytes())
    with (evidence / "trials.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(config["required_columns"])
    (evidence / "provenance.json").write_text("{}\n", encoding="utf-8")
    (evidence / "calibration.json").write_text('{"records": []}\n', encoding="utf-8")
    (evidence / "media_manifest.json").write_text('{"items": []}\n', encoding="utf-8")
    report = audit_hardware_gate(CONFIG_PATH, evidence, output)
    assert report["passed"] is False
    assert report["policy_summaries"] == {}
    assert (output / "hardware_gate_audit.json").is_file()
