from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_hardware_gate import audit_hardware_gate
from scripts.hardware_study import (
    append_trial,
    complete_collection,
    freeze_study,
    initialize_study,
    register_calibration,
    register_media,
    start_collection,
    study_status,
)

CONFIG_PATH = Path("configs/hardware/insertbot_gate.json")


def _initialize_and_freeze(root: Path) -> None:
    initialize_study(CONFIG_PATH, root)
    freeze_study(
        CONFIG_PATH,
        root,
        study_id="insertbot-test-v1",
        site_id="lab-test",
        operator_ids=["operator-a"],
        reviewer_id="reviewer-b",
        robot_serial="robot-001",
        software_revision="revision-001",
        safety_review_approved=True,
        at="2026-01-10T00:00:00+00:00",
    )


def _register_required_calibrations(root: Path) -> None:
    for kind in ("force_torque", "hand_eye", "robot_tcp"):
        artifact = root / "calibration" / f"{kind}.txt"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_text(f"calibration evidence for {kind}\n", encoding="utf-8")
        register_calibration(
            CONFIG_PATH,
            root,
            kind=kind,
            device_id=f"device-{kind}",
            artifact=artifact,
            performed_at="2026-01-15T00:00:00+00:00",
            valid_through="2027-01-15T00:00:00+00:00",
        )


def _trial() -> dict[str, object]:
    return {
        "trial_id": "trial-001-contact",
        "pair_id": "pair-001",
        "timestamp_utc": "2026-02-15T00:00:00+00:00",
        "specimen_id": "specimen-001",
        "setup_id": "setup-a",
        "connector_family": "family-a",
        "condition": "nominal",
        "policy": "contact_belief",
        "success": 1,
        "locked": 1,
        "pull_test_pass": 1,
        "damage": 0,
        "safety_stop": 0,
        "operator_intervention": 0,
        "peak_force_n": 8.4,
        "peak_torque_nm": 0.3,
        "cycle_time_s": 11.2,
        "retries": 0,
        "insertion_depth_mm": 40.0,
        "failure_type": "none",
        "notes": "",
    }


def test_hardware_study_state_machine_and_fail_closed_audit(tmp_path: Path) -> None:
    root = tmp_path / "study"
    _initialize_and_freeze(root)
    _register_required_calibrations(root)
    start_collection(CONFIG_PATH, root, at="2026-02-01T00:00:00+00:00")
    append_trial(CONFIG_PATH, root, _trial())
    media = root / "media" / "trial-001.mp4"
    media.parent.mkdir()
    media.write_bytes(b"traceable test media")
    register_media(
        CONFIG_PATH,
        root,
        kind="robot_overview_video",
        artifact=media,
        trial_id="trial-001-contact",
    )
    complete_collection(root, at="2026-03-01T00:00:00+00:00")
    status = study_status(root)
    assert status == {
        "robot": "insertbot",
        "study_id": "insertbot-test-v1",
        "state": "COMPLETED",
        "trials": 1,
        "calibrations": 3,
        "media_items": 1,
    }
    report = audit_hardware_gate(CONFIG_PATH, root, tmp_path / "audit")
    assert report["passed"] is False
    failed_names = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "minimum_trials_per_policy" in failed_names
    assert "study_state_completed" not in failed_names
    assert "trials_within_collection_window" not in failed_names


def test_collection_cannot_start_without_required_calibration(tmp_path: Path) -> None:
    root = tmp_path / "study"
    _initialize_and_freeze(root)
    with pytest.raises(ValueError, match="calibrations are missing"):
        start_collection(CONFIG_PATH, root, at="2026-02-01T00:00:00+00:00")


def test_duplicate_trial_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "study"
    _initialize_and_freeze(root)
    _register_required_calibrations(root)
    start_collection(CONFIG_PATH, root, at="2026-02-01T00:00:00+00:00")
    append_trial(CONFIG_PATH, root, _trial())
    with pytest.raises(ValueError, match="duplicate trial_id"):
        append_trial(CONFIG_PATH, root, _trial())
