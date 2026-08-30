from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_submission_bundle import _hardware_gate


def _write_gate(path: Path, protocol_hash: str) -> None:
    path.write_text(
        json.dumps(
            {
                "robot": "insertbot",
                "evidence_kind": "real_robot_hardware",
                "protocol": {"sha256": protocol_hash},
                "passed": True,
                "checks": [{"name": "traceability", "passed": True}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_hardware_gate_binds_robot_protocol_and_checks(tmp_path: Path) -> None:
    path = tmp_path / "hardware_gate_audit.json"
    protocol_hash = "a" * 64
    _write_gate(path, protocol_hash)
    passed, count, binding = _hardware_gate(path, "insertbot", protocol_hash)
    assert passed is True
    assert count == 1
    assert all(binding.values())


def test_hardware_gate_rejects_stale_protocol(tmp_path: Path) -> None:
    path = tmp_path / "hardware_gate_audit.json"
    _write_gate(path, "a" * 64)
    passed, _, binding = _hardware_gate(path, "insertbot", "b" * 64)
    assert passed is False
    assert binding["protocol_matches"] is False
