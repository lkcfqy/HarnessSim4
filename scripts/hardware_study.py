#!/usr/bin/env python3
"""Stateful, fail-closed workflow for collecting HarnessSim4 hardware evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROBOTS = ("inspectbot", "insertbot", "routebot", "branchbot")
STATES = ("INITIALIZED", "FROZEN", "COLLECTING", "COMPLETED")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(value: str | None = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(_timestamp(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def _safe_path(root: Path, value: Path) -> Path:
    resolved_root = root.resolve()
    candidate = value.resolve() if value.is_absolute() else (root / value).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"artifact must remain inside the study directory: {value}")
    return candidate


def _config_path(robot: str) -> Path:
    return Path(f"configs/hardware/{robot}_gate.json")


def _study_files(root: Path) -> dict[str, Path]:
    return {
        "trials": root / "trials.csv",
        "provenance": root / "provenance.json",
        "calibration": root / "calibration.json",
        "media": root / "media_manifest.json",
        "protocol_snapshot": root / "protocol_snapshot.json",
        "trial_template": root / "trial_record_template.json",
    }


def _require_state(provenance: dict[str, Any], expected: str) -> None:
    observed = provenance.get("state")
    if observed != expected:
        raise ValueError(f"study state must be {expected}; observed {observed!r}")


def initialize_study(config_path: Path, root: Path) -> dict[str, Any]:
    config = _read_json(config_path)
    root.mkdir(parents=True, exist_ok=True)
    files = _study_files(root)
    existing = [str(path) for path in files.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing study files: " + ", ".join(existing))
    with files["trials"].open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(config["required_columns"])
    provenance = {
        "schema_version": 1,
        "robot": config["robot"],
        "state": "INITIALIZED",
        "study_id": "",
        "site_id": "",
        "protocol_sha256": _sha256(config_path),
        "protocol_frozen_before_collection": False,
        "protocol_frozen_at_utc": "",
        "collection_started_at_utc": "",
        "collection_completed_at_utc": "",
        "operator_ids": [],
        "independent_reviewer_id": "",
        "safety_review_approved": False,
        "robot_serial": "",
        "software_revision": "",
    }
    _write_json(files["provenance"], provenance)
    _write_json(files["calibration"], {"schema_version": 1, "records": []})
    _write_json(files["media"], {"schema_version": 1, "items": []})
    _copy_atomic(config_path, files["protocol_snapshot"])
    _write_json(files["trial_template"], {column: "" for column in config["required_columns"]})
    return provenance


def freeze_study(
    config_path: Path,
    root: Path,
    *,
    study_id: str,
    site_id: str,
    operator_ids: list[str],
    reviewer_id: str,
    robot_serial: str,
    software_revision: str,
    safety_review_approved: bool,
    at: str | None = None,
) -> dict[str, Any]:
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "INITIALIZED")
    required_text = (study_id, site_id, reviewer_id, robot_serial, software_revision)
    if not all(value.strip() for value in required_text) or not operator_ids:
        raise ValueError("study, site, operators, reviewer, robot serial, and revision are required")
    if not safety_review_approved:
        raise ValueError("a documented safety review is required before protocol freeze")
    _copy_atomic(config_path, files["protocol_snapshot"])
    provenance.update(
        {
            "state": "FROZEN",
            "study_id": study_id,
            "site_id": site_id,
            "protocol_sha256": _sha256(config_path),
            "protocol_frozen_before_collection": True,
            "protocol_frozen_at_utc": _timestamp(at),
            "operator_ids": sorted(set(operator_ids)),
            "independent_reviewer_id": reviewer_id,
            "safety_review_approved": True,
            "robot_serial": robot_serial,
            "software_revision": software_revision,
        }
    )
    _write_json(files["provenance"], provenance)
    return provenance


def register_calibration(
    config_path: Path,
    root: Path,
    *,
    kind: str,
    device_id: str,
    artifact: Path,
    performed_at: str,
    valid_through: str,
) -> dict[str, Any]:
    config = _read_json(config_path)
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "FROZEN")
    if kind not in config["required_calibration_kinds"]:
        raise ValueError(f"unexpected calibration kind: {kind}")
    artifact_path = _safe_path(root, artifact)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    performed = _parse_time(performed_at)
    valid = _parse_time(valid_through)
    if valid <= performed:
        raise ValueError("calibration expiry must be after calibration time")
    calibration = _read_json(files["calibration"])
    records = calibration.setdefault("records", [])
    if any(item.get("kind") == kind for item in records):
        raise ValueError(f"calibration kind already registered: {kind}")
    record = {
        "kind": kind,
        "device_id": device_id,
        "performed_at_utc": performed.isoformat(),
        "valid_through_utc": valid.isoformat(),
        "passed": True,
        "artifact_path": artifact_path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(artifact_path),
    }
    records.append(record)
    _write_json(files["calibration"], calibration)
    return record


def start_collection(config_path: Path, root: Path, *, at: str | None = None) -> dict[str, Any]:
    config = _read_json(config_path)
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "FROZEN")
    if provenance.get("protocol_sha256") != _sha256(config_path):
        raise ValueError("protocol changed after freeze")
    if _sha256(files["protocol_snapshot"]) != provenance["protocol_sha256"]:
        raise ValueError("frozen protocol snapshot no longer matches provenance")
    started = _parse_time(_timestamp(at))
    frozen = _parse_time(provenance["protocol_frozen_at_utc"])
    if started <= frozen:
        raise ValueError("collection must start after protocol freeze")
    calibration = _read_json(files["calibration"])
    by_kind = {item.get("kind"): item for item in calibration.get("records", [])}
    missing = sorted(set(config["required_calibration_kinds"]) - set(by_kind))
    if missing:
        raise ValueError("required calibrations are missing: " + ", ".join(missing))
    for kind in config["required_calibration_kinds"]:
        item = by_kind[kind]
        artifact = _safe_path(root, Path(item["artifact_path"]))
        if not item.get("passed") or not artifact.is_file() or _sha256(artifact) != item["sha256"]:
            raise ValueError(f"calibration evidence is invalid: {kind}")
        if _parse_time(item["performed_at_utc"]) > started:
            raise ValueError(f"calibration was performed after collection start: {kind}")
        if _parse_time(item["valid_through_utc"]) <= started:
            raise ValueError(f"calibration expired before collection start: {kind}")
    provenance["state"] = "COLLECTING"
    provenance["collection_started_at_utc"] = started.isoformat()
    _write_json(files["provenance"], provenance)
    return provenance


def _read_trial_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _validate_trial(config: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    columns = config["required_columns"]
    missing_columns = [column for column in columns if column not in row]
    if missing_columns:
        raise ValueError("trial fields missing: " + ", ".join(missing_columns))
    optional = set(config.get("optional_value_columns", []))
    blank = [column for column in columns if column not in optional and str(row[column]).strip() == ""]
    if blank:
        raise ValueError("trial values missing: " + ", ".join(blank))
    normalized = {column: str(row[column]).strip() for column in columns}
    expected_policies = {config["primary_policy"], *config["comparators"]}
    if normalized["policy"] not in expected_policies:
        raise ValueError(f"unknown policy: {normalized['policy']}")
    if normalized["condition"] not in config["required_conditions"]:
        raise ValueError(f"unknown condition: {normalized['condition']}")
    for column in config["binary_columns"]:
        value = int(normalized[column])
        if value not in (0, 1):
            raise ValueError(f"{column} must be binary")
    for column in config["numeric_columns"]:
        value = float(normalized[column])
        if not math.isfinite(value):
            raise ValueError(f"{column} must be finite")
        if column in config["nonnegative_columns"] and value < 0:
            raise ValueError(f"{column} must be nonnegative")
    if int(normalized["success"]) == 1:
        for column in config["success_rules"]["must_be_one"]:
            if int(normalized[column]) != 1:
                raise ValueError(f"successful trial requires {column}=1")
        for column in config["success_rules"]["must_be_zero"]:
            if float(normalized[column]) != 0:
                raise ValueError(f"successful trial requires {column}=0")
    normalized["timestamp_utc"] = _timestamp(normalized["timestamp_utc"])
    return normalized


def append_trial(config_path: Path, root: Path, row: dict[str, Any]) -> dict[str, str]:
    config = _read_json(config_path)
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "COLLECTING")
    if provenance.get("protocol_sha256") != _sha256(config_path):
        raise ValueError("protocol changed after freeze")
    normalized = _validate_trial(config, row)
    timestamp = _parse_time(normalized["timestamp_utc"])
    if timestamp < _parse_time(provenance["collection_started_at_utc"]):
        raise ValueError("trial timestamp precedes collection start")
    columns, existing = _read_trial_rows(files["trials"])
    if columns != config["required_columns"]:
        raise ValueError("trial header no longer matches the frozen protocol")
    if any(item["trial_id"] == normalized["trial_id"] for item in existing):
        raise ValueError(f"duplicate trial_id: {normalized['trial_id']}")
    if any(
        item["pair_id"] == normalized["pair_id"] and item["policy"] == normalized["policy"]
        for item in existing
    ):
        raise ValueError("duplicate (pair_id, policy) assignment")
    with files["trials"].open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerow(normalized)
        handle.flush()
    return normalized


def register_media(
    config_path: Path,
    root: Path,
    *,
    kind: str,
    artifact: Path,
    trial_id: str | None = None,
) -> dict[str, Any]:
    del config_path  # The final audit checks required kinds; extra traceable media are allowed.
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "COLLECTING")
    artifact_path = _safe_path(root, artifact)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if trial_id:
        _, rows = _read_trial_rows(files["trials"])
        if trial_id not in {row["trial_id"] for row in rows}:
            raise ValueError(f"media references an unknown trial: {trial_id}")
    manifest = _read_json(files["media"])
    items = manifest.setdefault("items", [])
    relative = artifact_path.relative_to(root.resolve()).as_posix()
    if any(item.get("path") == relative for item in items):
        raise ValueError(f"media already registered: {relative}")
    item = {
        "trial_id": trial_id,
        "kind": kind,
        "path": relative,
        "sha256": _sha256(artifact_path),
    }
    items.append(item)
    _write_json(files["media"], manifest)
    return item


def complete_collection(root: Path, *, at: str | None = None) -> dict[str, Any]:
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _require_state(provenance, "COLLECTING")
    completed = _parse_time(_timestamp(at))
    started = _parse_time(provenance["collection_started_at_utc"])
    if completed <= started:
        raise ValueError("completion must be after collection start")
    _, rows = _read_trial_rows(files["trials"])
    if any(_parse_time(row["timestamp_utc"]) > completed for row in rows):
        raise ValueError("completion precedes one or more trial timestamps")
    provenance["state"] = "COMPLETED"
    provenance["collection_completed_at_utc"] = completed.isoformat()
    _write_json(files["provenance"], provenance)
    return provenance


def study_status(root: Path) -> dict[str, Any]:
    files = _study_files(root)
    provenance = _read_json(files["provenance"])
    _, rows = _read_trial_rows(files["trials"])
    calibration = _read_json(files["calibration"])
    media = _read_json(files["media"])
    return {
        "robot": provenance.get("robot"),
        "study_id": provenance.get("study_id"),
        "state": provenance.get("state"),
        "trials": len(rows),
        "calibrations": len(calibration.get("records", [])),
        "media_items": len(media.get("items", [])),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, choices=ROBOTS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--root", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--study-id", required=True)
    freeze_parser.add_argument("--site-id", required=True)
    freeze_parser.add_argument("--operator-id", action="append", required=True)
    freeze_parser.add_argument("--reviewer-id", required=True)
    freeze_parser.add_argument("--robot-serial", required=True)
    freeze_parser.add_argument("--software-revision", required=True)
    freeze_parser.add_argument("--safety-review-approved", action="store_true")
    freeze_parser.add_argument("--at")

    calibration_parser = subparsers.add_parser("register-calibration")
    calibration_parser.add_argument("--root", type=Path, required=True)
    calibration_parser.add_argument("--kind", required=True)
    calibration_parser.add_argument("--device-id", required=True)
    calibration_parser.add_argument("--artifact", type=Path, required=True)
    calibration_parser.add_argument("--performed-at", required=True)
    calibration_parser.add_argument("--valid-through", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--root", type=Path, required=True)
    start_parser.add_argument("--at")

    trial_parser = subparsers.add_parser("add-trial")
    trial_parser.add_argument("--root", type=Path, required=True)
    trial_parser.add_argument("--record-json", type=Path, required=True)

    media_parser = subparsers.add_parser("register-media")
    media_parser.add_argument("--root", type=Path, required=True)
    media_parser.add_argument("--kind", required=True)
    media_parser.add_argument("--artifact", type=Path, required=True)
    media_parser.add_argument("--trial-id")

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("--root", type=Path, required=True)
    complete_parser.add_argument("--at")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = _config_path(args.robot)
    if args.command == "init":
        result = initialize_study(config_path, args.root)
    elif args.command == "freeze":
        result = freeze_study(
            config_path,
            args.root,
            study_id=args.study_id,
            site_id=args.site_id,
            operator_ids=args.operator_id,
            reviewer_id=args.reviewer_id,
            robot_serial=args.robot_serial,
            software_revision=args.software_revision,
            safety_review_approved=args.safety_review_approved,
            at=args.at,
        )
    elif args.command == "register-calibration":
        result = register_calibration(
            config_path,
            args.root,
            kind=args.kind,
            device_id=args.device_id,
            artifact=args.artifact,
            performed_at=args.performed_at,
            valid_through=args.valid_through,
        )
    elif args.command == "start":
        result = start_collection(config_path, args.root, at=args.at)
    elif args.command == "add-trial":
        result = append_trial(config_path, args.root, _read_json(args.record_json))
    elif args.command == "register-media":
        result = register_media(
            config_path,
            args.root,
            kind=args.kind,
            artifact=args.artifact,
            trial_id=args.trial_id,
        )
    elif args.command == "complete":
        result = complete_collection(args.root, at=args.at)
    else:
        result = study_status(args.root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
