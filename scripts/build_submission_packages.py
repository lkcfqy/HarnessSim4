#!/usr/bin/env python3
"""Build four anonymous, venue-specific staging submission packages.

The packages deliberately remain marked as staging-only while the corresponding
real-robot evidence gate is open. This script never removes manuscript draft
warnings and never fabricates hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


PAPER_COPY = {
    "inspectbot": {
        "question": (
            "how an inspection robot should allocate a limited scan budget over a "
            "branched wire harness when defect detectability and process risk differ by site"
        ),
        "contributions": [
            "a frozen public-image anomaly audit on MVTec Cable and an independent stripped-wire dataset",
            "paired scan-budget experiments with exact correction for multiple comparisons and retained negative controls",
            "an executable UR5e digital-twin scan audit with an explicit boundary between simulation and factory evidence",
        ],
        "fit": (
            "The work targets automated quality control in a structured manufacturing cell "
            "and emphasizes efficiency, inspection quality, reliability, and deployment limits."
        ),
        "video_contents": (
            "A UR5e digital twin executes bounded active-inspection viewpoints around a "
            "branched harness fixture while the visualization identifies the scan task."
        ),
        "hardware_blocker": (
            "calibrated real-camera multiview trials on representative automotive harness "
            "defects, suppliers, lighting, and fixtures"
        ),
    },
    "insertbot": {
        "question": (
            "whether contact-state recovery can reduce failed or damaging cable-connector "
            "insertions under pose error, cable load, and frictional contact"
        ),
        "contributions": [
            "a paired force-safety benchmark with shared physical seeds and prespecified success and damage proxies",
            "contact-state, orientation, retract, and guarded-admittance comparisons with multiplicity-corrected statistics",
            "a native-contact KUKA iiwa digital-twin audit that preserves the boundary between synthetic contact and real connector reliability",
        ],
        "fit": (
            "The work addresses structured robotic assembly, contact-state diagnosis, "
            "force safety, production quality, and reliability."
        ),
        "video_contents": (
            "A KUKA LBR iiwa 14 digital twin with a Robotiq gripper approaches and "
            "inserts a terminal into a collision-enabled fixture."
        ),
        "hardware_blocker": (
            "force/torque-calibrated real-connector trials with cable reaction, latch "
            "verification, damage inspection, and post-insertion pull tests"
        ),
    },
    "routebot": {
        "question": (
            "whether observable process relations are necessary to prevent geometrically "
            "plausible but semantically wrong long-horizon cable routing"
        ),
        "contributions": [
            "paired semantic interventions that preserve geometry, physics, and random seed while changing only segment-to-fixture feasibility",
            "relation, architecture, and negative-control comparisons that separate relation availability from a particular graph network",
            "public real-robot trajectory, external DLO-Lab, and direct MuJoCo audits with explicit claim boundaries",
        ],
        "fit": (
            "The paper isolates a general manipulation and robot-learning problem in "
            "deformable-object control: grounding process semantics into long-horizon actions."
        ),
        "video_contents": (
            "A UR10e digital twin with a Robotiq gripper performs ordered clip routing "
            "over a fixture while the cable and target sequence remain visible."
        ),
        "hardware_blocker": (
            "force-controlled semantic clip-routing replication on hardware or a genuinely "
            "semantic external benchmark"
        ),
    },
    "branchbot": {
        "question": (
            "whether semantic identity and typed action grounding prevent endpoint swaps "
            "in bimanual manipulation of branched deformable objects"
        ),
        "contributions": [
            "an exact semantic counterfactual that preserves physical state while swapping observable endpoint-to-arm-to-target bindings",
            "paired relational, typed-action, geometry-only, and ablation tests that expose a branch-identity shortcut",
            "a direct dual-UR10e MuJoCo audit and an explicit boundary around the current two-branch abstraction",
        ],
        "fit": (
            "The paper addresses a general cooperative-manipulation and robot-learning "
            "question about semantic grounding in branched deformable objects."
        ),
        "video_contents": (
            "Two UR10e digital twins with Robotiq grippers coordinate branch separation "
            "and placement while preserving distinct endpoint identities."
        ),
        "hardware_blocker": (
            "vision-based three-to-five-branch identity trials with collision-aware dual-arm "
            "hardware execution"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def locate_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError(
        "ffmpeg is unavailable. Install the visual extra with "
        "`python -m pip install -e '.[visual]'`."
    )


def build_video(
    ffmpeg: str, source: Path, destination: Path, start: float, duration: float
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{start:.10f}",
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def hardware_gate_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("passed") is True and value.get("evidence_kind") == "real_robot_hardware"


def cover_letter(key: str, paper: dict[str, Any], venue: dict[str, Any]) -> str:
    copy = PAPER_COPY[key]
    contributions = "\n".join(
        f"{index}. {item}." for index, item in enumerate(copy["contributions"], start=1)
    )
    codes = ""
    if paper["venue"] == "T-ASE":
        methods = paper["methodology_codes"]
        applications = paper["application_codes"]
        codes = f"""

Requested T-ASE classification codes:

- Primary methodology: {methods["primary"]}
- Secondary methodology: {methods["secondary"]}
- Primary application: {applications["primary"]}
- Secondary application: {applications["secondary"]}
"""
    return f"""# STAGING COVER LETTER — DO NOT SUBMIT

Dear Editor-in-Chief,

Please consider the enclosed anonymous {venue["paper_type"]}, “{paper["title"]},” for
{venue["full_name"]}. The manuscript asks {copy["question"]}.

Its principal contributions are:

{contributions}

{copy["fit"]} The manuscript reports its negative controls and scientific limitations,
including the fact that the accompanying robot clip is a MuJoCo digital twin rather than
hardware validation.{codes}

Before upload, the corresponding author must replace this staging heading and personally
confirm authorship, originality, absence of concurrent review, conflicts of interest,
permissions, funding disclosure, generative-AI disclosure if required, and completion of
the real-robot evidence gate. The anonymous review package must not contain identifying
names, affiliations, acknowledgments, repository identities, or metadata.

Sincerely,

Anonymous Authors
"""


def submission_checklist(
    key: str, paper: dict[str, Any], venue: dict[str, Any], hardware_passed: bool
) -> str:
    copy = PAPER_COPY[key]
    ntp_line = (
        "- [x] A 100–300 word Note to Practitioners follows the abstract.\n"
        if venue["note_to_practitioners"]["required"]
        else "- [x] This target venue does not require a Note to Practitioners.\n"
    )
    hardware_box = "x" if hardware_passed else " "
    return f"""# {key} submission checklist

Target: {venue["full_name"]} — {venue["paper_type"]}

## Machine-checkable format

- [x] IEEE Transactions two-column class is used.
- [x] Review manuscript is anonymous.
- [x] Abstract target is at most {venue["abstract_max_words"]} words.
{ntp_line}- [x] IEEE keywords are present.
- [x] PDF uses consistent US Letter pages and embedded fonts.
- [x] Project-specific H.264 digital-twin clip, ReadMe, and short Summary are present.
- [x] Multimedia is explicitly labeled as simulation rather than hardware evidence.

## Scientific release gate

- [{hardware_box}] Real-robot evidence gate passed: {copy["hardware_blocker"]}.
- [ ] Red `DRAFT---NOT FOR SUBMISSION` warning removed only after the hardware gate passes.
- [ ] Every hardware-derived number in the manuscript independently audited against raw logs.
- [ ] Frozen statistical analysis rerun after hardware data are locked.
- [ ] Claims, abstract, tables, figures, limitations, and cover letter updated consistently.

## Corresponding-author confirmations

- [ ] Author order, affiliations, corresponding author, and ORCIDs confirmed outside the anonymous PDF.
- [ ] Originality and absence of concurrent submission personally confirmed.
- [ ] Conflicts, funding, permissions, data/code availability, and required AI-use disclosure confirmed.
- [ ] Anonymous source, supplement, media metadata, and repository links inspected for identity leakage.
- [ ] Current official author instructions rechecked on the actual upload date.

Status: **STAGING ONLY — DO NOT UPLOAD** while any unchecked scientific-release item remains.
"""


def multimedia_readme(key: str, paper: dict[str, Any], video_path: Path) -> str:
    copy = PAPER_COPY[key]
    video = paper["video"]
    return f"""TITLE
{paper["title"]}

FILE
{video_path.name}

PLAYBACK
H.264 video, yuv420p pixel format, no audio. Any current browser or VLC-compatible player
should play the file. Expected excerpt duration: {video["duration_s"]:.2f} seconds.

CONTENTS
{copy["video_contents"]}

PROVENANCE
The clip is a project-specific excerpt from the audited HarnessSim4 four-robot showcase.
It was rendered from scripted MuJoCo digital-twin trajectories and encoded from the source
declared in PACKAGE_MANIFEST.json.

SCIENTIFIC BOUNDARY
This file visualizes scene geometry, robot configuration, and executable scripted motion.
It is not real-robot evidence, a force/torque validation, a safety certification, a cycle-time
measurement, or proof of production reliability. The manuscript's real-hardware gate remains
authoritative.
"""


def multimedia_summary(key: str, paper: dict[str, Any]) -> str:
    copy = PAPER_COPY[key]
    return (
        f"This video accompanies “{paper['title']}.” "
        f"{copy['video_contents']} "
        "The clip is generated from a MuJoCo digital twin and demonstrates executable scripted motion. "
        "It does not constitute hardware validation, force-safety evidence, factory cycle-time evidence, or production performance."
    )


def claim_boundary(key: str, paper: dict[str, Any]) -> str:
    copy = PAPER_COPY[key]
    return f"""Package: {key}
Target venue: {paper["venue"]}

ALLOWED MULTIMEDIA CLAIM
The clip is an executable MuJoCo digital-twin visualization of the task and robot geometry.

PROHIBITED CLAIMS UNTIL THE HARDWARE GATE PASSES
- real-robot success rate
- production cycle time or labor replacement
- collision or functional safety certification
- connector, cable, or harness damage rate in production
- factory reliability, service life, return on investment, or customer readiness

OPEN HARDWARE REQUIREMENT
{copy["hardware_blocker"]}.
"""


def package_status(key: str, paper: dict[str, Any], passed: bool) -> str:
    state = "hardware gate passed" if passed else "hardware gate open"
    return f"""# Package status

- Paper: `{key}`
- Target: `{paper["venue"]}`
- State: `{state}`
- Intended use: anonymous venue-format staging and internal review
- Upload authorization: **NO** unless `scripts/audit_venue_packages.py --require-submission-ready`
  exits with code 0

The package builder does not remove the manuscript's draft warning. A human author must
review the current official journal instructions and all scientific claims before upload.
"""


def build_package(
    key: str,
    paper: dict[str, Any],
    venues: dict[str, Any],
    config_path: Path,
    ffmpeg: str | None,
) -> dict[str, Any]:
    venue = venues[paper["venue"]]
    package_dir = ROOT / paper["package_dir"]
    multimedia_dir = package_dir / "multimedia"
    package_dir.mkdir(parents=True, exist_ok=True)
    multimedia_dir.mkdir(parents=True, exist_ok=True)

    release_pdf = ROOT / paper["release_pdf"]
    source_tex = ROOT / paper["source_tex"]
    if not release_pdf.is_file() or not source_tex.is_file():
        raise FileNotFoundError(f"Missing manuscript input for {key}")

    manuscript = package_dir / "manuscript.pdf"
    shutil.copy2(release_pdf, manuscript)

    gate_path = ROOT / paper["hardware_gate"]
    gate_passed = hardware_gate_passed(gate_path)
    write_text(package_dir / "COVER_LETTER.md", cover_letter(key, paper, venue))
    write_text(
        package_dir / "SUBMISSION_CHECKLIST.md",
        submission_checklist(key, paper, venue, gate_passed),
    )
    write_text(package_dir / "PACKAGE_STATUS.md", package_status(key, paper, gate_passed))

    video = paper["video"]
    source_video = ROOT / video["source"]
    destination_video = multimedia_dir / video["filename"]
    if ffmpeg is not None:
        build_video(
            ffmpeg,
            source_video,
            destination_video,
            float(video["start_s"]),
            float(video["duration_s"]),
        )
    elif not destination_video.is_file():
        raise FileNotFoundError(
            f"Video build skipped but {destination_video.relative_to(ROOT)} does not exist"
        )
    write_text(multimedia_dir / "ReadMe.txt", multimedia_readme(key, paper, destination_video))
    write_text(multimedia_dir / "Summary.txt", multimedia_summary(key, paper))
    write_text(multimedia_dir / "CLAIM_BOUNDARY.txt", claim_boundary(key, paper))

    source_text = source_tex.read_text(encoding="utf-8")
    draft_warning_present = "DRAFT---NOT FOR SUBMISSION" in source_text
    files = [
        manuscript,
        package_dir / "COVER_LETTER.md",
        package_dir / "SUBMISSION_CHECKLIST.md",
        package_dir / "PACKAGE_STATUS.md",
        destination_video,
        multimedia_dir / "ReadMe.txt",
        multimedia_dir / "Summary.txt",
        multimedia_dir / "CLAIM_BOUNDARY.txt",
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": key,
        "title": paper["title"],
        "venue": paper["venue"],
        "package_state": (
            "submission_candidate"
            if gate_passed and not draft_warning_present
            else "staging_only_not_for_submission"
        ),
        "claim_scope": (
            "Anonymous venue-format staging package. Multimedia is MuJoCo digital-twin "
            "evidence only and is not hardware or factory validation."
        ),
        "source_config": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "source_manuscript": {
            "path": paper["release_pdf"],
            "sha256": sha256(release_pdf),
        },
        "source_video": {
            "path": video["source"],
            "sha256": sha256(source_video),
            "excerpt_start_s": video["start_s"],
            "excerpt_duration_s": video["duration_s"],
        },
        "hardware_gate": {
            "path": paper["hardware_gate"],
            "present": gate_path.is_file(),
            "passed": gate_passed,
        },
        "draft_warning_present": draft_warning_present,
        "files": [
            {
                "path": str(path.relative_to(package_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    manifest_path = package_dir / "PACKAGE_MANIFEST.json"
    write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    return {
        "paper": key,
        "venue": paper["venue"],
        "package_dir": str(package_dir),
        "files": len(files) + 1,
        "video_bytes": destination_video.stat().st_size,
        "hardware_gate_passed": gate_passed,
        "package_state": manifest["package_state"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/submission/venue_profiles.json",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Reuse existing package videos instead of encoding them again.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ffmpeg = None if args.skip_video else locate_ffmpeg()
    results = [
        build_package(key, paper, config["venues"], config_path, ffmpeg)
        for key, paper in config["papers"].items()
    ]
    print(json.dumps({"packages": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
