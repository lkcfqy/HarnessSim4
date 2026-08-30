#!/usr/bin/env python3
"""Audit venue format and scientific release gates for all four staging packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_submission_bundle import PAPERS, _audit_paper, _build_log_audit, _font_audit

ROOT = Path(__file__).resolve().parents[1]

TASE_METHOD_CODES = {
    "2 AI and Machine Learning",
    "4 Computer Vision",
    "5 Control Theory",
    "7 Digital Twins",
    "11 Modeling and Simulation",
    "15 Risk Assessment",
    "16 Planning",
    "19 System Diagnosis",
}
TASE_APPLICATION_CODES = {
    "2 Automotive",
    "13 Manufacturing",
    "19 Robotics",
    "23 Testing and Quality Control",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_block(source: str, environment: str) -> str | None:
    match = re.search(
        rf"\\begin\{{{re.escape(environment)}\}}(.*?)\\end\{{{re.escape(environment)}\}}",
        source,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def latex_word_count(text: str | None) -> int:
    if text is None:
        return 0
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(?:cite|ref|label|url|href)\*?(?:\[[^]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " TOKEN ", text)
    text = text.replace("~", " ").replace("--", "-")
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text, flags=re.UNICODE))


def extract_note_to_practitioners(source: str) -> str | None:
    match = re.search(
        r"Note to Practitioners---\}(.*?)(?=\\begin\{IEEEkeywords\}|\\section\{)",
        source,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def keyword_list(source: str) -> list[str]:
    block = extract_block(source, "IEEEkeywords")
    if block is None:
        return []
    cleaned = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", block)
    return [item.strip().rstrip(".") for item in cleaned.split(",") if item.strip()]


def pdf_version(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(16).decode("ascii", errors="replace")
    match = re.search(r"%PDF-(\d+\.\d+)", header)
    return match.group(1) if match else "0.0"


def version_at_least(actual: str, minimum: str) -> bool:
    def parts(value: str) -> tuple[int, ...]:
        return tuple(int(piece) for piece in value.split("."))

    return parts(actual) >= parts(minimum)


def sentence_count(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text.strip())
    return len(re.findall(r"(?:[.!?](?:[\"'”’)]*)\s+|[.!?](?:[\"'”’)]*)$)", normalized))


def video_metadata(path: Path) -> dict[str, Any]:
    try:
        import imageio_ffmpeg

        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(frames)
        frames.close()
        return {
            "codec": metadata.get("codec"),
            "duration_s": float(metadata.get("duration", 0.0)),
            "fps": float(metadata.get("fps", 0.0)),
            "source_size": list(metadata.get("source_size", (0, 0))),
        }
    except (ImportError, RuntimeError, OSError, StopIteration, ValueError):
        pass

    # The audit must also run in the lightweight paper-only Python environment.
    # Read the MP4 movie header directly when imageio-ffmpeg is not installed.
    data = path.read_bytes()
    mvhd = data.find(b"mvhd")
    if mvhd < 0 or mvhd + 32 > len(data):
        return {"error": "MP4 mvhd atom not found", "duration_s": 0.0}
    payload = mvhd + 4
    version = data[payload]
    if version == 0:
        timescale = struct.unpack(">I", data[payload + 12 : payload + 16])[0]
        duration = struct.unpack(">I", data[payload + 16 : payload + 20])[0]
    elif version == 1:
        timescale = struct.unpack(">I", data[payload + 20 : payload + 24])[0]
        duration = struct.unpack(">Q", data[payload + 24 : payload + 32])[0]
    else:
        return {"error": f"unsupported mvhd version {version}", "duration_s": 0.0}
    return {
        "codec": "h264" if b"avc1" in data or b"avc3" in data else "unknown",
        "duration_s": duration / timescale if timescale else 0.0,
        "fps": None,
        "source_size": None,
        "metadata_reader": "iso_bmff_mvhd_fallback",
    }


def manifest_audit(package_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    entries = manifest.get("files", [])
    results["manifest_has_files"] = bool(entries)
    all_exist = True
    hashes_match = True
    sizes_match = True
    for entry in entries:
        path = package_dir / entry.get("path", "")
        if not path.is_file():
            all_exist = False
            hashes_match = False
            sizes_match = False
            continue
        hashes_match &= sha256(path) == entry.get("sha256")
        sizes_match &= path.stat().st_size == entry.get("bytes")
    results["manifest_files_exist"] = all_exist
    results["manifest_hashes_match"] = hashes_match
    results["manifest_sizes_match"] = sizes_match
    return results


def audit_paper(
    key: str,
    paper: dict[str, Any],
    venue: dict[str, Any],
    base_audit: dict[str, Any],
) -> dict[str, Any]:
    source_path = ROOT / paper["source_tex"]
    release_pdf = ROOT / paper["release_pdf"]
    build_log = ROOT / paper["build_log"]
    package_dir = ROOT / paper["package_dir"]
    manifest_path = package_dir / "PACKAGE_MANIFEST.json"
    source = source_path.read_text(encoding="utf-8")
    abstract = extract_block(source, "abstract")
    ntp = extract_note_to_practitioners(source)
    keywords = keyword_list(source)
    abstract_words = latex_word_count(abstract)
    ntp_words = latex_word_count(ntp)

    from pypdf import PdfReader

    pages = len(PdfReader(str(release_pdf)).pages)
    fonts = _font_audit(release_pdf)
    log = _build_log_audit(build_log)
    version = pdf_version(release_pdf)
    manuscript = package_dir / "manuscript.pdf"
    cover_letter = package_dir / "COVER_LETTER.md"
    checklist = package_dir / "SUBMISSION_CHECKLIST.md"
    status = package_dir / "PACKAGE_STATUS.md"
    readme = package_dir / "multimedia/ReadMe.txt"
    summary = package_dir / "multimedia/Summary.txt"
    claim_boundary = package_dir / "multimedia/CLAIM_BOUNDARY.txt"
    video = package_dir / "multimedia" / paper["video"]["filename"]
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    manifest_checks = manifest_audit(package_dir, manifest)
    readme_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    summary_text = summary.read_text(encoding="utf-8") if summary.is_file() else ""
    cover_text = cover_letter.read_text(encoding="utf-8") if cover_letter.is_file() else ""
    checklist_text = checklist.read_text(encoding="utf-8") if checklist.is_file() else ""
    boundary_text = claim_boundary.read_text(encoding="utf-8") if claim_boundary.is_file() else ""
    video_meta = (
        video_metadata(video) if video.is_file() else {"duration_s": 0.0, "error": "missing"}
    )

    if paper["venue"] == "T-RO":
        page_limit = venue["first_submission_max_pages"]
    else:
        page_limit = venue["target_max_pages"]
    ntp_spec = venue["note_to_practitioners"]
    ntp_valid = (
        ntp_spec["min_words"] <= ntp_words <= ntp_spec["max_words"]
        if ntp_spec["required"]
        else ntp is None
    )
    summary_max_sentences = int(venue.get("multimedia_summary_max_sentences", 5))

    format_checks = {
        "documentclass_exact": venue["documentclass"] in source,
        "anonymous_author_exact": "\\author{Anonymous Authors}" in source,
        "abstract_present": abstract is not None,
        "abstract_within_limit": 1 <= abstract_words <= venue["abstract_max_words"],
        "note_to_practitioners_valid": ntp_valid,
        "keyword_count_valid": venue["keywords_min"] <= len(keywords) <= venue["keywords_max"],
        "page_count_within_target": 1 <= pages <= page_limit,
        "pdf_version_valid": version_at_least(version, venue["minimum_pdf_version"]),
        "letter_page_geometry": fonts["letter_page_geometry"],
        "consistent_page_geometry": fonts["consistent_page_geometry"],
        "all_fonts_embedded": fonts["all_fonts_embedded"],
        "build_log_no_latex_errors": log["no_latex_errors"],
        "build_log_no_unresolved_references": log["no_unresolved_references"],
        "build_log_no_overfull_boxes": log["no_overfull_boxes"],
    }
    package_checks = {
        "manifest_present": manifest_path.is_file(),
        **manifest_checks,
        "manuscript_present": manuscript.is_file(),
        "manuscript_matches_release": manuscript.is_file()
        and sha256(manuscript) == sha256(release_pdf),
        "anonymous_cover_letter_present": cover_letter.is_file()
        and paper["title"] in cover_text
        and "Anonymous Authors" in cover_text,
        "staging_status_explicit": status.is_file()
        and "Upload authorization: **NO**" in status.read_text(encoding="utf-8"),
        "checklist_present": checklist.is_file() and "STAGING ONLY" in checklist_text,
        "video_present": video.is_file() and video.stat().st_size > 100_000,
        "video_within_size_limit": video.is_file()
        and video.stat().st_size <= venue["multimedia_max_bytes"],
        "video_duration_valid": abs(
            float(video_meta.get("duration_s", 0.0)) - float(paper["video"]["duration_s"])
        )
        <= 0.20,
        "readme_present": readme.is_file(),
        "readme_claim_boundary": "not real-robot evidence" in readme_text.lower(),
        "summary_present": summary.is_file(),
        "summary_sentence_limit": 1 <= sentence_count(summary_text) <= summary_max_sentences,
        "summary_claim_boundary": "does not constitute hardware validation" in summary_text.lower(),
        "claim_boundary_present": claim_boundary.is_file() and "PROHIBITED CLAIMS" in boundary_text,
        "manifest_staging_state": manifest.get("package_state")
        in {"staging_only_not_for_submission", "submission_candidate"},
    }

    classification_checks: dict[str, bool] = {}
    if paper["venue"] == "T-ASE":
        methods = paper["methodology_codes"]
        applications = paper["application_codes"]
        classification_checks = {
            "primary_methodology_code_valid": methods["primary"] in TASE_METHOD_CODES,
            "secondary_methodology_code_valid": methods["secondary"] in TASE_METHOD_CODES,
            "primary_application_code_valid": applications["primary"] in TASE_APPLICATION_CODES,
            "secondary_application_code_valid": applications["secondary"] in TASE_APPLICATION_CODES,
            "classification_codes_in_cover_letter": all(
                value in cover_text for value in (*methods.values(), *applications.values())
            ),
        }

    venue_format_ready = all(
        (*format_checks.values(), *package_checks.values(), *classification_checks.values())
    )
    draft_warning_present = "DRAFT---NOT FOR SUBMISSION" in source
    hardware_passed = base_audit["hardware_gate"]["passed"]
    submission_ready = (
        venue_format_ready
        and base_audit["research_draft_ready"]
        and hardware_passed
        and not draft_warning_present
    )
    blockers: list[str] = []
    blockers.extend(name for name, passed in format_checks.items() if not passed)
    blockers.extend(name for name, passed in package_checks.items() if not passed)
    blockers.extend(name for name, passed in classification_checks.items() if not passed)
    if not base_audit["research_draft_ready"]:
        blockers.append("base_research_draft_audit_failed")
    if not hardware_passed:
        blockers.append("real_robot_hardware_gate_open")
    if draft_warning_present:
        blockers.append("draft_warning_intentionally_present")
    return {
        "key": key,
        "venue": paper["venue"],
        "paper_type": venue["paper_type"],
        "source": paper["source_tex"],
        "release_pdf": paper["release_pdf"],
        "package_dir": paper["package_dir"],
        "pages": pages,
        "page_limit": page_limit,
        "abstract_words": abstract_words,
        "abstract_limit": venue["abstract_max_words"],
        "note_to_practitioners_words": ntp_words,
        "keywords": keywords,
        "pdf_version": version,
        "video": {
            "path": str(video.relative_to(ROOT)).replace("\\", "/"),
            "bytes": video.stat().st_size if video.is_file() else 0,
            "metadata": video_meta,
            "summary_sentences": sentence_count(summary_text),
        },
        "format_checks": format_checks,
        "package_checks": package_checks,
        "classification_checks": classification_checks,
        "base_research_draft_ready": base_audit["research_draft_ready"],
        "hardware_gate_passed": hardware_passed,
        "draft_warning_present": draft_warning_present,
        "venue_format_ready": venue_format_ready,
        "submission_ready": submission_ready,
        "blockers": blockers,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HarnessSim4 venue package audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "| Paper | Venue | Pages | Abstract | NtP | Keywords | Venue format | Research draft | Hardware | Submission |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for paper in report["papers"]:
        ntp = (
            str(paper["note_to_practitioners_words"])
            if paper["note_to_practitioners_words"]
            else "n/a"
        )
        lines.append(
            f"| {paper['key']} | {paper['venue']} | {paper['pages']}/{paper['page_limit']} | "
            f"{paper['abstract_words']}/{paper['abstract_limit']} | {ntp} | "
            f"{len(paper['keywords'])} | {'PASS' if paper['venue_format_ready'] else 'FAIL'} | "
            f"{'PASS' if paper['base_research_draft_ready'] else 'FAIL'} | "
            f"{'PASS' if paper['hardware_gate_passed'] else 'OPEN'} | "
            f"{'YES' if paper['submission_ready'] else 'NO'} |"
        )
    lines.extend(("", "## Open gates", ""))
    for paper in report["papers"]:
        lines.append(f"### {paper['key']}")
        lines.extend(f"- `{item}`" for item in paper["blockers"])
        lines.append("")
    lines.extend(
        (
            "## Interpretation",
            "",
            (
                "`venue_format_ready` audits the anonymous IEEE layout and the staging "
                "package. `submission_ready` additionally requires a passing bound "
                "real-robot hardware gate and removal of the draft warning. The latter "
                "must remain false while hardware evidence is absent."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_by_key = {spec.key: _audit_paper(spec) for spec in PAPERS}
    papers = [
        audit_paper(key, paper, config["venues"][paper["venue"]], base_by_key[key])
        for key, paper in config["papers"].items()
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requirements_verified_on": config["verified_on"],
        "official_sources": config["sources"],
        "all_venue_format_ready": all(paper["venue_format_ready"] for paper in papers),
        "all_research_drafts_ready": all(paper["base_research_draft_ready"] for paper in papers),
        "all_submission_ready": all(paper["submission_ready"] for paper in papers),
        "papers": papers,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "venue_package_audit.json"
    markdown_path = output_dir / "VENUE_PACKAGE_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "all_venue_format_ready": report["all_venue_format_ready"],
                "all_research_drafts_ready": report["all_research_drafts_ready"],
                "all_submission_ready": report["all_submission_ready"],
                "venue_format_ready": sum(paper["venue_format_ready"] for paper in papers),
                "hardware_gates_passed": sum(paper["hardware_gate_passed"] for paper in papers),
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/submission/venue_profiles.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/submission_bundle",
    )
    parser.add_argument("--require-format-ready", action="store_true")
    parser.add_argument("--require-submission-ready", action="store_true")
    args = parser.parse_args()
    report = audit(args.config.resolve(), args.output.resolve())
    if args.require_format_ready and not report["all_venue_format_ready"]:
        return 1
    if args.require_submission_ready and not report["all_submission_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
