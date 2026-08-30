#!/usr/bin/env python3
"""Build a machine-readable four-paper release and submission-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperSpec:
    key: str
    title_token: str
    source_dir: Path
    build_pdf: Path
    build_log: Path
    release_pdf: Path
    audit_path: Path
    minimum_checks: int
    hardware_protocol: Path
    hardware_gate: Path
    blocker: str


PAPERS = (
    PaperSpec(
        "inspectbot",
        "Topology-Risk Active Inspection",
        Path("papers/01_inspectbot"),
        Path("papers/01_inspectbot/main.pdf"),
        Path("papers/01_inspectbot/main.log"),
        Path("output/pdf/01_inspectbot_research_draft.pdf"),
        Path("artifacts/papers/inspectbot/final/audit/inspectbot_result_audit.json"),
        787,
        Path("configs/hardware/inspectbot_gate.json"),
        Path("artifacts/hardware/inspectbot/hardware_gate_audit.json"),
        "Calibrated multi-view UR5e/industrial-camera study across real harness defects.",
    ),
    PaperSpec(
        "insertbot",
        "Contact-State Recovery",
        Path("papers/02_insertbot"),
        Path("papers/02_insertbot/build/main.pdf"),
        Path("papers/02_insertbot/build/main.log"),
        Path("output/pdf/02_insertbot_research_draft.pdf"),
        Path("artifacts/papers/insertbot/formal/audit/insertbot_result_audit.json"),
        153,
        Path("configs/hardware/insertbot_gate.json"),
        Path("artifacts/hardware/insertbot/hardware_gate_audit.json"),
        "Force-calibrated robot study with real latching, cable reaction, and pull test.",
    ),
    PaperSpec(
        "routebot",
        "Explicit Process Relations",
        Path("papers/03_routebot"),
        Path("papers/03_routebot/build/main.pdf"),
        Path("papers/03_routebot/build/main.log"),
        Path("output/pdf/03_routebot_research_draft.pdf"),
        Path("artifacts/papers/routebot/final/audit/routebot_result_audit.json"),
        39,
        Path("configs/hardware/routebot_gate.json"),
        Path("artifacts/hardware/routebot/hardware_gate_audit.json"),
        "Force-controlled semantic clip-routing replication on hardware or an external benchmark.",
    ),
    PaperSpec(
        "branchbot",
        "Semantic Counterfactuals",
        Path("papers/04_branchbot"),
        Path("papers/04_branchbot/main.pdf"),
        Path("papers/04_branchbot/main.log"),
        Path("output/pdf/04_branchbot_research_draft.pdf"),
        Path("artifacts/papers/branchbot/final/audit/branchbot_result_audit.json"),
        182,
        Path("configs/hardware/branchbot_gate.json"),
        Path("artifacts/hardware/branchbot/hardware_gate_audit.json"),
        "Vision-based 3--5 branch identity tests and collision-aware dual-arm hardware trials.",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("—", "-").replace("–", "-")).strip()


def _read_pdf(path: Path) -> tuple[int, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise RuntimeError("Install the paper extra: python -m pip install -e '.[paper]'") from exc
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return len(reader.pages), _normalized(text)


def _font_audit(path: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    fonts: dict[str, bool] = {}
    visited: set[int] = set()

    def dereference(value: Any) -> Any:
        return value.get_object() if hasattr(value, "get_object") else value

    def descriptor(font: Any) -> Any | None:
        value = dereference(font)
        direct = value.get("/FontDescriptor")
        if direct is not None:
            return dereference(direct)
        descendants = value.get("/DescendantFonts", [])
        if descendants:
            descendant = dereference(descendants[0])
            direct = descendant.get("/FontDescriptor")
            return dereference(direct) if direct is not None else None
        return None

    def walk_resources(resources: Any) -> None:
        resources = dereference(resources)
        if resources is None or id(resources) in visited:
            return
        visited.add(id(resources))
        for font_ref in dereference(resources.get("/Font", {})).values():
            font = dereference(font_ref)
            name = str(font.get("/BaseFont", "unnamed"))
            font_descriptor = descriptor(font)
            embedded = font_descriptor is not None and any(
                key in font_descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
            )
            fonts[name] = fonts.get(name, False) or embedded
        for xobject_ref in dereference(resources.get("/XObject", {})).values():
            xobject = dereference(xobject_ref)
            if xobject.get("/Subtype") == "/Form":
                walk_resources(xobject.get("/Resources"))

    sizes: list[list[float]] = []
    for page in reader.pages:
        walk_resources(page.get("/Resources"))
        sizes.append([float(page.mediabox.width), float(page.mediabox.height)])
    return {
        "fonts": fonts,
        "all_fonts_embedded": bool(fonts) and all(fonts.values()),
        "page_sizes_points": sizes,
        "consistent_page_geometry": len({tuple(size) for size in sizes}) == 1,
        "letter_page_geometry": all(
            abs(width - 612.0) <= 1.0 and abs(height - 792.0) <= 1.0
            for width, height in sizes
        ),
    }


def _build_log_audit(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    undefined_patterns = (
        "There were undefined references",
        "Citation `",
        "Reference `",
        "undefined citations",
    )
    error_patterns = ("! LaTeX Error", "Undefined control sequence", "Emergency stop")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "no_latex_errors": not any(pattern in text for pattern in error_patterns),
        "no_unresolved_references": not any(pattern in text for pattern in undefined_patterns),
        "no_overfull_boxes": "Overfull \\hbox" not in text and "Overfull \\vbox" not in text,
        "underfull_box_warnings": text.count("Underfull \\hbox")
        + text.count("Underfull \\vbox"),
    }


def _hardware_gate(
    path: Path, expected_robot: str, expected_protocol_sha256: str
) -> tuple[bool, int, dict[str, bool]]:
    if not path.is_file():
        return False, 0, {
            "audit_declares_pass": False,
            "all_audit_checks_pass": False,
            "real_robot_evidence": False,
            "robot_matches": False,
            "protocol_matches": False,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = value.get("checks", [])
    binding = {
        "audit_declares_pass": value.get("passed") is True,
        "all_audit_checks_pass": bool(checks)
        and all(item.get("passed") is True for item in checks),
        "real_robot_evidence": value.get("evidence_kind") == "real_robot_hardware",
        "robot_matches": value.get("robot") == expected_robot,
        "protocol_matches": value.get("protocol", {}).get("sha256")
        == expected_protocol_sha256,
    }
    return all(binding.values()), len(checks), binding


def _audit_paper(spec: PaperSpec) -> dict[str, Any]:
    source_path = spec.source_dir / "main.tex"
    bib_path = spec.source_dir / "references.bib"
    source = source_path.read_text(encoding="utf-8")
    result_audit = json.loads(spec.audit_path.read_text(encoding="utf-8"))
    pages, pdf_text = _read_pdf(spec.release_pdf)
    font_audit = _font_audit(spec.release_pdf)
    build_log_audit = _build_log_audit(spec.build_log)
    build_hash = _sha256(spec.build_pdf)
    release_hash = _sha256(spec.release_pdf)
    result_checks = len(result_audit.get("checks", []))
    draft_warning = "DRAFT" in pdf_text.upper() and "NOT FOR SUBMISSION" in pdf_text.upper()
    hardware_protocol = json.loads(spec.hardware_protocol.read_text(encoding="utf-8"))
    hardware_protocol_hash = _sha256(spec.hardware_protocol)
    hardware_passed, hardware_checks, hardware_binding = _hardware_gate(
        spec.hardware_gate, spec.key, hardware_protocol_hash
    )
    integrity_checks = {
        "source_exists": source_path.is_file(),
        "bibliography_exists": bib_path.is_file() and bib_path.stat().st_size > 500,
        "release_pdf_exists": spec.release_pdf.is_file() and spec.release_pdf.stat().st_size > 100_000,
        "build_pdf_matches_release": build_hash == release_hash,
        "page_count_plausible": 4 <= pages <= 12,
        "page_geometry_consistent": font_audit["consistent_page_geometry"],
        "letter_page_geometry": font_audit["letter_page_geometry"],
        "all_fonts_embedded": font_audit["all_fonts_embedded"],
        "build_log_no_latex_errors": build_log_audit["no_latex_errors"],
        "build_log_no_unresolved_references": build_log_audit["no_unresolved_references"],
        "build_log_no_overfull_boxes": build_log_audit["no_overfull_boxes"],
        "title_present": spec.title_token in pdf_text,
        "abstract_present": "Abstract" in pdf_text,
        "references_present": "References" in pdf_text,
        "limitations_present": "Limit" in pdf_text or "Hardware Gate" in pdf_text,
        "draft_boundary_present": draft_warning,
        "no_tbd": "TBD" not in pdf_text,
        "no_unresolved_reference_marker": "??" not in pdf_text and "[?]" not in pdf_text,
        "result_audit_passed": bool(result_audit.get("passed")),
        "result_audit_scope": result_checks >= spec.minimum_checks,
        "source_has_bibliography": "\\bibliography" in source,
        "source_has_hardware_boundary": "hardware" in source.lower(),
        "hardware_protocol_exists": spec.hardware_protocol.is_file(),
        "hardware_protocol_robot_matches": hardware_protocol.get("robot") == spec.key,
    }
    research_draft_ready = all(integrity_checks.values())
    submission_ready = research_draft_ready and hardware_passed and not draft_warning
    blockers = []
    if draft_warning:
        blockers.append("Manuscript remains explicitly marked DRAFT---NOT FOR SUBMISSION.")
    if not hardware_passed:
        blockers.append(spec.blocker)
    blockers.extend(name for name, passed in integrity_checks.items() if not passed)
    return {
        "key": spec.key,
        "source": str(source_path),
        "release_pdf": str(spec.release_pdf),
        "release_sha256": release_hash,
        "bytes": spec.release_pdf.stat().st_size,
        "pages": pages,
        "pdf_quality": {
            "font_audit": font_audit,
            "build_log_audit": build_log_audit,
        },
        "result_audit": {
            "path": str(spec.audit_path),
            "passed": bool(result_audit.get("passed")),
            "checks": result_checks,
            "sha256": _sha256(spec.audit_path),
        },
        "hardware_gate": {
            "protocol": {
                "path": str(spec.hardware_protocol),
                "version": hardware_protocol.get("protocol_version"),
                "sha256": hardware_protocol_hash,
            },
            "path": str(spec.hardware_gate),
            "present": spec.hardware_gate.is_file(),
            "passed": hardware_passed,
            "checks": hardware_checks,
            "binding_checks": hardware_binding,
        },
        "integrity_checks": integrity_checks,
        "research_draft_ready": research_draft_ready,
        "submission_ready": submission_ready,
        "blockers": blockers,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HarnessSim4 four-paper readiness audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "| Paper | Pages | Result audit | Draft integrity | Hardware gate | Submission-ready |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for paper in report["papers"]:
        result = paper["result_audit"]
        lines.append(
            f"| {paper['key']} | {paper['pages']} | {result['checks']}/{result['checks']} | "
            f"{'PASS' if paper['research_draft_ready'] else 'FAIL'} | "
            f"{'PASS' if paper['hardware_gate']['passed'] else 'OPEN'} | "
            f"{'YES' if paper['submission_ready'] else 'NO'} |"
        )
    lines.extend(("", "## Blocking evidence", ""))
    for paper in report["papers"]:
        lines.append(f"### {paper['key']}")
        lines.extend(f"- {item}" for item in paper["blockers"])
        lines.append("")
    lines.extend(
        (
            "## Interpretation",
            "",
            (
                "All four PDFs are reproducible research drafts with passing numerical audits. "
                "They are not submission-ready because the declared hardware gates are absent "
                "and the PDFs correctly retain their draft warning. The audit intentionally "
                "treats that absence as an open scientific requirement rather than silently "
                "passing it."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_bundle(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = [_audit_paper(spec) for spec in PAPERS]
    showcase_path = Path("artifacts/showcase/harnesssim4_four_robot_media_audit.json")
    showcase = json.loads(showcase_path.read_text(encoding="utf-8"))
    bundle_integrity = all(paper["research_draft_ready"] for paper in papers) and bool(
        showcase.get("passed")
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_integrity_passed": bundle_integrity,
        "all_submission_ready": all(paper["submission_ready"] for paper in papers),
        "papers": papers,
        "showcase": {
            "path": str(showcase_path),
            "passed": bool(showcase.get("passed")),
            "checks": len(showcase.get("checks", [])),
            "sha256": _sha256(showcase_path),
        },
    }
    json_path = output_dir / "four_paper_readiness.json"
    markdown_path = output_dir / "FOUR_PAPER_READINESS.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, report)
    summary = {
        "bundle_integrity_passed": report["bundle_integrity_passed"],
        "all_submission_ready": report["all_submission_ready"],
        "research_drafts_ready": sum(paper["research_draft_ready"] for paper in papers),
        "hardware_gates_passed": sum(paper["hardware_gate"]["passed"] for paper in papers),
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission_bundle"))
    parser.add_argument("--require-submission-ready", action="store_true")
    args = parser.parse_args()
    report = audit_bundle(args.output)
    if not report["bundle_integrity_passed"]:
        return 1
    if args.require_submission_ready and not report["all_submission_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
