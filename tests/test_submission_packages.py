from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_venue_packages import (
    extract_block,
    extract_note_to_practitioners,
    keyword_list,
    latex_word_count,
    sentence_count,
    sha256,
    video_metadata,
)


def load_config() -> dict:
    return json.loads((ROOT / "configs/submission/venue_profiles.json").read_text(encoding="utf-8"))


def test_all_sources_satisfy_frozen_venue_text_limits() -> None:
    config = load_config()
    for paper in config["papers"].values():
        venue = config["venues"][paper["venue"]]
        source = (ROOT / paper["source_tex"]).read_text(encoding="utf-8")
        assert venue["documentclass"] in source
        assert "\\author{Anonymous Authors}" in source
        assert 1 <= latex_word_count(extract_block(source, "abstract")) <= 200
        assert 2 <= len(keyword_list(source)) <= 5
        ntp = extract_note_to_practitioners(source)
        if venue["note_to_practitioners"]["required"]:
            assert 100 <= latex_word_count(ntp) <= 300
        else:
            assert ntp is None


def test_staging_manifests_bind_every_packaged_file() -> None:
    config = load_config()
    for key, paper in config["papers"].items():
        package_dir = ROOT / paper["package_dir"]
        manifest = json.loads((package_dir / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["paper"] == key
        assert manifest["package_state"] == "staging_only_not_for_submission"
        assert manifest["draft_warning_present"] is True
        assert manifest["hardware_gate"]["passed"] is False
        assert manifest["files"]
        for item in manifest["files"]:
            path = package_dir / item["path"]
            assert path.stat().st_size == item["bytes"]
            assert sha256(path) == item["sha256"]


def test_multimedia_is_short_h264_and_claim_bounded() -> None:
    config = load_config()
    for paper in config["papers"].values():
        venue = config["venues"][paper["venue"]]
        package_dir = ROOT / paper["package_dir"]
        video = package_dir / "multimedia" / paper["video"]["filename"]
        metadata = video_metadata(video)
        assert metadata["codec"] == "h264"
        assert abs(metadata["duration_s"] - paper["video"]["duration_s"]) <= 0.20
        assert video.stat().st_size <= venue["multimedia_max_bytes"]
        summary = (package_dir / "multimedia/Summary.txt").read_text(encoding="utf-8")
        readme = (package_dir / "multimedia/ReadMe.txt").read_text(encoding="utf-8")
        assert 1 <= sentence_count(summary) <= 5
        assert "does not constitute hardware validation" in summary.lower()
        assert "not real-robot evidence" in readme.lower()


def test_requirement_sources_are_official_https_pages() -> None:
    config = load_config()
    assert config["verified_on"] == "2026-08-27"
    assert config["sources"]
    for url in config["sources"].values():
        assert url.startswith("https://www.ieee-ras.org/")
