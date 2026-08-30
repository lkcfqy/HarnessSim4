#!/usr/bin/env python3
"""Audit the encoded HarnessSim4 showcase and render a decoded-frame storyboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for directory in (Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts")):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, passed: bool, name: str, observed: object = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "observed": observed})

    @property
    def passed(self) -> bool:
        return all(bool(item["passed"]) for item in self.checks)


def _decode_frame(capture: cv2.VideoCapture, time_s: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode showcase frame at {time_s:.3f} s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _storyboard(
    frames: list[np.ndarray],
    labels: list[str],
    output_path: Path,
) -> None:
    columns = 3
    tile_width = 640
    image_height = 360
    label_height = 58
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "#07101b")
    title_font = _font(21, bold=True)
    small_font = _font(15)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        image = Image.fromarray(frame).resize((tile_width, image_height), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x, y + image_height, x + tile_width, y + image_height + label_height), fill="#0c1725")
        draw.text((x + 18, y + image_height + 8), label, font=title_font, fill="#edf4fb")
        draw.text(
            (x + 18, y + image_height + 34),
            "decoded from final H.264 output",
            font=small_font,
            fill="#8fa5ba",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def audit_showcase(manifest_path: Path, output_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    video_path = root / manifest["video"]["path"]
    audit = Audit()
    audit.require(video_path.is_file(), "video exists", str(video_path))
    audit.require(
        _sha256(video_path) == manifest["video"]["sha256"],
        "video hash matches manifest",
    )
    audit.require(video_path.stat().st_size >= 10_000_000, "video is nontrivial")
    audit.require(
        manifest["claim_scope"].endswith("not hardware validation or factory performance."),
        "evidence boundary is explicit",
    )

    for filename, expected_hash in manifest["inputs"].items():
        source = root.parent / "mujoco_validation" / filename
        audit.require(
            source.is_file() and _sha256(source) == expected_hash,
            f"input hash {filename}",
        )
    for filename, expected_hash in manifest["outputs"].items():
        output = root / filename
        audit.require(
            output.is_file() and _sha256(output) == expected_hash,
            f"companion output hash {filename}",
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        audit.require(
            width == int(manifest["video"]["width"])
            and height == int(manifest["video"]["height"]),
            "decoded video dimensions",
            [width, height],
        )
        audit.require(abs(fps - float(manifest["video"]["fps"])) < 1e-6, "decoded frame rate", fps)
        audit.require(
            frame_count == int(manifest["video"]["frames"]),
            "decoded frame count",
            frame_count,
        )
        duration_s = frame_count / fps
        audit.require(
            abs(duration_s - float(manifest["video"]["duration_s"])) < 1e-6,
            "decoded duration",
            duration_s,
        )
        audit.require(fourcc.lower() in {"avc1", "h264"}, "decoded H.264 codec", fourcc)

        sample_times: list[float] = []
        labels: list[str] = []
        chapters = manifest["chapters"]
        for index, chapter in enumerate(chapters):
            start = float(chapter["start_s"])
            end = (
                float(chapters[index + 1]["start_s"])
                if index + 1 < len(chapters)
                else duration_s
            )
            sample_times.append(min(end - 0.05, start + max(0.25, 0.45 * (end - start))))
            labels.append(f"{chapter['label']} | {sample_times[-1]:.1f} s")
        frames = [_decode_frame(capture, time_s) for time_s in sample_times]
    finally:
        capture.release()

    audit.require(
        all(frame.shape == (height, width, 3) for frame in frames),
        "all chapter samples decode at full resolution",
    )
    audit.require(
        all(np.isfinite(frame).all() and float(frame.std()) >= 20.0 for frame in frames),
        "all chapter samples are finite and nonblank",
    )
    differences = [
        float(np.mean(np.abs(left.astype(float) - right.astype(float))))
        for left, right in pairwise(frames)
    ]
    audit.require(
        all(value >= 5.0 for value in differences),
        "chapter samples are visually distinct",
        differences,
    )
    _storyboard(frames, labels, output_path)
    audit.require(output_path.stat().st_size >= 500_000, "decoded storyboard is nontrivial")

    html_path = root / "harnesssim4_four_robot_showcase.html"
    html = html_path.read_text(encoding="utf-8")
    audit.require(
        manifest["video"]["path"] in html and "data-time=" in html and "<video" in html,
        "interactive player references the audited video",
    )
    audit.require(
        html.count("button type=\"button\" data-time=") == len(manifest["chapters"]),
        "one seek control per chapter",
    )
    vtt_path = root / "harnesssim4_four_robot_chapters.vtt"
    vtt = vtt_path.read_text(encoding="utf-8")
    audit.require(
        vtt.startswith("WEBVTT")
        and all(str(chapter["label"]) in vtt for chapter in manifest["chapters"]),
        "chapter track covers all sections",
    )

    result = {
        "name": "HarnessSim4 showcase independent media audit",
        "passed": audit.passed,
        "checks": audit.checks,
        "decoded": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration_s": duration_s,
            "fourcc": fourcc,
            "sample_times_s": sample_times,
        },
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "video_sha256": _sha256(video_path),
        },
        "storyboard": {
            "path": output_path.name,
            "sha256": _sha256(output_path),
            "bytes": output_path.stat().st_size,
        },
    }
    report_path = root / "harnesssim4_four_robot_media_audit.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(audit.checks),
                "report": str(report_path.resolve()),
                "storyboard": str(output_path.resolve()),
            },
            indent=2,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/showcase/harnesssim4_four_robot_manifest.json"),
    )
    parser.add_argument(
        "--storyboard",
        type=Path,
        default=Path("artifacts/showcase/harnesssim4_four_robot_storyboard.png"),
    )
    args = parser.parse_args()
    result = audit_showcase(args.manifest, args.storyboard)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
