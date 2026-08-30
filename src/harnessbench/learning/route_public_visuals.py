"""Generate an attributed four-view figure from the public RouteBot data."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from harnessbench.route_public_data import REVISION

VIEWS = (
    ("Side / global", "observation.images.image"),
    ("Top / global", "observation.images.top_image"),
    ("Wrist 225 deg", "observation.images.wrist225_image"),
    ("Wrist 45 deg", "observation.images.wrist45_image"),
)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_frame(video: Path, timestamp_seconds: float) -> Image.Image:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install the project 'visual' extra to decode public videos") from exc
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, timeout=120)
    if not result.stdout:
        raise RuntimeError(f"FFmpeg returned no frame for {video}")
    with Image.open(io.BytesIO(result.stdout)) as frame:
        return frame.convert("RGB")


def compose_multiview(
    labeled_frames: list[tuple[str, Image.Image]],
    *,
    footer: str,
    panel_size: int = 384,
) -> Image.Image:
    if len(labeled_frames) != 4:
        raise ValueError("Exactly four labeled frames are required")
    margin = 18
    label_height = 42
    footer_height = 48
    panel_total = panel_size + label_height
    width = margin * 3 + panel_size * 2
    height = margin * 3 + panel_total * 2 + footer_height
    sheet = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(sheet)
    label_font = _font(24)
    footer_font = _font(17)
    for index, (label, frame) in enumerate(labeled_frames):
        row, column = divmod(index, 2)
        x = margin + column * (panel_size + margin)
        y = margin + row * (panel_total + margin)
        draw.rounded_rectangle(
            (x, y, x + panel_size, y + panel_total),
            radius=8,
            fill="white",
            outline="#aab4c3",
            width=2,
        )
        resized = frame.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x, y + label_height))
        draw.text((x + 12, y + 8), label, fill="#152236", font=label_font)
    footer_y = height - footer_height
    draw.text((margin, footer_y + 12), footer, fill="#42516a", font=footer_font)
    return sheet


def render_route_public_multiview(
    data_dir: Path,
    output_dir: Path,
    *,
    timestamp_seconds: float = 120.0,
) -> dict:
    if timestamp_seconds < 0:
        raise ValueError("Timestamp must be non-negative")
    labeled_frames: list[tuple[str, Image.Image]] = []
    source_records = []
    for label, key in VIEWS:
        video = data_dir / "videos" / key / "chunk-000" / "file-000.mp4"
        if not video.exists():
            raise FileNotFoundError(
                f"Missing {video}; rerun sim-download-route-public --include-videos"
            )
        labeled_frames.append((label, _extract_frame(video, timestamp_seconds)))
        source_records.append(
            {
                "view": key,
                "path": str(video.resolve()),
                "bytes": video.stat().st_size,
                "sha256": _sha256(video),
            }
        )

    footer = (
        f"Berkeley Cable Routing, CC BY 4.0 | pinned {REVISION[:9]}... | "
        f"aligned t={timestamp_seconds:.1f}s"
    )
    sheet = compose_multiview(labeled_frames, footer=footer)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "berkeley_real_multiview.png"
    sheet.save(image_path, format="PNG", optimize=True)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": REVISION,
        "timestamp_seconds": timestamp_seconds,
        "license": "CC-BY-4.0",
        "sources": source_records,
        "output": {
            "path": str(image_path.resolve()),
            "width": sheet.width,
            "height": sheet.height,
            "sha256": _sha256(image_path),
        },
    }
    (output_dir / "berkeley_real_multiview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
