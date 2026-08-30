#!/usr/bin/env python3
"""Compose the four MuJoCo robot replays into one cinematic evidence video."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageSequence


@dataclass(frozen=True)
class TaskSpec:
    key: str
    chapter: str
    robot: str
    process: str
    evidence: str


TASKS = (
    TaskSpec(
        "inspect",
        "01 INSPECT",
        "UR5e + dual-lens inspection head",
        "Active scan and anomaly localization",
        "Public-image audit + direct MuJoCo scan",
    ),
    TaskSpec(
        "insert",
        "02 INSERT",
        "KUKA LBR iiwa 14 + Robotiq 2F-85",
        "Contact-aware terminal insertion",
        "Paired force-safety proxy + native-contact audit",
    ),
    TaskSpec(
        "route",
        "03 ROUTE",
        "UR10e + Robotiq 2F-85",
        "Ordered clip routing",
        "Public trajectories + semantic counterfactuals",
    ),
    TaskSpec(
        "branch",
        "04 BRANCH",
        "Dual UR10e + dual Robotiq 2F-85",
        "Branch identity, separation, and placement",
        "A/B semantics + direct dual-arm MuJoCo audit",
    ),
)

CLAIM_SCOPE = (
    "MuJoCo digital-twin visualization and scripted trajectory sanity check; "
    "not hardware validation or factory performance."
)


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
    paths = (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("C:/Windows/Fonts"),
    )
    for directory in paths:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    canvas_width: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text(((canvas_width - width) // 2, y), text, font=font, fill=fill)


def _load_gif(path: Path) -> tuple[list[Image.Image], int]:
    with Image.open(path) as image:
        duration_ms = int(image.info.get("duration", 80))
        frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames, duration_ms


def _fit(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    if frame.size == size:
        return frame.copy()
    return frame.resize(size, Image.Resampling.LANCZOS)


def _sample_index(index: int, count: int, source_count: int) -> int:
    if count <= 1 or source_count <= 1:
        return 0
    return min(source_count - 1, round(index * (source_count - 1) / (count - 1)))


def _task_frame(
    frame: Image.Image,
    spec: TaskSpec,
    size: tuple[int, int],
) -> Image.Image:
    image = _fit(frame, size).convert("RGBA")
    width, height = size
    draw = ImageDraw.Draw(image, "RGBA")
    label_height = max(50, round(height * 0.062))
    y0 = height - max(92, round(height * 0.105))
    draw.rounded_rectangle(
        (round(width * 0.022), y0, round(width * 0.69), y0 + label_height),
        radius=max(8, round(height * 0.011)),
        fill=(6, 14, 24, 218),
    )
    draw.rectangle(
        (
            round(width * 0.022),
            y0,
            round(width * 0.027),
            y0 + label_height,
        ),
        fill=(26, 211, 157, 255),
    )
    font = _font(max(18, round(height * 0.021)), bold=True)
    small = _font(max(14, round(height * 0.016)))
    draw.text(
        (round(width * 0.039), y0 + round(label_height * 0.13)),
        spec.chapter,
        font=font,
        fill=(242, 247, 252, 255),
    )
    draw.text(
        (round(width * 0.17), y0 + round(label_height * 0.19)),
        spec.process,
        font=small,
        fill=(191, 205, 220, 255),
    )
    warning = "MUJOCO DIGITAL TWIN | NOT HARDWARE VALIDATION"
    warning_font = _font(max(12, round(height * 0.013)), bold=True)
    warning_box = draw.textbbox((0, 0), warning, font=warning_font)
    warning_width = warning_box[2] - warning_box[0]
    draw.text(
        (width - warning_width - round(width * 0.022), y0 + round(label_height * 0.25)),
        warning,
        font=warning_font,
        fill=(255, 208, 92, 255),
    )
    return image.convert("RGB")


def _grid_frame(
    task_frames: dict[str, Image.Image],
    size: tuple[int, int],
) -> Image.Image:
    width, height = size
    half = (width // 2, height // 2)
    canvas = Image.new("RGB", size, (7, 13, 22))
    positions = ((0, 0), (half[0], 0), (0, half[1]), (half[0], half[1]))
    for spec, position in zip(TASKS, positions):
        canvas.paste(_fit(task_frames[spec.key], half), position)
    draw = ImageDraw.Draw(canvas)
    gutter = max(3, round(min(size) * 0.004))
    draw.rectangle((half[0] - gutter, 0, half[0] + gutter, height), fill=(7, 13, 22))
    draw.rectangle((0, half[1] - gutter, width, half[1] + gutter), fill=(7, 13, 22))
    return canvas


def _intro_frame(contact_sheet: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = _fit(contact_sheet, size).filter(ImageFilter.GaussianBlur(radius=4)).convert("RGBA")
    overlay = Image.new("RGBA", size, (4, 10, 18, 174))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image, "RGBA")
    title = _font(max(48, round(height * 0.072)), bold=True)
    subtitle = _font(max(25, round(height * 0.032)))
    process = _font(max(17, round(height * 0.021)), bold=True)
    small = _font(max(13, round(height * 0.015)))
    _centered_text(draw, width, round(height * 0.36), "HARNESS SIM 4", title, (245, 249, 253, 255))
    _centered_text(
        draw,
        width,
        round(height * 0.46),
        "Four robotic cells for wire-harness assembly",
        subtitle,
        (202, 215, 229, 255),
    )
    _centered_text(
        draw,
        width,
        round(height * 0.535),
        "INSPECT  ->  INSERT  ->  ROUTE  ->  BRANCH",
        process,
        (45, 220, 168, 255),
    )
    _centered_text(
        draw,
        width,
        round(height * 0.61),
        "REALISTIC MUJOCO DIGITAL TWINS | RESEARCH EVIDENCE, NOT FACTORY VALIDATION",
        small,
        (255, 210, 103, 255),
    )
    return image.convert("RGB")


def _outro_frame(grid: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = _fit(grid, size).convert("RGBA")
    shade = Image.new("RGBA", size, (3, 8, 15, 120))
    image = Image.alpha_composite(image, shade)
    draw = ImageDraw.Draw(image, "RGBA")
    title = _font(max(35, round(height * 0.048)), bold=True)
    subtitle = _font(max(18, round(height * 0.024)))
    _centered_text(
        draw,
        width,
        round(height * 0.44),
        "ONE WIRE-HARNESS RESEARCH PIPELINE",
        title,
        (245, 249, 253, 255),
    )
    _centered_text(
        draw,
        width,
        round(height * 0.52),
        "Four auditable projects | hardware validation remains the final gate",
        subtitle,
        (255, 214, 115, 255),
    )
    return image.convert("RGB")


def _blend(first: Image.Image, second: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(first.convert("RGB"), second.convert("RGB"), min(max(alpha, 0.0), 1.0))


def _video_frames(
    task_frames: dict[str, list[Image.Image]],
    durations_ms: dict[str, int],
    contact_sheet: Image.Image,
    *,
    size: tuple[int, int],
    fps: int,
) -> tuple[Iterator[Image.Image], list[dict[str, object]]]:
    transition_count = max(6, round(0.5 * fps))
    intro_count = round(2.4 * fps)
    outro_count = round(2.2 * fps)
    task_counts = {
        key: max(1, round(len(frames) * durations_ms[key] / 1000.0 * fps))
        for key, frames in task_frames.items()
    }
    grid_count = max(task_counts.values())
    chapters: list[dict[str, object]] = []

    def generate() -> Iterator[Image.Image]:
        elapsed = 0
        intro = _intro_frame(contact_sheet, size)
        black = Image.new("RGB", size, (3, 8, 15))
        chapters.append({"label": "Overview", "start_s": 0.0})
        for index in range(intro_count):
            fade = min(1.0, (index + 1) / max(1, round(0.6 * fps)))
            yield _blend(black, intro, fade)
            elapsed += 1

        previous = intro
        for spec in TASKS:
            source = task_frames[spec.key]
            first = _task_frame(source[0], spec, size)
            for index in range(transition_count):
                yield _blend(previous, first, (index + 1) / transition_count)
                elapsed += 1
            chapters.append({"label": spec.chapter.title(), "start_s": elapsed / fps})
            count = task_counts[spec.key]
            for index in range(count):
                source_index = _sample_index(index, count, len(source))
                previous = _task_frame(source[source_index], spec, size)
                yield previous
                elapsed += 1

        first_grid = _grid_frame({key: frames[0] for key, frames in task_frames.items()}, size)
        for index in range(transition_count):
            yield _blend(previous, first_grid, (index + 1) / transition_count)
            elapsed += 1
        chapters.append({"label": "Synchronized four-cell view", "start_s": elapsed / fps})
        grid = first_grid
        for index in range(grid_count):
            current: dict[str, Image.Image] = {}
            for key, frames in task_frames.items():
                source_index = _sample_index(index, grid_count, len(frames))
                current[key] = frames[source_index]
            grid = _grid_frame(current, size)
            yield grid
            elapsed += 1

        outro = _outro_frame(grid, size)
        for index in range(outro_count):
            alpha = min(1.0, (index + 1) / max(1, round(0.5 * fps)))
            yield _blend(grid, outro, alpha)

    return generate(), chapters


def _timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _write_vtt(path: Path, chapters: list[dict[str, object]], duration_s: float) -> None:
    lines = ["WEBVTT", ""]
    for index, chapter in enumerate(chapters):
        start = float(chapter["start_s"])
        end = (
            float(chapters[index + 1]["start_s"])
            if index + 1 < len(chapters)
            else duration_s
        )
        lines.extend(
            (
                f"{_timestamp(start)} --> {_timestamp(end)}",
                str(chapter["label"]),
                "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(
    path: Path,
    *,
    video_name: str,
    poster_name: str,
    vtt_name: str,
    chapters: list[dict[str, object]],
) -> None:
    chapter_buttons = "\n".join(
        (
            f'<button type="button" data-time="{float(item["start_s"]):.3f}">'
            f'{item["label"]}</button>'
        )
        for item in chapters
    )
    cards = "\n".join(
        f"""
        <article class="robot-card">
          <img src="../mujoco_validation/{spec.key}_mujoco_final.png"
               alt="{spec.chapter.title()} realistic MuJoCo final frame">
          <div class="robot-copy">
            <h2>{spec.chapter.title()}</h2>
            <p>{spec.process}</p>
            <small>{spec.robot}<br>{spec.evidence}</small>
          </div>
        </article>"""
        for spec in TASKS
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HarnessSim4 four-robot showcase</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #07101b; color: #edf4fb; }}
    main {{ width: min(1440px, 100%); margin: 0 auto; padding: 28px; }}
    header {{ display: flex; gap: 20px; align-items: end; justify-content: space-between; flex-wrap: wrap; }}
    h1 {{ margin: 0; font-size: clamp(30px, 5vw, 58px); font-weight: 600; letter-spacing: .04em; }}
    header p {{ max-width: 620px; margin: 8px 0 0; color: #aebfd0; }}
    .scope {{ color: #ffd170; font-size: 13px; max-width: 420px; text-align: right; }}
    .player {{ margin-top: 24px; background: #02070d; border: 1px solid #213044; border-radius: 18px; overflow: hidden; box-shadow: 0 24px 80px #0008; }}
    video {{ display: block; width: 100%; aspect-ratio: 16 / 9; background: #02070d; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 14px; border-top: 1px solid #213044; }}
    button {{ border: 1px solid #31455d; border-radius: 999px; color: #dfeaf5; background: #101c2a; padding: 9px 13px; cursor: pointer; }}
    button:hover, button:focus-visible {{ background: #193249; border-color: #25d8a6; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 22px; }}
    .robot-card {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(190px, .75fr); background: #0e1927; border: 1px solid #213044; border-radius: 16px; overflow: hidden; }}
    .robot-card img {{ width: 100%; height: 100%; min-height: 220px; object-fit: cover; }}
    .robot-copy {{ padding: 18px; align-self: center; }}
    .robot-copy h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .robot-copy p {{ margin: 0 0 14px; color: #31dca9; }}
    .robot-copy small {{ color: #9cb0c4; line-height: 1.55; }}
    footer {{ color: #71879c; font-size: 12px; margin: 20px 0 6px; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }} .grid {{ grid-template-columns: 1fr; }}
      .scope {{ text-align: left; }}
    }}
    @media (max-width: 560px) {{
      .robot-card {{ grid-template-columns: 1fr; }} .robot-card img {{ min-height: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>HARNESS SIM 4</h1><p>Inspect, insert, route, and branch handling in one synchronized wire-harness digital-twin film.</p></div>
      <div class="scope">{CLAIM_SCOPE}</div>
    </header>
    <section class="player" aria-label="Four-robot showcase player">
      <video controls preload="metadata" poster="{poster_name}">
        <source src="{video_name}" type="video/mp4">
        <track kind="chapters" src="{vtt_name}" srclang="en" label="Workflow chapters" default>
      </video>
      <nav aria-label="Video chapters">{chapter_buttons}</nav>
    </section>
    <section class="grid" aria-label="Robot cells">{cards}</section>
    <footer>All rendered robots are visual or task-space MuJoCo twins. See each research draft for evidence and hardware gates.</footer>
  </main>
  <script>
    const video = document.querySelector('video');
    for (const button of document.querySelectorAll('button[data-time]')) {{
      button.addEventListener('click', () => {{
        video.currentTime = Number(button.dataset.time);
        video.play();
      }});
    }}
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def render_showcase(
    input_dir: Path,
    output_dir: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> dict[str, object]:
    if width % 2 or height % 2:
        raise ValueError("Video width and height must be even for yuv420p output")
    size = (width, height)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {spec.key: input_dir / f"{spec.key}_mujoco.gif" for spec in TASKS}
    contact_path = input_dir / "harnesssim4_mujoco_contact_sheet.png"
    task_frames: dict[str, list[Image.Image]] = {}
    durations_ms: dict[str, int] = {}
    for key, path in input_paths.items():
        task_frames[key], durations_ms[key] = _load_gif(path)
    with Image.open(contact_path) as contact:
        contact_sheet = contact.convert("RGB").copy()

    frames, chapters = _video_frames(
        task_frames,
        durations_ms,
        contact_sheet,
        size=size,
        fps=fps,
    )
    video_path = output_dir / "harnesssim4_four_robot_showcase.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video_path),
        size,
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=["-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    frame_count = 0
    try:
        for frame in frames:
            writer.send(np.asarray(frame, dtype=np.uint8))
            frame_count += 1
    finally:
        writer.close()
    duration_s = frame_count / fps

    poster_path = output_dir / "harnesssim4_four_robot_poster.png"
    _intro_frame(contact_sheet, size).save(poster_path, optimize=True)
    vtt_path = output_dir / "harnesssim4_four_robot_chapters.vtt"
    _write_vtt(vtt_path, chapters, duration_s)
    html_path = output_dir / "harnesssim4_four_robot_showcase.html"
    _write_html(
        html_path,
        video_name=video_path.name,
        poster_name=poster_path.name,
        vtt_name=vtt_path.name,
        chapters=chapters,
    )
    manifest = {
        "name": "HarnessSim4 four-robot cinematic showcase",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": CLAIM_SCOPE,
        "video": {
            "path": video_path.name,
            "sha256": _sha256(video_path),
            "bytes": video_path.stat().st_size,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": frame_count,
            "duration_s": duration_s,
            "codec": "H.264 / yuv420p",
        },
        "chapters": chapters,
        "tasks": [asdict(spec) for spec in TASKS],
        "inputs": {
            **{path.name: _sha256(path) for path in input_paths.values()},
            contact_path.name: _sha256(contact_path),
        },
        "outputs": {
            poster_path.name: _sha256(poster_path),
            vtt_path.name: _sha256(vtt_path),
            html_path.name: _sha256(html_path),
        },
    }
    manifest_path = output_dir / "harnesssim4_four_robot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path.resolve()), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/mujoco_validation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/showcase"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    report = render_showcase(
        args.input,
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
