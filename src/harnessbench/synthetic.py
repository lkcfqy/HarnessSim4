"""Lightweight 2-D wire-terminal insertion scenes for schema and vision prototyping."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SyntheticSpec:
    episodes: int = 12
    frames_per_episode: int = 40
    image_size: int = 256
    seed: int = 202609


def _rotated_rectangle(
    center: np.ndarray, width: float, height: float, angle: float
) -> list[tuple[float, float]]:
    corners = np.asarray(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = corners @ rotation.T + center
    return [(float(x), float(y)) for x, y in points]


def _angle_difference(target: float, source: float) -> float:
    return math.atan2(math.sin(target - source), math.cos(target - source))


def _bezier_points(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
    count: int = 32,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)[:, None]
    return (
        (1 - t) ** 3 * start
        + 3 * (1 - t) ** 2 * t * control_1
        + 3 * (1 - t) * t**2 * control_2
        + t**3 * end
    )


def _episode_split(index: int, episodes: int) -> str:
    fraction = (index + 0.5) / episodes
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def _draw_scene(
    size: int,
    rng: np.random.Generator,
    socket_center: np.ndarray,
    socket_angle: float,
    terminal_center: np.ndarray,
    terminal_angle: float,
    cable_anchor: np.ndarray,
    cable_bend: np.ndarray,
    inserted: bool,
) -> tuple[Image.Image, np.ndarray]:
    base = np.full((size, size, 3), rng.integers(190, 218), dtype=np.int16)
    noise = rng.normal(0, 3.0, size=base.shape)
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)

    # Fixture plate and mounting holes create nuisance geometry.
    margin = int(size * 0.055)
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=max(4, size // 40),
        outline=(105, 110, 115),
        width=max(2, size // 100),
    )
    for x, y in ((margin * 2, margin * 2), (size - margin * 2, margin * 2)):
        r = max(3, size // 45)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(80, 82, 86))

    socket_width, socket_height = size * 0.19, size * 0.11
    socket_poly = _rotated_rectangle(socket_center, socket_width, socket_height, socket_angle)
    socket_color = (35, int(rng.integers(80, 121)), int(rng.integers(125, 171)))
    draw.polygon(socket_poly, fill=socket_color, outline=(18, 38, 58), width=max(2, size // 100))
    opening_center = (
        socket_center
        - np.asarray([math.cos(socket_angle), math.sin(socket_angle)]) * socket_width * 0.37
    )
    opening = _rotated_rectangle(
        opening_center, socket_width * 0.16, socket_height * 0.54, socket_angle
    )
    draw.polygon(opening, fill=(22, 24, 26))

    terminal_length, terminal_width = size * 0.115, size * 0.030
    tail = (
        terminal_center
        - np.asarray([math.cos(terminal_angle), math.sin(terminal_angle)]) * terminal_length * 0.54
    )
    control_1 = cable_anchor + cable_bend
    control_2 = (cable_anchor + tail) * 0.5 - cable_bend * 0.45
    cable_points = _bezier_points(cable_anchor, control_1, control_2, tail)
    cable_color = (
        int(rng.integers(95, 150)),
        int(rng.integers(35, 65)),
        int(rng.integers(25, 50)),
    )
    draw.line(
        [tuple(point) for point in cable_points],
        fill=cable_color,
        width=max(5, size // 35),
        joint="curve",
    )

    terminal_poly = _rotated_rectangle(
        terminal_center, terminal_length, terminal_width, terminal_angle
    )
    terminal_color = (206, 183, 92) if not inserted else (173, 158, 92)
    draw.polygon(
        terminal_poly,
        fill=terminal_color,
        outline=(76, 66, 42),
        width=max(1, size // 128),
    )
    return image, cable_points


def generate_synthetic_dataset(output_dir: Path, spec: SyntheticSpec) -> dict:
    """Generate an idempotent labeled dataset; refuse incompatible silent overwrites."""

    if spec.episodes < 3 or spec.frames_per_episode < 3:
        raise ValueError("Use at least 3 episodes and 3 frames per episode")
    if spec.image_size < 96:
        raise ValueError("image_size must be at least 96 pixels")

    info_path = output_dir / "dataset_info.json"
    requested = asdict(spec)
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("spec") != requested:
            raise FileExistsError(
                f"{output_dir} already contains a dataset with a different spec. "
                "Choose a new output directory instead of silently overwriting it."
            )
        expected = spec.episodes * spec.frames_per_episode
        if existing.get("frames") == expected and (output_dir / "labels.jsonl").exists():
            return {**existing, "status": "cached"}

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    labels: list[dict] = []
    size = spec.image_size

    for episode in range(spec.episodes):
        socket_center = np.asarray([rng.uniform(0.60, 0.76) * size, rng.uniform(0.34, 0.66) * size])
        socket_angle = float(rng.uniform(-0.42, 0.42))
        axis = np.asarray([math.cos(socket_angle), math.sin(socket_angle)])
        lateral = np.asarray([-axis[1], axis[0]])
        initial_distance = rng.uniform(0.30, 0.43) * size
        initial_lateral = rng.uniform(-0.12, 0.12) * size
        initial_center = socket_center - axis * initial_distance + lateral * initial_lateral
        initial_angle = socket_angle + rng.uniform(-0.65, 0.65)
        cable_anchor = np.asarray([rng.uniform(0.08, 0.22) * size, rng.uniform(0.18, 0.82) * size])
        cable_bend = np.asarray([rng.uniform(-0.04, 0.08), rng.uniform(-0.18, 0.18)]) * size
        split = _episode_split(episode, spec.episodes)

        for frame in range(spec.frames_per_episode):
            progress = frame / (spec.frames_per_episode - 1)
            approach = min(progress / 0.82, 1.0)
            smooth = approach * approach * (3.0 - 2.0 * approach)
            insertion_depth = max((progress - 0.82) / 0.18, 0.0) * size * 0.035
            jitter_scale = (1.0 - smooth) * size * 0.006
            terminal_center = (
                initial_center * (1.0 - smooth)
                + socket_center * smooth
                + axis * insertion_depth
                + rng.normal(0.0, jitter_scale, size=2)
            )
            terminal_angle = initial_angle + _angle_difference(socket_angle, initial_angle) * smooth
            terminal_angle += float(rng.normal(0.0, (1.0 - smooth) * 0.012))
            inserted = bool(progress >= 0.96)
            image, cable_points = _draw_scene(
                size,
                rng,
                socket_center,
                socket_angle,
                terminal_center,
                terminal_angle,
                cable_anchor,
                cable_bend,
                inserted,
            )

            relative_path = Path("images") / f"episode_{episode:04d}_frame_{frame:04d}.png"
            image.save(output_dir / relative_path, optimize=True)
            delta_xy = (socket_center - terminal_center) / size
            delta_angle = _angle_difference(socket_angle, terminal_angle)
            labels.append(
                {
                    "image": relative_path.as_posix(),
                    "episode_index": episode,
                    "frame_index": frame,
                    "split": split,
                    "terminal_xy": [float(value / size) for value in terminal_center],
                    "terminal_angle_rad": float(terminal_angle),
                    "socket_xy": [float(value / size) for value in socket_center],
                    "socket_angle_rad": float(socket_angle),
                    "cable_polyline_xy": [
                        [float(x / size), float(y / size)] for x, y in cable_points[::4]
                    ],
                    "target_delta_xy": [float(value) for value in delta_xy],
                    "target_delta_angle_rad": float(delta_angle),
                    "progress": float(progress),
                    "inserted": inserted,
                }
            )

    with (output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for record in labels:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_counts = {
        split: sum(record["split"] == split for record in labels)
        for split in ("train", "validation", "test")
    }
    info = {
        "name": "harness_insert_synthetic_v0",
        "purpose": "schema, keypoint, pose and insertion-action prototyping only",
        "license": "MIT (generated by this project)",
        "spec": requested,
        "frames": len(labels),
        "split_frame_counts": split_counts,
        "label_file": "labels.jsonl",
        "limitations": [
            "2-D appearance is not photorealistic",
            "cable shape is kinematic rather than physics-accurate",
            "results on this data cannot establish real-robot performance",
        ],
    }
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {**info, "status": "generated"}
