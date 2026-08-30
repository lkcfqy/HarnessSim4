"""Optional MuJoCo high-fidelity validation backend for all HarnessSim4 tasks.

The fast 2-D PBD backend is used for large paired benchmarks.  This module keeps
MuJoCo optional and provides a second, independent physics backend for visual
and numerical sanity checks.  It deliberately does not turn scripted motion
into a learned-policy result.
"""

from __future__ import annotations

import base64
import io
import json
import os
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harnessbench.config import PROJECT_ROOT
from harnessbench.sim.realistic_robots import ROBOT_VISUAL_SPECS, load_realistic_model


@dataclass(frozen=True)
class MuJoCoSceneSpec:
    key: str
    robot: str
    task: str
    xml_path: Path
    tracks: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]]
    grasp_pairs: tuple[tuple[str, str], ...] = ()


def _scene(
    key: str,
    robot: str,
    task: str,
    relative_xml: str,
    tracks: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]],
    grasp_pairs: tuple[tuple[str, str], ...] = (),
) -> MuJoCoSceneSpec:
    return MuJoCoSceneSpec(
        key=key,
        robot=robot,
        task=task,
        xml_path=PROJECT_ROOT / relative_xml,
        tracks=tracks,
        grasp_pairs=grasp_pairs,
    )


SCENE_SPECS: dict[str, MuJoCoSceneSpec] = {
    "inspect": _scene(
        "inspect",
        "InspectBot",
        "active topology-aware defect inspection",
        "projects/01_inspectbot/mujoco/scene.xml",
        {
            "scanner": (
                (0.00, (-0.40, -0.09, 0.24)),
                (0.22, (-0.20, -0.09, 0.24)),
                (0.50, (0.08, -0.09, 0.24)),
                (0.73, (0.30, -0.09, 0.24)),
                (1.00, (0.42, -0.09, 0.24)),
            )
        },
    ),
    "insert": _scene(
        "insert",
        "InsertBot",
        "terminal alignment, insertion and lock verification",
        "projects/02_insertbot/mujoco/scene.xml",
        {
            "gripper": (
                (0.00, (0.20, 0.00, 0.13)),
                (0.45, (0.21, 0.00, 0.13)),
                (0.78, (0.22, 0.00, 0.13)),
                (1.00, (0.23, 0.00, 0.13)),
            )
        },
        (("gripper", "insert_B_last"),),
    ),
    "route": _scene(
        "route",
        "RouteBot",
        "semantic cable routing through ordered clips",
        "projects/03_routebot/mujoco/scene.xml",
        {
            "route_gripper": (
                (0.00, (0.67, -0.23, 0.14)),
                (0.16, (-0.24, -0.12, 0.16)),
                (0.32, (-0.13, 0.02, 0.18)),
                (0.50, (-0.02, 0.08, 0.16)),
                (0.72, (0.13, -0.02, 0.16)),
                (1.00, (0.34, 0.14, 0.15)),
            )
        },
        (("route_gripper", "route_B_last"),),
    ),
    "branch": _scene(
        "branch",
        "BranchBot",
        "dual-arm branched-harness disentanglement and placement",
        "projects/04_branchbot/mujoco/scene.xml",
        {
            "gripper_a": (
                (0.00, (0.49, -0.30, 0.15)),
                (0.18, (0.08, -0.38, 0.18)),
                (0.42, (0.02, 0.36, 0.20)),
                (0.68, (0.34, 0.34, 0.17)),
                (1.00, (0.43, 0.30, 0.15)),
            ),
            "gripper_b": (
                (0.00, (0.49, 0.30, 0.15)),
                (0.42, (0.49, 0.30, 0.17)),
                (0.58, (0.08, 0.38, 0.19)),
                (0.78, (0.04, -0.36, 0.20)),
                (1.00, (0.43, -0.30, 0.15)),
            ),
        },
        (("gripper_a", "branch_a_B_last"), ("gripper_b", "branch_b_B_last")),
    ),
}


def _load_mujoco():
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
    except Exception as exc:  # includes Windows Application Control failures
        raise RuntimeError(
            "MuJoCo is unavailable in this Python runtime. On this workspace use WSL: "
            "MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench "
            "sim-mujoco-validate"
        ) from exc
    return mujoco


def _mocap_ids(mujoco: Any, model: Any, spec: MuJoCoSceneSpec) -> dict[str, int]:
    result: dict[str, int] = {}
    for body_name in spec.tracks:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Missing mocap body {body_name!r} in {spec.xml_path}")
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body {body_name!r} is not a mocap body")
        result[body_name] = mocap_id
    return result


def _interpolate(
    track: tuple[tuple[float, tuple[float, float, float]], ...], progress: float
) -> np.ndarray:
    if progress <= track[0][0]:
        return np.asarray(track[0][1], dtype=float)
    for (time_a, point_a), (time_b, point_b) in pairwise(track):
        if progress <= time_b:
            weight = (progress - time_a) / max(time_b - time_a, 1e-9)
            # Cubic easing avoids velocity discontinuities at scripted waypoints.
            weight = weight * weight * (3.0 - 2.0 * weight)
            return np.asarray(point_a) * (1.0 - weight) + np.asarray(point_b) * weight
    return np.asarray(track[-1][1], dtype=float)


def _set_mocap_targets(
    data: Any,
    mocap_ids: dict[str, int],
    spec: MuJoCoSceneSpec,
    progress: float,
) -> None:
    for body_name, mocap_id in mocap_ids.items():
        data.mocap_pos[mocap_id] = _interpolate(spec.tracks[body_name], progress)


def _assert_finite(data: Any, task: str) -> None:
    for name in ("qpos", "qvel", "qacc"):
        values = np.asarray(getattr(data, name))
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{task}: non-finite MuJoCo state in {name}")


def _grasp_errors(mujoco: Any, model: Any, data: Any, spec: MuJoCoSceneSpec) -> list[float]:
    errors: list[float] = []
    for mocap_body, cable_body in spec.grasp_pairs:
        mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body)
        cable_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cable_body)
        if mocap_id < 0 or cable_id < 0:
            raise ValueError(f"Invalid grasp pair {(mocap_body, cable_body)}")
        equality_index = next(
            (
                index
                for index in range(model.neq)
                if int(model.eq_obj1id[index]) == cable_id
                and int(model.eq_obj2id[index]) == mocap_id
            ),
            -1,
        )
        if equality_index < 0:
            raise ValueError(f"No equality constraint found for {(mocap_body, cable_body)}")
        equality_data = model.eq_data[equality_index]
        cable_anchor = data.xpos[cable_id] + data.xmat[cable_id].reshape(3, 3) @ equality_data[:3]
        mocap_anchor = data.xpos[mocap_id] + data.xmat[mocap_id].reshape(3, 3) @ equality_data[3:6]
        errors.append(float(np.linalg.norm(cable_anchor - mocap_anchor)))
    return errors


def _model_stats(model: Any, spec: MuJoCoSceneSpec) -> dict[str, Any]:
    return {
        "task_key": spec.key,
        "robot": spec.robot,
        "xml": str(spec.xml_path.resolve()),
        "bodies": int(model.nbody),
        "position_coordinates": int(model.nq),
        "velocity_coordinates": int(model.nv),
        "plugin_instances": int(model.nplugin),
        "mesh_assets": int(model.nmesh),
        "visual_robot": ROBOT_VISUAL_SPECS[spec.key].display_name,
    }


def validate_mujoco_scenes(
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    *,
    stationary_steps: int = 100,
) -> dict[str, Any]:
    """Compile and step each scene without rendering."""

    mujoco = _load_mujoco()
    results: list[dict[str, Any]] = []
    for task in tasks:
        spec = SCENE_SPECS[task]
        model, robot_rig = load_realistic_model(mujoco, task, spec.xml_path)
        data = mujoco.MjData(model)
        mocap_ids = _mocap_ids(mujoco, model, spec)
        _set_mocap_targets(data, mocap_ids, spec, 0.0)
        robot_rig.initialize(data)
        for _ in range(stationary_steps):
            robot_rig.hold(data)
            mujoco.mj_step(model, data)
        _assert_finite(data, task)
        grasp_errors = _grasp_errors(mujoco, model, data, spec)
        results.append(
            {
                **_model_stats(model, spec),
                "stationary_steps": stationary_steps,
                "finite_state": True,
                "max_grasp_error_m": max(grasp_errors, default=0.0),
                "visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
            }
        )
    return {
        "backend": f"MuJoCo {mujoco.__version__}",
        "validated": len(results),
        "scenes": results,
    }


def _annotate(image: Image.Image, spec: MuJoCoSceneSpec, progress: float) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", max(16, image.width // 48))
        detail_font = ImageFont.truetype("DejaVuSans.ttf", max(12, image.width // 72))
    except OSError:
        title_font = detail_font = ImageFont.load_default()
    panel_height = max(66, image.height // 8)
    draw.rounded_rectangle(
        (16, 14, image.width - 16, panel_height),
        radius=14,
        fill=(9, 16, 26, 218),
    )
    draw.text(
        (32, 23),
        f"{spec.robot}  |  realistic MuJoCo digital twin",
        fill=(246, 249, 252, 255),
        font=title_font,
    )
    draw.text(
        (32, 50),
        ROBOT_VISUAL_SPECS[spec.key].display_name,
        fill=(174, 195, 214, 255),
        font=detail_font,
    )
    bar_left, bar_right = 28, image.width - 28
    draw.rounded_rectangle(
        (bar_left, image.height - 26, bar_right, image.height - 16),
        radius=5,
        fill=(28, 40, 56, 210),
    )
    draw.rounded_rectangle(
        (
            bar_left,
            image.height - 26,
            bar_left + (bar_right - bar_left) * progress,
            image.height - 16,
        ),
        radius=5,
        fill=(34, 197, 132, 245),
    )
    return image


def _jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _add_process_inset(image: Image.Image, process_view: Image.Image) -> Image.Image:
    composed = image.copy()
    inset_width = max(220, round(image.width * 0.29))
    inset_height = round(inset_width * process_view.height / process_view.width)
    inset = process_view.resize((inset_width, inset_height), Image.Resampling.LANCZOS)
    margin = max(14, image.width // 60)
    border = max(3, image.width // 320)
    left = image.width - inset_width - margin
    header_height = max(66, image.height // 8)
    top = header_height + max(18, image.height // 30)
    draw = ImageDraw.Draw(composed, "RGBA")
    draw.rounded_rectangle(
        (
            left - border,
            top - border - 22,
            image.width - margin + border,
            top + inset_height + border,
        ),
        radius=10,
        fill=(6, 12, 20, 232),
    )
    draw.text((left + 8, top - 20), "PROCESS VIEW", fill=(205, 220, 232, 255))
    composed.paste(inset, (left, top))
    return composed


def _render_scene(
    mujoco: Any,
    spec: MuJoCoSceneSpec,
    *,
    frame_count: int,
    steps_per_frame: int,
    width: int,
    height: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    model, robot_rig = load_realistic_model(mujoco, spec.key, spec.xml_path)
    data = mujoco.MjData(model)
    mocap_ids = _mocap_ids(mujoco, model, spec)
    _set_mocap_targets(data, mocap_ids, spec, 0.0)
    robot_rig.initialize(data)
    for _ in range(80):
        robot_rig.hold(data)
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    process_renderer = mujoco.Renderer(
        model,
        height=max(180, height // 3),
        width=max(320, width // 3),
    )
    frames: list[Image.Image] = []
    peak_grasp_error = 0.0
    peak_visual_ik_error = 0.0
    try:
        for frame_index in range(frame_count):
            progress = frame_index / max(frame_count - 1, 1)
            _set_mocap_targets(data, mocap_ids, spec, progress)
            robot_rig.sync(data)
            peak_visual_ik_error = max(
                peak_visual_ik_error,
                max(robot_rig.errors(data), default=0.0),
            )
            for _ in range(steps_per_frame):
                robot_rig.hold(data)
                mujoco.mj_step(model, data)
            _assert_finite(data, spec.key)
            peak_grasp_error = max(
                peak_grasp_error,
                max(_grasp_errors(mujoco, model, data, spec), default=0.0),
            )
            renderer.update_scene(data, camera="overview")
            rgb = renderer.render()
            process_renderer.update_scene(data, camera="process_view")
            process_rgb = process_renderer.render()
            annotated = _annotate(Image.fromarray(rgb), spec, progress)
            frames.append(_add_process_inset(annotated, Image.fromarray(process_rgb)))
    finally:
        renderer.close()
        process_renderer.close()

    final_grasp_error = max(_grasp_errors(mujoco, model, data, spec), default=0.0)
    metrics = {
        **_model_stats(model, spec),
        "finite_state": True,
        "rendered_frames": frame_count,
        "physics_steps": 80 + frame_count * steps_per_frame,
        "simulated_seconds": round((80 + frame_count * steps_per_frame) * model.opt.timestep, 3),
        "peak_grasp_error_m": peak_grasp_error,
        "final_grasp_error_m": final_grasp_error,
        "peak_visual_ik_error_m": peak_visual_ik_error,
        "final_visual_ik_error_m": max(robot_rig.errors(data), default=0.0),
    }
    return frames, metrics


def generate_mujoco_demos(
    output_dir: Path,
    *,
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    frame_count: int = 72,
    steps_per_frame: int = 32,
    width: int = 960,
    height: int = 540,
) -> dict[str, Any]:
    """Render all requested scripted high-fidelity validation trajectories."""

    mujoco = _load_mujoco()
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_data: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    final_images: list[tuple[str, Image.Image]] = []

    for task in tasks:
        spec = SCENE_SPECS[task]
        frames, metrics = _render_scene(
            mujoco,
            spec,
            frame_count=frame_count,
            steps_per_frame=steps_per_frame,
            width=width,
            height=height,
        )
        gif_path = output_dir / f"{task}_mujoco.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=85,
            loop=0,
            optimize=False,
        )
        final_path = output_dir / f"{task}_mujoco_final.png"
        frames[-1].save(final_path, optimize=True)
        final_images.append((spec.robot, frames[-1]))
        entry = {
            **metrics,
            "gif": gif_path.name,
            "final_frame": final_path.name,
            "validation_only": True,
        }
        entries.append(entry)
        replay_stride = max(1, len(frames) // 36)
        replay_data[task] = {
            "robot": spec.robot,
            "task": spec.task,
            "policy": "scripted MuJoCo validation trajectory",
            "seed": 0,
            "difficulty": "high-fidelity check",
            "success": True,
            "steps": metrics["physics_steps"],
            "metrics": {
                "bodies": metrics["bodies"],
                "velocity_coordinates": metrics["velocity_coordinates"],
                "plugin_instances": metrics["plugin_instances"],
                "mesh_assets": metrics["mesh_assets"],
                "visual_robot": metrics["visual_robot"],
                "peak_grasp_error_m": metrics["peak_grasp_error_m"],
                "final_grasp_error_m": metrics["final_grasp_error_m"],
                "peak_visual_ik_error_m": metrics["peak_visual_ik_error_m"],
                "finite_state": metrics["finite_state"],
                "claim_scope": "physics/visual validation, not learned-policy evidence",
            },
            "frames": [_jpeg_data_url(frame) for frame in frames[::replay_stride]],
        }

    template_path = PROJECT_ROOT / "web" / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    payload = json.dumps(replay_data, ensure_ascii=False, separators=(",", ":"))
    dashboard_path = output_dir / "mujoco_hifi_dashboard.html"
    dashboard_path.write_text(
        template.replace("/*__REPLAY_DATA__*/", f"const REPLAYS = {payload};"),
        encoding="utf-8",
    )

    contact_width = width * 2
    contact_height = height * 2
    contact = Image.new("RGB", (contact_width, contact_height), (235, 238, 242))
    for index, (_, image) in enumerate(final_images):
        contact.paste(image, ((index % 2) * width, (index // 2) * height))
    contact_path = output_dir / "harnesssim4_mujoco_contact_sheet.png"
    contact.save(contact_path, optimize=True)

    manifest = {
        "name": "HarnessSim4 MuJoCo high-fidelity validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": f"MuJoCo {mujoco.__version__} elasticity cable plugin",
        "claim_scope": (
            "Independent physics and visual sanity check of scripted trajectories; "
            "not a learned-policy benchmark and not real-world validation."
        ),
        "dashboard": dashboard_path.name,
        "contact_sheet": contact_path.name,
        "scenes": entries,
    }
    manifest_path = output_dir / "mujoco_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dashboard": str(dashboard_path.resolve()),
        "contact_sheet": str(contact_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "scenes": entries,
    }
