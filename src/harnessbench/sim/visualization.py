"""Generate GIFs, final frames and a self-contained interactive replay dashboard."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from harnessbench.sim.benchmark import run_episode
from harnessbench.sim.policies import POLICY_REGISTRY, Policy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = PROJECT_ROOT / "web" / "dashboard_template.html"


def _jpeg_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_visual_demos(
    output_dir: Path,
    *,
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    policy_name: str = "topology",
    seed: int = 202609,
    difficulty: float = 0.5,
    record_every: int = 4,
    policy_factory: Callable[[str, int], Policy] | None = None,
    seed_stride: int = 1_000,
    dashboard_filename: str = "harnesssim4_dashboard.html",
    manifest_filename: str = "replay_manifest.json",
    manifest_name: str = "HarnessSim4 visual demos",
    backend_description: str = "2-D position-based dynamics fast backend",
    claim_scope: str = "scripted-policy visual development evidence",
) -> dict:
    if policy_factory is None and policy_name not in POLICY_REGISTRY:
        raise KeyError(policy_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_data: dict[str, dict] = {}
    manifest_tasks: list[dict] = []

    for task_index, task in enumerate(tasks):
        episode_seed = seed + task_index * seed_stride
        policy = (
            policy_factory(task, episode_seed)
            if policy_factory is not None
            else POLICY_REGISTRY[policy_name](episode_seed)
        )
        result = run_episode(
            task,
            policy,
            seed=episode_seed,
            difficulty=difficulty,
            record_every=record_every,
            render_size=(640, 480),
        )
        frames = list(result.frames)
        if not frames:
            raise RuntimeError(f"No frames captured for {task}")
        gif_path = output_dir / f"{task}_{policy_name}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=90,
            loop=0,
            optimize=False,
        )
        final_path = output_dir / f"{task}_{policy_name}_final.png"
        frames[-1].save(final_path, optimize=True)
        replay_data[task] = {
            "robot": result.robot,
            "task": result.task,
            "policy": result.policy,
            "seed": result.seed,
            "difficulty": result.difficulty,
            "success": result.success,
            "steps": result.steps,
            "metrics": result.metrics,
            "frames": [_jpeg_data_url(frame) for frame in frames],
        }
        manifest_tasks.append(
            {
                "task_key": task,
                "robot": result.robot,
                "task": result.task,
                "policy": result.policy,
                "seed": result.seed,
                "difficulty": result.difficulty,
                "success": result.success,
                "steps": result.steps,
                "metrics": result.metrics,
                "frame_count": len(frames),
                "gif": gif_path.name,
                "final_frame": final_path.name,
            }
        )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = json.dumps(replay_data, ensure_ascii=False, separators=(",", ":"))
    dashboard = template.replace("/*__REPLAY_DATA__*/", f"const REPLAYS = {payload};")
    if "/*__REPLAY_DATA__*/" in dashboard:
        raise RuntimeError("dashboard placeholder was not replaced")
    dashboard_path = output_dir / dashboard_filename
    dashboard_path.write_text(dashboard, encoding="utf-8")

    manifest = {
        "name": manifest_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend_description,
        "claim_scope": claim_scope,
        "dashboard": dashboard_path.name,
        "tasks": manifest_tasks,
    }
    manifest_path = output_dir / manifest_filename
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dashboard": str(dashboard_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "tasks": manifest_tasks,
    }
