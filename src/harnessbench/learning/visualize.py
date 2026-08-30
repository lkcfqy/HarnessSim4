"""Visual replay artifacts for frozen TopoHarness checkpoints."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.learning.train import load_policy_model
from harnessbench.sim.visualization import generate_visual_demos


def generate_learned_visual_demos(
    output_dir: Path,
    checkpoint: Path,
    *,
    tasks: Iterable[str] = ("inspect", "insert", "route", "branch"),
    mode: str = "topology",
    seed: int = 31_010_000,
    difficulty: float = 0.5,
    record_every: int = 4,
    device_name: str = "cpu",
) -> dict:
    """Render one held-out closed-loop episode per robot from a checkpoint."""

    if mode not in {"topology", "geometry"}:
        raise ValueError("mode must be topology or geometry")
    checkpoint = checkpoint.resolve()
    model = load_policy_model(checkpoint, device_name=device_name)
    expected_topology = mode == "topology"
    if bool(model.config.use_topology) is not expected_topology:
        raise ValueError(
            f"checkpoint topology mode is {model.config.use_topology}, requested {mode}"
        )
    policy_name = f"learned_{mode}"

    def policy_factory(task: str, episode_seed: int) -> LearnedGraphPolicy:
        return LearnedGraphPolicy(
            checkpoint,
            task,
            name=policy_name,
            device_name=device_name,
        )

    result = generate_visual_demos(
        output_dir,
        tasks=tasks,
        policy_name=policy_name,
        seed=seed,
        difficulty=difficulty,
        record_every=record_every,
        policy_factory=policy_factory,
        seed_stride=100_000,
        dashboard_filename=f"harnesssim4_{policy_name}_dashboard.html",
        manifest_filename=f"{policy_name}_replay_manifest.json",
        manifest_name=f"HarnessSim4 {policy_name} closed-loop visual demos",
        backend_description="2-D PBD with frozen TopoHarness graph-pointer policy",
        claim_scope=("held-out PBD closed-loop visual evidence; not MuJoCo or real-world evidence"),
    )
    result["checkpoint"] = str(checkpoint)
    result["mode"] = mode
    return result
