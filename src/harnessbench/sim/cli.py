"""CLI integration for HarnessSim4."""

from __future__ import annotations

from harnessbench.config import project_path
from harnessbench.sim.benchmark import run_benchmark, save_benchmark
from harnessbench.sim.envs import ENV_REGISTRY
from harnessbench.sim.policies import POLICY_REGISTRY
from harnessbench.sim.visualization import generate_visual_demos

PAPER_ROUTE_POLICIES = (
    "learned_topology",
    "learned_geometry",
    "learned_physical",
    "learned_semantic",
    "learned_no_features",
    "learned_data20",
    "learned_data50",
    "act_chunk",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
    "random",
)

DIRECT_MUJOCO_ROUTE_POLICIES = (
    "learned_topology",
    "learned_geometry",
    "teacher_topology",
)

PAPER_BRANCH_POLICIES = (
    "learned_topology",
    "learned_geometry",
    "learned_physical",
    "learned_semantic",
    "learned_no_features",
    "act_chunk",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
    "random",
)

DIRECT_MUJOCO_BRANCH_POLICIES = (
    "learned_topology",
    "learned_geometry",
    "teacher_topology",
)


def add_sim_subparsers(subparsers) -> None:
    subparsers.add_parser("sim-list", help="List all four simulated robot tasks")

    demo = subparsers.add_parser("sim-demo", help="Generate GIFs and an interactive replay lab")
    demo.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", *ENV_REGISTRY],
        default=["all"],
    )
    demo.add_argument("--policy", choices=sorted(POLICY_REGISTRY), default="topology")
    demo.add_argument("--seed", type=int, default=202609)
    demo.add_argument("--difficulty", type=float, default=0.5)
    demo.add_argument("--record-every", type=int, default=4)
    demo.add_argument("--output", default="artifacts/simulations")

    benchmark = subparsers.add_parser(
        "sim-benchmark", help="Run paired multi-seed experiments and export JSON/CSV"
    )
    benchmark.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    benchmark.add_argument(
        "--policies", nargs="+", choices=sorted(POLICY_REGISTRY), default=list(POLICY_REGISTRY)
    )
    benchmark.add_argument("--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    benchmark.add_argument("--episodes", type=int, default=5)
    benchmark.add_argument("--seed", type=int, default=202609)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument("--output", default="artifacts/sim_benchmark")

    validate = subparsers.add_parser(
        "sim-mujoco-validate", help="Compile and numerically validate all MuJoCo scenes"
    )
    validate.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    validate.add_argument("--steps", type=int, default=100)

    mujoco_demo = subparsers.add_parser(
        "sim-mujoco-demo", help="Render all high-fidelity MuJoCo validation trajectories"
    )
    mujoco_demo.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    mujoco_demo.add_argument("--frames", type=int, default=72)
    mujoco_demo.add_argument("--steps-per-frame", type=int, default=32)
    mujoco_demo.add_argument("--width", type=int, default=960)
    mujoco_demo.add_argument("--height", type=int, default=540)
    mujoco_demo.add_argument("--output", default="artifacts/mujoco_validation")

    demos = subparsers.add_parser(
        "sim-generate-demos", help="Generate leakage-safe graph demonstrations"
    )
    demos.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    demos.add_argument("--train-episodes", type=int, default=12)
    demos.add_argument("--validation-episodes", type=int, default=4)
    demos.add_argument("--test-episodes", type=int, default=8)
    demos.add_argument("--record-every", type=int, default=2)
    demos.add_argument(
        "--route-assignment-indices",
        nargs="+",
        type=int,
        choices=range(9),
        default=list(range(9)),
        help="Indices of the predeclared RouteBot relational assignments to collect",
    )
    demos.add_argument("--seed", type=int, default=430000)
    demos.add_argument("--output", default="artifacts/learning/topoharness_demos.npz")

    successful_demos = subparsers.add_parser(
        "sim-filter-successful-demos",
        help="Select a balanced, leakage-safe dataset of successful expert episodes",
    )
    successful_demos.add_argument("--source", required=True)
    successful_demos.add_argument("--train-episodes", type=int, default=100)
    successful_demos.add_argument("--validation-episodes", type=int, default=20)
    successful_demos.add_argument("--test-episodes", type=int, default=20)
    successful_demos.add_argument(
        "--output", default="artifacts/learning/topoharness_successful_demos.npz"
    )

    train = subparsers.add_parser(
        "sim-train-policy", help="Train topology and/or geometry graph policies"
    )
    train.add_argument("--dataset", default="artifacts/learning/topoharness_demos.npz")
    train.add_argument(
        "--mode",
        choices=["topology", "geometry", "physical", "semantic", "features", "both"],
        default="both",
    )
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.002)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--seed", type=int, default=202609)
    train.add_argument("--device", default="cpu")
    train.add_argument("--output", default="artifacts/learning")

    act_train = subparsers.add_parser(
        "sim-train-act",
        help="Train the independent state-based ACT action-chunking baseline",
    )
    act_train.add_argument("--dataset", default="artifacts/learning/topoharness_demos.npz")
    act_train.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    act_train.add_argument("--epochs", type=int, default=120)
    act_train.add_argument("--batch-size", type=int, default=128)
    act_train.add_argument("--learning-rate", type=float, default=0.001)
    act_train.add_argument("--weight-decay", type=float, default=0.0001)
    act_train.add_argument("--hidden-dim", type=int, default=96)
    act_train.add_argument("--heads", type=int, default=4)
    act_train.add_argument("--layers", type=int, default=2)
    act_train.add_argument("--chunk-size", type=int, default=16)
    act_train.add_argument("--latent-dim", type=int, default=16)
    act_train.add_argument("--kl-weight", type=float, default=0.1)
    act_train.add_argument("--pointer-loss-weight", type=float, default=1.0)
    act_train.add_argument(
        "--use-relations",
        action="store_true",
        help="Add fixed one-hop semantic-relation pooling before the ACT Transformer",
    )
    act_train.add_argument("--patience", type=int, default=18)
    act_train.add_argument("--seed", type=int, default=202610)
    act_train.add_argument("--device", default="cpu")
    act_train.add_argument("--output", default="artifacts/learning/act_chunk_policy.pt")

    learned_eval = subparsers.add_parser(
        "sim-eval-learned", help="Run paired held-out closed-loop learned-policy evaluation"
    )
    learned_eval.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    learned_eval.add_argument("--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    learned_eval.add_argument("--episodes", type=int, default=5)
    learned_eval.add_argument("--seed", type=int, default=31000000)
    learned_eval.add_argument("--device", default="cpu")
    learned_eval.add_argument(
        "--topology-checkpoint", default="artifacts/learning/topology_policy.pt"
    )
    learned_eval.add_argument(
        "--geometry-checkpoint", default="artifacts/learning/geometry_policy.pt"
    )
    learned_eval.add_argument("--output", default="artifacts/learning/evaluation")

    route_paper = subparsers.add_parser(
        "sim-eval-route-paper",
        help="Run a paired RouteBot table with TopoHarness, ACT and controls",
    )
    route_paper.add_argument("--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    route_paper.add_argument("--episodes", type=int, default=100)
    route_paper.add_argument("--seed", type=int, default=61000000)
    route_paper.add_argument("--device", default="cpu")
    route_paper.add_argument("--workers", type=int, default=1)
    route_paper.add_argument("--policies", nargs="+", choices=PAPER_ROUTE_POLICIES, default=None)
    route_paper.add_argument(
        "--topology-checkpoint", default="artifacts/learning/topology_policy.pt"
    )
    route_paper.add_argument(
        "--geometry-checkpoint", default="artifacts/learning/geometry_policy.pt"
    )
    route_paper.add_argument("--act-checkpoint", default="artifacts/learning/act_chunk_policy.pt")
    route_paper.add_argument("--act-relational-checkpoint", default=None)
    route_paper.add_argument("--physical-checkpoint", default=None)
    route_paper.add_argument("--semantic-checkpoint", default=None)
    route_paper.add_argument("--no-features-checkpoint", default=None)
    route_paper.add_argument("--data20-checkpoint", default=None)
    route_paper.add_argument("--data50-checkpoint", default=None)
    route_paper.add_argument("--output", default="artifacts/papers/routebot/main_table")

    route_counterfactual = subparsers.add_parser(
        "sim-eval-route-counterfactual",
        help="Change feasible segment-to-clip assignments on identical RouteBot physics",
    )
    route_counterfactual.add_argument(
        "--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8]
    )
    route_counterfactual.add_argument("--physical-seeds", type=int, default=10)
    route_counterfactual.add_argument(
        "--assignment-indices",
        nargs="+",
        type=int,
        choices=range(9),
        default=list(range(9)),
        help="Predeclared assignments to compare; default is the frozen nine-assignment set",
    )
    route_counterfactual.add_argument("--seed", type=int, default=62000000)
    route_counterfactual.add_argument("--device", default="cpu")
    route_counterfactual.add_argument("--workers", type=int, default=1)
    route_counterfactual.add_argument(
        "--policies", nargs="+", choices=PAPER_ROUTE_POLICIES, default=None
    )
    route_counterfactual.add_argument(
        "--topology-checkpoint", default="artifacts/learning/topology_policy.pt"
    )
    route_counterfactual.add_argument(
        "--geometry-checkpoint", default="artifacts/learning/geometry_policy.pt"
    )
    route_counterfactual.add_argument(
        "--act-checkpoint", default="artifacts/learning/act_chunk_policy.pt"
    )
    route_counterfactual.add_argument("--act-relational-checkpoint", default=None)
    route_counterfactual.add_argument("--physical-checkpoint", default=None)
    route_counterfactual.add_argument("--semantic-checkpoint", default=None)
    route_counterfactual.add_argument("--no-features-checkpoint", default=None)
    route_counterfactual.add_argument("--data20-checkpoint", default=None)
    route_counterfactual.add_argument("--data50-checkpoint", default=None)
    route_counterfactual.add_argument(
        "--output", default="artifacts/papers/routebot/counterfactual"
    )

    route_audit = subparsers.add_parser(
        "sim-audit-route-assignments",
        help="Screen all predeclared RouteBot assignments with the privileged teacher",
    )
    route_audit.add_argument("--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    route_audit.add_argument("--physical-seeds", type=int, default=10)
    route_audit.add_argument(
        "--assignment-indices",
        nargs="+",
        type=int,
        choices=range(9),
        default=list(range(9)),
    )
    route_audit.add_argument("--seed", type=int, default=60500000)
    route_audit.add_argument("--output", default="artifacts/papers/routebot/assignment_audit")

    route_public_download = subparsers.add_parser(
        "sim-download-route-public",
        help="Download the pinned real Berkeley cable-routing dataset",
    )
    route_public_download.add_argument(
        "--include-videos",
        action="store_true",
        help="Also download all four 128x128 real-robot camera streams (about 316 MB)",
    )
    route_public_download.add_argument("--workers", type=int, default=4)
    route_public_download.add_argument(
        "--output", default="data/public/berkeley_cable_routing"
    )

    route_public_inspect = subparsers.add_parser(
        "sim-inspect-route-public",
        help="Validate cardinality, dimensions, finiteness, and episode keys",
    )
    route_public_inspect.add_argument(
        "--data", default="data/public/berkeley_cable_routing"
    )

    route_public_eval = subparsers.add_parser(
        "sim-eval-route-public",
        help="Run a leakage-safe offline baseline on held-out real cable-routing episodes",
    )
    route_public_eval.add_argument("--data", default="data/public/berkeley_cable_routing")
    route_public_eval.add_argument("--seed", type=int, default=202613)
    route_public_eval.add_argument(
        "--output", default="artifacts/papers/routebot/public_real"
    )

    route_public_render = subparsers.add_parser(
        "sim-render-route-public",
        help="Extract an attributed aligned four-camera real-data paper figure",
    )
    route_public_render.add_argument("--data", default="data/public/berkeley_cable_routing")
    route_public_render.add_argument("--timestamp", type=float, default=120.0)
    route_public_render.add_argument(
        "--output", default="artifacts/papers/routebot/public_real"
    )

    route_mujoco_direct = subparsers.add_parser(
        "sim-eval-route-mujoco-direct",
        help="Close the RouteBot policy loop directly on current MuJoCo poses",
    )
    route_mujoco_direct.add_argument("--physical-seeds", type=int, default=2)
    route_mujoco_direct.add_argument(
        "--assignment-indices",
        nargs="+",
        type=int,
        choices=range(9),
        default=list(range(9)),
    )
    route_mujoco_direct.add_argument(
        "--policies",
        nargs="+",
        choices=DIRECT_MUJOCO_ROUTE_POLICIES,
        default=list(DIRECT_MUJOCO_ROUTE_POLICIES),
    )
    route_mujoco_direct.add_argument("--difficulty", type=float, default=0.5)
    route_mujoco_direct.add_argument("--seed", type=int, default=63000000)
    route_mujoco_direct.add_argument("--physics-steps", type=int, default=6)
    route_mujoco_direct.add_argument("--max-steps", type=int, default=360)
    route_mujoco_direct.add_argument("--device", default="cpu")
    route_mujoco_direct.add_argument(
        "--topology-checkpoint",
        default="artifacts/papers/routebot/final/models/seed_202611/topology_policy.pt",
    )
    route_mujoco_direct.add_argument(
        "--geometry-checkpoint",
        default="artifacts/papers/routebot/final/models/seed_202611/geometry_policy.pt",
    )
    route_mujoco_direct.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the representative high-detail UR10e final frame",
    )
    route_mujoco_direct.add_argument(
        "--output", default="artifacts/papers/routebot/mujoco_direct"
    )

    learned_demo = subparsers.add_parser(
        "sim-demo-learned", help="Render frozen learned policies as GIFs and a replay lab"
    )
    learned_demo.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    learned_demo.add_argument("--mode", choices=["topology", "geometry"], default="topology")
    learned_demo.add_argument("--checkpoint", default=None)
    learned_demo.add_argument("--seed", type=int, default=31010000)
    learned_demo.add_argument("--difficulty", type=float, default=0.5)
    learned_demo.add_argument("--record-every", type=int, default=4)
    learned_demo.add_argument("--device", default="cpu")
    learned_demo.add_argument("--output", default="artifacts/learning/visuals")

    counterfactual = subparsers.add_parser(
        "sim-eval-branch-counterfactual",
        help="Pair both A/B semantic assignments on identical BranchBot physics",
    )
    counterfactual.add_argument("--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    counterfactual.add_argument("--episodes", type=int, default=10)
    counterfactual.add_argument("--seed", type=int, default=41000000)
    counterfactual.add_argument("--device", default="cpu")
    counterfactual.add_argument(
        "--topology-checkpoint", default="artifacts/learning/topology_policy.pt"
    )
    counterfactual.add_argument(
        "--geometry-checkpoint", default="artifacts/learning/geometry_policy.pt"
    )
    counterfactual.add_argument("--output", default="artifacts/learning/branch_counterfactual")

    branch_paper = subparsers.add_parser(
        "sim-eval-branch-paper",
        help="Run paired BranchBot A/B interventions with ACT and controls",
    )
    branch_paper.add_argument(
        "--difficulties", nargs="+", type=float, default=[0.2, 0.5, 0.8]
    )
    branch_paper.add_argument("--physical-seeds", type=int, default=100)
    branch_paper.add_argument("--seed", type=int, default=66000000)
    branch_paper.add_argument("--device", default="cpu")
    branch_paper.add_argument("--workers", type=int, default=1)
    branch_paper.add_argument(
        "--policies", nargs="+", choices=PAPER_BRANCH_POLICIES, default=None
    )
    branch_paper.add_argument(
        "--topology-checkpoint",
        default="artifacts/papers/branchbot/final/models/graph_topology/topology_policy.pt",
    )
    branch_paper.add_argument(
        "--geometry-checkpoint",
        default="artifacts/papers/branchbot/final/models/graph_geometry/geometry_policy.pt",
    )
    branch_paper.add_argument(
        "--act-checkpoint",
        default="artifacts/papers/branchbot/final/models/act_geometry.pt",
    )
    branch_paper.add_argument(
        "--act-relational-checkpoint",
        default="artifacts/papers/branchbot/final/models/act_relational.pt",
    )
    branch_paper.add_argument("--physical-checkpoint", default=None)
    branch_paper.add_argument("--semantic-checkpoint", default=None)
    branch_paper.add_argument("--no-features-checkpoint", default=None)
    branch_paper.add_argument(
        "--output", default="artifacts/papers/branchbot/final/counterfactual"
    )

    branch_mujoco_direct = subparsers.add_parser(
        "sim-eval-branch-mujoco-direct",
        help="Run BranchBot policies directly from MuJoCo observations",
    )
    branch_mujoco_direct.add_argument("--physical-seeds", type=int, default=20)
    branch_mujoco_direct.add_argument(
        "--policies",
        nargs="+",
        choices=DIRECT_MUJOCO_BRANCH_POLICIES,
        default=list(DIRECT_MUJOCO_BRANCH_POLICIES),
    )
    branch_mujoco_direct.add_argument("--difficulty", type=float, default=0.5)
    branch_mujoco_direct.add_argument("--seed", type=int, default=65000000)
    branch_mujoco_direct.add_argument("--physics-steps", type=int, default=6)
    branch_mujoco_direct.add_argument("--max-steps", type=int, default=300)
    branch_mujoco_direct.add_argument("--device", default="cpu")
    branch_mujoco_direct.add_argument(
        "--topology-checkpoint",
        default="artifacts/papers/branchbot/final/models/graph_topology/topology_policy.pt",
    )
    branch_mujoco_direct.add_argument(
        "--geometry-checkpoint",
        default="artifacts/papers/branchbot/final/models/graph_geometry/geometry_policy.pt",
    )
    branch_mujoco_direct.add_argument("--no-render", action="store_true")
    branch_mujoco_direct.add_argument(
        "--output", default="artifacts/papers/branchbot/final/mujoco_direct"
    )

    transfer = subparsers.add_parser(
        "sim-mujoco-transfer",
        help="Audit frozen PBD policy decisions through MuJoCo motion primitives",
    )
    transfer.add_argument("--tasks", nargs="+", choices=["all", *ENV_REGISTRY], default=["all"])
    transfer.add_argument("--episodes", type=int, default=2)
    transfer.add_argument("--difficulty", type=float, default=0.5)
    transfer.add_argument("--seed", type=int, default=51000000)
    transfer.add_argument("--physics-steps", type=int, default=6)
    transfer.add_argument("--device", default="cpu")
    transfer.add_argument("--topology-checkpoint", default="artifacts/learning/topology_policy.pt")
    transfer.add_argument("--geometry-checkpoint", default="artifacts/learning/geometry_policy.pt")
    transfer.add_argument("--output", default="artifacts/learning/mujoco_transfer")


def _task_list(values: list[str]) -> list[str]:
    return list(ENV_REGISTRY) if "all" in values else values


def handle_sim_command(args) -> dict:
    if args.command == "sim-list":
        return {
            "benchmark": "HarnessSim4",
            "robots": [
                {
                    "key": "inspect",
                    "robot": "InspectBot",
                    "task": "active topology-aware defect inspection",
                },
                {
                    "key": "insert",
                    "robot": "InsertBot",
                    "task": "terminal alignment, insertion and lock verification",
                },
                {
                    "key": "route",
                    "robot": "RouteBot",
                    "task": "semantic cable routing into ordered clips",
                },
                {
                    "key": "branch",
                    "robot": "BranchBot",
                    "task": "dual-arm branched-harness disentanglement and placement",
                },
            ],
        }
    if args.command == "sim-demo":
        return generate_visual_demos(
            project_path(args.output),
            tasks=_task_list(args.tasks),
            policy_name=args.policy,
            seed=args.seed,
            difficulty=args.difficulty,
            record_every=args.record_every,
        )
    if args.command == "sim-benchmark":
        report = run_benchmark(
            tasks=_task_list(args.tasks),
            policies=args.policies,
            difficulties=args.difficulties,
            episodes=args.episodes,
            base_seed=args.seed,
            workers=args.workers,
        )
        exported = save_benchmark(report, project_path(args.output))
        return {
            **exported,
            "config": report["config"],
            "summary": report["summary"],
            "paired_topology_advantage": report["paired_topology_advantage"],
        }
    if args.command == "sim-mujoco-validate":
        from harnessbench.sim.mujoco_backend import validate_mujoco_scenes

        return validate_mujoco_scenes(tasks=_task_list(args.tasks), stationary_steps=args.steps)
    if args.command == "sim-mujoco-demo":
        from harnessbench.sim.mujoco_backend import generate_mujoco_demos

        return generate_mujoco_demos(
            project_path(args.output),
            tasks=_task_list(args.tasks),
            frame_count=args.frames,
            steps_per_frame=args.steps_per_frame,
            width=args.width,
            height=args.height,
        )
    if args.command == "sim-generate-demos":
        from harnessbench.learning.dataset import generate_expert_dataset
        from harnessbench.sim.envs.route import ROUTE_TARGET_ASSIGNMENTS

        return generate_expert_dataset(
            project_path(args.output),
            tasks=_task_list(args.tasks),
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
            record_every=args.record_every,
            base_seed=args.seed,
            route_target_assignments=tuple(
                ROUTE_TARGET_ASSIGNMENTS[index] for index in args.route_assignment_indices
            ),
        )
    if args.command == "sim-filter-successful-demos":
        from harnessbench.learning.dataset import filter_successful_expert_dataset

        return filter_successful_expert_dataset(
            project_path(args.source),
            project_path(args.output),
            train_episodes=args.train_episodes,
            validation_episodes=args.validation_episodes,
            test_episodes=args.test_episodes,
        )
    if args.command == "sim-train-policy":
        from harnessbench.learning.train import train_policy

        output_dir = project_path(args.output)
        modes = ["topology", "geometry"] if args.mode == "both" else [args.mode]
        mode_configs = {
            "topology": {
                "use_topology": True,
                "adjacency_mode": "combined",
                "use_topology_features": True,
            },
            "geometry": {
                "use_topology": False,
                "adjacency_mode": "none",
                "use_topology_features": False,
            },
            "physical": {
                "use_topology": True,
                "adjacency_mode": "physical",
                "use_topology_features": True,
            },
            "semantic": {
                "use_topology": True,
                "adjacency_mode": "semantic",
                "use_topology_features": True,
            },
            "features": {
                "use_topology": True,
                "adjacency_mode": "combined",
                "use_topology_features": False,
            },
        }
        results = {}
        for mode in modes:
            mode_config = mode_configs[mode]
            results[mode] = train_policy(
                project_path(args.dataset),
                output_dir / f"{mode}_policy.pt",
                use_topology=mode_config["use_topology"],
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                message_passing_layers=args.layers,
                adjacency_mode=mode_config["adjacency_mode"],
                use_topology_features=mode_config["use_topology_features"],
                patience=args.patience,
                device_name=args.device,
            )
        return {"models": results}
    if args.command == "sim-train-act":
        from harnessbench.learning.act_baseline import train_act_chunk_policy

        return train_act_chunk_policy(
            project_path(args.dataset),
            project_path(args.output),
            tasks=tuple(_task_list(args.tasks)),
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_heads=args.heads,
            layers=args.layers,
            chunk_size=args.chunk_size,
            latent_dim=args.latent_dim,
            kl_weight=args.kl_weight,
            pointer_loss_weight=args.pointer_loss_weight,
            use_relations=args.use_relations,
            patience=args.patience,
            device_name=args.device,
        )
    if args.command == "sim-eval-learned":
        from harnessbench.learning.evaluate import (
            evaluate_learned_policies,
            save_learning_evaluation,
        )

        report = evaluate_learned_policies(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            tasks=_task_list(args.tasks),
            difficulties=args.difficulties,
            episodes=args.episodes,
            base_seed=args.seed,
            device_name=args.device,
        )
        return save_learning_evaluation(report, project_path(args.output))
    if args.command == "sim-eval-route-paper":
        from harnessbench.learning.route_paper import (
            evaluate_route_paper_table,
            save_route_paper_table,
        )

        report = evaluate_route_paper_table(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            project_path(args.act_checkpoint),
            difficulties=args.difficulties,
            episodes=args.episodes,
            base_seed=args.seed,
            device_name=args.device,
            workers=args.workers,
            act_relational_checkpoint=(
                project_path(args.act_relational_checkpoint)
                if args.act_relational_checkpoint
                else None
            ),
            policy_names=args.policies,
            extra_checkpoints={
                name: project_path(value)
                for name, value in (
                    ("learned_physical", args.physical_checkpoint),
                    ("learned_semantic", args.semantic_checkpoint),
                    ("learned_no_features", args.no_features_checkpoint),
                    ("learned_data20", args.data20_checkpoint),
                    ("learned_data50", args.data50_checkpoint),
                )
                if value
            },
        )
        return save_route_paper_table(report, project_path(args.output))
    if args.command == "sim-eval-route-counterfactual":
        from harnessbench.learning.route_paper import (
            evaluate_route_counterfactual,
            save_route_counterfactual,
        )
        from harnessbench.sim.envs.route import ROUTE_TARGET_ASSIGNMENTS

        report = evaluate_route_counterfactual(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            project_path(args.act_checkpoint),
            difficulties=args.difficulties,
            physical_seeds=args.physical_seeds,
            base_seed=args.seed,
            device_name=args.device,
            target_assignments=tuple(
                ROUTE_TARGET_ASSIGNMENTS[index] for index in args.assignment_indices
            ),
            workers=args.workers,
            act_relational_checkpoint=(
                project_path(args.act_relational_checkpoint)
                if args.act_relational_checkpoint
                else None
            ),
            policy_names=args.policies,
            extra_checkpoints={
                name: project_path(value)
                for name, value in (
                    ("learned_physical", args.physical_checkpoint),
                    ("learned_semantic", args.semantic_checkpoint),
                    ("learned_no_features", args.no_features_checkpoint),
                    ("learned_data20", args.data20_checkpoint),
                    ("learned_data50", args.data50_checkpoint),
                )
                if value
            },
        )
        return save_route_counterfactual(report, project_path(args.output))
    if args.command == "sim-audit-route-assignments":
        from harnessbench.learning.route_paper import (
            audit_route_assignment_feasibility,
            save_route_assignment_audit,
        )
        from harnessbench.sim.envs.route import ROUTE_TARGET_ASSIGNMENTS

        report = audit_route_assignment_feasibility(
            difficulties=args.difficulties,
            physical_seeds=args.physical_seeds,
            base_seed=args.seed,
            target_assignments=tuple(
                ROUTE_TARGET_ASSIGNMENTS[index] for index in args.assignment_indices
            ),
        )
        return save_route_assignment_audit(report, project_path(args.output))
    if args.command == "sim-download-route-public":
        from harnessbench.route_public_data import download_route_public_dataset

        return download_route_public_dataset(
            project_path(args.output),
            include_videos=args.include_videos,
            workers=args.workers,
        )
    if args.command == "sim-inspect-route-public":
        from harnessbench.route_public_data import inspect_route_public_dataset

        return inspect_route_public_dataset(project_path(args.data))
    if args.command == "sim-eval-route-public":
        from harnessbench.learning.route_public import evaluate_route_public_dataset

        return evaluate_route_public_dataset(
            project_path(args.data),
            project_path(args.output),
            seed=args.seed,
        )
    if args.command == "sim-render-route-public":
        from harnessbench.learning.route_public_visuals import render_route_public_multiview

        return render_route_public_multiview(
            project_path(args.data),
            project_path(args.output),
            timestamp_seconds=args.timestamp,
        )
    if args.command == "sim-eval-route-mujoco-direct":
        from harnessbench.learning.route_mujoco_direct import (
            evaluate_route_mujoco_direct,
            save_route_mujoco_direct,
        )

        output_dir = project_path(args.output)
        report = evaluate_route_mujoco_direct(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            output_dir=output_dir,
            physical_seeds=args.physical_seeds,
            assignment_indices=args.assignment_indices,
            policies=args.policies,
            difficulty=args.difficulty,
            base_seed=args.seed,
            physics_steps=args.physics_steps,
            max_steps=args.max_steps,
            device_name=args.device,
            render_representative=not args.no_render,
        )
        return save_route_mujoco_direct(report, output_dir)
    if args.command == "sim-demo-learned":
        from harnessbench.learning.visualize import generate_learned_visual_demos

        checkpoint = args.checkpoint or f"artifacts/learning/{args.mode}_policy.pt"
        return generate_learned_visual_demos(
            project_path(args.output),
            project_path(checkpoint),
            tasks=_task_list(args.tasks),
            mode=args.mode,
            seed=args.seed,
            difficulty=args.difficulty,
            record_every=args.record_every,
            device_name=args.device,
        )
    if args.command == "sim-eval-branch-counterfactual":
        from harnessbench.learning.evaluate import (
            evaluate_branch_counterfactual,
            save_learning_evaluation,
        )

        report = evaluate_branch_counterfactual(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            difficulties=args.difficulties,
            episodes=args.episodes,
            base_seed=args.seed,
            device_name=args.device,
        )
        return save_learning_evaluation(report, project_path(args.output))
    if args.command == "sim-eval-branch-paper":
        from harnessbench.learning.branch_paper import (
            evaluate_branch_paper_table,
            save_branch_paper_table,
        )

        report = evaluate_branch_paper_table(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            project_path(args.act_checkpoint),
            act_relational_checkpoint=(
                project_path(args.act_relational_checkpoint)
                if args.act_relational_checkpoint
                else None
            ),
            difficulties=args.difficulties,
            physical_seeds=args.physical_seeds,
            base_seed=args.seed,
            device_name=args.device,
            workers=args.workers,
            policy_names=args.policies,
            extra_checkpoints={
                name: project_path(value)
                for name, value in (
                    ("learned_physical", args.physical_checkpoint),
                    ("learned_semantic", args.semantic_checkpoint),
                    ("learned_no_features", args.no_features_checkpoint),
                )
                if value
            },
        )
        return save_branch_paper_table(report, project_path(args.output))
    if args.command == "sim-eval-branch-mujoco-direct":
        from harnessbench.learning.branch_mujoco_direct import (
            evaluate_branch_mujoco_direct,
            save_branch_mujoco_direct,
        )

        output_dir = project_path(args.output)
        report = evaluate_branch_mujoco_direct(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            output_dir=output_dir,
            physical_seeds=args.physical_seeds,
            policies=args.policies,
            difficulty=args.difficulty,
            base_seed=args.seed,
            physics_steps=args.physics_steps,
            max_steps=args.max_steps,
            device_name=args.device,
            render_representative=not args.no_render,
        )
        return save_branch_mujoco_direct(report, output_dir)
    if args.command == "sim-mujoco-transfer":
        from harnessbench.learning.mujoco_transfer import (
            evaluate_mujoco_transfer,
            save_mujoco_transfer,
        )

        output_dir = project_path(args.output)
        report = evaluate_mujoco_transfer(
            project_path(args.topology_checkpoint),
            project_path(args.geometry_checkpoint),
            output_dir=output_dir,
            tasks=_task_list(args.tasks),
            episodes=args.episodes,
            difficulty=args.difficulty,
            base_seed=args.seed,
            physics_steps_per_action=args.physics_steps,
            device_name=args.device,
        )
        return save_mujoco_transfer(report, output_dir)
    raise KeyError(args.command)
