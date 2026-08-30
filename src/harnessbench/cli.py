"""Command-line entry point for the first reproducible project."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from harnessbench.config import load_config, project_path
from harnessbench.public_data import download_public_dataset
from harnessbench.sim.cli import add_sim_subparsers, handle_sim_command


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _doctor() -> int:
    core_modules = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "PIL")}
    parquet = {"installed": importlib.util.find_spec("pyarrow") is not None, "loadable": False}
    if parquet["installed"]:
        try:
            importlib.import_module("pyarrow.parquet")
            parquet["loadable"] = True
        except Exception as exc:  # noqa: BLE001 - diagnostic must report loader failures
            parquet["error"] = f"{type(exc).__name__}: {exc}"
    ready = all(core_modules.values())
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "core_dependencies": core_modules,
        "optional_parquet_reader": parquet,
        "portable_read_chain": (
            "existing NumPy cache -> pure-JavaScript hyparquet -> PyArrow -> "
            "Hugging Face Dataset Viewer API"
        ),
        "ready": ready,
        "next_step": ("python -m harnessbench run-all" if ready else "python -m pip install -e ."),
    }
    _print(result)
    return 0 if result["ready"] else 1


def _paths(config: dict) -> tuple[Path, Path]:
    return project_path(config["public_data_dir"]), project_path(config["artifact_dir"])


def _download(config: dict, include_video: bool) -> dict:
    data_dir, _ = _paths(config)
    return download_public_dataset(data_dir, include_video=include_video)


def _inspect(config: dict) -> dict:
    from harnessbench.lerobot import dataset_summary, load_trajectories

    data_dir, _ = _paths(config)
    summary = dataset_summary(load_trajectories(data_dir))
    return {"data_dir": str(data_dir.resolve()), **summary}


def _train(config: dict) -> dict:
    from harnessbench.training import train_public_baseline

    data_dir, artifact_dir = _paths(config)
    return train_public_baseline(
        data_dir=data_dir,
        artifact_dir=artifact_dir,
        train_fraction=float(config["train_episode_fraction"]),
        seed=int(config["seed"]),
        alpha=float(config["ridge_alpha"]),
    )


def _synthetic(config: dict) -> dict:
    from harnessbench.synthetic import SyntheticSpec, generate_synthetic_dataset

    synthetic = config["synthetic"]
    return generate_synthetic_dataset(
        project_path(synthetic["output_dir"]),
        SyntheticSpec(
            episodes=int(synthetic["episodes"]),
            frames_per_episode=int(synthetic["frames_per_episode"]),
            image_size=int(synthetic["image_size"]),
            seed=int(config["seed"]),
        ),
    )


def _run_all(config: dict, include_video: bool) -> dict:
    manifest = _download(config, include_video)
    inspection = _inspect(config)
    report = _train(config)
    synthetic = _synthetic(config)
    summary = {
        "project": config["project"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_download_files": len(manifest["files"]),
        "public_dataset": inspection,
        "held_out_metrics": report["held_out_metrics"],
        "synthetic_dataset": synthetic,
        "next_milestone": (
            "Collect 30-60 successful and 20-30 failed trials for one real connector SKU, "
            "following docs/REAL_DATA_PROTOCOL.md."
        ),
    }
    output = project_path("artifacts/run_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harnessbench",
        description="HarnessSim4 topology-aware wire-harness simulation benchmark",
    )
    parser.add_argument("--config", help="Path to a JSON config file")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check the local Python environment")

    download = subparsers.add_parser("download-public", help="Download the MIT public dataset")
    download.add_argument(
        "--include-video", action="store_true", help="Also download the 88 MB video"
    )
    subparsers.add_parser("inspect-public", help="Validate and summarize downloaded parquet data")
    subparsers.add_parser("train-public", help="Train and evaluate the state-action baseline")
    subparsers.add_parser("generate-synthetic", help="Generate wire-terminal visual scenes")
    run_all = subparsers.add_parser("run-all", help="Download, validate, train and generate")
    run_all.add_argument(
        "--include-video", action="store_true", help="Also download the 88 MB video"
    )
    add_sim_subparsers(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        raise SystemExit(_doctor())
    if args.command.startswith("sim-"):
        _print(handle_sim_command(args))
        return

    config = load_config(args.config)
    if args.command == "download-public":
        result = _download(config, args.include_video)
    elif args.command == "inspect-public":
        result = _inspect(config)
    elif args.command == "train-public":
        result = _train(config)
    elif args.command == "generate-synthetic":
        result = _synthetic(config)
    elif args.command == "run-all":
        result = _run_all(config, args.include_video)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(result)
