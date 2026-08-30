#!/usr/bin/env python3
"""Run the InspectBot direct MuJoCo scan-execution audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.inspect_mujoco_direct import evaluate_inspect_mujoco_direct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=78_000_000)
    parser.add_argument("--physics-steps", type=int, default=4)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    report = evaluate_inspect_mujoco_direct(
        args.scores,
        args.output,
        physical_seeds=args.physical_seeds,
        base_seed=args.base_seed,
        physics_steps=args.physics_steps,
        render=not args.no_render,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
