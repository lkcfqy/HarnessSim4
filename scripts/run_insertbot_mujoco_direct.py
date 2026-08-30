#!/usr/bin/env python3
"""Run the direct closed-loop InsertBot MuJoCo contact audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.insert_mujoco_direct import (
    DirectInsertConfig,
    run_insert_mujoco_direct,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=91_000_000)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    config = DirectInsertConfig(
        physical_seeds_per_difficulty=args.physical_seeds,
        base_seed=args.base_seed,
    )
    report = run_insert_mujoco_direct(
        args.output,
        config=config,
        render_representative=not args.no_render,
    )
    print(
        json.dumps(
            {
                "report": report["report"],
                "episode_rows": report["episode_rows"],
                "episodes_sha256": report["episodes_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
