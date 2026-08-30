#!/usr/bin/env python3
"""Run the frozen paired InsertBot contact benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.insert_paper import InsertPaperConfig, run_insert_paper_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-seeds", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=83_000_000)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    config = InsertPaperConfig(
        physical_seeds_per_difficulty=args.physical_seeds,
        base_seed=args.base_seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    report = run_insert_paper_benchmark(args.output, config=config)
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
