#!/usr/bin/env python3
"""Run the frozen MVTec cable perception experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.inspect_perception import run_inspect_perception_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_inspect_perception_experiment(
        args.data,
        args.output,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

