#!/usr/bin/env python3
"""Backward-compatible wrapper for the hardware-study initializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardware_study import initialize_study


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, choices=("inspectbot", "insertbot", "routebot", "branchbot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path(f"configs/hardware/{args.robot}_gate.json")
    initialize_study(config_path, args.output)
    print(json.dumps({"robot": args.robot, "study_root": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
