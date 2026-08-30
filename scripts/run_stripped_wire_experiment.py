#!/usr/bin/env python3
"""Run InspectBot's independent Stripped Wire validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.inspect_wire_external import run_stripped_wire_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_stripped_wire_experiment(
        args.data,
        args.zip,
        args.output,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
