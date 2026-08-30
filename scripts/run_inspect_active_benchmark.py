#!/usr/bin/env python3
"""Run the frozen paired InspectBot active-inspection benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.inspect_active import (
    inspect_deployment_v2_config,
    run_inspect_active_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("stress_v1", "deployment_v2"),
        default="stress_v1",
    )
    args = parser.parse_args()
    config = inspect_deployment_v2_config() if args.protocol == "deployment_v2" else None
    report = run_inspect_active_benchmark(args.scores, args.output, config=config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
