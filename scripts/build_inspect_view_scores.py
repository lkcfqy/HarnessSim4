#!/usr/bin/env python3
"""Build the frozen four-view score table for InspectBot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.learning.inspect_active import build_inspect_view_score_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = build_inspect_view_score_cache(
        args.checkpoint,
        args.data,
        args.output,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
