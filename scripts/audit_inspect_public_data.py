#!/usr/bin/env python3
"""Audit and index the pinned MVTec AD cable data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.inspect_public_data import audit_mvtec_cable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_mvtec_cable(args.data, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

