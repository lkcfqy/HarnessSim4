#!/usr/bin/env python3
"""Audit the public FAU/FAPS Stripped Wire Dataset release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnessbench.inspect_wire_data import audit_stripped_wire


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_stripped_wire(args.data, args.output, zip_path=args.zip)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
