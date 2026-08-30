#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

root="artifacts/papers/inspectbot/final"
paper_dir="papers/01_inspectbot"
release_pdf="output/pdf/01_inspectbot_research_draft.pdf"
manifest="$root/inspectbot_research_draft.sha256"

test -s "$root/perception/inspect_perception_report.json"
test -s "$root/stripped_wire/stripped_wire_report.json"
test -s "$root/active/stress_v1/inspect_active_report.json"
test -s "$root/active/deployment_v2_corrected/inspect_active_report.json"
test -s "$root/mujoco_direct/inspect_mujoco_direct_report.json"

.venv-wsl/bin/ruff check src scripts tests

# The core simulation suite is dependency-light; the perception/active tests use
# the separate frozen torchvision/scikit-learn environment.
PYTHONPATH=src .venv-wsl/bin/python -m pytest -q \
  --ignore=tests/test_inspect_active.py \
  --ignore=tests/test_inspect_wire_external.py \
  --ignore=tests/test_inspect_mujoco_direct.py
PYTHONPATH=src .venv-dlolab/bin/python -m pytest -q \
  tests/test_inspect_active.py \
  tests/test_inspect_wire_external.py \
  tests/test_inspect_mujoco_direct.py \
  tests/test_mujoco_assets.py \
  tests/test_realistic_robots.py

PYTHONPATH=src .venv-dlolab/bin/python scripts/audit_inspectbot_results.py
PYTHONPATH=src .venv-dlolab/bin/python scripts/render_inspect_paper_assets.py

(
  cd "$paper_dir"
  ../../third_party/tectonic-0.16.9/tectonic main.tex --keep-logs --keep-intermediates
)

mkdir -p "$(dirname "$release_pdf")" "$(dirname "$manifest")"
cp "$paper_dir/main.pdf" "$release_pdf"
sha256sum "$release_pdf" > "$manifest"

printf '%s\n' "$release_pdf"
cat "$manifest"
