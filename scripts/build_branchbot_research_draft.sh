#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

pbd_report="artifacts/papers/branchbot/final/counterfactual/branch_paper_report.json"
mujoco_report="artifacts/papers/branchbot/final/mujoco_direct/branch_mujoco_direct_report.json"
paper_dir="papers/04_branchbot"
release_pdf="output/pdf/04_branchbot_research_draft.pdf"
manifest="artifacts/papers/branchbot/final/branchbot_research_draft.sha256"

test -s "$pbd_report"
test -s "$mujoco_report"

.venv-wsl/bin/ruff check src scripts tests
PYTHONPATH=src .venv-wsl/bin/python -m pytest -q

PYTHONPATH=src .venv-dlolab/bin/python scripts/render_branch_paper_assets.py \
  --report "$pbd_report" \
  --mujoco-report "$mujoco_report" \
  --output artifacts/papers/branchbot/final/counterfactual

PYTHONPATH=src .venv-wsl/bin/python scripts/audit_branchbot_results.py

(
  cd "$paper_dir"
  ../../third_party/tectonic-0.16.9/tectonic main.tex --keep-logs --keep-intermediates
)

mkdir -p "$(dirname "$release_pdf")" "$(dirname "$manifest")"
cp "$paper_dir/main.pdf" "$release_pdf"
sha256sum "$release_pdf" > "$manifest"

printf '%s\n' "$release_pdf"
cat "$manifest"
