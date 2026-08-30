#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core_python="$workspace_dir/.venv-wsl/bin/python"
plot_python="$workspace_dir/.venv-dlolab/bin/python"
test_python="$workspace_dir/.venv-dlolab/bin/python"
ruff_bin="$workspace_dir/.venv-wsl/bin/ruff"
tectonic_bin="$workspace_dir/third_party/tectonic-0.16.9/tectonic"
paper_dir="$workspace_dir/papers/03_routebot"
output_pdf="$workspace_dir/output/pdf/03_routebot_research_draft.pdf"
audit_file="$workspace_dir/artifacts/papers/routebot/final/audit/routebot_result_audit.json"
manifest_file="$workspace_dir/artifacts/papers/routebot/final/routebot_research_draft.sha256"

for executable in "$core_python" "$plot_python" "$test_python" "$ruff_bin" "$tectonic_bin"; do
  if [[ ! -x "$executable" ]]; then
    echo "Missing required executable: $executable" >&2
    exit 1
  fi
done

cd "$workspace_dir"
bash -n scripts/run_routebot_dlolab_external.sh
"$ruff_bin" check src scripts tests
PYTHONPATH=src "$test_python" -m pytest -q
"$core_python" scripts/render_dlolab_wiring_post_table.py
"$plot_python" scripts/plot_route_counterfactual_heatmap.py
PYTHONPATH=src "$core_python" scripts/audit_routebot_results.py

mkdir -p "$paper_dir/build" "$workspace_dir/output/pdf"
(
  cd "$paper_dir"
  "$tectonic_bin" --keep-logs --outdir build main.tex
)
cp "$paper_dir/build/main.pdf" "$output_pdf"

manifest_files=(
  "output/pdf/03_routebot_research_draft.pdf"
  "artifacts/papers/routebot/final/audit/routebot_result_audit.json"
  "artifacts/papers/routebot/final/main_table/route_paper_report.json"
  "artifacts/papers/routebot/final/counterfactual/route_counterfactual_report.json"
  "artifacts/papers/routebot/public_real/report.json"
  "artifacts/papers/routebot/final/mujoco_direct/route_mujoco_direct_report.json"
  "artifacts/papers/routebot/final/dlolab_external/wiring_post_smoke_report.json"
  "artifacts/papers/routebot/final/dlolab_external/matched_paths/matched_path_report.json"
  "artifacts/papers/routebot/final/counterfactual/route_counterfactual_heatmap.pdf"
  "artifacts/papers/routebot/final/counterfactual/route_counterfactual_heatmap.png"
  "artifacts/papers/routebot/final/data/route_paired_source_432.npz"
  "artifacts/papers/routebot/final/data/route_paired_success_432.npz"
)
sha256sum "${manifest_files[@]}" > "$manifest_file"

echo "Audit: $audit_file"
echo "PDF: $output_pdf"
echo "Manifest: $manifest_file"
