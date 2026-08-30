#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tectonic_bin="$workspace_dir/third_party/tectonic-0.16.9/tectonic"

if [[ ! -x "$tectonic_bin" ]]; then
  echo "Missing executable Tectonic compiler: $tectonic_bin" >&2
  exit 1
fi

paper_dirs=(
  "papers/01_inspectbot"
  "papers/02_insertbot"
  "papers/03_routebot"
  "papers/04_branchbot"
)

mkdir -p "$workspace_dir/output/pdf"

for relative_dir in "${paper_dirs[@]}"; do
  paper_dir="$workspace_dir/$relative_dir"
  if [[ ! -f "$paper_dir/main.tex" ]]; then
    echo "Skipping $relative_dir (main.tex not created yet)"
    continue
  fi
  mkdir -p "$paper_dir/build"
  (
    cd "$paper_dir"
    "$tectonic_bin" --keep-logs --outdir build main.tex
  )
  paper_name="$(basename "$relative_dir")"
  cp "$paper_dir/build/main.pdf" "$workspace_dir/output/pdf/${paper_name}.pdf"
done
