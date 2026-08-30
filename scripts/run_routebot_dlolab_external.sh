#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$workspace_dir/.venv-dlolab/bin/python"
target_file="$workspace_dir/third_party/DLO-Lab/genesis/assets/dlo-lab/target_pos/wiring_post_finalpos.npy"
expected_target_sha="4e852122d01b4d0e0e1aa59cd286f749031f22a962ef375a8ab1c3a4a9953072"

# Quadrants loads the unversioned CUDA driver library.  WSL exposes it under
# /usr/lib/wsl/lib, which is visible to PyTorch but is not always on dlopen's
# default search path.
if [[ -f /usr/lib/wsl/lib/libcuda.so ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Missing isolated DLO-Lab interpreter: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$target_file" ]]; then
  echo "Missing official DLO-Lab wiring-post target: $target_file" >&2
  exit 1
fi

actual_target_sha="$(sha256sum "$target_file" | cut -d ' ' -f 1)"
if [[ "$actual_target_sha" != "$expected_target_sha" ]]; then
  echo "DLO-Lab wiring-post target hash mismatch" >&2
  exit 1
fi

cd "$workspace_dir"
"$python_bin" scripts/smoke_dlolab_wiring_post.py \
  --output artifacts/papers/routebot/final/dlolab_external
"$python_bin" scripts/evaluate_dlolab_wiring_post.py \
  --output artifacts/papers/routebot/final/dlolab_external/matched_paths \
  --scene-steps-per-action 20 --max-step-m 0.006 --grasp-height-m 0.03
