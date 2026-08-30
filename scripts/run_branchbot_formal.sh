#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

topology="artifacts/papers/branchbot/final/models/graph_topology/topology_policy.pt"
geometry="artifacts/papers/branchbot/final/models/graph_geometry/geometry_policy.pt"
physical="artifacts/papers/branchbot/final/models/graph_physical/physical_policy.pt"
semantic="artifacts/papers/branchbot/final/models/graph_semantic/semantic_policy.pt"
no_features="artifacts/papers/branchbot/final/models/graph_no_features/features_policy.pt"
act_geometry="artifacts/papers/branchbot/final/models/act_geometry.pt"
act_relational="artifacts/papers/branchbot/final/models/act_relational.pt"

for checkpoint in \
  "$topology" "$geometry" "$physical" "$semantic" "$no_features" \
  "$act_geometry" "$act_relational"; do
  test -s "$checkpoint"
done

PYTHONPATH=src .venv-wsl/bin/python -m harnessbench sim-eval-branch-paper \
  --difficulties 0.2 0.5 0.8 \
  --physical-seeds 100 \
  --seed 66000000 \
  --device cpu \
  --workers 12 \
  --policies \
    learned_topology learned_geometry learned_physical learned_semantic \
    learned_no_features act_chunk act_chunk_typed act_relational_typed \
    teacher_topology random \
  --topology-checkpoint "$topology" \
  --geometry-checkpoint "$geometry" \
  --physical-checkpoint "$physical" \
  --semantic-checkpoint "$semantic" \
  --no-features-checkpoint "$no_features" \
  --act-checkpoint "$act_geometry" \
  --act-relational-checkpoint "$act_relational" \
  --output artifacts/papers/branchbot/final/counterfactual

MUJOCO_GL=egl PYTHONPATH=src .venv-wsl/bin/python -m harnessbench \
  sim-eval-branch-mujoco-direct \
  --physical-seeds 20 \
  --policies learned_topology learned_geometry teacher_topology \
  --difficulty 0.5 \
  --seed 65000000 \
  --physics-steps 6 \
  --max-steps 300 \
  --topology-checkpoint "$topology" \
  --geometry-checkpoint "$geometry" \
  --output artifacts/papers/branchbot/final/mujoco_direct

scripts/build_branchbot_research_draft.sh
