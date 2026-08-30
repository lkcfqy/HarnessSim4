#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$workspace_dir/.venv-wsl/bin/python"
worker_count="${HARNESS_ROUTE_WORKERS:-8}"

cd "$workspace_dir"
export PYTHONPATH=src

"$python_bin" -m harnessbench sim-generate-demos \
  --tasks route \
  --train-episodes 270 --validation-episodes 81 --test-episodes 81 \
  --route-assignment-indices 0 1 2 3 4 5 6 7 8 \
  --record-every 2 --seed 433000 \
  --output artifacts/papers/routebot/final/data/route_paired_source_432.npz

"$python_bin" -m harnessbench sim-filter-successful-demos \
  --source artifacts/papers/routebot/final/data/route_paired_source_432.npz \
  --train-episodes 270 --validation-episodes 81 --test-episodes 81 \
  --output artifacts/papers/routebot/final/data/route_paired_success_432.npz

"$python_bin" -m harnessbench sim-filter-successful-demos \
  --source artifacts/papers/routebot/final/data/route_paired_source_432.npz \
  --train-episodes 54 --validation-episodes 81 --test-episodes 81 \
  --output artifacts/papers/routebot/final/data/route_paired_train20.npz

"$python_bin" -m harnessbench sim-filter-successful-demos \
  --source artifacts/papers/routebot/final/data/route_paired_source_432.npz \
  --train-episodes 135 --validation-episodes 81 --test-episodes 81 \
  --output artifacts/papers/routebot/final/data/route_paired_train50.npz

"$python_bin" -m harnessbench sim-train-policy \
  --dataset artifacts/papers/routebot/final/data/route_paired_success_432.npz \
  --mode both --epochs 150 --batch-size 512 --learning-rate 0.0005 \
  --weight-decay 0.0001 --hidden-dim 64 --layers 3 --patience 25 \
  --seed 202611 --output artifacts/papers/routebot/final/models/seed_202611

for ablation_mode in physical semantic features; do
  "$python_bin" -m harnessbench sim-train-policy \
    --dataset artifacts/papers/routebot/final/data/route_paired_success_432.npz \
    --mode "$ablation_mode" --epochs 150 --batch-size 512 --learning-rate 0.0005 \
    --weight-decay 0.0001 --hidden-dim 64 --layers 3 --patience 25 \
    --seed 202611 --output artifacts/papers/routebot/final/models/ablations
done

for data_fraction in 20 50; do
  "$python_bin" -m harnessbench sim-train-policy \
    --dataset "artifacts/papers/routebot/final/data/route_paired_train${data_fraction}.npz" \
    --mode topology --epochs 150 --batch-size 512 --learning-rate 0.0005 \
    --weight-decay 0.0001 --hidden-dim 64 --layers 3 --patience 25 \
    --seed 202611 --output "artifacts/papers/routebot/final/models/data${data_fraction}"
done

"$python_bin" -m harnessbench sim-train-act \
  --dataset artifacts/papers/routebot/final/data/route_paired_success_432.npz \
  --tasks route --epochs 120 --batch-size 256 --learning-rate 0.001 \
  --weight-decay 0.0001 --hidden-dim 48 --heads 4 --layers 1 \
  --chunk-size 8 --latent-dim 8 --kl-weight 0.1 --pointer-loss-weight 1.0 \
  --patience 20 --seed 202612 \
  --output artifacts/papers/routebot/final/models/seed_202612/act_geometry.pt

"$python_bin" -m harnessbench sim-train-act \
  --dataset artifacts/papers/routebot/final/data/route_paired_success_432.npz \
  --tasks route --use-relations --epochs 120 --batch-size 256 \
  --learning-rate 0.001 --weight-decay 0.0001 --hidden-dim 48 --heads 4 \
  --layers 1 --chunk-size 8 --latent-dim 8 --kl-weight 0.1 \
  --pointer-loss-weight 1.0 --patience 20 --seed 202612 \
  --output artifacts/papers/routebot/final/models/seed_202612/act_relational.pt

"$python_bin" -m harnessbench sim-eval-route-paper \
  --episodes 100 --difficulties 0.2 0.5 0.8 --workers "$worker_count" \
  --topology-checkpoint artifacts/papers/routebot/final/models/seed_202611/topology_policy.pt \
  --geometry-checkpoint artifacts/papers/routebot/final/models/seed_202611/geometry_policy.pt \
  --act-checkpoint artifacts/papers/routebot/final/models/seed_202612/act_geometry.pt \
  --act-relational-checkpoint artifacts/papers/routebot/final/models/seed_202612/act_relational.pt \
  --physical-checkpoint artifacts/papers/routebot/final/models/ablations/physical_policy.pt \
  --semantic-checkpoint artifacts/papers/routebot/final/models/ablations/semantic_policy.pt \
  --no-features-checkpoint artifacts/papers/routebot/final/models/ablations/features_policy.pt \
  --data20-checkpoint artifacts/papers/routebot/final/models/data20/topology_policy.pt \
  --data50-checkpoint artifacts/papers/routebot/final/models/data50/topology_policy.pt \
  --seed 61000000 --output artifacts/papers/routebot/final/main_table

"$python_bin" -m harnessbench sim-eval-route-counterfactual \
  --physical-seeds 100 --difficulties 0.2 0.5 0.8 --workers "$worker_count" \
  --assignment-indices 0 1 2 3 4 5 6 7 8 \
  --topology-checkpoint artifacts/papers/routebot/final/models/seed_202611/topology_policy.pt \
  --geometry-checkpoint artifacts/papers/routebot/final/models/seed_202611/geometry_policy.pt \
  --act-checkpoint artifacts/papers/routebot/final/models/seed_202612/act_geometry.pt \
  --act-relational-checkpoint artifacts/papers/routebot/final/models/seed_202612/act_relational.pt \
  --policies learned_topology learned_geometry act_chunk_typed \
    act_relational_typed teacher_topology \
  --seed 62000000 --output artifacts/papers/routebot/final/counterfactual

"$python_bin" -m harnessbench sim-eval-route-mujoco-direct \
  --physical-seeds 20 --assignment-indices 0 1 2 3 4 5 6 7 8 \
  --policies learned_topology learned_geometry teacher_topology \
  --physics-steps 4 --max-steps 220 --no-render \
  --topology-checkpoint artifacts/papers/routebot/final/models/seed_202611/topology_policy.pt \
  --geometry-checkpoint artifacts/papers/routebot/final/models/seed_202611/geometry_policy.pt \
  --output artifacts/papers/routebot/final/mujoco_direct

MUJOCO_GL=egl "$python_bin" -m harnessbench sim-eval-route-mujoco-direct \
  --physical-seeds 1 --assignment-indices 1 --policies learned_topology \
  --physics-steps 4 --max-steps 220 \
  --topology-checkpoint artifacts/papers/routebot/final/models/seed_202611/topology_policy.pt \
  --geometry-checkpoint artifacts/papers/routebot/final/models/seed_202611/geometry_policy.pt \
  --output artifacts/papers/routebot/final/mujoco_direct_visual
