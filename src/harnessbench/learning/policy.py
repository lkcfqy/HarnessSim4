"""Closed-loop policy adapter for a trained TopoHarness checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from harnessbench.learning.graph import (
    CONFIDENCE,
    ROLE_CABLE,
    ROLE_CLIP,
    ROLE_SITE,
    ROLE_TARGET,
    SEMANTIC_A,
    SEMANTIC_B,
    STATE_ACTIVE,
    STATE_DETECTED,
    STATE_INSPECTED,
    TASK_TO_ID,
    EncodedGraph,
    binary_action_mask,
    denormalize_action,
    encode_environment,
)
from harnessbench.learning.train import load_policy_model
from harnessbench.sim.envs.base import HarnessEnv


class LearnedGraphPolicy:
    """Type-constrained graph-pointer policy with observable action grounding."""

    route_grasp_distance = 0.032

    def __init__(
        self,
        checkpoint: Path,
        task: str,
        *,
        name: str,
        device_name: str = "cpu",
    ) -> None:
        self.model = load_policy_model(checkpoint, device_name=device_name)
        self.task = task
        self.name = name
        self.seed = 0
        self.device = next(self.model.parameters()).device
        self.last_debug: dict = {}

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def _pointer_candidates(
        self,
        graph: EncodedGraph,
        env: HarnessEnv,
        pointer_slot: int,
    ) -> np.ndarray:
        features = graph.node_features
        candidates = graph.node_mask > 0.0
        if self.task == "inspect":
            sites = candidates & (features[:, ROLE_SITE] > 0.5)
            uninspected = sites & (features[:, STATE_INSPECTED] < 0.5)
            if np.any(uninspected):
                return uninspected
            unresolved = (
                sites
                & (features[:, STATE_DETECTED] < 0.5)
                & (features[:, STATE_ACTIVE] < 0.999)
                & (features[:, CONFIDENCE] > 0.30)
            )
            candidates = unresolved if np.any(unresolved) else sites
        elif self.task == "insert":
            candidates &= features[:, ROLE_TARGET] > 0.5
        elif self.task == "route":
            observation = env.observation()
            if observation["grasp_idx"] is None:
                candidates &= features[:, ROLE_CABLE] > 0.5
                for particle in observation["bindings"].values():
                    candidates[int(particle)] = False
            else:
                target = features[:, ROLE_TARGET] > 0.5
                cable = features[:, ROLE_CABLE] > 0.5
                active_clip = (features[:, ROLE_CLIP] > 0.5) & (features[:, STATE_ACTIVE] > 0.5)
                candidates &= target & ~cable & ~active_clip
        elif self.task == "branch":
            candidates &= features[:, ROLE_TARGET] > 0.5
            if self.model.config.use_topology:
                semantic_column = SEMANTIC_A if pointer_slot == 0 else SEMANTIC_B
                typed_candidates = candidates & (features[:, semantic_column] > 0.5)
                # The no-unary-feature ablation can still infer endpoint--target
                # identity through semantic edges.  Keep its legal action set at
                # target nodes instead of falling back to arbitrary cable nodes.
                if np.any(typed_candidates):
                    candidates = typed_candidates
        if not np.any(candidates):
            return graph.node_mask > 0.0
        return candidates

    def _ground_inspection_trigger(
        self,
        env: HarnessEnv,
        graph: EncodedGraph,
        pointer_index: int,
        normalized: np.ndarray,
    ) -> dict:
        observation = env.observation()
        feature = graph.node_features[pointer_index]
        distance = float(np.linalg.norm(np.asarray(observation["camera"]) - feature[:2]))
        needs_scan = bool(
            feature[STATE_INSPECTED] < 0.5
            or (
                feature[STATE_DETECTED] < 0.5
                and feature[STATE_ACTIVE] < 0.999
                and feature[CONFIDENCE] > 0.30
            )
        )
        trigger_distance = 0.35 * float(observation["fov_radius"])
        normalized[2] = float(needs_scan and distance <= trigger_distance)
        return {
            "phase": "approach_or_scan_observable_site",
            "needs_scan": needs_scan,
            "camera_pointer_distance": distance,
            "trigger_distance": trigger_distance,
        }

    def _ground_route_gripper(
        self,
        env: HarnessEnv,
        graph: EncodedGraph,
        pointer_index: int,
        normalized: np.ndarray,
    ) -> dict:
        observation = env.observation()
        pointer_is_cable = bool(graph.node_features[pointer_index, ROLE_CABLE] > 0.5)
        distance = float(
            np.linalg.norm(
                np.asarray(observation["arm_ee"]) - graph.node_features[pointer_index, :2]
            )
        )
        if observation["grasp_idx"] is None:
            normalized[2] = float(pointer_is_cable and distance <= self.route_grasp_distance)
            phase = "approach_cable"
        else:
            normalized[2] = 1.0
            phase = "transport_grasped_cable"
        return {
            "phase": phase,
            "pointer_is_cable": pointer_is_cable,
            "arm_pointer_distance": distance,
            "grasp_distance": self.route_grasp_distance,
        }

    @torch.no_grad()
    def act(self, env: HarnessEnv) -> np.ndarray:
        graph = encode_environment(self.task, env)

        def tensor(value: np.ndarray, *, integer: bool = False) -> torch.Tensor:
            dtype = torch.long if integer else torch.float32
            return torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)

        action_logits, pointer_logits = self.model.forward_with_pointers(
            tensor(graph.node_features),
            tensor(graph.physical_adjacency),
            tensor(graph.semantic_adjacency),
            tensor(graph.node_mask),
            tensor(graph.global_features),
            tensor(np.asarray(graph.task_id), integer=True),
        )
        raw_probabilities = torch.sigmoid(action_logits[0]).cpu().numpy()
        normalized = raw_probabilities.copy()
        thresholds = np.asarray(self.model.binary_thresholds[TASK_TO_ID[self.task]])
        for dimension in np.flatnonzero(binary_action_mask(self.task) > 0.0):
            normalized[dimension] = float(normalized[dimension] >= thresholds[dimension])
        pointer_indices: list[int] = []
        pointer_candidate_counts: list[int] = []
        action_grounding = None
        pointer_slots = 2 if self.task == "branch" else 1
        for pointer_slot in range(pointer_slots):
            candidates = self._pointer_candidates(graph, env, pointer_slot)
            pointer_candidate_counts.append(int(np.sum(candidates)))
            scores = pointer_logits[0, pointer_slot].cpu().numpy()
            scores = np.where(candidates, scores, -np.inf)
            pointer_indices.append(int(np.argmax(scores)))
        pointer_index = pointer_indices[0]
        normalized[:2] = graph.node_features[pointer_index, :2]
        if self.task == "branch":
            normalized[3:5] = graph.node_features[pointer_indices[1], :2]
            normalized[2] = normalized[5] = 1.0
            action_grounding = {
                "phase": "typed_dual_arm_pointer",
                "topology_candidates": bool(self.model.config.use_topology),
            }
        elif self.task == "inspect":
            action_grounding = self._ground_inspection_trigger(
                env, graph, pointer_index, normalized
            )
        elif self.task == "route":
            action_grounding = self._ground_route_gripper(env, graph, pointer_index, normalized)
        action = denormalize_action(self.task, normalized)
        self.last_debug = {
            "pointer_index": pointer_index,
            "pointer_indices": pointer_indices,
            "pointer_candidate_count": pointer_candidate_counts[0],
            "pointer_candidate_counts": pointer_candidate_counts,
            "pointer_position": (
                graph.node_features[pointer_index, :2].tolist()
                if pointer_index is not None
                else None
            ),
            "pointer_positions": [
                graph.node_features[index, :2].tolist() for index in pointer_indices
            ],
            "raw_action_probabilities": raw_probabilities.tolist(),
            "normalized_action": normalized.tolist(),
            "action": action.tolist(),
            "action_grounding": action_grounding,
        }
        return action
