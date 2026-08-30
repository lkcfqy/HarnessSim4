from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harnessbench.learning.dataset import (
    filter_successful_expert_dataset,
    generate_expert_dataset,
    load_expert_dataset,
)
from harnessbench.learning.graph import (
    ACTION_DIMS,
    FEATURE_DIM,
    GLOBAL_DIM,
    MAX_NODES,
    TASK_ORDER,
    TOPOLOGY_FEATURE_COLUMNS,
    denormalize_action,
    encode_environment,
    normalize_action,
)
from harnessbench.sim.envs import make_env


class LearningGraphTests(unittest.TestCase):
    def test_all_tasks_encode_to_shared_schema(self) -> None:
        for task_index, task in enumerate(TASK_ORDER):
            with self.subTest(task=task):
                env = make_env(task)
                env.reset(seed=100 + task_index, difficulty=0.5)
                graph = encode_environment(task, env)
                self.assertEqual(graph.node_features.shape, (MAX_NODES, FEATURE_DIM))
                self.assertEqual(graph.physical_adjacency.shape, (MAX_NODES, MAX_NODES))
                self.assertEqual(graph.semantic_adjacency.shape, (MAX_NODES, MAX_NODES))
                self.assertEqual(graph.global_features.shape, (GLOBAL_DIM,))
                self.assertEqual(graph.task_id, task_index)
                self.assertGreater(graph.node_mask.sum(), 0)
                self.assertGreater(graph.physical_adjacency.sum(), 0)
                self.assertGreater(graph.semantic_adjacency.sum(), 0)
                np.testing.assert_allclose(graph.physical_adjacency, graph.physical_adjacency.T)
                np.testing.assert_allclose(graph.semantic_adjacency, graph.semantic_adjacency.T)

    def test_action_normalization_round_trip(self) -> None:
        examples = {
            "inspect": np.asarray([0.2, 0.7, 1.0]),
            "insert": np.asarray([0.4, 0.6, -0.7, 1.0]),
            "route": np.asarray([0.8, 0.3, 0.0]),
            "branch": np.asarray([0.2, 0.3, 1.0, 0.8, 0.7, 1.0]),
        }
        for task, action in examples.items():
            with self.subTest(task=task):
                restored = denormalize_action(task, normalize_action(task, action))
                np.testing.assert_allclose(restored, action[: ACTION_DIMS[task]], atol=1e-6)

    def test_inspect_expert_does_not_read_hidden_defect_truth(self) -> None:
        env = make_env("inspect")
        env.reset(seed=7, difficulty=0.5)
        env.inspected = set(range(len(env.sites)))
        env.scan_attempts = {index: 1 for index in range(len(env.sites))}
        env.site_scores[:] = 0.6
        env.detections.clear()
        rng = np.random.default_rng(1)
        action_before = env.policy_action("topology", rng)
        env.defects = np.logical_not(env.defects)
        action_after = env.policy_action("topology", rng)
        np.testing.assert_allclose(action_before, action_after)

    def test_branch_semantic_counterfactual_preserves_geometry(self) -> None:
        encoded = []
        target_sets = []
        for swap in (False, True):
            env = make_env("branch")
            env.semantic_swap_override = swap
            observation, _ = env.reset(seed=91, difficulty=0.5)
            graph = encode_environment("branch", env)
            features = graph.node_features[graph.node_mask > 0.0].copy()
            features[:, list(TOPOLOGY_FEATURE_COLUMNS)] = 0.0
            encoded.append(sorted(tuple(np.round(row, 7)) for row in features))
            target_sets.append(
                sorted(
                    tuple(np.round(value, 7)) for value in observation["semantic_targets"].values()
                )
            )
        self.assertEqual(target_sets[0], target_sets[1])
        self.assertEqual(encoded[0], encoded[1])

    def test_route_target_assignment_is_encoded_only_by_relational_edges(self) -> None:
        encoded = []
        target_indices = []
        semantic_adjacencies = []
        for assignment in ((7, 15, 23), (5, 17, 25)):
            env = make_env("route")
            env.target_indices_override = assignment
            observation, _ = env.reset(seed=92, difficulty=0.5)
            graph = encode_environment("route", env)
            encoded.append(graph.node_features.copy())
            target_indices.append(observation["target_particle_indices"])
            semantic_adjacencies.append(graph.semantic_adjacency)
        np.testing.assert_allclose(encoded[0], encoded[1])
        self.assertFalse(np.array_equal(target_indices[0], target_indices[1]))
        self.assertFalse(np.array_equal(semantic_adjacencies[0], semantic_adjacencies[1]))

    def test_dataset_split_is_episode_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.npz"
            result = generate_expert_dataset(
                path,
                tasks=["insert"],
                train_episodes=1,
                validation_episodes=1,
                test_episodes=1,
                record_every=10,
                base_seed=80,
            )
            arrays = load_expert_dataset(path)
            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            split_seed_sets = [set(values) for values in metadata["split_seeds"].values()]
            self.assertTrue(all(split_seed_sets))
            self.assertFalse(split_seed_sets[0] & split_seed_sets[1])
            self.assertFalse(split_seed_sets[0] & split_seed_sets[2])
            self.assertFalse(split_seed_sets[1] & split_seed_sets[2])
            self.assertEqual(len(arrays["task_id"]), result["samples"])

    def test_successful_filter_balances_route_relational_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            assignments = ([5, 13, 21], [7, 15, 23])
            difficulties = (0.2, 0.8)
            episodes = []
            seeds = []
            split_ids = []
            for split_id, split_name in enumerate(("train", "validation", "test")):
                for difficulty in difficulties:
                    for assignment in assignments:
                        seed = 1000 + 100 * split_id + len(seeds)
                        seeds.append(seed)
                        split_ids.append(split_id)
                        episodes.append(
                            {
                                "task": "route",
                                "split": split_name,
                                "seed": seed,
                                "difficulty": difficulty,
                                "success": True,
                                "steps": 10,
                                "samples": 1,
                                "semantic_target_swap": None,
                                "route_target_assignment": assignment,
                            }
                        )
            np.savez_compressed(
                source,
                episode_seed=np.asarray(seeds, dtype=np.int64),
                split=np.asarray(split_ids, dtype=np.int64),
                task_id=np.full(len(seeds), 2, dtype=np.int64),
                value=np.arange(len(seeds), dtype=np.float32),
            )
            source.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "tasks": ["route"],
                        "difficulties": list(difficulties),
                        "record_every": 1,
                        "observation_scope": "observable only",
                        "episodes": episodes,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "filtered.npz"
            result = filter_successful_expert_dataset(
                source,
                output,
                train_episodes=4,
                validation_episodes=4,
                test_episodes=4,
            )
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(result["selected_episode_count"], 12)
            self.assertTrue(
                all(
                    count == 1
                    for split_counts in metadata["selection_counts"]["route"].values()
                    for count in split_counts.values()
                )
            )
            self.assertEqual(metadata["episode_success_rate"], {name: 1.0 for name in ("train", "validation", "test")})


if __name__ == "__main__":
    unittest.main()
