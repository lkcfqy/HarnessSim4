from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harnessbench.learning.graph import (
    ROLE_TARGET,
    SEMANTIC_A,
    SEMANTIC_B,
    encode_environment,
)
from harnessbench.learning.policy import LearnedGraphPolicy
from harnessbench.sim.envs import make_env


class LearningModelTests(unittest.TestCase):
    def test_branch_no_unary_ablation_falls_back_to_legal_target_nodes(self) -> None:
        env = make_env("branch")
        env.reset(seed=7, difficulty=0.5)
        graph = encode_environment("branch", env)
        graph.node_features[:, (SEMANTIC_A, SEMANTIC_B)] = 0.0
        policy = object.__new__(LearnedGraphPolicy)
        policy.task = "branch"
        policy.model = SimpleNamespace(config=SimpleNamespace(use_topology=True))
        candidates = policy._pointer_candidates(graph, env, pointer_slot=0)
        expected = (graph.node_mask > 0.0) & (graph.node_features[:, ROLE_TARGET] > 0.5)
        np.testing.assert_array_equal(candidates, expected)

    def test_topology_and_geometry_forward_have_same_shape_and_parameters(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is an optional learning dependency")
        from harnessbench.learning.model import ModelConfig, TopoHarnessNet

        env = make_env("branch")
        env.reset(seed=3, difficulty=0.5)
        graph = encode_environment("branch", env)
        inputs = [
            torch.from_numpy(value).unsqueeze(0)
            for value in (
                graph.node_features,
                graph.physical_adjacency,
                graph.semantic_adjacency,
                graph.node_mask,
                graph.global_features,
            )
        ]
        task_id = torch.as_tensor([graph.task_id], dtype=torch.long)
        topology = TopoHarnessNet(ModelConfig(use_topology=True))
        geometry = TopoHarnessNet(ModelConfig(use_topology=False))
        physical_only = TopoHarnessNet(ModelConfig(adjacency_mode="physical"))
        semantic_only = TopoHarnessNet(ModelConfig(adjacency_mode="semantic"))
        no_features = TopoHarnessNet(ModelConfig(use_topology_features=False))
        output_topology = topology(*inputs, task_id)
        output_geometry = geometry(*inputs, task_id)
        self.assertEqual(tuple(output_topology.shape), (1, 6))
        self.assertEqual(tuple(output_geometry.shape), (1, 6))
        self.assertTrue(np.isfinite(output_topology.detach().numpy()).all())
        self.assertEqual(
            sum(parameter.numel() for parameter in topology.parameters()),
            sum(parameter.numel() for parameter in geometry.parameters()),
        )
        expected_parameters = sum(parameter.numel() for parameter in topology.parameters())
        for model in (physical_only, semantic_only, no_features):
            with self.subTest(mode=model.config):
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), expected_parameters)
                self.assertEqual(tuple(model(*inputs, task_id).shape), (1, 6))


if __name__ == "__main__":
    unittest.main()
