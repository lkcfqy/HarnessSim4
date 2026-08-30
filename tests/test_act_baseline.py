from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harnessbench.learning.act_baseline import (
    ActChunkConfig,
    ActChunkNet,
    ActChunkPolicy,
    ActionChunkDataset,
)
from harnessbench.learning.graph import (
    FEATURE_DIM,
    GLOBAL_DIM,
    MAX_ACTION_DIM,
    MAX_NODES,
    ROLE_TARGET,
    SEMANTIC_A,
    SEMANTIC_B,
)
from harnessbench.learning.route_paper import (
    _holm_adjust,
    _json_default,
    _mcnemar_exact,
    _order_policy_rows,
    _paired_counterfactual_comparisons,
)


def _tiny_arrays() -> dict[str, np.ndarray]:
    samples = 6
    actions = np.arange(samples * MAX_ACTION_DIM, dtype=np.float32).reshape(samples, MAX_ACTION_DIM)
    actions /= actions.max()
    return {
        "node_features": np.zeros((samples, MAX_NODES, FEATURE_DIM), dtype=np.float32),
        "semantic_adjacency": np.zeros((samples, MAX_NODES, MAX_NODES), dtype=np.float32),
        "node_mask": np.ones((samples, MAX_NODES), dtype=np.float32),
        "global_features": np.zeros((samples, GLOBAL_DIM), dtype=np.float32),
        "task_id": np.full(samples, 2, dtype=np.int64),
        "action": actions,
        "action_mask": np.ones((samples, MAX_ACTION_DIM), dtype=np.float32),
        "binary_action_mask": np.zeros((samples, MAX_ACTION_DIM), dtype=np.float32),
        "pointer_target": np.zeros((samples, 2), dtype=np.int64),
        "pointer_mask": np.ones((samples, 2), dtype=np.float32),
        "split": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        "episode_seed": np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64),
        "episode_step": np.asarray([0, 2, 4, 0, 2, 4], dtype=np.int64),
    }


def test_action_chunks_pad_but_never_cross_episode_boundaries() -> None:
    arrays = _tiny_arrays()
    dataset = ActionChunkDataset(arrays, tasks=("route",), chunk_size=3)
    second = dataset[1]
    np.testing.assert_allclose(second["action_chunk"].numpy(), arrays["action"][[1, 2, 2]])
    np.testing.assert_array_equal(second["chunk_valid"].numpy(), [1.0, 1.0, 0.0])
    third = dataset[2]
    np.testing.assert_allclose(third["action_chunk"].numpy(), arrays["action"][[2, 2, 2]])
    np.testing.assert_array_equal(third["chunk_valid"].numpy(), [1.0, 0.0, 0.0])


def test_act_chunk_model_outputs_finite_action_sequence() -> None:
    config = ActChunkConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        action_encoder_layers=1,
        chunk_size=3,
        latent_dim=8,
        dropout=0.0,
    )
    model = ActChunkNet(config)
    model.eval()
    logits, pointer_logits, mu, log_variance = model(
        torch.zeros((2, MAX_NODES, FEATURE_DIM)),
        torch.ones((2, MAX_NODES)),
        torch.zeros((2, GLOBAL_DIM)),
        torch.as_tensor([2, 2]),
    )
    assert tuple(logits.shape) == (2, 3, MAX_ACTION_DIM)
    assert tuple(pointer_logits.shape) == (2, 3, 2, MAX_NODES)
    assert tuple(mu.shape) == (2, 8)
    assert tuple(log_variance.shape) == (2, 8)
    assert torch.isfinite(logits).all()


def test_relational_act_requires_and_accepts_semantic_adjacency() -> None:
    config = ActChunkConfig(
        hidden_dim=32,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        action_encoder_layers=1,
        chunk_size=2,
        latent_dim=8,
        dropout=0.0,
        use_relational_pooling=True,
    )
    model = ActChunkNet(config).eval()
    inputs = (
        torch.zeros((1, MAX_NODES, FEATURE_DIM)),
        torch.ones((1, MAX_NODES)),
        torch.zeros((1, GLOBAL_DIM)),
        torch.as_tensor([2]),
    )
    with pytest.raises(ValueError, match="semantic_adjacency"):
        model(*inputs)
    logits, _, _, _ = model(
        *inputs,
        semantic_adjacency=torch.eye(MAX_NODES).unsqueeze(0),
    )
    assert tuple(logits.shape) == (1, 2, MAX_ACTION_DIM)


def test_typed_act_branch_grounding_keeps_semantic_pointer_slots_separate() -> None:
    features = np.zeros((MAX_NODES, FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros(MAX_NODES, dtype=np.float32)
    for index, semantic_column, xy in (
        (2, SEMANTIC_A, (0.3, 0.2)),
        (3, SEMANTIC_A, (0.8, 0.2)),
        (4, SEMANTIC_B, (0.3, 0.8)),
        (5, SEMANTIC_B, (0.8, 0.8)),
    ):
        node_mask[index] = 1.0
        features[index, :2] = xy
        features[index, ROLE_TARGET] = 1.0
        features[index, semantic_column] = 1.0
    graph = SimpleNamespace(node_features=features, node_mask=node_mask)
    scores = np.zeros((2, MAX_NODES), dtype=np.float32)
    scores[0, 3] = 4.0
    scores[0, 5] = 100.0  # Must be masked out of semantic slot A.
    scores[1, 4] = 5.0
    scores[1, 2] = 100.0  # Must be masked out of semantic slot B.

    policy = object.__new__(ActChunkPolicy)
    grounded, debug = policy._ground_branch_action(
        graph,
        np.zeros(MAX_ACTION_DIM, dtype=np.float64),
        pointer_scores=scores,
    )

    np.testing.assert_allclose(grounded[:2], features[3, :2])
    np.testing.assert_allclose(grounded[3:5], features[4, :2])
    np.testing.assert_array_equal(grounded[[2, 5]], [1.0, 1.0])
    assert debug["pointer_indices"] == [3, 4]
    assert debug["candidate_counts"] == [2, 2]


def test_mcnemar_exact_uses_only_discordant_pairs() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert _mcnemar_exact(2, 2) == 1.0


def test_paper_policy_rows_use_declared_display_order() -> None:
    rows = [
        {"policy": "random"},
        {"policy": "learned_geometry"},
        {"policy": "learned_topology"},
        {"policy": "future_baseline"},
    ]
    ordered = _order_policy_rows(rows)
    assert [row["policy"] for row in ordered] == [
        "learned_topology",
        "learned_geometry",
        "random",
        "future_baseline",
    ]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    rows = [
        {"mcnemar_exact_p": 0.04},
        {"mcnemar_exact_p": 0.01},
        {"mcnemar_exact_p": 0.03},
    ]
    adjusted = _holm_adjust(rows)
    assert [row["mcnemar_exact_p_holm"] for row in adjusted] == pytest.approx(
        [0.06, 0.03, 0.06]
    )


def test_counterfactual_pairing_keeps_assignment_in_key() -> None:
    records = []
    for assignment, topology, baseline in (("5-13-21", True, False), ("7-15-23", False, True)):
        for policy, success in (("learned_topology", topology), ("baseline", baseline)):
            records.append(
                {
                    "difficulty": 0.5,
                    "target_assignment": assignment,
                    "seed": 10,
                    "policy": policy,
                    "success": success,
                }
            )
    rows = _paired_counterfactual_comparisons(records)
    assert len(rows) == 2
    indexed = {row["target_assignment"]: row for row in rows}
    assert indexed["5-13-21"]["topology_success_baseline_failure"] == 1
    assert indexed["7-15-23"]["topology_failure_baseline_success"] == 1


def test_route_report_json_converts_numpy_scalars_and_arrays() -> None:
    encoded = json.dumps(
        {
            "success": np.bool_(True),
            "episodes": np.int64(3),
            "rate": np.float32(0.5),
            "assignment": np.asarray([5, 13, 21], dtype=np.int64),
        },
        default=_json_default,
    )
    assert json.loads(encoded) == {
        "success": True,
        "episodes": 3,
        "rate": 0.5,
        "assignment": [5, 13, 21],
    }


def test_route_report_json_still_rejects_unknown_types() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"invalid": object()}, default=_json_default)
