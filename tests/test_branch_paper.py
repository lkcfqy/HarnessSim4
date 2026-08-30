from __future__ import annotations

import pytest

from harnessbench.learning.branch_paper import (
    _branch_failure_type,
    _paired_comparisons,
    _semantic_invariance,
)


def _record(
    policy: str,
    *,
    seed: int,
    swap: bool,
    success: bool,
    difficulty: float = 0.5,
) -> dict:
    return {
        "policy": policy,
        "seed": seed,
        "semantic_target_swap": swap,
        "success": success,
        "difficulty": difficulty,
        "endpoint_a_error": 0.0 if success else 0.2,
        "endpoint_b_error": 0.0 if success else 0.2,
        "crossings": 0,
    }


def test_branch_pairing_keeps_semantic_assignment_in_key() -> None:
    records = []
    for swap, topology, baseline in ((False, True, False), (True, False, True)):
        records.extend(
            (
                _record("learned_topology", seed=10, swap=swap, success=topology),
                _record("baseline", seed=10, swap=swap, success=baseline),
            )
        )
    rows = _paired_comparisons(records)
    assert len(rows) == 2
    indexed = {row["semantic_target_swap"]: row for row in rows}
    assert indexed[False]["topology_success_baseline_failure"] == 1
    assert indexed[True]["topology_failure_baseline_success"] == 1


def test_semantic_invariance_compares_same_physics_across_both_names() -> None:
    records = [
        _record("method", seed=1, swap=False, success=True),
        _record("method", seed=1, swap=True, success=True),
        _record("method", seed=2, swap=False, success=True),
        _record("method", seed=2, swap=True, success=False),
    ]
    row = _semantic_invariance(records)[0]
    assert row["paired_physical_seeds"] == 2
    assert row["outcome_agreement"] == pytest.approx(0.5)
    assert row["canonical_success_swapped_failure"] == 1
    assert row["canonical_failure_swapped_success"] == 0


def test_branch_failure_taxonomy_prioritizes_endpoint_errors() -> None:
    assert _branch_failure_type({"success": True}) == "success"
    assert (
        _branch_failure_type(
            {
                "success": False,
                "endpoint_a_error": 0.2,
                "endpoint_b_error": 0.2,
                "crossings": 2,
            }
        )
        == "both_endpoints_misplaced"
    )
    assert (
        _branch_failure_type(
            {
                "success": False,
                "endpoint_a_error": 0.01,
                "endpoint_b_error": 0.01,
                "crossings": 1,
            }
        )
        == "residual_crossing"
    )
