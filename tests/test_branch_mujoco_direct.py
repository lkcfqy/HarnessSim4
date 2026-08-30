from __future__ import annotations

import numpy as np
import pytest

from harnessbench.learning.branch_mujoco_direct import (
    POLICY_TO_WORLD,
    WORLD_TO_POLICY,
    _affine_homogeneous,
    _BranchObservationAdapter,
    _cable_body_names,
    _transform_xy,
)


def test_branch_policy_world_transform_round_trip() -> None:
    points = np.asarray(((-0.50, 0.00), (-0.16, 0.00), (0.43, 0.30), (0.43, -0.30)))
    recovered = _transform_xy(
        _transform_xy(points, WORLD_TO_POLICY),
        POLICY_TO_WORLD,
    )
    np.testing.assert_allclose(recovered, points, atol=1e-12)


def test_branch_cable_names_match_mujoco_composite_convention() -> None:
    trunk = _cable_body_names("trunk", 18)
    branch = _cable_body_names("branch_a", 24)
    assert len(trunk) == 18
    assert trunk[:2] == ("trunk_B_first", "trunk_B_1")
    assert trunk[-2:] == ("trunk_B_16", "trunk_B_last")
    assert len(branch) == 24
    assert branch[-2:] == ("branch_a_B_22", "branch_a_B_last")


def test_branch_affine_rejects_rank_deficiency_and_adapter_tracks_step() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        _affine_homogeneous(np.zeros((3, 2)), np.zeros((3, 2)))
    adapter = _BranchObservationAdapter(0.5)
    observation = {"semantic_target_swap": True}
    adapter.set_observation(observation, 23)
    assert adapter.observation() is observation
    assert adapter.step_count == 23
    assert adapter.max_steps == 300
