from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from harnessbench.learning.route_mujoco_direct import (
    DIRECT_ROUTE_SEGMENTS,
    POLICY_TO_WORLD,
    WORLD_TO_POLICY,
    _affine_homogeneous,
    _direct_route_body_name,
    _RouteObservationAdapter,
    _transform_xy,
)


def test_direct_arc_map_is_strict_and_covers_endpoints() -> None:
    assert len(DIRECT_ROUTE_SEGMENTS) == 29
    assert len(set(DIRECT_ROUTE_SEGMENTS)) == 29
    assert all(
        left < right
        for left, right in pairwise(DIRECT_ROUTE_SEGMENTS)
    )
    assert _direct_route_body_name(0) == "route_B_first"
    assert _direct_route_body_name(28) == "route_B_last"
    target_names = {
        _direct_route_body_name(index)
        for index in (5, 7, 9, 13, 15, 17, 21, 23, 25)
    }
    assert len(target_names) == 9
    with pytest.raises(IndexError):
        _direct_route_body_name(29)


def test_policy_world_transform_round_trip() -> None:
    points = np.asarray(
        ((-0.50, -0.23), (-0.24, -0.12), (-0.02, 0.08), (0.34, 0.14))
    )
    recovered = _transform_xy(
        _transform_xy(points, WORLD_TO_POLICY),
        POLICY_TO_WORLD,
    )
    np.testing.assert_allclose(recovered, points, atol=1e-12)


def test_affine_validation_and_adapter() -> None:
    with pytest.raises(ValueError):
        _affine_homogeneous(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        _affine_homogeneous(np.zeros((3, 2)), np.zeros((3, 2)))

    adapter = _RouteObservationAdapter(0.5)
    observation = {"grasp_idx": None}
    adapter.set_observation(observation, 17)
    assert adapter.observation() is observation
    assert adapter.step_count == 17
    assert adapter.max_steps == 360
