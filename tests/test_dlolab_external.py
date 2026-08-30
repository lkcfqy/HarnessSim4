from __future__ import annotations

import numpy as np
import pytest

from harnessbench.learning.dlolab_external import (
    resample_polyline_deltas,
    wiring_post_curve_metrics,
)


def test_wiring_post_metrics_are_exact_for_target_curve() -> None:
    target = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, 0.2, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    fixtures = np.asarray([[-0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    metrics = wiring_post_curve_metrics(target, target, fixtures)
    assert metrics["orientation"] == "direct"
    assert metrics["ordered_point_rmse_m"] == pytest.approx(0.0)
    assert metrics["symmetric_chamfer_sum_m"] == pytest.approx(0.0)
    assert metrics["fixture_side_relation_accuracy"] == pytest.approx(1.0)
    assert metrics["fixture_angular_mae_rad"] == pytest.approx(0.0)
    assert metrics["fixture_relation_score"] == pytest.approx(1.0)
    assert metrics["winding_change_mae_rad"] == pytest.approx(0.0)


def test_wiring_post_metrics_align_reversed_target_once() -> None:
    target = np.asarray(
        [[-1.0, 1.0, 0.0], [0.0, 0.2, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    fixtures = np.asarray([[-0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    metrics = wiring_post_curve_metrics(target, target[::-1], fixtures)
    assert metrics["orientation"] == "reversed"
    assert metrics["ordered_point_rmse_m"] == pytest.approx(0.0)
    assert metrics["fixture_side_relation_accuracy"] == pytest.approx(1.0)


def test_resample_polyline_deltas_preserves_endpoint_and_step_bound() -> None:
    path = np.asarray(
        [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.03, 0.04, 0.0]], dtype=np.float64
    )
    deltas = resample_polyline_deltas(path, max_step_m=0.011)
    np.testing.assert_allclose(deltas.sum(axis=0), path[-1] - path[0])
    assert np.linalg.norm(deltas, axis=1).max() <= 0.011
    assert deltas.shape == (7, 3)


def test_resample_polyline_deltas_rejects_invalid_bound() -> None:
    path = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive finite"):
        resample_polyline_deltas(path, max_step_m=0.0)


@pytest.mark.parametrize(
    ("vertices", "target", "fixtures", "message"),
    [
        (np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 2)), "at least two"),
        (np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((1, 2)), "shape"),
        (np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((1, 2)), "matching shapes"),
        (np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((0, 2)), "at least one"),
    ],
)
def test_wiring_post_metrics_reject_invalid_inputs(
    vertices: np.ndarray,
    target: np.ndarray,
    fixtures: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wiring_post_curve_metrics(vertices, target, fixtures)
