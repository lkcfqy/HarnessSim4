"""Pure NumPy metrics for the external DLO-Lab wiring-post audit.

The functions in this module deliberately do not import Genesis.  This keeps the
metric definition testable in the lightweight HarnessBench environment while the
official simulator runs in its own isolated environment.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_curve(points: np.ndarray, *, name: str) -> np.ndarray:
    curve = np.asarray(points, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_points, 3), got {curve.shape}")
    if curve.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.isfinite(curve).all():
        raise ValueError(f"{name} contains non-finite values")
    return curve


def _ordered_point_rmse(candidate: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((candidate - target) ** 2, axis=1))))


def _winding_change(curve_xy: np.ndarray, fixture_xy: np.ndarray) -> float:
    angles = np.unwrap(
        np.arctan2(curve_xy[:, 1] - fixture_xy[1], curve_xy[:, 0] - fixture_xy[0])
    )
    return float(angles[-1] - angles[0])


def resample_polyline_deltas(points: np.ndarray, max_step_m: float) -> np.ndarray:
    """Convert a Cartesian polyline to bounded per-step displacements."""

    path = _as_curve(points, name="points")
    if not np.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("max_step_m must be a positive finite number")
    deltas: list[np.ndarray] = []
    for start, end in pairwise(path):
        displacement = end - start
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-12:
            continue
        count = max(1, int(np.ceil(distance / max_step_m)))
        deltas.extend([displacement / count] * count)
    if not deltas:
        return np.zeros((0, 3), dtype=np.float64)
    output = np.asarray(deltas, dtype=np.float64)
    if np.linalg.norm(output, axis=1).max() > max_step_m * (1.0 + 1e-12):
        raise AssertionError("resampling produced an out-of-bound displacement")
    return output


def wiring_post_curve_metrics(
    vertices: np.ndarray,
    target: np.ndarray,
    fixtures: np.ndarray,
) -> dict[str, object]:
    """Measure geometry and ordered cable-to-fixture relations.

    DLO-Lab's native wiring-post reward is set based: it uses a symmetric nearest-
    neighbour distance.  RouteBot additionally audits ordered relations between
    cable vertices and fixtures.  Direction is selected once, using the lower
    ordered RMSE, so reversing the target array cannot change the result.
    """

    candidate = _as_curve(vertices, name="vertices")
    reference = _as_curve(target, name="target")
    if candidate.shape != reference.shape:
        raise ValueError(
            f"vertices and target must have matching shapes, got "
            f"{candidate.shape} and {reference.shape}"
        )
    fixture_array = np.asarray(fixtures, dtype=np.float64)
    if fixture_array.ndim != 2 or fixture_array.shape[1] not in (2, 3):
        raise ValueError(f"fixtures must have shape (n_fixtures, 2 or 3), got {fixture_array.shape}")
    if fixture_array.shape[0] < 1 or not np.isfinite(fixture_array).all():
        raise ValueError("fixtures must contain at least one finite point")
    fixture_xy = fixture_array[:, :2]

    direct_rmse = _ordered_point_rmse(candidate, reference)
    reversed_reference = reference[::-1]
    reversed_rmse = _ordered_point_rmse(candidate, reversed_reference)
    if reversed_rmse < direct_rmse:
        aligned_target = reversed_reference
        orientation = "reversed"
        ordered_rmse = reversed_rmse
    else:
        aligned_target = reference
        orientation = "direct"
        ordered_rmse = direct_rmse

    pairwise = np.linalg.norm(candidate[:, None, :] - reference[None, :, :], axis=-1)
    current_to_target = float(pairwise.min(axis=1).mean())
    target_to_current = float(pairwise.min(axis=0).mean())

    current_offsets = candidate[None, :, :2] - fixture_xy[:, None, :]
    target_offsets = aligned_target[None, :, :2] - fixture_xy[:, None, :]
    valid_side_labels = np.abs(target_offsets) > 1e-9
    matching_sides = np.signbit(current_offsets) == np.signbit(target_offsets)
    side_accuracy = float(matching_sides[valid_side_labels].mean())

    current_norm = np.linalg.norm(current_offsets, axis=-1)
    target_norm = np.linalg.norm(target_offsets, axis=-1)
    valid_angles = (current_norm > 1e-9) & (target_norm > 1e-9)
    dot = np.sum(current_offsets * target_offsets, axis=-1)
    cross = (
        current_offsets[..., 0] * target_offsets[..., 1]
        - current_offsets[..., 1] * target_offsets[..., 0]
    )
    angular_errors = np.abs(np.arctan2(cross[valid_angles], dot[valid_angles]))
    angular_mae = float(angular_errors.mean())

    current_winding = [
        _winding_change(candidate[:, :2], fixture) for fixture in fixture_xy
    ]
    target_winding = [
        _winding_change(aligned_target[:, :2], fixture) for fixture in fixture_xy
    ]
    winding_mae = float(
        np.mean(np.abs(np.asarray(current_winding) - np.asarray(target_winding)))
    )

    return {
        "orientation": orientation,
        "ordered_point_rmse_m": ordered_rmse,
        "symmetric_chamfer_sum_m": current_to_target + target_to_current,
        "current_to_target_mean_m": current_to_target,
        "target_to_current_mean_m": target_to_current,
        "fixture_side_relation_accuracy": side_accuracy,
        "fixture_angular_mae_rad": angular_mae,
        "fixture_relation_score": float(1.0 - angular_mae / np.pi),
        "winding_change_current_rad": current_winding,
        "winding_change_target_rad": target_winding,
        "winding_change_mae_rad": winding_mae,
    }
