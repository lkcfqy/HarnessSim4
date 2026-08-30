"""A deterministic 2-D position-based dynamics core for branched cables.

The fast backend is intentionally small and dependency-light.  It is designed for
large ablations, curriculum experiments and visual debugging.  High-fidelity 3-D
validation is kept as a separate MuJoCo/Isaac backend milestone; metrics from this
backend must not be represented as real-world performance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

EPSILON = 1e-9


@dataclass(frozen=True)
class CircleObstacle:
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64))
        if self.center.shape != (2,) or self.radius <= 0:
            raise ValueError("CircleObstacle requires a 2-D center and positive radius")


@dataclass
class CableGraph:
    """Particles and graph constraints for chains or branched DLOs."""

    positions: np.ndarray
    edges: np.ndarray
    bend_edges: np.ndarray | None = None
    particle_radius: float = 0.009
    stretch_stiffness: float = 0.96
    bend_stiffness: float = 0.18
    damping: float = 0.91
    self_collision: bool = True

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64).copy()
        self.edges = np.asarray(self.edges, dtype=np.int64).copy()
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape [particles, 2]")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape [edges, 2]")
        if self.edges.size and (self.edges.min() < 0 or self.edges.max() >= len(self.positions)):
            raise ValueError("edge index outside particle array")
        if self.bend_edges is None:
            self.bend_edges = np.empty((0, 2), dtype=np.int64)
        else:
            self.bend_edges = np.asarray(self.bend_edges, dtype=np.int64).copy()
        self.velocities = np.zeros_like(self.positions)
        self.rest_lengths = self._edge_lengths(self.edges)
        self.bend_rest_lengths = self._edge_lengths(self.bend_edges)
        self._adjacent = {tuple(sorted(edge)) for edge in self.edges.tolist()}
        self._collision_pairs = np.asarray(
            [
                (i, j)
                for i in range(len(self.positions) - 1)
                for j in range(i + 1, len(self.positions))
                if (i, j) not in self._adjacent
            ],
            dtype=np.int64,
        )
        self._crossing_edge_pairs = np.asarray(
            [
                (first_index, second_index)
                for first_index, first in enumerate(self.edges)
                for second_index, second in enumerate(
                    self.edges[first_index + 1 :], first_index + 1
                )
                if not (set(first.tolist()) & set(second.tolist()))
            ],
            dtype=np.int64,
        )

    @classmethod
    def from_paths(
        cls,
        positions: np.ndarray,
        paths: Iterable[Iterable[int]],
        **kwargs: object,
    ) -> CableGraph:
        edges: list[tuple[int, int]] = []
        bends: list[tuple[int, int]] = []
        for raw_path in paths:
            path = list(raw_path)
            edges.extend(pairwise(path))
            bends.extend(zip(path[:-2], path[2:]))
        # Preserve order while removing edges shared by two paths at a branch.
        unique_edges = list(dict.fromkeys(tuple(sorted(edge)) for edge in edges))
        unique_bends = list(dict.fromkeys(tuple(sorted(edge)) for edge in bends))
        return cls(
            positions=np.asarray(positions),
            edges=np.asarray(unique_edges, dtype=np.int64),
            bend_edges=np.asarray(unique_bends, dtype=np.int64),
            **kwargs,
        )

    @classmethod
    def chain(cls, positions: np.ndarray, **kwargs: object) -> CableGraph:
        count = len(positions)
        return cls.from_paths(positions, [range(count)], **kwargs)

    def _edge_lengths(self, edges: np.ndarray) -> np.ndarray:
        if edges.size == 0:
            return np.empty(0, dtype=np.float64)
        delta = self.positions[edges[:, 1]] - self.positions[edges[:, 0]]
        return np.linalg.norm(delta, axis=1)

    def clone(self) -> CableGraph:
        cloned = CableGraph(
            positions=self.positions.copy(),
            edges=self.edges.copy(),
            bend_edges=self.bend_edges.copy(),
            particle_radius=self.particle_radius,
            stretch_stiffness=self.stretch_stiffness,
            bend_stiffness=self.bend_stiffness,
            damping=self.damping,
            self_collision=self.self_collision,
        )
        cloned.velocities = self.velocities.copy()
        cloned.rest_lengths = self.rest_lengths.copy()
        cloned.bend_rest_lengths = self.bend_rest_lengths.copy()
        return cloned

    def step(
        self,
        anchors: Mapping[int, np.ndarray] | None = None,
        obstacles: Iterable[CircleObstacle] = (),
        *,
        dt: float = 0.025,
        substeps: int = 2,
        iterations: int = 10,
        bounds: tuple[float, float, float, float] = (0.035, 0.965, 0.05, 0.95),
    ) -> None:
        """Advance with distance, bending, collision and kinematic constraints."""

        if dt <= 0 or substeps < 1 or iterations < 1:
            raise ValueError("dt, substeps and iterations must be positive")
        anchor_map = {
            int(index): np.asarray(target, dtype=np.float64)
            for index, target in (anchors or {}).items()
        }
        for index, target in anchor_map.items():
            if index < 0 or index >= len(self.positions) or target.shape != (2,):
                raise ValueError("invalid anchor")

        obstacle_tuple = tuple(obstacles)
        h = dt / substeps
        for _ in range(substeps):
            old = self.positions.copy()
            self.positions += self.velocities * h
            for _ in range(iterations):
                self._solve_distances(
                    self.edges, self.rest_lengths, self.stretch_stiffness, anchor_map
                )
                self._solve_distances(
                    self.bend_edges,
                    self.bend_rest_lengths,
                    self.bend_stiffness,
                    anchor_map,
                )
                self._solve_obstacles(obstacle_tuple, anchor_map)
                if self.self_collision:
                    self._solve_self_collision(anchor_map)
                self._solve_bounds(bounds, anchor_map)
                for index, target in anchor_map.items():
                    self.positions[index] = target
            self.velocities = (self.positions - old) / h * self.damping
            for index in anchor_map:
                self.velocities[index] = 0.0

    def _solve_distances(
        self,
        edges: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: float,
        anchors: Mapping[int, np.ndarray],
    ) -> None:
        if edges.size == 0:
            return
        i = edges[:, 0]
        j = edges[:, 1]
        delta = self.positions[j] - self.positions[i]
        distance = np.linalg.norm(delta, axis=1)
        safe_distance = np.maximum(distance, EPSILON)
        anchor_indices = np.fromiter(anchors.keys(), dtype=np.int64, count=len(anchors))
        wi = (~np.isin(i, anchor_indices)).astype(np.float64)
        wj = (~np.isin(j, anchor_indices)).astype(np.float64)
        weight = wi + wj
        active = (distance > EPSILON) & (weight > 0)
        correction = np.zeros_like(delta)
        correction[active] = (
            stiffness
            * ((distance[active] - rest_lengths[active]) / safe_distance[active])[:, None]
            * delta[active]
            / weight[active, None]
        )
        accumulated = np.zeros_like(self.positions)
        np.add.at(accumulated, i, correction * wi[:, None])
        np.add.at(accumulated, j, -correction * wj[:, None])
        self.positions += accumulated

    def _solve_obstacles(
        self, obstacles: tuple[CircleObstacle, ...], anchors: Mapping[int, np.ndarray]
    ) -> None:
        for obstacle in obstacles:
            delta = self.positions - obstacle.center
            distances = np.linalg.norm(delta, axis=1)
            minimum = obstacle.radius + self.particle_radius
            movable = np.ones(len(self.positions), dtype=bool)
            if anchors:
                movable[np.fromiter(anchors.keys(), dtype=np.int64, count=len(anchors))] = False
            inside = (distances < minimum) & movable
            if not np.any(inside):
                continue
            safe = np.maximum(distances[inside], EPSILON)
            direction = delta[inside] / safe[:, None]
            direction[distances[inside] <= EPSILON] = np.asarray([1.0, 0.0])
            self.positions[inside] = obstacle.center + direction * minimum

    def _solve_self_collision(self, anchors: Mapping[int, np.ndarray]) -> None:
        if self._collision_pairs.size == 0:
            return
        minimum = 2.0 * self.particle_radius
        i_all = self._collision_pairs[:, 0]
        j_all = self._collision_pairs[:, 1]
        delta_all = self.positions[j_all] - self.positions[i_all]
        distance_all = np.linalg.norm(delta_all, axis=1)
        mask = distance_all < minimum
        if not np.any(mask):
            return
        i = i_all[mask]
        j = j_all[mask]
        delta = delta_all[mask]
        distance = np.maximum(distance_all[mask], EPSILON)
        direction = delta / distance[:, None]
        zero = distance_all[mask] <= EPSILON
        direction[zero] = np.asarray([1.0, 0.0])
        overlap = minimum - distance_all[mask]
        wi = np.asarray([0.0 if int(index) in anchors else 1.0 for index in i])
        wj = np.asarray([0.0 if int(index) in anchors else 1.0 for index in j])
        weight = wi + wj
        active = weight > 0
        correction = np.zeros_like(direction)
        correction[active] = direction[active] * overlap[active, None] / weight[active, None]
        accumulated = np.zeros_like(self.positions)
        np.add.at(accumulated, i, -correction * wi[:, None])
        np.add.at(accumulated, j, correction * wj[:, None])
        self.positions += accumulated

    def _solve_bounds(
        self,
        bounds: tuple[float, float, float, float],
        anchors: Mapping[int, np.ndarray],
    ) -> None:
        xmin, xmax, ymin, ymax = bounds
        movable = np.ones(len(self.positions), dtype=bool)
        if anchors:
            movable[np.fromiter(anchors.keys(), dtype=np.int64, count=len(anchors))] = False
        self.positions[movable, 0] = np.clip(self.positions[movable, 0], xmin, xmax)
        self.positions[movable, 1] = np.clip(self.positions[movable, 1], ymin, ymax)

    def endpoint_angle(self, endpoint: int, neighbor: int) -> float:
        vector = self.positions[endpoint] - self.positions[neighbor]
        return float(np.arctan2(vector[1], vector[0]))

    def stretch_error(self) -> float:
        lengths = self._edge_lengths(self.edges)
        relative = np.abs(lengths - self.rest_lengths) / np.maximum(self.rest_lengths, EPSILON)
        return float(relative.mean()) if relative.size else 0.0

    def crossing_count(self) -> int:
        if self._crossing_edge_pairs.size == 0:
            return 0
        first = self.edges[self._crossing_edge_pairs[:, 0]]
        second = self.edges[self._crossing_edge_pairs[:, 1]]
        a, b = self.positions[first[:, 0]], self.positions[first[:, 1]]
        c, d = self.positions[second[:, 0]], self.positions[second[:, 1]]
        o1 = _orientation_batch(a, b, c)
        o2 = _orientation_batch(a, b, d)
        o3 = _orientation_batch(c, d, a)
        o4 = _orientation_batch(c, d, b)
        return int(np.count_nonzero((o1 * o2 < -EPSILON) & (o3 * o4 < -EPSILON)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    return float(first[0] * second[1] - first[1] * second[0])


def _orientation_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = b - a
    second = c - a
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return o1 * o2 < -EPSILON and o3 * o4 < -EPSILON


def wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))
