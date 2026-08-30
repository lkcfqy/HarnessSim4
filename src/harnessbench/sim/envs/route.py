"""RouteBot: topology-constrained cable placement into ordered clips."""

from __future__ import annotations

import numpy as np

from harnessbench.sim.arm import PlanarArm
from harnessbench.sim.core import CableGraph, CircleObstacle
from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.rendering import SceneCanvas

# Strictly increasing internal cable-particle assignments.  They stay within
# local arc-length windows around the nominal p7/p15/p23 routing points, so the
# semantic intervention does not reverse cable order or require a topological
# crossing by construction.
ROUTE_TARGET_ASSIGNMENTS = (
    (5, 13, 21),
    (7, 15, 23),
    (9, 17, 25),
    (5, 15, 25),
    (7, 17, 21),
    (9, 13, 23),
    (5, 17, 23),
    (7, 13, 25),
    (9, 15, 21),
)

# Frozen after the independent semantic-completion audit in
# artifacts/papers/routebot/assignment_audit_semantic_final. All nine cells
# completed on ten held-out physical seeds at each difficulty. The earlier
# crossing-gated audits are retained to document why crossing-free completion
# is reported as a stricter secondary 2-D shape metric.
ROUTE_SCREENED_ASSIGNMENT_INDICES = tuple(range(len(ROUTE_TARGET_ASSIGNMENTS)))
ROUTE_SCREENED_TARGET_ASSIGNMENTS = tuple(
    ROUTE_TARGET_ASSIGNMENTS[index] for index in ROUTE_SCREENED_ASSIGNMENT_INDICES
)


class RouteEnv(HarnessEnv):
    task_name = "cable_routing"
    robot_name = "RouteBot"
    max_steps = 360
    action_dim = 3

    def _reset_task(self) -> None:
        count = 29
        t = np.linspace(0.0, 1.0, count)
        positions = np.column_stack(
            (
                0.10 + 0.39 * t,
                0.78 + (0.10 + 0.035 * self.difficulty) * np.sin(5.5 * np.pi * t) - 0.08 * t,
            )
        )
        positions += self.rng.normal(0.0, 0.002 + 0.002 * self.difficulty, positions.shape)
        self.cable = CableGraph.chain(
            positions,
            particle_radius=0.0085,
            stretch_stiffness=0.96,
            bend_stiffness=0.10 + 0.10 * (1.0 - self.difficulty),
            self_collision=True,
        )
        lateral = self.rng.normal(0.0, 0.025 * self.difficulty, size=(3, 2))
        self.clips = np.asarray([[0.39, 0.61], [0.58, 0.39], [0.77, 0.58]]) + lateral
        base_target_indices = np.asarray([7, 15, 23], dtype=np.int64)
        target_indices_override = getattr(self, "target_indices_override", None)
        if target_indices_override is None:
            self.target_indices = base_target_indices.copy()
        else:
            self.target_indices = np.asarray(target_indices_override, dtype=np.int64)
            if self.target_indices.shape != (3,) or not np.all(np.diff(self.target_indices) > 0):
                raise ValueError("target_indices_override must contain three increasing indices")
            if self.target_indices[0] < 1 or self.target_indices[-1] >= count - 1:
                raise ValueError("target_indices_override must select internal cable particles")
        self.finish = np.asarray([0.88, 0.72]) + self.rng.normal(0, 0.012 * self.difficulty, 2)
        self.bindings: dict[int, int] = {}
        self.grasp_idx: int | None = None
        self.endpoint_placed = False
        self.arm = PlanarArm(
            base=np.asarray([0.17, 0.90]),
            # Keep the physical initial state fixed under semantic target
            # permutation interventions. The first process target may change,
            # but the arm always starts beside physical particle 7.
            ee=self.cable.positions[base_target_indices[0]].copy(),
            link_lengths=(0.42, 0.38),
            max_speed=0.78 - 0.12 * self.difficulty,
            elbow_up=False,
            name="route_arm",
        )
        self.obstacles = (
            CircleObstacle(np.asarray([0.50, 0.73]), 0.038 + 0.010 * self.difficulty),
            CircleObstacle(np.asarray([0.68, 0.27]), 0.035 + 0.008 * self.difficulty),
        )
        self._geometry_choices: dict[int, int] = {}
        self._new_latch_correct = False

    def _available_particles(self) -> list[int]:
        bound = set(self.bindings.values())
        return [index for index in range(1, len(self.cable.positions)) if index not in bound]

    def _step_task(self, action: np.ndarray) -> float:
        before = len(self.bindings)
        self._new_latch_correct = False
        target = np.clip(action[:2], 0.04, 0.96)
        grip = bool(action[2] >= 0.5)
        self.arm.move_toward(target, self.dt)

        if not grip:
            self.grasp_idx = None
        elif self.grasp_idx is None:
            available = self._available_particles()
            if available:
                distances = np.linalg.norm(self.cable.positions[available] - self.arm.ee, axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] < 0.036:
                    self.grasp_idx = int(available[nearest])

        anchors = {particle: self.clips[clip] for clip, particle in self.bindings.items()}
        if self.grasp_idx is not None and grip:
            anchors[self.grasp_idx] = self.arm.ee.copy()
        self.cable.step(
            anchors,
            self.obstacles,
            dt=self.dt,
            substeps=2,
            iterations=11,
        )

        if self.grasp_idx is not None and grip:
            unbound = [index for index in range(len(self.clips)) if index not in self.bindings]
            if unbound:
                distances = np.linalg.norm(
                    self.clips[unbound] - self.cable.positions[self.grasp_idx], axis=1
                )
                nearest_clip = unbound[int(np.argmin(distances))]
                tolerance = 0.030 - 0.008 * self.difficulty
                if float(np.min(distances)) < tolerance:
                    self.bindings[nearest_clip] = int(self.grasp_idx)
                    self._new_latch_correct = bool(
                        self.grasp_idx == int(self.target_indices[nearest_clip])
                    )
                    self.grasp_idx = None

        if (
            self.grasp_idx == len(self.cable.positions) - 1
            and np.linalg.norm(self.cable.positions[-1] - self.finish) < 0.035
        ):
            self.endpoint_placed = True

        added = len(self.bindings) - before
        correct = sum(
            int(particle == int(self.target_indices[clip]))
            for clip, particle in self.bindings.items()
        )
        nearest_unbound_distance = 0.0
        unbound_clips = [index for index in range(len(self.clips)) if index not in self.bindings]
        if unbound_clips:
            nearest_unbound_distance = min(
                float(np.linalg.norm(self.arm.ee - self.clips[index])) for index in unbound_clips
            )
        return (
            2.0 * added
            + 4.0 * int(self._new_latch_correct)
            + 0.08 * correct
            - 0.05 * nearest_unbound_distance
            - 0.02 * self.cable.stretch_error()
        )

    def observation(self) -> dict:
        return {
            "cable_positions": self.cable.positions.copy(),
            "cable_edges": self.cable.edges.copy(),
            "arm_ee": self.arm.ee.copy(),
            "clips": self.clips.copy(),
            "finish": self.finish.copy(),
            "target_particle_indices": self.target_indices.copy(),
            "bindings": dict(self.bindings),
            "grasp_idx": self.grasp_idx,
            "endpoint_placed": self.endpoint_placed,
        }

    def metrics(self) -> dict[str, float | int | bool]:
        correct = sum(
            int(particle == int(self.target_indices[clip]))
            for clip, particle in self.bindings.items()
        )
        total = len(self.clips)
        semantic_complete = (
            len(self.bindings) == total
            and all(
                self.bindings[index] == int(self.target_indices[index])
                for index in range(total)
            )
            and np.linalg.norm(self.cable.positions[-1] - self.finish) < 0.035
            and self.endpoint_placed
        )
        crossing_free = self.cable.crossing_count() == 0
        return {
            "clip_coverage": len(self.bindings) / total,
            "topology_accuracy": correct / total,
            "wrong_latches": len(self.bindings) - correct,
            "stretch_error": self.cable.stretch_error(),
            "crossings": self.cable.crossing_count(),
            "crossing_free": crossing_free,
            "endpoint_error": float(np.linalg.norm(self.cable.positions[-1] - self.finish)),
            "endpoint_placed": self.endpoint_placed,
            "semantic_complete": semantic_complete,
            "strict_crossing_free_success": semantic_complete and crossing_free,
        }

    def success(self) -> bool:
        # The primary paper outcome isolates semantic routing correctness.
        # A crossing in a 2-D projection is not necessarily a topological
        # crossing in the real 3-D harness; it remains a separately reported
        # strict shape/safety metric instead of invalidating the semantic task.
        return (
            len(self.bindings) == len(self.clips)
            and all(
                self.bindings[index] == int(self.target_indices[index])
                for index in range(len(self.clips))
            )
            and np.linalg.norm(self.cable.positions[-1] - self.finish) < 0.035
            and self.endpoint_placed
        )

    def _manipulate_particle(self, particle: int, destination: np.ndarray) -> np.ndarray:
        if self.grasp_idx is not None and self.grasp_idx != particle:
            return np.asarray([self.arm.ee[0], self.arm.ee[1], 0.0])
        particle_position = self.cable.positions[particle]
        if self.grasp_idx is None:
            distance = float(np.linalg.norm(self.arm.ee - particle_position))
            return np.asarray(
                [particle_position[0], particle_position[1], 1.0 if distance < 0.025 else 0.0]
            )
        return np.asarray([destination[0], destination[1], 1.0])

    def policy_action(self, policy_name: str, policy_rng: np.random.Generator) -> np.ndarray:
        unbound = [index for index in range(len(self.clips)) if index not in self.bindings]
        if not unbound:
            if policy_name in {"topology", "geometry"}:
                return self._manipulate_particle(len(self.cable.positions) - 1, self.finish)
            if policy_name == "random":
                return np.asarray([self.arm.ee[0], self.arm.ee[1], 0.0])
            raise KeyError(policy_name)
        clip_index = unbound[0]
        if policy_name == "topology":
            particle = int(self.target_indices[clip_index])
            return self._manipulate_particle(particle, self.clips[clip_index])
        if policy_name == "geometry":
            if clip_index not in self._geometry_choices:
                available = self._available_particles()
                distance = np.linalg.norm(
                    self.cable.positions[available] - self.clips[clip_index], axis=1
                )
                self._geometry_choices[clip_index] = int(available[int(np.argmin(distance))])
            return self._manipulate_particle(
                self._geometry_choices[clip_index], self.clips[clip_index]
            )
        if policy_name == "random":
            if self.step_count % 20 == 0 or not hasattr(self, "_random_target"):
                self._random_target = policy_rng.uniform([0.12, 0.16], [0.88, 0.84])
                self._random_grip = float(policy_rng.random() > 0.25)
            return np.asarray([self._random_target[0], self._random_target[1], self._random_grip])
        raise KeyError(policy_name)

    def render(self, width: int = 640, height: int = 480):
        canvas = SceneCanvas(width, height)
        for obstacle in self.obstacles:
            canvas.obstacle(obstacle)
        for index, clip in enumerate(self.clips):
            bound = self.bindings.get(index)
            canvas.clip(
                clip,
                latched=bound is not None,
                correct=bound == int(self.target_indices[index]) if bound is not None else True,
            )
            x, y = canvas.xy(clip)
            canvas.draw.text(
                (x + 12, y - 18), f"C{index + 1}:p{self.target_indices[index]}", fill=(35, 39, 43)
            )
        canvas.target(self.finish, "finish")
        canvas.cable(self.cable)
        canvas.arm(self.arm, gripped=self.grasp_idx is not None)
        metrics = self.metrics()
        canvas.text(
            "RouteBot - ordered clip routing",
            f"step {self.step_count:03d} | coverage {metrics['clip_coverage']:.2f} | "
            f"topology {metrics['topology_accuracy']:.2f} | wrong {metrics['wrong_latches']}",
        )
        return canvas.image
