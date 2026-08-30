"""BranchBot: semantic branch assignment and dual-arm disentanglement."""

from __future__ import annotations

import numpy as np

from harnessbench.sim.arm import PlanarArm
from harnessbench.sim.core import CableGraph
from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.rendering import PALETTE, SceneCanvas

BRANCH_WAYPOINT_X = np.asarray([0.54, 0.82])
BRANCH_CLEARANCE_Y = (0.16, 0.84)
BRANCH_PHASE_BOUNDARIES = (0.28, 0.58)


class BranchEnv(HarnessEnv):
    task_name = "branched_harness"
    robot_name = "BranchBot"
    max_steps = 300
    action_dim = 6

    def _reset_task(self) -> None:
        trunk_count = 10
        branch_count = 9
        trunk = np.column_stack(
            (np.linspace(0.08, 0.42, trunk_count), np.linspace(0.52, 0.52, trunk_count))
        )
        junction = trunk_count - 1
        # Semantic branch A begins below and B above, so nearest-target geometry swaps them.
        a_points = np.column_stack(
            (
                np.linspace(0.44, 0.66, branch_count),
                0.52
                - 0.25 * np.sin(np.linspace(0, np.pi, branch_count))
                + 0.24 * np.linspace(0, 1, branch_count),
            )
        )
        b_points = np.column_stack(
            (
                np.linspace(0.44, 0.66, branch_count),
                0.52
                + 0.25 * np.sin(np.linspace(0, np.pi, branch_count))
                - 0.24 * np.linspace(0, 1, branch_count),
            )
        )
        positions = np.concatenate((trunk, a_points, b_points), axis=0)
        positions += self.rng.normal(0, 0.002 + 0.003 * self.difficulty, positions.shape)
        a_path = [junction] + list(range(trunk_count, trunk_count + branch_count))
        b_path = [junction] + list(
            range(trunk_count + branch_count, trunk_count + 2 * branch_count)
        )
        physical_endpoint_a = a_path[-1]
        physical_endpoint_b = b_path[-1]
        self.cable = CableGraph.from_paths(
            positions,
            [range(trunk_count), a_path, b_path],
            particle_radius=0.0085,
            stretch_stiffness=0.965,
            bend_stiffness=0.11,
            self_collision=True,
        )
        offset = self.rng.normal(0, 0.018 * self.difficulty, size=(2, 2))
        lower_target = np.asarray([0.86, 0.28]) + offset[0]
        upper_target = np.asarray([0.86, 0.72]) + offset[1]
        # Keep the same physical routing problem while randomly renaming both
        # branches and their matched targets. The geometry is identical under
        # both assignments, so only observable A/B semantics identify which
        # arm output must follow which route.
        swap_override = getattr(self, "semantic_swap_override", None)
        self.semantic_target_swap = (
            bool(self.rng.integers(0, 2)) if swap_override is None else bool(swap_override)
        )
        if self.semantic_target_swap:
            self.endpoint_a, self.target_a = physical_endpoint_b, upper_target
            self.endpoint_b, self.target_b = physical_endpoint_a, lower_target
        else:
            self.endpoint_a, self.target_a = physical_endpoint_a, lower_target
            self.endpoint_b, self.target_b = physical_endpoint_b, upper_target
        self.waypoints_a = self._semantic_waypoints(self.target_a)
        self.waypoints_b = self._semantic_waypoints(self.target_b)
        self.arm_a = PlanarArm(
            base=np.asarray([0.14, 0.10]),
            ee=self.cable.positions[self.endpoint_a].copy(),
            link_lengths=(0.48, 0.42),
            max_speed=0.76 - 0.10 * self.difficulty,
            elbow_up=False,
            name="branch_arm_a",
        )
        self.arm_b = PlanarArm(
            base=np.asarray([0.14, 0.90]),
            ee=self.cable.positions[self.endpoint_b].copy(),
            link_lengths=(0.48, 0.42),
            max_speed=0.76 - 0.10 * self.difficulty,
            elbow_up=False,
            name="branch_arm_b",
        )
        self.initial_crossings = self.cable.crossing_count()
        self.stable_steps = 0

    @staticmethod
    def _semantic_waypoints(target: np.ndarray) -> np.ndarray:
        clearance_y = BRANCH_CLEARANCE_Y[0] if float(target[1]) < 0.5 else BRANCH_CLEARANCE_Y[1]
        return np.column_stack((BRANCH_WAYPOINT_X, np.full(len(BRANCH_WAYPOINT_X), clearance_y)))

    def _step_task(self, action: np.ndarray) -> float:
        target_a = np.clip(action[0:2], 0.04, 0.96)
        grip_a = bool(action[2] >= 0.5)
        target_b = np.clip(action[3:5], 0.04, 0.96)
        grip_b = bool(action[5] >= 0.5)
        self.arm_a.move_toward(target_a, self.dt)
        self.arm_b.move_toward(target_b, self.dt)
        anchors: dict[int, np.ndarray] = {}
        if grip_a:
            anchors[self.endpoint_a] = self.arm_a.ee.copy()
        if grip_b:
            anchors[self.endpoint_b] = self.arm_b.ee.copy()
        self.cable.step(anchors, dt=self.dt, substeps=2, iterations=12)

        distance_a = float(np.linalg.norm(self.cable.positions[self.endpoint_a] - self.target_a))
        distance_b = float(np.linalg.norm(self.cable.positions[self.endpoint_b] - self.target_b))
        crossings = self.cable.crossing_count()
        tolerance = 0.036 - 0.010 * self.difficulty
        if distance_a < tolerance and distance_b < tolerance and crossings == 0:
            self.stable_steps += 1
        else:
            self.stable_steps = max(0, self.stable_steps - 1)
        return (
            -0.5 * (distance_a + distance_b)
            - 0.15 * crossings
            - 0.02 * self.cable.stretch_error()
            + (8.0 if self.stable_steps >= 4 else 0.0)
        )

    def observation(self) -> dict:
        return {
            "cable_positions": self.cable.positions.copy(),
            "cable_edges": self.cable.edges.copy(),
            "semantic_endpoints": {"A": self.endpoint_a, "B": self.endpoint_b},
            "semantic_targets": {"A": self.target_a.copy(), "B": self.target_b.copy()},
            "semantic_waypoints": {
                "A": self.waypoints_a.copy(),
                "B": self.waypoints_b.copy(),
            },
            "semantic_target_swap": self.semantic_target_swap,
            "arm_a_ee": self.arm_a.ee.copy(),
            "arm_b_ee": self.arm_b.ee.copy(),
            "crossings": self.cable.crossing_count(),
        }

    def metrics(self) -> dict[str, float | int | bool]:
        distance_a = float(np.linalg.norm(self.cable.positions[self.endpoint_a] - self.target_a))
        distance_b = float(np.linalg.norm(self.cable.positions[self.endpoint_b] - self.target_b))
        return {
            "endpoint_a_error": distance_a,
            "endpoint_b_error": distance_b,
            "mean_endpoint_error": 0.5 * (distance_a + distance_b),
            "crossings": self.cable.crossing_count(),
            "initial_crossings": self.initial_crossings,
            "topology_preserved": distance_a < 0.05 and distance_b < 0.05,
            "stretch_error": self.cable.stretch_error(),
            "semantic_target_swap": self.semantic_target_swap,
        }

    def success(self) -> bool:
        return self.stable_steps >= 4

    def policy_action(self, policy_name: str, policy_rng: np.random.Generator) -> np.ndarray:
        if policy_name == "topology":
            progress = self.step_count / self.max_steps
            if progress < BRANCH_PHASE_BOUNDARIES[0]:
                target_a = self.waypoints_a[0]
                target_b = self.waypoints_b[0]
            elif progress < BRANCH_PHASE_BOUNDARIES[1]:
                target_a = self.waypoints_a[1]
                target_b = self.waypoints_b[1]
            else:
                target_a, target_b = self.target_a, self.target_b
            return np.asarray([*target_a, 1.0, *target_b, 1.0])
        if policy_name == "geometry":
            endpoint_a = self.cable.positions[self.endpoint_a]
            endpoint_b = self.cable.positions[self.endpoint_b]
            # Assign solely by nearest Euclidean target, ignoring branch identity.
            costs_direct = np.linalg.norm(endpoint_a - self.target_a) + np.linalg.norm(
                endpoint_b - self.target_b
            )
            costs_swapped = np.linalg.norm(endpoint_a - self.target_b) + np.linalg.norm(
                endpoint_b - self.target_a
            )
            target_a, target_b = (
                (self.target_b, self.target_a)
                if costs_swapped < costs_direct
                else (self.target_a, self.target_b)
            )
            return np.asarray([*target_a, 1.0, *target_b, 1.0])
        if policy_name == "random":
            if self.step_count % 20 == 0 or not hasattr(self, "_random_targets"):
                self._random_targets = policy_rng.uniform(
                    [[0.32, 0.12], [0.32, 0.12]], [[0.92, 0.88], [0.92, 0.88]]
                )
            return np.asarray([*self._random_targets[0], 1.0, *self._random_targets[1], 1.0])
        raise KeyError(policy_name)

    def render(self, width: int = 640, height: int = 480):
        canvas = SceneCanvas(width, height)
        canvas.target(self.target_a, "A", PALETTE["arm"])
        canvas.target(self.target_b, "B", PALETTE["arm_alt"])
        canvas.cable(self.cable)
        for endpoint, label, color in (
            (self.endpoint_a, "A", PALETTE["arm"]),
            (self.endpoint_b, "B", PALETTE["arm_alt"]),
        ):
            x, y = canvas.xy(self.cable.positions[endpoint])
            r = canvas.radius(0.018)
            canvas.draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            canvas.draw.text((x + r + 2, y - r), label, fill=PALETTE["text"])
        canvas.arm(self.arm_a, alternate=False, gripped=True)
        canvas.arm(self.arm_b, alternate=True, gripped=True)
        metrics = self.metrics()
        canvas.text(
            "BranchBot - semantic branch disentanglement",
            f"step {self.step_count:03d} | mean error {metrics['mean_endpoint_error']:.3f} | "
            f"crossings {metrics['crossings']} | topology {metrics['topology_preserved']}",
        )
        return canvas.image
