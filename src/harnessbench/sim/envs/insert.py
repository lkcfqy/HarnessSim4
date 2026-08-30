"""InsertBot: topology-guided terminal alignment and insertion."""

from __future__ import annotations

import numpy as np

from harnessbench.sim.arm import PlanarArm
from harnessbench.sim.core import CableGraph, CircleObstacle, wrap_angle
from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.rendering import PALETTE, SceneCanvas


class InsertEnv(HarnessEnv):
    task_name = "terminal_insertion"
    robot_name = "InsertBot"
    max_steps = 220
    action_dim = 4

    def _reset_task(self) -> None:
        count = 22
        t = np.linspace(0.0, 1.0, count)
        y_center = float(self.rng.uniform(0.44, 0.60))
        start = np.asarray([0.10, y_center + self.rng.uniform(-0.10, 0.10)])
        end = np.asarray([0.43, y_center + self.rng.uniform(-0.07, 0.07)])
        positions = np.column_stack(
            (
                start[0] + (end[0] - start[0]) * t,
                start[1]
                + (end[1] - start[1]) * t
                + (0.035 + 0.025 * self.difficulty) * np.sin(t * np.pi * 2.0),
            )
        )
        self.cable = CableGraph.chain(
            positions,
            particle_radius=0.009,
            stretch_stiffness=0.97,
            bend_stiffness=0.10 + 0.16 * (1.0 - self.difficulty),
            self_collision=True,
        )
        self.endpoint = count - 1
        self.neighbor = count - 2
        self.socket_angle = float(self.rng.uniform(-0.28, 0.28) * (0.4 + self.difficulty))
        self.socket = np.asarray([0.82, y_center])
        self.axis = np.asarray([np.cos(self.socket_angle), np.sin(self.socket_angle)])
        self.preinsert = self.socket - self.axis * (0.13 + 0.035 * self.difficulty)
        self.arm = PlanarArm(
            base=np.asarray([0.18, 0.88]),
            ee=self.cable.positions[self.endpoint].copy(),
            link_lengths=(0.40, 0.36),
            max_speed=0.66 - 0.10 * self.difficulty,
            elbow_up=False,
            name="insert_arm",
        )
        obstacle_offset = np.asarray([-self.axis[1], self.axis[0]])
        clearance = 0.095 - 0.020 * self.difficulty
        self.obstacles = (
            CircleObstacle(self.preinsert + obstacle_offset * clearance, 0.040),
            CircleObstacle(self.preinsert - obstacle_offset * clearance, 0.040),
        )
        self.peak_force = 0.0
        self.aligned_steps = 0
        self.locked = False
        self._topology_phase = 0
        self.terminal_orientation = self.cable.endpoint_angle(self.endpoint, self.neighbor)

    def _step_task(self, action: np.ndarray) -> float:
        target = np.clip(action[:2], 0.04, 0.96)
        desired_orientation = float(action[2])
        grip = bool(action[3] >= 0.5)
        self.arm.move_toward(target, self.dt)
        angular_error = wrap_angle(desired_orientation - self.terminal_orientation)
        max_angular_step = (2.6 - 0.6 * self.difficulty) * self.dt
        self.terminal_orientation += float(
            np.clip(angular_error, -max_angular_step, max_angular_step)
        )
        anchors: dict[int, np.ndarray] = {}
        if grip:
            anchors[self.endpoint] = self.arm.ee.copy()
            terminal_length = float(self.cable.rest_lengths[-1])
            direction = np.asarray(
                [np.cos(self.terminal_orientation), np.sin(self.terminal_orientation)]
            )
            anchors[self.neighbor] = self.arm.ee - direction * terminal_length
        self.cable.step(
            anchors,
            self.obstacles,
            dt=self.dt,
            substeps=2,
            iterations=12,
        )
        distance = float(np.linalg.norm(self.cable.positions[self.endpoint] - self.socket))
        angle_error = abs(
            wrap_angle(self.cable.endpoint_angle(self.endpoint, self.neighbor) - self.socket_angle)
        )
        socket_offset = self.cable.positions[self.endpoint] - self.socket
        lateral_error = abs(
            float(self.axis[0] * socket_offset[1] - self.axis[1] * socket_offset[0])
        )
        force_proxy = (
            3.0 * self.cable.stretch_error()
            + 4.0 * angle_error * max(0.0, 0.12 - distance)
            + 7.0 * lateral_error * max(0.0, 0.14 - distance)
        )
        self.peak_force = max(self.peak_force, float(force_proxy))

        tolerance = 0.032 - 0.010 * self.difficulty
        angle_tolerance = 0.30 - 0.10 * self.difficulty
        if distance < tolerance and angle_error < angle_tolerance and grip:
            self.aligned_steps += 1
        else:
            self.aligned_steps = max(0, self.aligned_steps - 1)
        if self.aligned_steps >= 4:
            self.locked = True
            self.cable.positions[self.endpoint] = self.socket + self.axis * 0.006

        return -distance - 0.12 * angle_error - 0.02 * force_proxy + (8.0 if self.locked else 0.0)

    def observation(self) -> dict:
        return {
            "cable_positions": self.cable.positions.copy(),
            "cable_edges": self.cable.edges.copy(),
            "arm_ee": self.arm.ee.copy(),
            "socket": self.socket.copy(),
            "socket_axis": self.axis.copy(),
            "terminal_angle": self.cable.endpoint_angle(self.endpoint, self.neighbor),
        }

    def metrics(self) -> dict[str, float | int | bool]:
        terminal = self.cable.positions[self.endpoint]
        distance = float(np.linalg.norm(terminal - self.socket))
        angle_error = abs(
            wrap_angle(self.cable.endpoint_angle(self.endpoint, self.neighbor) - self.socket_angle)
        )
        return {
            "terminal_error": distance,
            "angle_error_rad": angle_error,
            "peak_force_proxy": self.peak_force,
            "stretch_error": self.cable.stretch_error(),
            "locked": self.locked,
        }

    def success(self) -> bool:
        return self.locked

    def policy_action(self, policy_name: str, policy_rng: np.random.Generator) -> np.ndarray:
        terminal = self.cable.positions[self.endpoint]
        if policy_name == "topology":
            if self._topology_phase == 0 and np.linalg.norm(terminal - self.preinsert) < 0.022:
                self._topology_phase = 1
            target = (
                self.preinsert if self._topology_phase == 0 else self.socket + self.axis * 0.008
            )
            return np.asarray([target[0], target[1], self.socket_angle, 1.0])
        if policy_name == "geometry":
            return np.asarray([self.socket[0], self.socket[1], 0.0, 1.0])
        if policy_name == "random":
            if self.step_count % 18 == 0 or not hasattr(self, "_random_target"):
                self._random_target = policy_rng.uniform([0.30, 0.18], [0.92, 0.82])
                self._random_orientation = float(policy_rng.uniform(-np.pi, np.pi))
            return np.asarray(
                [
                    self._random_target[0],
                    self._random_target[1],
                    self._random_orientation,
                    1.0,
                ]
            )
        raise KeyError(policy_name)

    def render(self, width: int = 640, height: int = 480):
        canvas = SceneCanvas(width, height)
        for obstacle in self.obstacles:
            canvas.obstacle(obstacle)
        canvas.target(self.preinsert, "pre-align", PALETTE["camera"])
        # Connector body and insertion slot.
        sx, sy = canvas.xy(self.socket)
        body_w, body_h = canvas.radius(0.075), canvas.radius(0.055)
        canvas.draw.rounded_rectangle(
            (sx - body_w, sy - body_h, sx + body_w, sy + body_h),
            radius=6,
            fill=(34, 105, 139),
            outline=PALETTE["border"],
            width=2,
        )
        slot = canvas.radius(0.018)
        canvas.draw.ellipse((sx - slot, sy - slot, sx + slot, sy + slot), fill=(25, 29, 33))
        canvas.cable(self.cable)
        canvas.arm(self.arm, gripped=True)
        metrics = self.metrics()
        canvas.text(
            "InsertBot - terminal insertion",
            f"step {self.step_count:03d} | error {metrics['terminal_error']:.3f} | "
            f"angle {metrics['angle_error_rad']:.2f} rad | locked {self.locked}",
        )
        return canvas.image
