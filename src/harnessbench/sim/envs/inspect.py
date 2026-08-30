"""InspectBot: active topology-aware inspection of harness-critical sites."""

from __future__ import annotations

import numpy as np

from harnessbench.sim.core import CableGraph
from harnessbench.sim.envs.base import HarnessEnv
from harnessbench.sim.rendering import PALETTE, SceneCanvas


class InspectEnv(HarnessEnv):
    task_name = "active_inspection"
    robot_name = "InspectBot"
    max_steps = 260
    action_dim = 3

    def _reset_task(self) -> None:
        trunk = np.column_stack((np.linspace(0.10, 0.72, 15), np.full(15, 0.55)))
        upper = np.column_stack((np.linspace(0.46, 0.83, 8), np.linspace(0.54, 0.28, 8)))
        lower = np.column_stack((np.linspace(0.46, 0.83, 8), np.linspace(0.56, 0.78, 8)))
        positions = np.concatenate((trunk, upper, lower), axis=0)
        junction = 8
        upper_path = [junction] + list(range(15, 23))
        lower_path = [junction] + list(range(23, 31))
        self.cable = CableGraph.from_paths(
            positions,
            [range(15), upper_path, lower_path],
            self_collision=False,
            bend_stiffness=0.1,
        )
        jitter = self.rng.normal(0.0, 0.006 * self.difficulty, self.cable.positions.shape)
        self.cable.positions += jitter
        self.site_names = [
            "input_terminal",
            "clip_1",
            "branch_junction",
            "clip_upper",
            "clip_lower",
            "terminal_upper",
            "terminal_lower",
        ]
        self.sites = np.asarray(
            [
                self.cable.positions[0],
                self.cable.positions[6],
                self.cable.positions[junction],
                self.cable.positions[18],
                self.cable.positions[26],
                self.cable.positions[22],
                self.cable.positions[30],
            ]
        )
        defect_probability = 0.18 + 0.22 * self.difficulty
        self.defects = self.rng.random(len(self.sites)) < defect_probability
        if not self.defects.any():
            self.defects[int(self.rng.integers(0, len(self.sites)))] = True
        self.camera = np.asarray([0.14, 0.18], dtype=np.float64)
        self.inspected: set[int] = set()
        self.detections: set[int] = set()
        self.scan_attempts: dict[int, int] = {}
        # Observable classifier confidence. This prevents the expert from using
        # hidden ground-truth defect labels when deciding which site to revisit.
        self.site_scores = np.full(len(self.sites), 0.5, dtype=np.float64)
        self.site_score_counts = np.zeros(len(self.sites), dtype=np.int64)
        self.fov_radius = 0.105 - 0.025 * self.difficulty
        self._geometry_waypoints = np.asarray(
            [[0.15, 0.50], [0.33, 0.50], [0.51, 0.50], [0.69, 0.50], [0.86, 0.50]]
        )
        self._geometry_index = 0

    def _step_task(self, action: np.ndarray) -> float:
        target = np.clip(action[:2], 0.04, 0.96)
        trigger = bool(action[2] >= 0.5)
        delta = target - self.camera
        distance = float(np.linalg.norm(delta))
        speed = 0.62 - 0.12 * self.difficulty
        if distance <= speed * self.dt:
            self.camera = target.copy()
        elif distance > 1e-9:
            self.camera += delta / distance * speed * self.dt

        newly_inspected = 0
        correct_detection = 0
        false_detection = 0
        if trigger:
            distances = np.linalg.norm(self.sites - self.camera, axis=1)
            visible = np.flatnonzero(distances <= self.fov_radius)
            for raw_index in visible:
                index = int(raw_index)
                if index not in self.inspected:
                    newly_inspected += 1
                self.inspected.add(index)
                self.scan_attempts[index] = self.scan_attempts.get(index, 0) + 1
                distance_factor = max(0.0, 1.0 - float(distances[index]) / self.fov_radius)
                score_mean = (
                    0.84 - 0.12 * self.difficulty
                    if self.defects[index]
                    else 0.12 + 0.10 * self.difficulty
                )
                score_noise = 0.08 + 0.06 * self.difficulty
                measurement = float(
                    np.clip(
                        self.rng.normal(score_mean + 0.05 * distance_factor, score_noise),
                        0.0,
                        1.0,
                    )
                )
                count = int(self.site_score_counts[index])
                self.site_scores[index] = (self.site_scores[index] * count + measurement) / (
                    count + 1
                )
                self.site_score_counts[index] = count + 1
                was_detected = index in self.detections
                # A noisy measurement already supplies stochasticity. Decisions
                # are made from the observable running confidence, not a second
                # hidden-label Bernoulli draw.
                if self.site_scores[index] >= 0.60:
                    self.detections.add(index)
                elif self.scan_attempts[index] >= 2:
                    self.detections.discard(index)
                if not was_detected and index in self.detections:
                    if self.defects[index]:
                        correct_detection += 1
                    else:
                        false_detection += 1
        return 0.25 * newly_inspected + correct_detection - 0.5 * false_detection - 0.002

    def observation(self) -> dict:
        return {
            "cable_positions": self.cable.positions.copy(),
            "cable_edges": self.cable.edges.copy(),
            "camera": self.camera.copy(),
            "sites": self.sites.copy(),
            "inspected": sorted(self.inspected),
            "detections": sorted(self.detections),
            "scan_attempts": np.asarray(
                [self.scan_attempts.get(index, 0) for index in range(len(self.sites))],
                dtype=np.int64,
            ),
            "site_scores": self.site_scores.copy(),
            "fov_radius": self.fov_radius,
        }

    def metrics(self) -> dict[str, float | int | bool]:
        truth = set(np.flatnonzero(self.defects).tolist())
        true_positive = len(truth & self.detections)
        false_positive = len(self.detections - truth)
        false_negative = len(truth - self.detections)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {
            "coverage": len(self.inspected) / len(self.sites),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_defects": len(truth),
            "detected_defects": true_positive,
            "false_positives": false_positive,
        }

    def success(self) -> bool:
        metrics = self.metrics()
        return bool(
            metrics["coverage"] >= 0.999
            and metrics["recall"] >= 0.999
            and metrics["precision"] >= 0.80
        )

    def policy_action(self, policy_name: str, policy_rng: np.random.Generator) -> np.ndarray:
        if policy_name == "topology":
            remaining = [index for index in range(len(self.sites)) if index not in self.inspected]
            if remaining:
                index = min(
                    remaining, key=lambda value: np.linalg.norm(self.sites[value] - self.camera)
                )
                target = self.sites[index]
                trigger = float(np.linalg.norm(target - self.camera) < self.fov_radius * 0.35)
                return np.asarray([target[0], target[1], trigger])
            # Revisit only sites that remain uncertain according to observable
            # classifier scores; never consult self.defects here.
            unresolved = [
                index
                for index in range(len(self.sites))
                if index not in self.detections
                and self.scan_attempts.get(index, 0) < 4
                and self.site_scores[index] > 0.30
            ]
            if unresolved:
                index = min(
                    unresolved,
                    key=lambda value: (
                        self.scan_attempts.get(value, 0),
                        np.linalg.norm(self.sites[value] - self.camera),
                    ),
                )
                target = self.sites[index]
                return np.asarray([target[0], target[1], 1.0])
            return np.asarray([self.camera[0], self.camera[1], 0.0])
        if policy_name == "geometry":
            target = self._geometry_waypoints[self._geometry_index]
            trigger = float(np.linalg.norm(target - self.camera) < 0.025)
            if trigger and self._geometry_index < len(self._geometry_waypoints) - 1:
                self._geometry_index += 1
            return np.asarray([target[0], target[1], trigger])
        if policy_name == "random":
            if self.step_count % 16 == 0 or not hasattr(self, "_random_target"):
                self._random_target = policy_rng.uniform([0.08, 0.12], [0.92, 0.88])
                self._random_trigger = float(policy_rng.random() > 0.55)
            return np.asarray(
                [self._random_target[0], self._random_target[1], self._random_trigger]
            )
        raise KeyError(policy_name)

    def render(self, width: int = 640, height: int = 480):
        canvas = SceneCanvas(width, height)
        canvas.cable(self.cable)
        for index, site in enumerate(self.sites):
            x, y = canvas.xy(site)
            r = canvas.radius(0.014)
            outline = (
                PALETTE["warning"]
                if self.defects[index]
                else (PALETTE["target"] if index in self.inspected else PALETTE["muted"])
            )
            canvas.draw.ellipse((x - r, y - r, x + r, y + r), outline=outline, width=3)
            if index in self.detections:
                canvas.draw.line((x - r, y - r, x + r, y + r), fill=PALETTE["warning"], width=2)
                canvas.draw.line((x - r, y + r, x + r, y - r), fill=PALETTE["warning"], width=2)
            canvas.draw.text((x + r + 2, y - r), str(index + 1), fill=PALETTE["text"])
        cx, cy = canvas.xy(self.camera)
        fov = canvas.radius(self.fov_radius)
        canvas.draw.ellipse(
            (cx - fov, cy - fov, cx + fov, cy + fov),
            outline=PALETTE["camera"],
            width=2,
        )
        camera_r = canvas.radius(0.018)
        canvas.draw.rectangle(
            (cx - camera_r, cy - camera_r, cx + camera_r, cy + camera_r),
            fill=PALETTE["camera"],
        )
        metrics = self.metrics()
        canvas.text(
            "InspectBot - active harness inspection",
            f"step {self.step_count:03d} | coverage {metrics['coverage']:.2f} | "
            f"precision {metrics['precision']:.2f} | recall {metrics['recall']:.2f}",
        )
        return canvas.image
