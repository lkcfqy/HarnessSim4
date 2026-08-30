"""Pillow rendering primitives shared by project demos and paper figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harnessbench.sim.arm import PlanarArm
from harnessbench.sim.core import CableGraph, CircleObstacle

PALETTE = {
    "background": (232, 235, 238),
    "board": (248, 249, 250),
    "border": (72, 80, 88),
    "grid": (218, 222, 226),
    "cable": (127, 45, 36),
    "cable_alt": (33, 105, 139),
    "node": (91, 35, 31),
    "arm": (48, 86, 122),
    "arm_alt": (170, 91, 42),
    "gripper": (28, 31, 34),
    "target": (47, 135, 86),
    "warning": (196, 58, 51),
    "clip": (78, 70, 153),
    "camera": (45, 121, 141),
    "text": (35, 39, 43),
    "muted": (92, 98, 104),
}


@dataclass
class SceneCanvas:
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        self.image = Image.new("RGB", (self.width, self.height), PALETTE["background"])
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default()
        self._draw_board()

    def xy(self, point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        return round(x * self.width), round(y * self.height)

    def radius(self, value: float) -> int:
        return max(1, round(value * min(self.width, self.height)))

    def _draw_board(self) -> None:
        margin = self.radius(0.03)
        self.draw.rounded_rectangle(
            (margin, margin, self.width - margin, self.height - margin),
            radius=self.radius(0.018),
            fill=PALETTE["board"],
            outline=PALETTE["border"],
            width=2,
        )
        for value in np.linspace(0.1, 0.9, 9):
            x = int(value * self.width)
            y = int(value * self.height)
            self.draw.line((x, margin, x, self.height - margin), fill=PALETTE["grid"], width=1)
            self.draw.line((margin, y, self.width - margin, y), fill=PALETTE["grid"], width=1)

    def cable(self, graph: CableGraph, color: tuple[int, int, int] | None = None) -> None:
        cable_color = color or PALETTE["cable"]
        width = max(4, self.radius(graph.particle_radius * 1.35))
        for i, j in graph.edges:
            self.draw.line(
                (self.xy(graph.positions[i]), self.xy(graph.positions[j])),
                fill=cable_color,
                width=width,
            )
        node_radius = max(2, width // 3)
        for point in graph.positions:
            x, y = self.xy(point)
            self.draw.ellipse(
                (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
                fill=PALETTE["node"],
            )

    def arm(self, arm: PlanarArm, alternate: bool = False, gripped: bool = True) -> None:
        points = [self.xy(point) for point in arm.joints()]
        color = PALETTE["arm_alt"] if alternate else PALETTE["arm"]
        self.draw.line((points[0], points[1]), fill=color, width=self.radius(0.022))
        self.draw.line((points[1], points[2]), fill=color, width=self.radius(0.018))
        for radius, point in ((0.026, points[0]), (0.020, points[1])):
            r = self.radius(radius)
            self.draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color)
        ee = self.xy(arm.ee)
        r = self.radius(0.014)
        gripper_color = PALETTE["gripper"] if gripped else PALETTE["muted"]
        self.draw.rectangle((ee[0] - r, ee[1] - r, ee[0] + r, ee[1] + r), fill=gripper_color)

    def obstacle(self, obstacle: CircleObstacle) -> None:
        x, y = self.xy(obstacle.center)
        r = self.radius(obstacle.radius)
        self.draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(176, 181, 186),
            outline=PALETTE["border"],
            width=2,
        )

    def target(
        self, point: np.ndarray, label: str = "", color: tuple[int, int, int] | None = None
    ) -> None:
        x, y = self.xy(point)
        r = self.radius(0.025)
        chosen = color or PALETTE["target"]
        self.draw.ellipse((x - r, y - r, x + r, y + r), outline=chosen, width=3)
        self.draw.line((x - r, y, x + r, y), fill=chosen, width=2)
        self.draw.line((x, y - r, x, y + r), fill=chosen, width=2)
        if label:
            self.draw.text((x + r + 3, y - r), label, fill=PALETTE["text"], font=self.font)

    def clip(self, point: np.ndarray, latched: bool = False, correct: bool = True) -> None:
        x, y = self.xy(point)
        r = self.radius(0.023)
        color = (
            PALETTE["target"]
            if latched and correct
            else (PALETTE["warning"] if latched else PALETTE["clip"])
        )
        self.draw.arc((x - r, y - r, x + r, y + r), 20, 160, fill=color, width=4)
        self.draw.arc((x - r, y - r, x + r, y + r), 200, 340, fill=color, width=4)

    def text(self, title: str, detail: str = "") -> None:
        self.draw.rectangle((14, 10, self.width - 14, 48), fill=(248, 249, 250))
        self.draw.text((22, 17), title, fill=PALETTE["text"], font=self.font)
        if detail:
            self.draw.text((22, 32), detail, fill=PALETTE["muted"], font=self.font)
