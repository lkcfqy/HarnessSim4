#!/usr/bin/env python3
"""Plot the frozen RouteBot assignment-by-difficulty success matrix.

Chart contract
--------------
Question: Does changing only the segment-to-clip relation expose geometric or
assignment shortcuts?
Takeaway: Relation-equipped policies remain high across all cells, Geometry is
near zero, and ACT-Pointer is strongly assignment-sensitive.
Family/variant: three vertically faceted heatmaps with a shared 0--100% scale.
Data: 81 paper-scale cells (3 policies x 3 difficulties x 9 assignments), each
with 100 paired physical seeds; ACT-RelPool is asserted identical to TopoHarness.
Surface: static Matplotlib PDF and PNG for a single-column LaTeX figure.
Palette: one blue root plus neutrals; numeric labels preserve non-color access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

WORKSPACE = Path(__file__).resolve().parents[1]
PANELS = (
    ("learned_topology", "TopoHarness / ACT-RelPool"),
    ("learned_geometry", "Geometry"),
    ("act_chunk_typed", "ACT-Pointer"),
)


def _matrix(
    rows: list[dict],
    policy: str,
    difficulties: list[float],
    assignments: list[str],
) -> np.ndarray:
    lookup = {
        (float(row["difficulty"]), str(row["target_assignment"])): 100.0
        * float(row["success_rate"])
        for row in rows
        if row["policy"] == policy
    }
    expected = {(difficulty, assignment) for difficulty in difficulties for assignment in assignments}
    if set(lookup) != expected:
        raise ValueError(f"Incomplete matrix for {policy}: {len(lookup)} cells")
    return np.asarray(
        [[lookup[(difficulty, assignment)] for assignment in assignments] for difficulty in difficulties]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            WORKSPACE
            / "artifacts"
            / "papers"
            / "routebot"
            / "final"
            / "counterfactual"
            / "route_counterfactual_report.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=(
            WORKSPACE
            / "artifacts"
            / "papers"
            / "routebot"
            / "final"
            / "counterfactual"
            / "route_counterfactual_heatmap"
        ),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if int(report["config"]["physical_seeds_per_difficulty"]) != 100:
        raise ValueError("The frozen paper figure requires exactly 100 seeds per cell")
    difficulties = [float(value) for value in report["config"]["difficulties"]]
    assignments = [str(value) for value in report["config"]["target_assignments"]]
    rows = report["aggregate_by_policy_assignment"]

    topology = _matrix(rows, "learned_topology", difficulties, assignments)
    relational = _matrix(rows, "act_relational_typed", difficulties, assignments)
    if not np.array_equal(topology, relational):
        raise ValueError("ACT-RelPool is no longer cellwise identical to TopoHarness")

    palette = LinearSegmentedColormap.from_list(
        "route_blue",
        ("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        len(PANELS),
        1,
        figsize=(3.45, 3.15),
        sharex=True,
        constrained_layout=False,
    )
    figure.patch.set_facecolor("white")
    image = None
    for axis, (policy, title) in zip(axes, PANELS, strict=True):
        values = _matrix(rows, policy, difficulties, assignments)
        image = axis.imshow(values, cmap=palette, vmin=0.0, vmax=100.0, aspect="auto")
        axis.set_title(title, loc="left", fontsize=7.2, fontweight="bold", color="#20252b", pad=2)
        axis.set_yticks(range(len(difficulties)), [f"{value:.1f}" for value in difficulties])
        axis.tick_params(axis="y", labelsize=5.5, length=0, pad=2)
        axis.set_ylabel("difficulty", fontsize=5.5, color="#38434f", labelpad=3)
        axis.set_xticks(np.arange(-0.5, len(assignments), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(difficulties), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.65)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=4.35,
                    fontweight="bold" if value >= 80 else "normal",
                    color="white" if value >= 58 else "#17212b",
                )
        for spine in axis.spines.values():
            spine.set_color("#6f7a85")
            spine.set_linewidth(0.45)

    axes[-1].set_xticks(range(len(assignments)), assignments, rotation=42, ha="right")
    axes[-1].tick_params(axis="x", labelsize=4.8, length=0, pad=1)
    axes[-1].set_xlabel("segment assignment to ordered fixtures", fontsize=5.5, labelpad=1)
    figure.text(
        0.11,
        0.985,
        "Semantic success per cell (%) — 100 paired physical seeds",
        ha="left",
        va="top",
        fontsize=6.1,
        color="#38434f",
    )
    assert image is not None
    color_axis = figure.add_axes((0.87, 0.12, 0.022, 0.78))
    colorbar = figure.colorbar(image, cax=color_axis, ticks=(0, 50, 100))
    colorbar.ax.tick_params(labelsize=5, length=2, width=0.45, pad=1)
    colorbar.outline.set_linewidth(0.45)
    figure.subplots_adjust(left=0.14, right=0.84, top=0.93, bottom=0.20, hspace=0.48)

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {prefix.with_suffix('.pdf')}")
    print(f"Saved {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
