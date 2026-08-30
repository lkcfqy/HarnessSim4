#!/usr/bin/env python3
"""Render frozen BranchBot paper tables, macros, and the key-policy heatmap.

Chart contract
--------------
Decision: show whether each policy preserves completion when only A/B semantic
binding changes, and how that pattern varies with physical difficulty.
Data: ``aggregate_by_policy_difficulty_assignment`` from the frozen paired
BranchBot report; every cell must contain the same declared number of physical
seeds.  The plot intentionally shows five decision-relevant policies while the
LaTeX table retains every trained ablation.
Encoding: rows are policies, columns are canonical/swapped assignments grouped
by difficulty, cell colour and annotation are success percentage on a fixed
0--100 scale.  Cividis is perceptually ordered and colour-vision robust.
Comparability: no panel-specific scales, no smoothing, no omitted cells, and no
error bars whose pairing structure would be ambiguous; exact paired tests live
in the CSV/report and manuscript text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POLICY_ORDER = (
    "learned_topology",
    "learned_physical",
    "learned_semantic",
    "learned_no_features",
    "learned_geometry",
    "act_chunk",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
    "random",
)
POLICY_LABELS = {
    "learned_topology": "TopoHarness",
    "learned_physical": "No semantic edges",
    "learned_semantic": "No physical edges",
    "learned_no_features": "Relations, no ID/order",
    "learned_geometry": "Geometry",
    "act_chunk": "ACT direct",
    "act_chunk_typed": "ACT-Pointer",
    "act_relational_typed": "ACT-RelPool",
    "teacher_topology": "Scripted teacher",
    "random": "Random",
}
HEATMAP_POLICIES = (
    "learned_topology",
    "learned_geometry",
    "act_chunk_typed",
    "act_relational_typed",
    "teacher_topology",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"${coefficient:.2f}\times10^{{{exponent}}}$"


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def _aggregate_index(report: dict) -> dict[str, dict]:
    return {str(row["policy"]): row for row in report["aggregate_by_policy"]}


def _write_main_table(report: dict, output_dir: Path) -> Path:
    aggregate = _aggregate_index(report)
    policies = [name for name in POLICY_ORDER if name in aggregate]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Policy & Succ. & Rate [95\% CI] \\",
        r"\midrule",
    ]
    for name in policies:
        row = aggregate[name]
        successes = round(float(row["success_rate"]) * int(row["episodes"]))
        lines.append(
            f"{_latex_escape(POLICY_LABELS.get(name, name))} & "
            f"{successes}/{int(row['episodes'])} & "
            f"{100 * float(row['success_rate']):.1f} "
            f"[{100 * float(row['success_ci95_low']):.1f},"
            f"{100 * float(row['success_ci95_high']):.1f}] \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path = output_dir / "branch_paper_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_mujoco_table(report: dict, output_dir: Path) -> Path:
    rows = {str(row["policy"]): row for row in report["aggregate_by_policy"]}
    policies = [name for name in ("learned_topology", "learned_geometry", "teacher_topology") if name in rows]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Policy & Succ. & Rate [95\% CI] \\",
        r"\midrule",
    ]
    for name in policies:
        row = rows[name]
        successes = round(float(row["success_rate"]) * int(row["episodes"]))
        lines.append(
            f"{_latex_escape(POLICY_LABELS[name])} & {successes}/{int(row['episodes'])} & "
            f"{100 * float(row['success_rate']):.1f} "
            f"[{100 * float(row['success_ci95_low']):.1f},"
            f"{100 * float(row['success_ci95_high']):.1f}] \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path = output_dir / "branch_mujoco_direct_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _count(report: dict, policy: str) -> int:
    row = _aggregate_index(report)[policy]
    return round(float(row["success_rate"]) * int(row["episodes"]))


def _rate(report: dict, policy: str) -> float:
    return 100.0 * float(_aggregate_index(report)[policy]["success_rate"])


def _write_numbers(report: dict, mujoco_report: dict | None, output_dir: Path) -> Path:
    comparisons = [
        row for row in report["paired_comparisons"] if row["baseline"] == "learned_geometry"
    ]
    max_holm = max(float(row["mcnemar_exact_p_holm"]) for row in comparisons)
    values = {
        "TopoCount": str(_count(report, "learned_topology")),
        "TopoRate": f"{_rate(report, 'learned_topology'):.1f}",
        "GeomCount": str(_count(report, "learned_geometry")),
        "GeomRate": f"{_rate(report, 'learned_geometry'):.1f}",
        "ACTPointerCount": str(_count(report, "act_chunk_typed")),
        "ACTRelCount": str(_count(report, "act_relational_typed")),
        "TeacherCount": str(_count(report, "teacher_topology")),
        "MaxHolmP": _format_scientific(max_holm),
        "MuJoCoTopoCount": (
            str(_count(mujoco_report, "learned_topology")) if mujoco_report else "TBD"
        ),
        "MuJoCoGeomCount": (
            str(_count(mujoco_report, "learned_geometry")) if mujoco_report else "TBD"
        ),
    }
    path = output_dir / "branch_numbers.tex"
    path.write_text(
        "\n".join(
            rf"\renewcommand{{\{name}}}{{{value}}}" for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _render_heatmap(report: dict, output_dir: Path) -> tuple[Path, Path]:
    physical_seeds = int(report["config"]["physical_seeds_per_difficulty"])
    cells = {
        (
            str(row["policy"]),
            float(row["difficulty"]),
            bool(row["semantic_target_swap"]),
        ): row
        for row in report["aggregate_by_policy_difficulty_assignment"]
    }
    difficulties = tuple(float(value) for value in report["config"]["difficulties"])
    columns = [(difficulty, swap) for difficulty in difficulties for swap in (False, True)]
    matrix = np.empty((len(HEATMAP_POLICIES), len(columns)), dtype=np.float64)
    for row_index, policy in enumerate(HEATMAP_POLICIES):
        for column_index, (difficulty, swap) in enumerate(columns):
            row = cells[(policy, difficulty, swap)]
            if int(row["episodes"]) != physical_seeds:
                raise ValueError(
                    f"unexpected cell size for {policy}, {difficulty}, {swap}: {row['episodes']}"
                )
            matrix[row_index, column_index] = 100.0 * float(row["success_rate"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.7, 2.9), constrained_layout=True)
    image = ax.imshow(matrix, cmap="cividis", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_yticks(range(len(HEATMAP_POLICIES)))
    ax.set_yticklabels([POLICY_LABELS[name] for name in HEATMAP_POLICIES])
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(["Canonical" if not swap else "A/B swap" for _, swap in columns])
    ax.tick_params(axis="x", rotation=30, pad=1)
    ax.set_ylabel("Policy")
    for difficulty_index, difficulty in enumerate(difficulties):
        center = 2 * difficulty_index + 0.5
        ax.text(
            center,
            -0.78,
            f"Difficulty {difficulty:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            transform=ax.transData,
            clip_on=False,
        )
        if difficulty_index:
            ax.axvline(2 * difficulty_index - 0.5, color="white", linewidth=1.8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value < 52 else "#101820",
                fontsize=8,
                fontweight="semibold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.018)
    colorbar.set_label("Success (%)")
    colorbar.set_ticks((0, 25, 50, 75, 100))
    ax.set_title("Exact semantic intervention on paired physical seeds", pad=29)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pdf_path = output_dir / "branch_counterfactual_heatmap.pdf"
    png_path = output_dir / "branch_counterfactual_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mujoco-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    mujoco_report = (
        json.loads(args.mujoco_report.read_text(encoding="utf-8"))
        if args.mujoco_report and args.mujoco_report.is_file()
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    main_table = _write_main_table(report, args.output)
    numbers = _write_numbers(report, mujoco_report, args.output)
    heatmap_pdf, heatmap_png = _render_heatmap(report, args.output)
    outputs = {
        "report_sha256": _sha256(args.report),
        "main_table": str(main_table.resolve()),
        "numbers": str(numbers.resolve()),
        "heatmap_pdf": str(heatmap_pdf.resolve()),
        "heatmap_png": str(heatmap_png.resolve()),
    }
    if mujoco_report is not None and args.mujoco_report is not None:
        mujoco_output = args.mujoco_report.parent
        outputs["mujoco_report_sha256"] = _sha256(args.mujoco_report)
        outputs["mujoco_table"] = str(_write_mujoco_table(mujoco_report, mujoco_output).resolve())
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
