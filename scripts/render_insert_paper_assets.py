#!/usr/bin/env python3
"""Render frozen InsertBot paper figures, tables, and number macros.

Chart contract
--------------
Questions: (1) how success changes with difficulty; (2) whether hard-condition
success trades against peak force; (3) whether the qualitative safety result
survives direct MuJoCo contact; and (4) what the representative native-contact
recovery looks like over time.
Data: 4,800 paired 2.5-D episodes (200 physical seeds per difficulty) and 300
paired direct-MuJoCo episodes (20 physical seeds per difficulty).  Every plotted
aggregate keeps its declared policy/difficulty grain; no episode is filtered.
Encoding: fixed 0--100 success axes, colour-plus-marker policy identity,
explicit bootstrap intervals for the primary benchmark, hatched damage bars,
and exact values in companion LaTeX tables.  The MuJoCo audit is shown as a
descriptive cross-physics check rather than pooled with the primary benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

POLICY_ORDER = (
    "contact_belief",
    "guarded_admittance",
    "vision_staged_no_force",
    "spiral_search",
    "direct_insertion",
    "contact_no_orientation",
    "contact_no_retract",
    "oracle_teacher",
)
POLICY_LABELS = {
    "contact_belief": "ContactBelief (ours)",
    "guarded_admittance": "Guarded admittance",
    "vision_staged_no_force": "Vision only",
    "spiral_search": "Spiral search",
    "direct_insertion": "Direct insertion",
    "contact_no_orientation": "No orientation update",
    "contact_no_retract": "No retract state",
    "oracle_teacher": "State-aware oracle",
}
SHORT_LABELS = {
    "contact_belief": "Ours",
    "guarded_admittance": "Guarded",
    "vision_staged_no_force": "Vision",
    "spiral_search": "Spiral",
    "direct_insertion": "Direct",
    "contact_no_orientation": "No orient.",
    "contact_no_retract": "No retract",
    "oracle_teacher": "Oracle",
}
COLORS = {
    "contact_belief": "#0072B2",
    "guarded_admittance": "#E69F00",
    "vision_staged_no_force": "#CC79A7",
    "spiral_search": "#009E73",
    "direct_insertion": "#D55E00",
    "contact_no_orientation": "#56B4E9",
    "contact_no_retract": "#202020",
    "oracle_teacher": "#777777",
}
MARKERS = {
    "contact_belief": "o",
    "guarded_admittance": "s",
    "vision_staged_no_force": "^",
    "spiral_search": "D",
    "direct_insertion": "X",
    "contact_no_orientation": "P",
    "contact_no_retract": "v",
    "oracle_teacher": "*",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def _format_p(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"${coefficient:.2f}\!\times\!10^{{{exponent}}}$"


def _index(report: dict) -> dict[tuple[float, str], dict]:
    return {(float(row["difficulty"]), str(row["policy"])): row for row in report["aggregate"]}


def _write_main_table(report: dict, output_dir: Path) -> Path:
    rows = _index(report)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & Easy & Mid & Hard & Hard dmg. & Hard force & Hard time \\",
        r" & \multicolumn{3}{c}{success (\%)} & (\%) & (N) & (s) \\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        hard = rows[(0.8, policy)]
        lines.append(
            f"{_latex_escape(POLICY_LABELS[policy])} & "
            f"{100 * float(rows[(0.2, policy)]['success_rate']):.1f} & "
            f"{100 * float(rows[(0.5, policy)]['success_rate']):.1f} & "
            f"{100 * float(hard['success_rate']):.1f} & "
            f"{100 * float(hard['damage_rate']):.1f} & "
            f"{float(hard['peak_force_mean_n']):.1f} & "
            f"{float(hard['cycle_time_mean_s']):.2f} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path = output_dir / "insert_main_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_paired_table(report: dict, output_dir: Path) -> Path:
    rows = [row for row in report["paired_comparisons"] if float(row["difficulty"]) == 0.8]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparator & $\Delta$succ. & Holm $p$ & $\Delta$force & Holm $p$ \\",
        r" & (pp) & McNemar & (N) & Wilcoxon \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(POLICY_LABELS[str(row['baseline'])])} & "
            f"{100 * float(row['success_rate_difference']):+.1f} & "
            f"{_format_p(float(row['mcnemar_exact_p_holm']))} & "
            f"{float(row['peak_force_difference_n']):+.1f} & "
            f"{_format_p(float(row['peak_force_wilcoxon_less_p_holm']))} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path = output_dir / "insert_paired_hard_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_mujoco_table(report: dict, output_dir: Path) -> Path:
    rows = _index(report)
    policies = tuple(report["config"]["policies"])
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & Succ. & Damage & Peak force & Time \\",
        r" & (\%) & (\%) & (N) & (s) \\",
        r"\midrule",
    ]
    for policy in policies:
        row = rows[(0.8, policy)]
        lines.append(
            f"{_latex_escape(POLICY_LABELS[policy])} & "
            f"{100 * float(row['success_rate']):.1f} & "
            f"{100 * float(row['damage_rate']):.1f} & "
            f"{float(row['peak_force_mean_n']):.1f} & "
            f"{float(row['cycle_time_mean_s']):.2f} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path = output_dir / "insert_mujoco_hard_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _paired_row(report: dict, baseline: str, difficulty: float = 0.8) -> dict:
    return next(
        row
        for row in report["paired_comparisons"]
        if row["baseline"] == baseline and float(row["difficulty"]) == difficulty
    )


def _write_numbers(
    report: dict,
    mujoco_report: dict,
    public_report: dict,
    audit: dict,
    output_dir: Path,
) -> Path:
    rows = _index(report)
    mujoco_rows = _index(mujoco_report)
    direct_pair = _paired_row(report, "direct_insertion")
    spiral_pair = _paired_row(report, "spiral_search")
    no_orient_pair = _paired_row(report, "contact_no_orientation")
    values = {
        "PrimaryEpisodes": str(int(report["episode_rows"])),
        "HardOursSuccess": f"{100 * rows[(0.8, 'contact_belief')]['success_rate']:.1f}",
        "HardGuardedSuccess": f"{100 * rows[(0.8, 'guarded_admittance')]['success_rate']:.1f}",
        "HardDirectSuccess": f"{100 * rows[(0.8, 'direct_insertion')]['success_rate']:.1f}",
        "HardSpiralSuccess": f"{100 * rows[(0.8, 'spiral_search')]['success_rate']:.1f}",
        "HardNoOrientSuccess": f"{100 * rows[(0.8, 'contact_no_orientation')]['success_rate']:.1f}",
        "HardNoRetractSuccess": f"{100 * rows[(0.8, 'contact_no_retract')]['success_rate']:.1f}",
        "HardOursDamage": f"{100 * rows[(0.8, 'contact_belief')]['damage_rate']:.1f}",
        "HardDirectDamage": f"{100 * rows[(0.8, 'direct_insertion')]['damage_rate']:.1f}",
        "HardDeltaDirect": f"{100 * direct_pair['success_rate_difference']:.1f}",
        "HardDeltaSpiral": f"{100 * spiral_pair['success_rate_difference']:.1f}",
        "HardDeltaNoOrient": f"{100 * no_orient_pair['success_rate_difference']:.1f}",
        "HardDirectHolm": _format_p(float(direct_pair["mcnemar_exact_p_holm"])).strip("$"),
        "HardDirectForceDelta": f"{float(direct_pair['peak_force_difference_n']):.1f}",
        "MuJoCoEpisodes": str(int(mujoco_report["episode_rows"])),
        "MuJoCoOursSuccess": f"{100 * mujoco_rows[(0.8, 'contact_belief')]['success_rate']:.1f}",
        "MuJoCoOursDamage": f"{100 * mujoco_rows[(0.8, 'contact_belief')]['damage_rate']:.1f}",
        "MuJoCoDirectSuccess": f"{100 * mujoco_rows[(0.8, 'direct_insertion')]['success_rate']:.1f}",
        "MuJoCoDirectDamage": f"{100 * mujoco_rows[(0.8, 'direct_insertion')]['damage_rate']:.1f}",
        "MuJoCoNoRetractDamage": f"{100 * mujoco_rows[(0.8, 'contact_no_retract')]['damage_rate']:.1f}",
        "MuJoCoGuardedSuccess": f"{100 * mujoco_rows[(0.8, 'guarded_admittance')]['success_rate']:.1f}",
        "PublicEpisodes": str(int(public_report["dataset"]["episodes"])),
        "PublicFrames": str(int(public_report["dataset"]["frames"])),
        "PublicTrainEpisodes": str(len(public_report["split"]["train_episodes"])),
        "PublicTestEpisodes": str(len(public_report["split"]["test_episodes"])),
        "PublicRidgeRMSE": f"{float(public_report['held_out_metrics']['rmse']):.5f}",
        "PublicIdentityRMSE": f"{float(public_report['identity_baseline_metrics']['rmse']):.5f}",
        "PublicRMSEGain": (
            f"{100 * float(public_report['comparison_to_identity']['rmse_reduction_fraction']):.2f}"
        ),
        "AuditChecks": str(len(audit["checks"])),
    }
    path = output_dir / "insert_numbers.tex"
    path.write_text(
        "\n".join(rf"\renewcommand{{\{name}}}{{{value}}}" for name, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return path


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _render_success(report: dict, output_dir: Path) -> tuple[Path, Path]:
    rows = _index(report)
    difficulties = tuple(float(value) for value in report["config"]["difficulties"])
    _style()
    fig, ax = plt.subplots(figsize=(7.05, 3.75), constrained_layout=True)
    for policy in POLICY_ORDER:
        values = [
            100 * float(rows[(difficulty, policy)]["success_rate"]) for difficulty in difficulties
        ]
        lows = [
            100 * float(rows[(difficulty, policy)]["success_ci95"][0])
            for difficulty in difficulties
        ]
        highs = [
            100 * float(rows[(difficulty, policy)]["success_ci95"][1])
            for difficulty in difficulties
        ]
        errors = np.asarray((np.asarray(values) - lows, np.asarray(highs) - values))
        ax.errorbar(
            difficulties,
            values,
            yerr=errors,
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=5.5 if policy != "oracle_teacher" else 7,
            linewidth=2.1 if policy == "contact_belief" else 1.25,
            linestyle="--" if policy in {"contact_no_orientation", "contact_no_retract"} else "-",
            capsize=2.5,
            label=POLICY_LABELS[policy],
            zorder=4 if policy == "contact_belief" else 2,
        )
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(-2, 104)
    ax.set_xticks(difficulties)
    ax.set_xticklabels(("Easy (0.2)", "Medium (0.5)", "Hard (0.8)"))
    ax.set_ylabel("Verified, undamaged success (%)")
    ax.set_xlabel("Scenario difficulty")
    ax.set_title("Paired synthetic contact benchmark (200 physical seeds per difficulty)")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.legend(ncol=2, loc="lower left", frameon=True, framealpha=0.96)
    pdf = output_dir / "insert_success_by_difficulty.pdf"
    png = output_dir / "insert_success_by_difficulty.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _render_pareto(report: dict, output_dir: Path) -> tuple[Path, Path]:
    rows = _index(report)
    _style()
    fig, ax = plt.subplots(figsize=(6.7, 3.45), constrained_layout=True)
    offsets = {
        "contact_belief": (5, 5),
        "guarded_admittance": (5, -13),
        "vision_staged_no_force": (5, -12),
        "spiral_search": (-49, -12),
        "direct_insertion": (5, 6),
        "contact_no_orientation": (5, 5),
        "contact_no_retract": (5, 6),
        "oracle_teacher": (5, -12),
    }
    for policy in POLICY_ORDER:
        row = rows[(0.8, policy)]
        x = float(row["peak_force_mean_n"])
        y = 100 * float(row["success_rate"])
        x_ci = row["peak_force_ci95"]
        y_ci = row["success_ci95"]
        ax.errorbar(
            x,
            y,
            xerr=np.asarray([[x - float(x_ci[0])], [float(x_ci[1]) - x]]),
            yerr=np.asarray([[y - 100 * float(y_ci[0])], [100 * float(y_ci[1]) - y]]),
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=7 if policy != "oracle_teacher" else 9,
            capsize=2.5,
            linestyle="none",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            SHORT_LABELS[policy],
            (x, y),
            xytext=offsets[policy],
            textcoords="offset points",
            fontsize=7.5,
        )
    ax.set_xlim(7.2, 16.3)
    ax.set_ylim(38, 104)
    ax.set_xlabel("Mean peak axial force (N; lower is better)")
    ax.set_ylabel("Verified, undamaged success (%)")
    ax.set_title("Hard condition: success--force operating points")
    ax.grid(color="#D8D8D8", linewidth=0.65)
    ax.annotate(
        "preferred region",
        xy=(7.7, 100),
        xytext=(9.2, 95),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#555555",
        fontsize=8,
    )
    pdf = output_dir / "insert_hard_pareto.pdf"
    png = output_dir / "insert_hard_pareto.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _render_mujoco_safety(report: dict, output_dir: Path) -> tuple[Path, Path]:
    rows = _index(report)
    policies = tuple(report["config"]["policies"])
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.15, 3.45), constrained_layout=True)
    x = np.arange(len(policies))
    width = 0.36
    success = [100 * float(rows[(0.8, policy)]["success_rate"]) for policy in policies]
    damage = [100 * float(rows[(0.8, policy)]["damage_rate"]) for policy in policies]
    left.bar(x - width / 2, success, width, color="#0072B2", label="Success")
    left.bar(
        x + width / 2,
        damage,
        width,
        color="#E69F00",
        hatch="///",
        edgecolor="#7A5200",
        label="Damage proxy",
    )
    left.set_ylim(0, 100)
    left.set_ylabel("Episodes (%)")
    left.set_title("Outcome rates (hard, $n=20$)")
    left.set_xticks(x)
    left.set_xticklabels([SHORT_LABELS[policy] for policy in policies], rotation=27, ha="right")
    left.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    left.legend(frameon=False, loc="upper right")
    forces = [float(rows[(0.8, policy)]["peak_force_mean_n"]) for policy in policies]
    right.bar(
        x,
        forces,
        color=[COLORS[policy] for policy in policies],
        edgecolor="white",
        linewidth=0.7,
    )
    for index, value in enumerate(forces):
        right.text(index, value + 2.5, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    right.set_ylim(0, max(forces) * 1.18)
    right.set_ylabel("Mean native peak contact force (N)")
    right.set_title("Collision-enabled guide contact")
    right.set_xticks(x)
    right.set_xticklabels([SHORT_LABELS[policy] for policy in policies], rotation=27, ha="right")
    right.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    fig.suptitle("Direct closed-loop MuJoCo audit; independent of the 2.5-D rollout", fontsize=10)
    pdf = output_dir / "insert_mujoco_safety.pdf"
    png = output_dir / "insert_mujoco_safety.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _render_trace(trace_report: dict, output_dir: Path) -> tuple[Path, Path]:
    trace = trace_report["trace"]
    steps = np.asarray([row["step"] for row in trace], dtype=np.int64)
    force = np.asarray([row["native_contact_force_n"] for row in trace], dtype=np.float64)
    depth = 1_000.0 * (np.asarray([row["tip_x_m"] for row in trace], dtype=np.float64) - 0.190)
    stages = [row["stage"] for row in trace]
    _style()
    fig, ax = plt.subplots(figsize=(7.05, 3.25), constrained_layout=True)
    ax.plot(steps, force, color="#0072B2", linewidth=1.8, label="Native contact force")
    ax.fill_between(steps, 0.0, force, color="#0072B2", alpha=0.15)
    ax.set_xlabel("Outer-loop step")
    ax.set_ylabel("Guide contact force (N)", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.set_ylim(bottom=0.0)
    secondary = ax.twinx()
    secondary.plot(steps, depth, color="#D55E00", linewidth=1.5, label="Tip advance")
    secondary.axhline(48.0, color="#D55E00", linestyle=":", linewidth=1.0)
    secondary.set_ylabel("Terminal-tip advance (mm)", color="#D55E00")
    secondary.tick_params(axis="y", labelcolor="#D55E00")
    retract = np.asarray([stage == "retract" for stage in stages], dtype=bool)
    start: int | None = None
    for index, active in enumerate(retract):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(retract) - 1):
            end = index if not active else index + 1
            ax.axvspan(steps[start], steps[min(end, len(steps) - 1)], color="#E69F00", alpha=0.14)
            start = None
    ax.text(
        0.99,
        0.96,
        "orange bands: retract",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.set_title(
        f"Representative hard MuJoCo recovery (seed {trace_report['seed']}; "
        f"peak {trace_report['peak_native_contact_force_n']:.1f} N)"
    )
    lines = ax.get_lines()[:1] + secondary.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", frameon=True)
    pdf = output_dir / "insert_mujoco_trace.pdf"
    png = output_dir / "insert_mujoco_trace.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.30) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.105,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + 0.0525, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color="#505050"))


def _render_overview(frame_path: Path, output_dir: Path) -> tuple[Path, Path]:
    _style()
    image = Image.open(frame_path).convert("RGB")
    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.22, 1.0))
    left = fig.add_subplot(grid[0, 0])
    left.imshow(image)
    left.axis("off")
    left.set_title("Collision-enabled terminal and socket guides")
    right = fig.add_subplot(grid[0, 1])
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.axis("off")
    right.set_title("ContactBelief recovery state machine")
    _box(right, (0.05, 0.80), "Vision pre-align", COLORS["contact_belief"])
    _box(right, (0.05, 0.59), "Guarded probe", COLORS["contact_belief"])
    _box(right, (0.58, 0.59), "Seat + verify", COLORS["spiral_search"])
    _box(right, (0.05, 0.32), "Contact belief\nupdate", COLORS["direct_insertion"])
    _box(right, (0.58, 0.32), "Retract", COLORS["guarded_admittance"])
    _box(right, (0.31, 0.09), "Re-align", COLORS["contact_no_orientation"])
    _arrow(right, (0.20, 0.80), (0.20, 0.70))
    _arrow(right, (0.36, 0.64), (0.57, 0.64))
    _arrow(right, (0.20, 0.59), (0.20, 0.44))
    _arrow(right, (0.36, 0.37), (0.57, 0.37))
    _arrow(right, (0.72, 0.32), (0.54, 0.20))
    right.add_patch(
        FancyArrowPatch(
            (0.31, 0.14),
            (0.05, 0.64),
            arrowstyle="-|>",
            mutation_scale=11,
            color="#505050",
            connectionstyle="angle3,angleA=180,angleB=90",
        )
    )
    right.text(0.46, 0.675, "clear", fontsize=7.5, color="#555555")
    right.text(0.24, 0.49, "force / torque\nthreshold", fontsize=7.3, color="#555555")
    right.text(
        0.04,
        0.01,
        "Primary: paired 2.5-D proxy\nAudit: direct MuJoCo observations",
        fontsize=7.5,
        color="#505050",
    )
    pdf = output_dir / "insert_system_overview.pdf"
    png = output_dir / "insert_system_overview.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mujoco-report", type=Path, required=True)
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("artifacts/aloha_ridge_v0/report.json"),
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    mujoco_report = json.loads(args.mujoco_report.read_text(encoding="utf-8"))
    public_report = json.loads(args.public_report.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise RuntimeError("refusing to render from a failed audit")
    expected_inputs = {
        "report_sha256": _sha256(args.report),
        "mujoco_report_sha256": _sha256(args.mujoco_report),
        "public_report_sha256": _sha256(args.public_report),
    }
    if any(audit["inputs"].get(key) != value for key, value in expected_inputs.items()):
        raise RuntimeError("refusing to render from reports that do not match the passing audit")
    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = Path(mujoco_report["representative_trace"])
    frame_path = Path(mujoco_report["representative_frame"])
    trace_report = json.loads(trace_path.read_text(encoding="utf-8"))

    outputs = [
        _write_main_table(report, args.output),
        _write_paired_table(report, args.output),
        _write_mujoco_table(mujoco_report, args.output),
        _write_numbers(report, mujoco_report, public_report, audit, args.output),
        *_render_success(report, args.output),
        *_render_pareto(report, args.output),
        *_render_mujoco_safety(mujoco_report, args.output),
        *_render_trace(trace_report, args.output),
        *_render_overview(frame_path, args.output),
    ]
    manifest = {
        "name": "InsertBot frozen paper assets",
        "source_hashes": {
            "primary_report": _sha256(args.report),
            "mujoco_report": _sha256(args.mujoco_report),
            "public_report": _sha256(args.public_report),
            "audit": _sha256(args.audit),
            "representative_trace": _sha256(trace_path),
            "representative_frame": _sha256(frame_path),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
        "failure_counts": dict(Counter(report["failure_counts"])),
    }
    manifest_path = args.output / "insert_paper_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputs": len(outputs),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
