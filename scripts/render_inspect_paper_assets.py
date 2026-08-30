#!/usr/bin/env python3
"""Render the frozen InspectBot manuscript assets.

Chart contract
--------------
Decision: determine whether the offline detector is credible enough to drive a
resource-limited inspection policy, where that policy helps, and where it
fails.  Data: untouched public test splits, frozen real-image score caches,
paired scenario rows, and direct MuJoCo execution logs.  Encoding: AUROC curves
use the full score ranking; budget curves use scenario means and paired
bootstrap intervals on a shared 0--75% axis; qualitative panels use a
deterministic median-score image per selected defect class and per-image robust
heat-map scaling.  Comparability: panels never use hidden smoothing or
different y-scales, and the adverse-view negative control is shown beside the
clean-first deployment protocol.  Uncertainty covers frozen held-out images or
scenario resampling only, never hardware or unseen-domain uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve

COLORS = {
    "rgb_statistics": "#8C8C8C",
    "resnet_global": "#4C78A8",
    "spatial_gaussian": "#E45756",
    "spatial_nn": "#54A24B",
    "topology_risk": "#D1495B",
    "geometry_coverage": "#4C78A8",
    "uncertainty": "#59A14F",
    "random": "#9C755F",
    "oracle": "#7A5195",
}
MARKERS = {
    "topology_risk": "o",
    "geometry_coverage": "s",
    "uncertainty": "^",
    "random": "D",
    "oracle": "P",
}
METHOD_LABELS = {
    "rgb_statistics": "RGB statistics",
    "resnet_global": "Global ResNet-18",
    "spatial_gaussian": "Spatial Gaussian",
    "spatial_nn": "Spatial nearest neighbor",
}
POLICY_LABELS = {
    "topology_risk": "Topology-risk",
    "geometry_coverage": "Geometry coverage",
    "uncertainty": "Uncertainty revisit",
    "random": "Random",
    "oracle": "Offline oracle",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10**exponent
    return rf"${coefficient:.2f}\!\times\!10^{{{exponent}}}$"


def _save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _write_perception_table(report: dict[str, Any], output: Path, methods: tuple[str, ...]) -> Path:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & AUROC & AUPR & Recall & F1 \\",
        r"\midrule",
    ]
    for method in methods:
        row = report["methods"][method]
        lines.append(
            f"{_escape(METHOD_LABELS[method])} & "
            f"{100 * row['auroc']:.1f} & {100 * row['aupr']:.1f} & "
            f"{100 * row['recall']:.1f} & {100 * row['f1']:.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _deployment_pairs(report: dict[str, Any], baseline: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["budget"]): row
        for row in report["paired_comparisons"]
        if row["baseline"] == baseline
    }


def _write_active_table(report: dict[str, Any], output: Path) -> Path:
    aggregate = {(int(row["budget"]), row["policy"]): row for row in report["aggregate"]}
    pairs = _deployment_pairs(report, "geometry_coverage")
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Budget & Topology & Geometry & $\Delta$ [95\% CI] & Holm $p$ \\",
        r"\midrule",
    ]
    for budget in report["config"]["budgets"]:
        topology = 100 * float(aggregate[(budget, "topology_risk")]["critical_weighted_recall"])
        geometry = 100 * float(aggregate[(budget, "geometry_coverage")]["critical_weighted_recall"])
        pair = pairs[budget]
        low, high = [100 * float(value) for value in pair["difference_ci95"]]
        lines.append(
            f"{budget} & {topology:.1f} & {geometry:.1f} & "
            f"{100 * float(pair['mean_weighted_recall_difference']):+.1f} "
            f"[{low:+.1f},{high:+.1f}] & {_scientific(float(pair['wilcoxon_p_holm']))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _view_metrics(
    score_rows: list[dict[str, str]], view_report: dict[str, Any]
) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        grouped[int(row["view_id"])].append(row)
    output = []
    for view_id in sorted(grouped):
        rows = grouped[view_id]
        labels = np.asarray([int(row["is_anomaly"]) for row in rows])
        scores = np.asarray([float(row["score"]) for row in rows])
        threshold = float(view_report["calibration"][str(view_id)]["threshold"])
        predictions = scores >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        tn = int(np.sum(~predictions & (labels == 0)))
        output.append(
            {
                "view_id": view_id,
                "auroc": float(roc_auc_score(labels, scores)),
                "recall": tp / max(1, tp + fn),
                "fpr": fp / max(1, fp + tn),
            }
        )
    return output


def _write_view_table(
    metrics: list[dict[str, float]], view_report: dict[str, Any], output: Path
) -> Path:
    lines = [
        r"\begin{tabular}{clrrr}",
        r"\toprule",
        r"View & Transform & AUROC & Recall & FPR \\",
        r"\midrule",
    ]
    for row in metrics:
        view_id = int(row["view_id"])
        label = str(view_report["views"][str(view_id)]).replace("deterministically ", "")
        lines.append(
            f"{view_id} & {_escape(label)} & {100 * row['auroc']:.1f} & "
            f"{100 * row['recall']:.1f} & {100 * row['fpr']:.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _write_mujoco_table(rows: list[dict[str, str]], output: Path) -> Path:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & Runs & Reach & Max IK [mm] & Travel [m] \\",
        r"\midrule",
    ]
    for policy in ("topology_risk", "geometry_coverage", "uncertainty", "random"):
        group = grouped[policy]
        reach = np.mean([row["native_reach_success"].lower() == "true" for row in group])
        max_ik = 1000 * max(float(row["peak_visual_ik_error_m"]) for row in group)
        mean_travel = np.mean([float(row["world_travel_distance_m"]) for row in group])
        lines.append(
            rf"{_escape(POLICY_LABELS[policy])} & {len(group)} & {100 * reach:.0f}\% & "
            f"{max_ik:.2f} & {mean_travel:.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _plot_rocs(
    mvtec_rows: list[dict[str, str]],
    wire_rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    datasets = (
        ("MVTec AD / cable", mvtec_rows, ("rgb_statistics", "resnet_global", "spatial_gaussian")),
        (
            "Stripped Wire / released test",
            wire_rows,
            ("rgb_statistics", "resnet_global", "spatial_nn", "spatial_gaussian"),
        ),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.05), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (title, rows, methods) in zip(axes, datasets):
        labels = np.asarray([int(row["is_anomaly"]) for row in rows])
        for method in methods:
            scores = np.asarray([float(row[f"{method}_score"]) for row in rows])
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = roc_auc_score(labels, scores)
            ax.plot(
                fpr,
                tpr,
                color=COLORS[method],
                linewidth=1.8,
                label=f"{METHOD_LABELS[method]} ({auroc:.3f})",
            )
        ax.plot((0, 1), (0, 1), linestyle="--", linewidth=1, color="#9E9E9E")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False-positive rate")
        ax.grid(True, color="#E5E7EB", linewidth=0.6)
        ax.legend(frameon=False, fontsize=7.1, loc="lower right")
    axes[0].set_ylabel("True-positive rate")
    fig.suptitle(
        "Frozen public-data image-level anomaly ranking", fontsize=10.5, fontweight="bold"
    )
    return _save_figure(fig, output_dir, "perception_roc")


def _plot_active_budget(
    stress: dict[str, Any],
    deployment: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.15), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, report, title in (
        (axes[0], stress, "Stress v1: adverse view first"),
        (axes[1], deployment, "Deployment v2: clean view first"),
    ):
        rows = {(int(row["budget"]), row["policy"]): row for row in report["aggregate"]}
        budgets = [int(value) for value in report["config"]["budgets"]]
        for policy in ("topology_risk", "geometry_coverage", "uncertainty", "random", "oracle"):
            means = np.asarray(
                [rows[(budget, policy)]["critical_weighted_recall"] for budget in budgets]
            )
            intervals = np.asarray(
                [rows[(budget, policy)]["critical_weighted_recall_ci95"] for budget in budgets]
            )
            ax.plot(
                budgets,
                100 * means,
                label=POLICY_LABELS[policy],
                color=COLORS[policy],
                marker=MARKERS[policy],
                markersize=4.2,
                linewidth=1.55,
            )
            ax.fill_between(
                budgets,
                100 * intervals[:, 0],
                100 * intervals[:, 1],
                color=COLORS[policy],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Scan budget")
        ax.set_xticks(budgets)
        ax.set_ylim(0, 75)
        ax.grid(True, color="#E5E7EB", linewidth=0.6)
    axes[0].set_ylabel("Critical-weighted recall (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="outside lower center", fontsize=7.2)
    fig.suptitle(
        "Perception quality bounds the value of active view planning",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.set_constrained_layout_pads(hspace=0.05, wspace=0.04, h_pad=0.12)
    return _save_figure(fig, output_dir, "active_budget")


def _preprocess_for_display(path: Path, *, preserve_aspect: bool) -> np.ndarray:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    if preserve_aspect:
        thumbnail = ImageOps.contain(image, (224, 224), method=Image.Resampling.BILINEAR)
        sample = np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR))
        fill = tuple(int(value) for value in np.median(sample, axis=(0, 1)))
        canvas = Image.new("RGB", (224, 224), fill)
        canvas.paste(thumbnail, ((224 - thumbnail.width) // 2, (224 - thumbnail.height) // 2))
        image = canvas
    else:
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        image = image.crop((16, 16, 240, 240))
    return np.asarray(image)


def _resize_heat(heat: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    low, high = np.quantile(heat, (0.05, 0.995))
    normalized = np.clip((heat - low) / max(1e-9, high - low), 0, 1)
    pil = Image.fromarray(np.uint8(255 * normalized))
    return np.asarray(pil.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _median_examples(
    rows: list[dict[str, str]],
    score_key: str,
    maximum_classes: int | None = None,
) -> list[dict[str, str]]:
    anomaly_rows = [row for row in rows if int(row["is_anomaly"]) == 1]
    counts = Counter(row["defect_type"] for row in anomaly_rows)
    classes = [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    if maximum_classes is not None:
        classes = classes[:maximum_classes]
    selected = []
    for defect in classes:
        candidates = [row for row in anomaly_rows if row["defect_type"] == defect]
        median = float(np.median([float(row[score_key]) for row in candidates]))
        selected.append(
            min(
                candidates,
                key=lambda row: (abs(float(row[score_key]) - median), row["relative_path"]),
            )
        )
    return selected


def _plot_qualitative(
    mvtec_rows: list[dict[str, str]],
    wire_rows: list[dict[str, str]],
    mvtec_maps_path: Path,
    wire_maps_path: Path,
    mvtec_root: Path,
    wire_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    mvtec_examples = _median_examples(mvtec_rows, "spatial_gaussian_score", maximum_classes=3)
    wire_examples = _median_examples(wire_rows, "spatial_nn_score")
    examples = [("MVTec", row, mvtec_root, False, "gaussian") for row in mvtec_examples]
    examples += [("Wire", row, wire_root, True, "nn") for row in wire_examples]

    mvtec_npz = np.load(mvtec_maps_path, allow_pickle=False)
    wire_npz = np.load(wire_maps_path, allow_pickle=False)
    mvtec_index = {str(path): index for index, path in enumerate(mvtec_npz["relative_paths"])}
    wire_index = {str(path): index for index, path in enumerate(wire_npz["relative_paths"])}

    fig, axes = plt.subplots(2, len(examples), figsize=(7.15, 3.15), constrained_layout=True)
    for column, (dataset, row, root, preserve_aspect, map_kind) in enumerate(examples):
        relative = row["relative_path"]
        image = _preprocess_for_display(root / relative, preserve_aspect=preserve_aspect)
        if map_kind == "gaussian":
            heat = mvtec_npz["maps"][mvtec_index[relative]]
        else:
            heat = wire_npz["spatial_nn_maps"][wire_index[relative]]
        resized = _resize_heat(heat, (image.shape[1], image.shape[0]))
        rgba = mpl.colormaps["magma"](resized)
        alpha = (0.15 + 0.65 * resized)[..., None]
        overlay = np.clip((1 - alpha) * (image / 255.0) + alpha * rgba[..., :3], 0, 1)
        axes[0, column].imshow(image)
        axes[1, column].imshow(overlay)
        title = row["defect_type"].replace("_", " ")
        axes[0, column].set_title(f"{dataset}: {title}\nmedian-score member", fontsize=7.4)
        for row_index in (0, 1):
            axes[row_index, column].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=8.5)
    axes[1, 0].set_ylabel("Anomaly map", fontsize=8.5)
    fig.suptitle(
        "Deterministic class-level qualitative audit (no best-case selection)",
        fontsize=10.2,
        fontweight="bold",
    )
    return _save_figure(fig, output_dir, "qualitative_anomalies")


def _plot_system_overview(
    active_report: dict[str, Any],
    render_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    positions = np.asarray(active_report["site_positions"], dtype=np.float64)
    display_positions = positions.copy()
    # Site 11 is a second process checkpoint colocated with the physical junction.
    # Offset only its schematic glyph so both labels and risks remain legible.
    display_positions[11] += np.asarray((0.0, 0.11))
    risks = np.asarray(active_report["site_risk"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95), constrained_layout=True)
    with Image.open(render_path) as raw:
        robot = raw.convert("RGB")
    axes[0].imshow(robot)
    axes[0].axis("off")
    axes[0].set_title("Executable UR5e inspection twin", fontweight="bold")

    ax = axes[1]
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (4, 8),
        (8, 9),
        (9, 10),
        (4, 11),
    )
    for left, right in edges:
        linestyle = ":" if right == 11 else "-"
        ax.plot(
            display_positions[[left, right], 0],
            display_positions[[left, right], 1],
            color="#64748B",
            linewidth=2.0,
            linestyle=linestyle,
            zorder=1,
        )
    scatter = ax.scatter(
        display_positions[:, 0],
        display_positions[:, 1],
        c=risks,
        cmap="plasma",
        vmin=float(risks.min()),
        vmax=float(risks.max()),
        s=90,
        edgecolors="white",
        linewidths=1,
        zorder=2,
    )
    for index, (x, y) in enumerate(display_positions):
        ax.text(
            x,
            y,
            str(index),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Frozen site risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.92)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Topology-risk allocation over 12 sites", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save_figure(fig, output_dir, "system_overview")


def _write_numbers(
    mvtec: dict[str, Any],
    wire: dict[str, Any],
    stress: dict[str, Any],
    deployment: dict[str, Any],
    view_metrics: list[dict[str, float]],
    mujoco: dict[str, Any],
    audit: dict[str, Any],
    output: Path,
) -> Path:
    deploy = {(int(row["budget"]), row["policy"]): row for row in deployment["aggregate"]}
    stress_index = {(int(row["budget"]), row["policy"]): row for row in stress["aggregate"]}
    pair = _deployment_pairs(deployment, "geometry_coverage")
    values = {
        "MVTecSpatialAUROC": f"{100 * mvtec['methods']['spatial_gaussian']['auroc']:.1f}",
        "MVTecPixelAUROC": f"{100 * mvtec['methods']['spatial_gaussian']['pixel']['auroc']:.1f}",
        "WireNNAUROC": f"{100 * wire['methods']['spatial_nn']['auroc']:.1f}",
        "WireNNRecall": f"{100 * wire['methods']['spatial_nn']['recall']:.1f}",
        "WorstViewRecall": f"{100 * min(row['recall'] for row in view_metrics):.1f}",
        "StressTopoBFour": f"{100 * stress_index[(4, 'topology_risk')]['critical_weighted_recall']:.1f}",
        "DeployTopoBFour": f"{100 * deploy[(4, 'topology_risk')]['critical_weighted_recall']:.1f}",
        "DeployGeomBFour": f"{100 * deploy[(4, 'geometry_coverage')]['critical_weighted_recall']:.1f}",
        "DeployDeltaBFour": f"{100 * pair[4]['mean_weighted_recall_difference']:.1f}",
        "DeployDeltaBSix": f"{100 * pair[6]['mean_weighted_recall_difference']:.1f}",
        "DeployDeltaBEight": f"{100 * pair[8]['mean_weighted_recall_difference']:.1f}",
        "DeployDeltaBTen": f"{100 * pair[10]['mean_weighted_recall_difference']:.1f}",
        "DeployBFourHolm": _scientific(float(pair[4]["wilcoxon_p_holm"])),
        "MuJoCoRuns": str(mujoco["episode_rows"]),
        "MuJoCoFOVmm": f"{1000 * mujoco['max_task_space_fov_error_m']:.2f}",
        "MuJoCoIKmm": f"{1000 * mujoco['max_visual_ik_error_m']:.2f}",
        "AuditChecks": str(len(audit["checks"])),
    }
    output.write_text(
        "\n".join(rf"\renewcommand{{\{name}}}{{{value}}}" for name, value in values.items()) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/papers/inspectbot/final"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/papers/inspectbot/final/paper_assets")
    )
    args = parser.parse_args()
    root = args.root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
        }
    )

    mvtec_report_path = root / "perception/inspect_perception_report.json"
    wire_report_path = root / "stripped_wire/stripped_wire_report.json"
    stress_report_path = root / "active/stress_v1/inspect_active_report.json"
    deployment_report_path = root / "active/deployment_v2_corrected/inspect_active_report.json"
    view_report_path = root / "active/stress_v1/inspect_view_score_report.json"
    mujoco_report_path = root / "mujoco_direct/inspect_mujoco_direct_report.json"
    audit_path = root / "audit/inspectbot_result_audit.json"
    mvtec = _read_json(mvtec_report_path)
    wire = _read_json(wire_report_path)
    stress = _read_json(stress_report_path)
    deployment = _read_json(deployment_report_path)
    view_report = _read_json(view_report_path)
    mujoco = _read_json(mujoco_report_path)
    audit = _read_json(audit_path)
    mvtec_rows = _read_csv(root / "perception/inspect_perception_predictions.csv")
    wire_rows = _read_csv(root / "stripped_wire/stripped_wire_predictions.csv")
    score_rows = _read_csv(root / "active/stress_v1/inspect_view_scores.csv")
    mujoco_rows = _read_csv(root / "mujoco_direct/inspect_mujoco_direct_episodes.csv")
    view_metrics = _view_metrics(score_rows, view_report)

    outputs: list[Path] = []
    outputs.append(
        _write_perception_table(
            mvtec,
            output / "mvtec_perception_table.tex",
            ("rgb_statistics", "resnet_global", "spatial_gaussian"),
        )
    )
    outputs.append(
        _write_perception_table(
            wire,
            output / "wire_perception_table.tex",
            ("rgb_statistics", "resnet_global", "spatial_nn", "spatial_gaussian"),
        )
    )
    outputs.append(_write_active_table(deployment, output / "active_budget_table.tex"))
    outputs.append(
        _write_view_table(view_metrics, view_report, output / "view_robustness_table.tex")
    )
    outputs.append(_write_mujoco_table(mujoco_rows, output / "mujoco_execution_table.tex"))
    outputs.append(
        _write_numbers(
            mvtec,
            wire,
            stress,
            deployment,
            view_metrics,
            mujoco,
            audit,
            output / "inspect_numbers.tex",
        )
    )
    outputs.extend(_plot_rocs(mvtec_rows, wire_rows, output))
    outputs.extend(_plot_active_budget(stress, deployment, output))
    outputs.extend(
        _plot_qualitative(
            mvtec_rows,
            wire_rows,
            root / "perception/inspect_spatial_maps.npz",
            root / "stripped_wire/stripped_wire_spatial_maps.npz",
            Path("data/public/mvtec_ad/cable"),
            Path("data/public/stripped_wire_zenodo_16686806/Insulated_wire_dataset"),
            output,
        )
    )
    outputs.extend(
        _plot_system_overview(
            deployment,
            root / "mujoco_direct/inspect_mujoco_direct_topology_final.png",
            output,
        )
    )

    manifest = {
        "name": "InspectBot paper asset manifest",
        "inputs": {
            str(path): _sha256(path)
            for path in (
                mvtec_report_path,
                wire_report_path,
                stress_report_path,
                deployment_report_path,
                view_report_path,
                mujoco_report_path,
                audit_path,
            )
        },
        "outputs": {str(path): _sha256(path) for path in outputs},
        "selection_rule": (
            "Top three MVTec anomaly classes by released test count and both Stripped Wire defect "
            "classes; within each class, the image nearest the class median anomaly score."
        ),
    }
    manifest_path = output / "paper_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": len(outputs), "manifest": str(manifest_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
