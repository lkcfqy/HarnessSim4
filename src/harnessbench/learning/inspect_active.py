"""Paired active-inspection benchmark grounded in real MVTec cable images.

Each latent inspection site references one held-out MVTec test image.  Four
deterministic appearance transforms emulate viewpoint/illumination quality;
they are not claimed to be real multi-view captures.  Policies receive only
calibrated anomaly scores, scan history, geometry, and (where applicable) an
engineering criticality map.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from harnessbench.inspect_public_data import file_sha256
from harnessbench.learning.inspect_perception import (
    SpatialGaussianDetector,
    _load_index,
)
from harnessbench.learning.route_paper import _holm_adjust, _mcnemar_exact

INSPECT_POLICY_NAMES = (
    "raster",
    "random",
    "geometry_coverage",
    "uncertainty",
    "topology_risk",
    "shuffled_risk",
    "oracle",
)
INSPECT_POLICY_LABELS = {
    "raster": "Raster",
    "random": "Random",
    "geometry_coverage": "Geometry coverage",
    "uncertainty": "Uncertainty revisit",
    "topology_risk": "Topology-risk + uncertainty",
    "shuffled_risk": "Shuffled-risk control",
    "oracle": "Offline score-aware upper bound",
}

DEFECT_SEVERITY = {
    "good": 0.0,
    "bent_wire": 1.0,
    "cable_swap": 1.5,
    "combined": 1.6,
    "cut_inner_insulation": 1.35,
    "cut_outer_insulation": 1.3,
    "missing_cable": 1.6,
    "missing_wire": 1.5,
    "poke_insulation": 1.15,
}


@dataclass(frozen=True)
class InspectActiveConfig:
    protocol_id: str = "stress_v1"
    site_count: int = 12
    budgets: tuple[int, ...] = (4, 6, 8, 10, 12, 16)
    episode_count: int = 1_000
    base_seed: int = 77_000_000
    anomaly_probability: float = 0.30
    max_views_per_site: int = 4
    view_order: tuple[int, ...] = (3, 1, 2, 0)
    bootstrap_draws: int = 10_000
    critical_success_tolerance: float = 0.999
    max_success_fpr: float = 0.20


def inspect_deployment_v2_config() -> InspectActiveConfig:
    """Frozen recovery protocol after the deliberately harsh stress-v1 result."""

    return InspectActiveConfig(
        protocol_id="deployment_v2",
        episode_count=2_000,
        base_seed=78_000_000,
        view_order=(0, 1, 2, 3),
    )


SITE_POSITIONS = np.asarray(
    [
        [0.05, 0.50],
        [0.18, 0.50],
        [0.31, 0.50],
        [0.44, 0.50],
        [0.56, 0.50],
        [0.68, 0.38],
        [0.81, 0.27],
        [0.94, 0.20],
        [0.68, 0.62],
        [0.81, 0.73],
        [0.94, 0.80],
        [0.56, 0.50],
    ],
    dtype=np.float64,
)
SITE_RISK = np.asarray(
    [1.55, 0.75, 0.90, 1.10, 1.90, 1.05, 1.45, 1.80, 0.95, 1.35, 1.70, 2.00],
    dtype=np.float64,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _calibration_paths(data_root: Path, detector: SpatialGaussianDetector) -> list[Path]:
    train_good, _, _, _ = _load_index(data_root)
    split_rng = np.random.default_rng(detector.config.seed)
    split_indices = split_rng.permutation(len(train_good))
    return [
        train_good[int(index)]
        for index in split_indices[detector.config.fit_good_images :]
    ]


def build_inspect_view_score_cache(
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device_name: str = "cuda",
) -> dict:
    """Score the frozen MVTec test set under four deterministic view transforms."""

    checkpoint = checkpoint.resolve()
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = SpatialGaussianDetector.load(checkpoint)
    calibration_paths = _calibration_paths(data_root, detector)
    _, _, test_paths, defect_labels = _load_index(data_root)
    rows: list[dict] = []
    calibration: dict[str, dict] = {}
    for view_id in range(4):
        calibration_scores, _ = detector.score_paths(
            calibration_paths,
            device_name=device_name,
            view_id=view_id,
        )
        test_scores, _ = detector.score_paths(
            test_paths,
            device_name=device_name,
            view_id=view_id,
        )
        threshold = float(
            np.quantile(calibration_scores, detector.config.calibration_quantile)
        )
        median = float(np.median(calibration_scores))
        mad = float(np.median(np.abs(calibration_scores - median)))
        robust_scale = max(1.4826 * mad, 1e-6)
        calibration[str(view_id)] = {
            "normal_images": len(calibration_paths),
            "median": median,
            "mad": mad,
            "robust_scale": robust_scale,
            "threshold": threshold,
            "calibration_quantile": detector.config.calibration_quantile,
        }
        for path, defect_type, score in zip(test_paths, defect_labels, test_scores):
            rows.append(
                {
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "defect_type": defect_type,
                    "is_anomaly": int(defect_type != "good"),
                    "view_id": view_id,
                    "score": float(score),
                    "normalized_score": float((score - threshold) / robust_scale),
                }
            )

    csv_path = output_dir / "inspect_view_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "name": "InspectBot deterministic synthetic-view score cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_root": str(data_root),
        "rows": len(rows),
        "test_images": len(test_paths),
        "views_per_image": 4,
        "views": {
            "0": "clean center crop",
            "1": "darkened and blurred",
            "2": "rotated and contrast shifted",
            "3": "deterministically occluded and darkened",
        },
        "calibration": calibration,
        "scores_csv": str(csv_path.resolve()),
        "scores_sha256": file_sha256(csv_path),
        "claim_scope": (
            "Scores use real held-out MVTec images with synthetic appearance transforms; "
            "the transforms are not real camera trajectories."
        ),
    }
    report_path = output_dir / "inspect_view_score_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path.resolve())
    return report


def _load_score_cache(path: Path) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    labels: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative = row["relative_path"]
            view_id = int(row["view_id"])
            scores[relative][view_id] = float(row["normalized_score"])
            labels[relative] = row["defect_type"]
    if not scores or any(set(views) != {0, 1, 2, 3} for views in scores.values()):
        raise ValueError("view score cache must contain exactly four views per image")
    return dict(scores), labels


def _scenario(
    seed: int,
    config: InspectActiveConfig,
    scores: dict[str, dict[int, float]],
    labels: dict[str, str],
) -> dict:
    rng = np.random.default_rng(seed)
    good_paths = sorted(path for path, label in labels.items() if label == "good")
    defect_paths: dict[str, list[str]] = defaultdict(list)
    for path, label in labels.items():
        if label != "good":
            defect_paths[label].append(path)
    anomaly_flags = rng.random(config.site_count) < config.anomaly_probability
    if not anomaly_flags.any():
        anomaly_flags[int(rng.integers(0, config.site_count))] = True
    if anomaly_flags.all():
        anomaly_flags[int(rng.integers(0, config.site_count))] = False
    selected_good = rng.choice(good_paths, size=int(np.sum(~anomaly_flags)), replace=False)
    defect_types = sorted(defect_paths)
    rows = []
    good_index = 0
    used_defect_paths: set[str] = set()
    for site_index, is_anomaly in enumerate(anomaly_flags):
        if is_anomaly:
            defect_type = str(rng.choice(defect_types))
            candidates = [path for path in defect_paths[defect_type] if path not in used_defect_paths]
            if not candidates:
                candidates = defect_paths[defect_type]
            relative_path = str(rng.choice(candidates))
            used_defect_paths.add(relative_path)
        else:
            defect_type = "good"
            relative_path = str(selected_good[good_index])
            good_index += 1
        rows.append(
            {
                "site_index": site_index,
                "relative_path": relative_path,
                "defect_type": defect_type,
                "is_anomaly": bool(is_anomaly),
                "severity": DEFECT_SEVERITY[defect_type],
                "risk": float(SITE_RISK[site_index]),
                "position": SITE_POSITIONS[site_index].copy(),
                "view_scores": scores[relative_path],
            }
        )
    return {"seed": seed, "sites": rows}


def _aggregated_score(observations: list[float]) -> float:
    if not observations:
        return float("nan")
    return float(np.mean(observations))


def _uncertainty(observations: list[float]) -> float:
    if not observations:
        return 1.0
    score = _aggregated_score(observations)
    return float(math.exp(-abs(score)) / math.sqrt(len(observations)))


def _travel(current: np.ndarray, site_index: int) -> float:
    return float(np.linalg.norm(SITE_POSITIONS[site_index] - current))


def _choose_site(
    policy: str,
    scenario: dict,
    observations: list[list[float]],
    scans: np.ndarray,
    current: np.ndarray,
    policy_rng: np.random.Generator,
    shuffled_risk: np.ndarray,
    config: InspectActiveConfig,
) -> int:
    candidates = np.flatnonzero(scans < config.max_views_per_site)
    unscanned = candidates[scans[candidates] == 0]
    available = unscanned if len(unscanned) else candidates
    if policy == "raster":
        return int(available[0])
    if policy == "random":
        return int(policy_rng.choice(candidates))
    if policy in {"geometry_coverage", "uncertainty"}:
        if len(unscanned):
            return min((int(index) for index in unscanned), key=lambda index: _travel(current, index))
        if policy == "geometry_coverage":
            return min((int(index) for index in candidates), key=lambda index: _travel(current, index))
        return max(
            (int(index) for index in candidates),
            key=lambda index: _uncertainty(observations[index]) / (1.0 + _travel(current, index)),
        )
    if policy in {"topology_risk", "shuffled_risk"}:
        risk = SITE_RISK if policy == "topology_risk" else shuffled_risk
        if len(unscanned):
            return max(
                (int(index) for index in unscanned),
                key=lambda index: risk[index] / (1.0 + _travel(current, index)),
            )
        return max(
            (int(index) for index in candidates),
            key=lambda index: (
                risk[index]
                * (0.35 + _uncertainty(observations[index]))
                / (1.0 + _travel(current, index))
            ),
        )
    if policy == "oracle":
        sites = scenario["sites"]
        return max(
            (int(index) for index in candidates),
            key=lambda index: (
                float(sites[index]["is_anomaly"]),
                sites[index]["risk"] * sites[index]["severity"],
                max(
                    sites[index]["view_scores"][view]
                    for view in config.view_order[int(scans[index]) :]
                ),
            ),
        )
    raise KeyError(policy)


def _episode_metrics(
    scenario: dict,
    observations: list[list[float]],
    scans: np.ndarray,
    travel_distance: float,
    config: InspectActiveConfig,
) -> dict:
    sites = scenario["sites"]
    truth = np.asarray([site["is_anomaly"] for site in sites], dtype=bool)
    predictions = np.asarray(
        [bool(values) and _aggregated_score(values) >= 0.0 for values in observations],
        dtype=bool,
    )
    true_positive = int(np.sum(truth & predictions))
    false_positive = int(np.sum(~truth & predictions))
    false_negative = int(np.sum(truth & ~predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    weights = np.asarray(
        [site["risk"] * site["severity"] if site["is_anomaly"] else 0.0 for site in sites]
    )
    weighted_recall = float(np.sum(weights[predictions]) / max(np.sum(weights), 1e-12))
    fpr = false_positive / max(int(np.sum(~truth)), 1)
    return {
        "true_anomalies": int(np.sum(truth)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "critical_weighted_recall": weighted_recall,
        "critical_miss_rate": 1.0 - weighted_recall,
        "false_positive_rate": fpr,
        "coverage": float(np.mean(scans > 0)),
        "revisit_fraction": float((np.sum(scans) - np.sum(scans > 0)) / max(np.sum(scans), 1)),
        "travel_distance": travel_distance,
        "scans_used": int(np.sum(scans)),
        "critical_success": bool(
            weighted_recall >= config.critical_success_tolerance
            and fpr <= config.max_success_fpr
        ),
    }


def _oracle_scan_plan(
    scenario: dict,
    budget: int,
    config: InspectActiveConfig,
) -> list[int]:
    """Solve the per-site view-prefix allocation exactly for weighted recall."""

    states: dict[int, tuple[tuple[float, int, int], list[int]]] = {
        0: ((0.0, 0, 0), [])
    }
    for site in scenario["sites"]:
        choices = []
        for scan_count in range(config.max_views_per_site + 1):
            values = [
                float(site["view_scores"][view_id])
                for view_id in config.view_order[:scan_count]
            ]
            predicted = bool(values) and _aggregated_score(values) >= 0.0
            true_positive = int(bool(site["is_anomaly"]) and predicted)
            false_positive = int(not bool(site["is_anomaly"]) and predicted)
            weighted = (
                float(site["risk"] * site["severity"]) if true_positive else 0.0
            )
            choices.append((scan_count, (weighted, true_positive, -false_positive)))
        next_states: dict[int, tuple[tuple[float, int, int], list[int]]] = {}
        for prior_cost, (prior_objective, prior_plan) in states.items():
            for scan_count, gain in choices:
                cost = prior_cost + scan_count
                if cost > budget:
                    continue
                objective = tuple(
                    left + right for left, right in zip(prior_objective, gain, strict=True)
                )
                candidate = (objective, [*prior_plan, scan_count])
                incumbent = next_states.get(cost)
                if incumbent is None or candidate[0] > incumbent[0]:
                    next_states[cost] = candidate
        states = next_states
    _, (_, plan) = max(
        states.items(),
        key=lambda item: (*item[1][0], -item[0]),
    )
    return plan


def _run_oracle(scenario: dict, config: InspectActiveConfig) -> list[dict]:
    output = []
    for budget in config.budgets:
        plan = _oracle_scan_plan(scenario, budget, config)
        observations: list[list[float]] = [[] for _ in range(config.site_count)]
        scans = np.asarray(plan, dtype=np.int64)
        current = np.asarray([0.0, 0.5], dtype=np.float64)
        travel_distance = 0.0
        for site_index, scan_count in enumerate(plan):
            if not scan_count:
                continue
            travel_distance += _travel(current, site_index)
            current = SITE_POSITIONS[site_index].copy()
            site = scenario["sites"][site_index]
            observations[site_index] = [
                float(site["view_scores"][view_id])
                for view_id in config.view_order[:scan_count]
            ]
        output.append(
            {
                "seed": int(scenario["seed"]),
                "policy": "oracle",
                "budget": budget,
                **_episode_metrics(
                    scenario,
                    observations,
                    scans,
                    travel_distance,
                    config,
                ),
            }
        )
    return output


def _run_policy(
    scenario: dict,
    policy: str,
    config: InspectActiveConfig,
) -> list[dict]:
    if policy == "oracle":
        return _run_oracle(scenario, config)
    seed = int(scenario["seed"])
    name_digest = int.from_bytes(hashlib.sha256(policy.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed ^ name_digest)
    shuffled_risk = rng.permutation(SITE_RISK)
    observations: list[list[float]] = [[] for _ in range(config.site_count)]
    scans = np.zeros(config.site_count, dtype=np.int64)
    current = np.asarray([0.0, 0.5], dtype=np.float64)
    travel_distance = 0.0
    output = []
    for scan_number in range(1, max(config.budgets) + 1):
        site_index = _choose_site(
            policy,
            scenario,
            observations,
            scans,
            current,
            rng,
            shuffled_risk,
            config,
        )
        view_id = config.view_order[int(scans[site_index])]
        site = scenario["sites"][site_index]
        travel_distance += _travel(current, site_index)
        current = SITE_POSITIONS[site_index].copy()
        observations[site_index].append(float(site["view_scores"][view_id]))
        scans[site_index] += 1
        if scan_number in config.budgets:
            output.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "budget": scan_number,
                    **_episode_metrics(
                        scenario,
                        observations,
                        scans,
                        travel_distance,
                        config,
                    ),
                }
            )
    return output


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> list[float]:
    sample_indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[sample_indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _aggregate(records: list[dict], config: InspectActiveConfig) -> list[dict]:
    output = []
    rng = np.random.default_rng(config.base_seed + 991)
    for budget in config.budgets:
        for policy in INSPECT_POLICY_NAMES:
            group = [
                row for row in records if row["budget"] == budget and row["policy"] == policy
            ]
            weighted = np.asarray([row["critical_weighted_recall"] for row in group])
            output.append(
                {
                    "budget": budget,
                    "policy": policy,
                    "episodes": len(group),
                    "critical_weighted_recall": float(np.mean(weighted)),
                    "critical_weighted_recall_ci95": _bootstrap_mean_ci(
                        weighted, rng, config.bootstrap_draws
                    ),
                    "recall": float(np.mean([row["recall"] for row in group])),
                    "precision": float(np.mean([row["precision"] for row in group])),
                    "f1": float(np.mean([row["f1"] for row in group])),
                    "false_positive_rate": float(
                        np.mean([row["false_positive_rate"] for row in group])
                    ),
                    "coverage": float(np.mean([row["coverage"] for row in group])),
                    "revisit_fraction": float(
                        np.mean([row["revisit_fraction"] for row in group])
                    ),
                    "scans_used": float(np.mean([row["scans_used"] for row in group])),
                    "travel_distance": float(np.mean([row["travel_distance"] for row in group])),
                    "critical_success_rate": float(
                        np.mean([row["critical_success"] for row in group])
                    ),
                }
            )
    return output


def _paired(records: list[dict], config: InspectActiveConfig) -> list[dict]:
    indexed = {
        (int(row["budget"]), int(row["seed"]), str(row["policy"])): row for row in records
    }
    output: list[dict] = []
    rng = np.random.default_rng(config.base_seed + 1_337)
    baselines = [name for name in INSPECT_POLICY_NAMES if name not in {"topology_risk", "oracle"}]
    for budget in config.budgets:
        seeds = sorted({int(row["seed"]) for row in records if row["budget"] == budget})
        topology = np.asarray(
            [indexed[(budget, seed, "topology_risk")]["critical_weighted_recall"] for seed in seeds]
        )
        topology_success = np.asarray(
            [indexed[(budget, seed, "topology_risk")]["critical_success"] for seed in seeds],
            dtype=bool,
        )
        for baseline in baselines:
            comparison = np.asarray(
                [indexed[(budget, seed, baseline)]["critical_weighted_recall"] for seed in seeds]
            )
            delta = topology - comparison
            if np.allclose(delta, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(delta, alternative="greater", zero_method="pratt", method="auto").pvalue
                )
            draws = rng.integers(0, len(delta), size=(config.bootstrap_draws, len(delta)))
            bootstrap = delta[draws].mean(axis=1)
            baseline_success = np.asarray(
                [indexed[(budget, seed, baseline)]["critical_success"] for seed in seeds],
                dtype=bool,
            )
            topology_only = int(np.sum(topology_success & ~baseline_success))
            baseline_only = int(np.sum(~topology_success & baseline_success))
            output.append(
                {
                    "budget": budget,
                    "baseline": baseline,
                    "paired_scenarios": len(seeds),
                    "mean_weighted_recall_difference": float(np.mean(delta)),
                    "difference_ci95": [
                        float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
                    ],
                    "wilcoxon_p": p_value,
                    "topology_success_baseline_failure": topology_only,
                    "topology_failure_baseline_success": baseline_only,
                    "mcnemar_exact_p": _mcnemar_exact(topology_only, baseline_only),
                }
            )
    output = _holm_adjust(output, key="wilcoxon_p")
    return _holm_adjust(output, key="mcnemar_exact_p")


def run_inspect_active_benchmark(
    score_cache: Path,
    output_dir: Path,
    *,
    config: InspectActiveConfig | None = None,
) -> dict:
    """Run a paired budget curve over deterministic real-image score tables."""

    config = config or InspectActiveConfig()
    if config.site_count != len(SITE_RISK) or config.site_count != len(SITE_POSITIONS):
        raise ValueError("site_count must match the frozen topology map")
    if max(config.budgets) > config.site_count * config.max_views_per_site:
        raise ValueError("scan budget exceeds available site views")
    score_cache = score_cache.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, labels = _load_score_cache(score_cache)
    records: list[dict] = []
    for episode_index in range(config.episode_count):
        scenario = _scenario(config.base_seed + episode_index, config, scores, labels)
        for policy in INSPECT_POLICY_NAMES:
            records.extend(_run_policy(scenario, policy, config))
    aggregates = _aggregate(records, config)
    paired = _paired(records, config)

    episodes_path = output_dir / "inspect_active_episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    aggregate_path = output_dir / "inspect_active_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    paired_path = output_dir / "inspect_active_paired.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    report = {
        "schema_version": 1,
        "name": "InspectBot paired topology-risk active-inspection benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "score_cache": str(score_cache),
        "score_cache_sha256": file_sha256(score_cache),
        "policy_labels": INSPECT_POLICY_LABELS,
        "site_positions": SITE_POSITIONS.tolist(),
        "site_risk": SITE_RISK.tolist(),
        "defect_severity": DEFECT_SEVERITY,
        "episode_rows": len(records),
        "paired_scenarios_per_budget": config.episode_count,
        "aggregate": aggregates,
        "paired_comparisons": paired,
        "episodes_csv": str(episodes_path.resolve()),
        "episodes_sha256": file_sha256(episodes_path),
        "aggregate_csv": str(aggregate_path.resolve()),
        "aggregate_sha256": file_sha256(aggregate_path),
        "paired_csv": str(paired_path.resolve()),
        "paired_sha256": file_sha256(paired_path),
        "claim_scope": (
            "Paired synthetic active-view benchmark driven by frozen scores from held-out real "
            "MVTec images. Confidence intervals quantify scenario resampling, not new-image or "
            "hardware uncertainty."
        ),
    }
    report_path = output_dir / "inspect_active_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path.resolve())
    return report
