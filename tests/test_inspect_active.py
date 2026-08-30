from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from harnessbench.learning.inspect_active import (
    InspectActiveConfig,
    _episode_metrics,
    _load_score_cache,
    _run_policy,
    _scenario,
)


def _write_score_cache(path: Path) -> None:
    rows = []
    labels = ["good"] * 20 + ["bent_wire", "missing_wire", "combined"] * 5
    for image_index, label in enumerate(labels):
        for view_id in range(4):
            rows.append(
                {
                    "relative_path": f"test/{label}/{image_index:03d}.png",
                    "defect_type": label,
                    "is_anomaly": int(label != "good"),
                    "view_id": view_id,
                    "score": float(image_index),
                    "normalized_score": 1.0 if label != "good" else -1.0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_score_cache_requires_exactly_four_views(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    _write_score_cache(path)
    scores, labels = _load_score_cache(path)
    assert len(scores) == len(labels) == 35
    assert all(set(views) == {0, 1, 2, 3} for views in scores.values())


def test_scenarios_are_deterministic_and_paired(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    _write_score_cache(path)
    scores, labels = _load_score_cache(path)
    config = InspectActiveConfig(episode_count=2, bootstrap_draws=20)
    first = _scenario(17, config, scores, labels)
    second = _scenario(17, config, scores, labels)
    assert [site["relative_path"] for site in first["sites"]] == [
        site["relative_path"] for site in second["sites"]
    ]
    topology = _run_policy(first, "topology_risk", config)
    raster = _run_policy(first, "raster", config)
    assert [row["seed"] for row in topology] == [row["seed"] for row in raster]
    assert [row["budget"] for row in topology] == list(config.budgets)


def test_episode_success_uses_supplied_configuration() -> None:
    scenario = {
        "sites": [
            {
                "is_anomaly": index == 0,
                "risk": 1.0,
                "severity": 1.0 if index == 0 else 0.0,
            }
            for index in range(12)
        ]
    }
    observations = [[1.0], [1.0], *([[-1.0]] * 10)]
    scans = np.ones(12, dtype=np.int64)
    strict = InspectActiveConfig(max_success_fpr=0.05)
    permissive = InspectActiveConfig(max_success_fpr=0.10)
    assert not _episode_metrics(scenario, observations, scans, 1.0, strict)[
        "critical_success"
    ]
    assert _episode_metrics(scenario, observations, scans, 1.0, permissive)[
        "critical_success"
    ]


def test_invalid_view_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "relative_path,defect_type,is_anomaly,view_id,score,normalized_score\n"
        "test/good/a.png,good,0,0,0,-1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly four views"):
        _load_score_cache(path)


def test_offline_oracle_solves_each_budget_as_an_upper_bound() -> None:
    sites = []
    for index in range(12):
        if index == 0:
            scores = {0: -1.0, 1: 3.0, 2: -1.0, 3: -1.0}
            anomaly, risk, severity = True, 2.0, 1.0
        elif index == 1:
            scores = {0: 1.0, 1: -1.0, 2: -1.0, 3: -1.0}
            anomaly, risk, severity = True, 1.0, 1.0
        else:
            scores = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
            anomaly, risk, severity = False, 1.0, 0.0
        sites.append(
            {
                "site_index": index,
                "is_anomaly": anomaly,
                "risk": risk,
                "severity": severity,
                "view_scores": scores,
            }
        )
    config = InspectActiveConfig(budgets=(1, 2), view_order=(0, 1, 2, 3))
    rows = _run_policy({"seed": 9, "sites": sites}, "oracle", config)
    assert rows[0]["critical_weighted_recall"] == pytest.approx(1.0 / 3.0)
    assert rows[1]["critical_weighted_recall"] == pytest.approx(2.0 / 3.0)
    assert all(row["scans_used"] <= row["budget"] for row in rows)
