#!/usr/bin/env python3
"""Render the frozen DLO-Lab matched-path report as a compact LaTeX table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
POLICY_ORDER = (
    "no_action",
    "endpoint_straight",
    "wrong_fixture_order",
    "ordered_fixture_route",
)
POLICY_LABELS = {
    "no_action": "No action",
    "endpoint_straight": "Endpoint straight",
    "wrong_fixture_order": "Wrong fixture order",
    "ordered_fixture_route": "Ordered fixture route",
}


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
            / "dlolab_external"
            / "matched_paths"
            / "matched_path_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            WORKSPACE
            / "artifacts"
            / "papers"
            / "routebot"
            / "final"
            / "dlolab_external"
            / "matched_paths"
            / "matched_path_table.tex"
        ),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = {str(row["policy"]): row for row in report["results"]}
    if set(rows) != set(POLICY_ORDER):
        raise ValueError(f"Unexpected policy set: {sorted(rows)}")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Path & Native $R$ $\uparrow$ & RMSE (cm) $\downarrow$ & Rel. $\uparrow$ & Wind. (rad) $\downarrow$ \\",
        r"\midrule",
    ]
    for policy in POLICY_ORDER:
        row = rows[policy]
        metrics = row["metrics"]
        lines.append(
            f"{POLICY_LABELS[policy]} & "
            f"{float(row['native_final_reward']):.3f} & "
            f"{100.0 * float(metrics['ordered_point_rmse_m']):.2f} & "
            f"{float(metrics['fixture_relation_score']):.3f} & "
            f"{float(metrics['winding_change_mae_rad']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
