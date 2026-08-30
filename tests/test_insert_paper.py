from __future__ import annotations

import math

from harnessbench.learning.insert_paper import (
    INSERT_POLICY_NAMES,
    InsertPaperConfig,
    insert_scenario,
    simulate_insert_episode,
)


def test_scenario_is_deterministic() -> None:
    config = InsertPaperConfig(max_steps=80)
    left = insert_scenario(1234, 0.5, config)
    right = insert_scenario(1234, 0.5, config)
    assert left == right
    assert 0.0 < left.clearance_m < 0.02
    assert len(left.noise_force_n) == config.max_steps


def test_all_insert_policies_return_finite_metrics() -> None:
    config = InsertPaperConfig(max_steps=100)
    scenario = insert_scenario(4321, 0.8, config)
    for policy in INSERT_POLICY_NAMES:
        row = simulate_insert_episode(scenario, policy, config)
        assert row["policy"] == policy
        assert 0 <= row["steps"] <= config.max_steps
        for key in (
            "peak_force_n",
            "force_impulse_ns",
            "final_depth_m",
            "final_lateral_error_m",
            "final_angle_error_rad",
        ):
            assert math.isfinite(float(row[key]))


def test_oracle_reaches_an_easy_connector() -> None:
    config = InsertPaperConfig(max_steps=320)
    scenario = insert_scenario(9001, 0.2, config)
    row = simulate_insert_episode(scenario, "oracle_teacher", config)
    assert row["success"]
    assert row["lock_verified"]
    assert not row["damage"]
