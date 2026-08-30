from __future__ import annotations

import math

from harnessbench.learning.insert_mujoco_direct import (
    DirectInsertConfig,
    direct_insert_scenario,
    run_direct_insert_episode,
)


def test_direct_scenario_is_deterministic() -> None:
    config = DirectInsertConfig(physical_seeds_per_difficulty=1)
    first = direct_insert_scenario(91_000_000, 0.5, config)
    second = direct_insert_scenario(91_000_000, 0.5, config)
    assert first == second


def test_easy_contact_controller_runs_native_mujoco() -> None:
    config = DirectInsertConfig(physical_seeds_per_difficulty=1)
    scenario = direct_insert_scenario(91_000_000, 0.2, config)
    result = run_direct_insert_episode(scenario, "contact_belief", config)
    assert result["finite_state"] is True
    assert result["success"] is True
    assert result["damage"] is False
    assert math.isfinite(result["peak_native_contact_force_n"])
    assert result["steps"] <= config.max_control_steps


def test_hard_policy_outputs_remain_finite() -> None:
    config = DirectInsertConfig(physical_seeds_per_difficulty=1)
    scenario = direct_insert_scenario(93_000_000, 0.8, config)
    for policy in ("contact_belief", "direct_insertion", "contact_no_retract"):
        result = run_direct_insert_episode(scenario, policy, config)
        assert result["finite_state"] is True
        assert math.isfinite(result["peak_native_contact_force_n"])
        assert math.isfinite(result["final_lateral_error_m"])
