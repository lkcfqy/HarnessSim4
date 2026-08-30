from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from harnessbench.learning.inspect_mujoco_direct import _policy_rng
from harnessbench.sim.mujoco_backend import SCENE_SPECS
from harnessbench.sim.realistic_robots import load_realistic_model


def test_inspection_scene_exposes_all_frozen_topology_sites() -> None:
    model, _ = load_realistic_model(mujoco, "inspect", SCENE_SPECS["inspect"].xml_path)
    for index in range(12):
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"inspection_site_{index:02d}",
        )
        assert site_id >= 0


def test_policy_rng_is_deterministic_and_policy_specific() -> None:
    first = _policy_rng(17, "topology_risk").random(4)
    second = _policy_rng(17, "topology_risk").random(4)
    other = _policy_rng(17, "random").random(4)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, other)
