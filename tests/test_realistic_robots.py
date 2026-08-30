from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from harnessbench.sim.mujoco_backend import SCENE_SPECS
from harnessbench.sim.realistic_robots import (
    ROBOT_VISUAL_SPECS,
    load_realistic_model,
)


@pytest.mark.parametrize("task_key", ("inspect", "insert", "route", "branch"))
def test_realistic_robot_composition_and_ik(task_key: str) -> None:
    scene = SCENE_SPECS[task_key]
    model, rig = load_realistic_model(mujoco, task_key, scene.xml_path)
    data = mujoco.MjData(model)

    for arm in rig.arms:
        track = scene.tracks[arm.spec.mocap_body]
        data.mocap_pos[arm.mocap_id] = np.asarray(track[0][1], dtype=float)
    rig.initialize(data)

    peak_error = max(rig.errors(data), default=0.0)
    progress_points = sorted({time for track in scene.tracks.values() for time, _ in track})
    for progress in progress_points:
        for arm in rig.arms:
            track = scene.tracks[arm.spec.mocap_body]
            nearest = min(track, key=lambda item: abs(item[0] - progress))
            data.mocap_pos[arm.mocap_id] = np.asarray(nearest[1], dtype=float)
        rig.sync(data, iterations=240)
        peak_error = max(peak_error, max(rig.errors(data), default=0.0))

    assert model.nmesh > 0
    assert peak_error < 0.005

    prefixes = tuple(
        [item.prefix for item in ROBOT_VISUAL_SPECS[task_key].models]
        + [item.prefix for item in ROBOT_VISUAL_SPECS[task_key].tools]
    )
    robot_geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if body_name and body_name.startswith(prefixes):
            robot_geom_ids.append(geom_id)
    assert robot_geom_ids
    assert np.all(model.geom_contype[robot_geom_ids] == 0)
    assert np.all(model.geom_conaffinity[robot_geom_ids] == 0)
