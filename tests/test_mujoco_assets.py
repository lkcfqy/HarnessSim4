from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from harnessbench.config import PROJECT_ROOT
from harnessbench.sim.mujoco_backend import SCENE_SPECS
from harnessbench.sim.realistic_robots import MENAGERIE_ROOT, ROBOT_VISUAL_SPECS


def test_all_four_mujoco_assets_exist_and_declare_cable_plugin() -> None:
    assert set(SCENE_SPECS) == {"inspect", "insert", "route", "branch"}
    for spec in SCENE_SPECS.values():
        assert spec.xml_path.is_file()
        root = ET.parse(spec.xml_path).getroot()
        plugins = root.findall("./extension/plugin")
        assert any(item.attrib.get("plugin") == "mujoco.elasticity.cable" for item in plugins)
        assert root.find("./worldbody/camera[@name='overview']") is not None
        composites = root.findall("./worldbody/composite")
        assert composites
        assert all(item.attrib.get("prefix") for item in composites)


def test_mujoco_tracks_are_bounded_and_monotonic_in_time() -> None:
    for spec in SCENE_SPECS.values():
        assert spec.tracks
        for track in spec.tracks.values():
            times = [item[0] for item in track]
            assert times[0] == 0.0
            assert times[-1] == 1.0
            assert times == sorted(times)
            assert all(len(point) == 3 for _, point in track)


def test_route_scene_declares_switchable_policy_constraints() -> None:
    root = ET.parse(SCENE_SPECS["route"].xml_path).getroot()
    names = {
        item.attrib["name"] for item in root.findall("./equality/connect") if "name" in item.attrib
    }
    assert {
        "route_grasp_p7",
        "route_grasp_p15",
        "route_grasp_p23",
        "route_latch_1",
        "route_latch_2",
        "route_latch_3",
        "route_finish_latch",
    } <= names


def test_realistic_visual_assets_retain_licenses_and_match_config() -> None:
    expected_directories = {
        "universal_robots_ur5e",
        "universal_robots_ur10e",
        "kuka_iiwa_14",
        "robotiq_2f85",
        "aloha",
    }
    for directory in expected_directories:
        assert (MENAGERIE_ROOT / directory / "LICENSE").is_file()

    assert set(ROBOT_VISUAL_SPECS) == set(SCENE_SPECS)
    for spec in ROBOT_VISUAL_SPECS.values():
        assert all(item.model_path.is_file() for item in spec.models)
        assert all(item.model_path is None or item.model_path.is_file() for item in spec.tools)

    config_path = PROJECT_ROOT / "configs" / "sim" / "realistic_visuals.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert set(config["tasks"]) == set(SCENE_SPECS)
    assert config["visual_twin_scope"]["robot_collisions"] is False
    assert config["visual_twin_scope"]["robot_dynamics"] is False
