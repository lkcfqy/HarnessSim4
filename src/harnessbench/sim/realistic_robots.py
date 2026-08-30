"""Attach licensed high-detail robot meshes and prescribe visual IK motion.

The task physics remains in the HarnessSim4 MJCF files.  This module composes
those models with curated MuJoCo Menagerie robot descriptions at load time and
drives the visible arm joints to the existing task-space mocap targets.  The
robot collision meshes are disabled deliberately: until full torque control is
implemented, the detailed arm is a kinematic visual twin and must not silently
change the cable benchmark physics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from harnessbench.config import PROJECT_ROOT


@dataclass(frozen=True)
class ArmVisualSpec:
    joint_names: tuple[str, ...]
    end_effector_site: str
    mocap_body: str
    home_qpos: tuple[float, ...]


@dataclass(frozen=True)
class ToolVisualSpec:
    model_path: Path | None
    attach_site: str
    prefix: str
    inline_xml: str | None = None


@dataclass(frozen=True)
class RobotModelVisualSpec:
    model_path: Path
    prefix: str
    mount_pos: tuple[float, float, float]
    mount_quat: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotVisualSpec:
    task_key: str
    display_name: str
    models: tuple[RobotModelVisualSpec, ...]
    arms: tuple[ArmVisualSpec, ...]
    tools: tuple[ToolVisualSpec, ...] = ()


MENAGERIE_ROOT = PROJECT_ROOT / "third_party" / "mujoco_menagerie"

_SCANNER_TOOL_XML = """
<mujoco model="harness_scanner_tool">
  <asset>
    <material name="scanner_shell" rgba="0.055 0.065 0.075 1" specular="0.65" shininess="0.5"/>
    <material name="scanner_glass" rgba="0.03 0.16 0.28 1" specular="0.95" shininess="0.9" reflectance="0.2"/>
    <material name="scanner_trim" rgba="0.12 0.15 0.18 1" metallic="0.4" roughness="0.28"/>
  </asset>
  <worldbody>
    <body name="scanner_head">
      <geom type="box" size="0.046 0.034 0.024" pos="0 0 0.026" material="scanner_shell" contype="0" conaffinity="0"/>
      <geom type="box" size="0.040 0.028 0.004" pos="0 0 0.053" material="scanner_trim" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.012 0.005" pos="-0.018 0 0.059" material="scanner_glass" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.012 0.005" pos="0.018 0 0.059" material="scanner_glass" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.004" pos="0 -0.023 0.055" rgba="0.1 0.95 0.45 1" contype="0" conaffinity="0"/>
      <site name="sensor_tip" pos="0 0 0.070" size="0.002" rgba="0 0 0 0"/>
      <camera name="inspection_camera" pos="0 0 0.061" quat="0 1 0 0" fovy="52"/>
    </body>
  </worldbody>
</mujoco>
"""


ROBOT_VISUAL_SPECS: dict[str, RobotVisualSpec] = {
    "inspect": RobotVisualSpec(
        task_key="inspect",
        display_name="Universal Robots UR5e + dual-lens inspection head",
        models=(
            RobotModelVisualSpec(
                model_path=MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml",
                prefix="inspect_robot/",
                mount_pos=(0.0, -0.72, 0.02),
                mount_quat=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        arms=(
            ArmVisualSpec(
                joint_names=(
                    "inspect_robot/shoulder_pan_joint",
                    "inspect_robot/shoulder_lift_joint",
                    "inspect_robot/elbow_joint",
                    "inspect_robot/wrist_1_joint",
                    "inspect_robot/wrist_2_joint",
                    "inspect_robot/wrist_3_joint",
                ),
                end_effector_site="inspect_tool/sensor_tip",
                mocap_body="scanner",
                home_qpos=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
            ),
        ),
        tools=(
            ToolVisualSpec(
                model_path=None,
                attach_site="inspect_robot/attachment_site",
                prefix="inspect_tool/",
                inline_xml=_SCANNER_TOOL_XML,
            ),
        ),
    ),
    "insert": RobotVisualSpec(
        task_key="insert",
        display_name="KUKA LBR iiwa 14 + Robotiq 2F-85",
        models=(
            RobotModelVisualSpec(
                model_path=MENAGERIE_ROOT / "kuka_iiwa_14" / "iiwa14.xml",
                prefix="insert_robot/",
                mount_pos=(0.0, -0.73, 0.02),
                mount_quat=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        arms=(
            ArmVisualSpec(
                joint_names=tuple(f"insert_robot/joint{index}" for index in range(1, 8)),
                end_effector_site="insert_tool/pinch",
                mocap_body="gripper",
                home_qpos=(0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0),
            ),
        ),
        tools=(
            ToolVisualSpec(
                model_path=MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml",
                attach_site="insert_robot/attachment_site",
                prefix="insert_tool/",
            ),
        ),
    ),
    "route": RobotVisualSpec(
        task_key="route",
        display_name="Universal Robots UR10e + Robotiq 2F-85",
        models=(
            RobotModelVisualSpec(
                model_path=MENAGERIE_ROOT / "universal_robots_ur10e" / "ur10e.xml",
                prefix="route_robot/",
                mount_pos=(0.0, -0.88, 0.02),
                mount_quat=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
        arms=(
            ArmVisualSpec(
                joint_names=(
                    "route_robot/shoulder_pan_joint",
                    "route_robot/shoulder_lift_joint",
                    "route_robot/elbow_joint",
                    "route_robot/wrist_1_joint",
                    "route_robot/wrist_2_joint",
                    "route_robot/wrist_3_joint",
                ),
                end_effector_site="route_tool/pinch",
                mocap_body="route_gripper",
                home_qpos=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
            ),
        ),
        tools=(
            ToolVisualSpec(
                model_path=MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml",
                attach_site="route_robot/attachment_site",
                prefix="route_tool/",
            ),
        ),
    ),
    "branch": RobotVisualSpec(
        task_key="branch",
        display_name="Dual Universal Robots UR10e + Robotiq 2F-85",
        models=(
            RobotModelVisualSpec(
                model_path=MENAGERIE_ROOT / "universal_robots_ur10e" / "ur10e.xml",
                prefix="branch_robot_a/",
                mount_pos=(-0.24, -0.78, 0.02),
                mount_quat=(1.0, 0.0, 0.0, 0.0),
            ),
            RobotModelVisualSpec(
                model_path=MENAGERIE_ROOT / "universal_robots_ur10e" / "ur10e.xml",
                prefix="branch_robot_b/",
                mount_pos=(-0.24, 0.78, 0.02),
                mount_quat=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        arms=(
            ArmVisualSpec(
                joint_names=(
                    "branch_robot_a/shoulder_pan_joint",
                    "branch_robot_a/shoulder_lift_joint",
                    "branch_robot_a/elbow_joint",
                    "branch_robot_a/wrist_1_joint",
                    "branch_robot_a/wrist_2_joint",
                    "branch_robot_a/wrist_3_joint",
                ),
                end_effector_site="branch_tool_a/pinch",
                mocap_body="gripper_a",
                home_qpos=(-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
            ),
            ArmVisualSpec(
                joint_names=(
                    "branch_robot_b/shoulder_pan_joint",
                    "branch_robot_b/shoulder_lift_joint",
                    "branch_robot_b/elbow_joint",
                    "branch_robot_b/wrist_1_joint",
                    "branch_robot_b/wrist_2_joint",
                    "branch_robot_b/wrist_3_joint",
                ),
                end_effector_site="branch_tool_b/pinch",
                mocap_body="gripper_b",
                home_qpos=(1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
            ),
        ),
        tools=(
            ToolVisualSpec(
                model_path=MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml",
                attach_site="branch_robot_a/attachment_site",
                prefix="branch_tool_a/",
            ),
            ToolVisualSpec(
                model_path=MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml",
                attach_site="branch_robot_b/attachment_site",
                prefix="branch_tool_b/",
            ),
        ),
    ),
}


@dataclass
class ArmVisualRuntime:
    spec: ArmVisualSpec
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    site_id: int
    mocap_id: int
    desired_qpos: np.ndarray


class RealisticRobotRig:
    """Kinematic visual twin synchronized to task-space mocap targets."""

    def __init__(self, mujoco: Any, model: Any, visual_spec: RobotVisualSpec):
        self.mujoco = mujoco
        self.model = model
        self.visual_spec = visual_spec
        self.arms: list[ArmVisualRuntime] = []
        for arm_spec in visual_spec.arms:
            joint_ids = np.asarray(
                [
                    _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    for name in arm_spec.joint_names
                ],
                dtype=np.int32,
            )
            qpos_ids = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
            dof_ids = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
            site_id = _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                arm_spec.end_effector_site,
            )
            mocap_body_id = _object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                arm_spec.mocap_body,
            )
            mocap_id = int(model.body_mocapid[mocap_body_id])
            if mocap_id < 0:
                raise ValueError(f"{arm_spec.mocap_body!r} is not a mocap body")
            self.arms.append(
                ArmVisualRuntime(
                    spec=arm_spec,
                    joint_ids=joint_ids,
                    qpos_ids=qpos_ids,
                    dof_ids=dof_ids,
                    site_id=site_id,
                    mocap_id=mocap_id,
                    desired_qpos=np.asarray(arm_spec.home_qpos, dtype=float),
                )
            )

    def initialize(self, data: Any) -> None:
        for arm in self.arms:
            data.qpos[arm.qpos_ids] = np.asarray(arm.spec.home_qpos, dtype=float)
            data.qvel[arm.dof_ids] = 0.0
            arm.desired_qpos = data.qpos[arm.qpos_ids].copy()
        self.mujoco.mj_forward(self.model, data)
        self.sync(data, iterations=180)

    def sync(self, data: Any, *, iterations: int = 45) -> None:
        for arm in self.arms:
            target = np.asarray(data.mocap_pos[arm.mocap_id], dtype=float)
            self._solve_arm(data, arm, target, iterations=iterations)
        self.mujoco.mj_forward(self.model, data)

    def hold(self, data: Any) -> None:
        for arm in self.arms:
            data.qpos[arm.qpos_ids] = arm.desired_qpos
            data.qvel[arm.dof_ids] = 0.0

    def errors(self, data: Any) -> list[float]:
        return [
            float(np.linalg.norm(data.site_xpos[arm.site_id] - data.mocap_pos[arm.mocap_id]))
            for arm in self.arms
        ]

    def _solve_arm(
        self,
        data: Any,
        arm: ArmVisualRuntime,
        target: np.ndarray,
        *,
        iterations: int,
    ) -> None:
        jacobian = np.zeros((3, self.model.nv), dtype=float)
        rotational = np.zeros((3, self.model.nv), dtype=float)
        for _ in range(iterations):
            self.mujoco.mj_forward(self.model, data)
            error = target - data.site_xpos[arm.site_id]
            if float(np.linalg.norm(error)) <= 0.0015:
                break
            self.mujoco.mj_jacSite(
                self.model,
                data,
                jacobian,
                rotational,
                arm.site_id,
            )
            reduced = jacobian[:, arm.dof_ids]
            normal = reduced @ reduced.T + np.eye(3) * 0.0025
            delta = reduced.T @ np.linalg.solve(normal, error * 0.72)
            delta = np.clip(delta, -0.10, 0.10)
            data.qpos[arm.qpos_ids] += delta
            for joint_id, qpos_id in zip(arm.joint_ids, arm.qpos_ids):
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], low, high)
        arm.desired_qpos = data.qpos[arm.qpos_ids].copy()
        data.qvel[arm.dof_ids] = 0.0


def _object_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise KeyError(name)
    return object_id


def _harmonize_options(parent: Any, child: Any) -> None:
    child.option.integrator = parent.option.integrator
    child.option.impratio = parent.option.impratio
    child.option.cone = parent.option.cone


def _attach_tool(mujoco: Any, parent: Any, tool: ToolVisualSpec) -> None:
    if tool.inline_xml is not None:
        child = mujoco.MjSpec.from_string(tool.inline_xml)
    elif tool.model_path is not None:
        child = mujoco.MjSpec.from_file(str(tool.model_path))
    else:
        raise ValueError("tool requires model_path or inline_xml")
    _harmonize_options(parent, child)
    parent.attach(child, site=tool.attach_site, prefix=tool.prefix)


def load_realistic_model(
    mujoco: Any,
    task_key: str,
    task_xml_path: Path,
) -> tuple[Any, RealisticRobotRig]:
    """Compose a task MJCF with its high-detail robot and return model + IK rig."""

    visual_spec = ROBOT_VISUAL_SPECS[task_key]
    missing = [
        path
        for path in [
            *(model.model_path for model in visual_spec.models),
            *(tool.model_path for tool in visual_spec.tools),
        ]
        if path is not None and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing realistic robot assets: {missing}")

    parent = mujoco.MjSpec.from_file(str(task_xml_path))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Attach conflict.*")
        for model_index, model_spec in enumerate(visual_spec.models):
            robot = mujoco.MjSpec.from_file(str(model_spec.model_path))
            _harmonize_options(parent, robot)
            mount = parent.worldbody.add_frame(name=f"{task_key}_visual_robot_mount_{model_index}")
            mount.pos = model_spec.mount_pos
            mount.quat = model_spec.mount_quat
            parent.attach(robot, frame=mount, prefix=model_spec.prefix)
        for tool in visual_spec.tools:
            _attach_tool(mujoco, parent, tool)
    model = parent.compile()

    # The arm is currently a visual kinematic twin.  Disable all collisions for
    # prefixed robot/tool bodies so it cannot alter benchmark physics.
    prefixes = (
        *(model.prefix for model in visual_spec.models),
        *(tool.prefix for tool in visual_spec.tools),
    )
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        )
        if body_name and body_name.startswith(prefixes):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    return model, RealisticRobotRig(mujoco, model, visual_spec)
