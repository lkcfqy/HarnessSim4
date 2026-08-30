# Vendored MuJoCo Menagerie models

Source: `google-deepmind/mujoco_menagerie` commit
`da76818e269b82289eba39808e2fb91d679d6994` (2026-08-09).

Upstream: <https://github.com/google-deepmind/mujoco_menagerie>

Vendored model directories:

- `universal_robots_ur5e` — InspectBot visual robot;
- `kuka_iiwa_14` — InsertBot visual robot;
- `universal_robots_ur10e` — RouteBot visual robot and both BranchBot arms;
- `robotiq_2f85` — gripper attached to InsertBot, RouteBot and both BranchBot arms;
- `aloha` — retained, currently unused reference model from an earlier BranchBot study.

Each directory retains its upstream `LICENSE`, `README.md`, `CHANGELOG.md`,
MJCF files, meshes and textures.  License terms differ by model; consult the
retained per-directory `LICENSE` before redistribution.  HarnessSim4 does not
claim ownership of these robot meshes or endorsement by their manufacturers.

When publishing results that use these assets, cite MuJoCo Menagerie as
requested by the upstream project and preserve the per-model attribution.

The current benchmark disables collision participation for all attached robot
and gripper geometries. They are kinematic visual twins driven by positional
inverse kinematics; cable and fixture dynamics remain those of HarnessSim4.
This boundary must be stated in papers and demonstrations that use the renders.
