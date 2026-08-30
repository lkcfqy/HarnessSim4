# InsertBot direct-MuJoCo contact audit protocol

**Frozen:** 2026-08-27  
**Version:** 1.0  
**Role in the paper:** independent cross-physics audit; not the primary efficacy benchmark

## Scope

This audit asks whether the qualitative safety/recovery behavior of InsertBot persists when the controller closes its loop on a collision-enabled MuJoCo terminal and socket rather than the paper's 2.5-D contact proxy. No 2.5-D trajectory, event, force, or success label is replayed. At every control step, each policy reads the current MuJoCo terminal pose and native guide-contact force and issues a new task-space target.

The KUKA/Robotiq geometry is visual-only. The commanded terminal is attached to a mocap target by a compliant weld. Consequently, this is evidence for cross-physics task-space behavior, not robot torque control, actuator limits, connector calibration, safety certification, or hardware performance.

## Geometry and dynamics

- A 50 x 12 x 12 mm collision-enabled terminal enters a four-rail socket mouth along world `+x`.
- The nominal opening is approximately 20 x 18 mm. Each scenario varies the `y` and `z` half-openings, friction, starting lateral/vertical offsets, yaw/pitch, visual bias, force bias, and sensor noise.
- The terminal has 0.25 kg effective mass and is attached to the gripper target by a compliant MuJoCo weld (`solref=0.008 1`).
- Native contact force is obtained from `mj_contactForce` for terminal/guide pairs only.
- Contact-aware controllers use an inner force stop during the 2 ms physics loop: 6.5 N for contact-belief variants and 22 N for guarded admittance. This prevents the outer task-space command from continuing after native contact is detected. Direct insertion has no force stop.
- The simulation uses six physics steps per control update and at most 220 control updates.

## Success and damage

Success requires the terminal-tip site to remain seated for five consecutive control updates, with:

- axial tip position within 2 mm of the seating site;
- lateral/vertical position inside the scenario opening after subtracting the 6 mm terminal half-width and adding 0.6 mm for compliant-contact numerical tolerance;
- absolute yaw and pitch no greater than 0.18 rad;
- no damage event.

Damage occurs when native peak guide-contact force exceeds the scenario limit, sampled uniformly from 90–115 N and reduced by 12 N times difficulty. These are engineering stress proxies, not connector specifications.

## Design

- Difficulties: `0.2`, `0.5`, `0.8`.
- Physical seeds: 20 per difficulty, with deterministic base seed `91,000,000` and one-million offsets between strata.
- Policies: `contact_belief`, `guarded_admittance`, `direct_insertion`, `contact_no_orientation`, and `contact_no_retract`.
- All five policies receive the same physical/sensor scenario for a given seed; 300 episodes total.
- No post-hoc exclusions.
- Primary audit quantities: success, damage, native peak contact force, contact-update count, cycle time, and final pose error.
- Exact paired McNemar tests compare `contact_belief` with each comparator within difficulty; Holm adjustment spans all 12 comparisons. These tests are descriptive because 20 seeds per stratum are intended as a cross-physics audit, not a replacement for the 200-seed primary benchmark.

The expected qualitative checks are: easy scenarios are solvable without contact; direct insertion becomes unsafe or unsuccessful as bias increases; contact-aware policies can arrest and recover from at least some native contacts; and removing retract state increases the hard-condition damage burden. Superiority over guarded admittance is not required.

## Pilot history

All pilots remain under `artifacts/papers/insertbot/pilots/` and are excluded from final inference.

- `mujoco_v1` showed that the original visualization-only socket contained no collision semantics and that the existing permissive InsertEnv result could not validate contact control.
- `mujoco_v2`–`v4` exposed insufficient time for recovery and an overly strict seating tolerance.
- `mujoco_v5` exposed terminal pitch sag from an underconstrained compliant weld and a pre-alignment state that could creep into the mouth before orientation convergence.
- `mujoco_v6` exposed an implementation error that compared visual residual with the commanded robot angle instead of the visual target angle.
- `mujoco_v7` verified direct native contacts but showed impulsive force growth between outer-loop updates.
- `mujoco_v8` added a physics-rate force stop and froze the present compliant fixture. In its 10-seed calibration run, hard-condition success was 50% for contact belief, 60% for guarded admittance, 50% without orientation update, 20% without retract, and 10% for direct insertion. Hard-condition damage was 0%, 0%, 10%, 80%, and 90%, respectively. These are calibration diagnostics, not final results.

No controller, fixture, threshold, tolerance, endpoint, or scenario distribution may change after this freeze without a new protocol version and a new formal result directory.

## Reproduction

```powershell
wsl -d Ubuntu-24.04 -- env PYTHONPATH=src .venv-dlolab/bin/python scripts/run_insertbot_mujoco_direct.py --output artifacts/papers/insertbot/formal/mujoco_direct_v1 --physical-seeds 20
```

The report hashes the scene, direct-controller source, runner, tests, episode CSV, aggregate CSV, representative trace, and rendered contact frame.
