# InsertBot paired-contact benchmark protocol

**Frozen for the formal synthetic run:** 2026-08-27  
**Protocol version:** 1.0  
**Status:** frozen after pilot calibration, before the 200-seed-per-difficulty run

## Purpose and claim boundary

This protocol evaluates whether a contact-belief insertion controller improves reliable connector insertion under pose bias, cable drag, friction, and contact uncertainty. The benchmark is a reproducible 2.5-D engineering proxy. It is **not** a calibrated connector model, finite-element simulation, camera validation, force/torque-sensor validation, or hardware result. Formal results may support algorithmic and experimental-design claims only.

## Experimental unit and pairing

- One experimental unit is one immutable physical/sensor scenario: clearance, friction, contact stiffness, jam and damage limits, cable drag, initial pose, fixed visual bias, force bias, and a time-indexed sensor-noise sequence.
- Every policy receives the same scenario and the same noise sequence for a given `(difficulty, seed)` pair. Policy comparisons are therefore paired.
- Difficulties are `0.2`, `0.5`, and `0.8`.
- The formal run uses 200 physical seeds per difficulty, beginning at `83,000,000` and offset by `1,000,000` between difficulty strata.
- Eight policies are run on each of 600 scenarios, giving 4,800 episodes. There are no post-hoc exclusions.
- Each episode is limited to 320 control steps at 20 ms per step. Nominal insertion depth is 40 mm.

## Contact proxy

The terminal state comprises axial depth `x`, lateral offset `y`, yaw, and pitch. For a 30 mm terminal, the tip offset and collision envelope are

```text
tip = y + 0.030 yaw + 0.45(0.030) pitch
envelope = |tip| + 0.38(0.030)(|yaw| + |pitch|)
violation = max(0, envelope - clearance)
```

Positive axial motion near the socket produces a friction/drag/speed-dependent seating load plus an edge-contact term proportional to clearance violation and penetration. Excess force can jam the axial motion. Damage is declared when either the scenario-specific peak-force limit or accumulated high-force impulse limit is exceeded. A terminal is successful only after it reaches full depth inside 90% of the clearance envelope and remains locked for six verification steps without damage.

The no-retract ablation performs lateral correction while still loaded against the guide. Its effective contact load is multiplied by 1.45 and impulse accumulation begins above 8 N rather than 10 N, representing added tangential friction and local edge pressure during sliding contact. This mechanism was fixed before the formal run.

## Policies

1. `contact_belief`: visual pre-alignment, force/torque-triggered contact update, orientation update, retract, realign, and verified insertion.
2. `guarded_admittance`: conservative force threshold with lateral-only correction and short retract.
3. `vision_staged_no_force`: identical staged visual approach without force observations.
4. `spiral_search`: deterministic lateral/orientation search during insertion.
5. `direct_insertion`: direct visual servo insertion without force recovery.
6. `contact_no_orientation`: contact-belief ablation with no yaw/pitch update.
7. `contact_no_retract`: contact-belief ablation that corrects while loaded and never enters a retract state.
8. `oracle_teacher`: state-aware upper-reference controller using the latent pose; it is not a deployable baseline.

## Endpoints and hypotheses

The primary endpoint is verified, undamaged insertion success. Secondary endpoints are first-pass success, peak axial force, 95th-percentile peak force, cycle time, damage, jam occurrence, retries, and failure type.

The frozen hypotheses are:

- H1: at difficulty 0.8, `contact_belief` has higher paired success than the three force-blind baselines (`vision_staged_no_force`, `spiral_search`, and `direct_insertion`).
- H2: at difficulty 0.8, `contact_belief` has lower paired peak force than those same force-blind baselines.
- H3: removing orientation update or retract state reduces hard-condition success and/or increases peak force, establishing that the full state machine—not contact detection alone—drives performance.
- `guarded_admittance` is the strongest classical comparator; superiority over it is exploratory rather than required.
- `oracle_teacher` is a ceiling check. The learned/contact controller is not expected to outperform it.

## Statistical analysis

- Report per-policy, per-difficulty success with deterministic 10,000-draw nonparametric bootstrap 95% confidence intervals.
- Report mean peak force with bootstrap 95% confidence intervals, its empirical 95th percentile, and mean cycle time.
- Compare `contact_belief` with each of seven comparators within each difficulty using exact paired McNemar tests for success.
- Compare paired peak force with a one-sided Wilcoxon signed-rank test (`contact_belief < comparator`).
- Apply Holm family-wise correction separately to all 21 McNemar tests and all 21 force tests.
- Report paired effect sizes and confidence intervals even when corrected tests are not significant. Do not convert a non-significant result into an equivalence claim.

## Pilot history and freeze rationale

Pilot data are retained under `artifacts/papers/insertbot/pilots/` and are excluded from formal inference.

- `contact_v1` exposed an invalid contact trigger: normal seating force was interpreted as collision by contact-aware policies.
- `contact_v2` separated lateral/edge load from normal seating load. It then exposed two weak ablations: no-retract could unrealistically correct under load without a friction penalty, and orientation error contributed too little to the collision envelope.
- `contact_v3` added the loaded-sliding penalty and terminal-length-based orientation envelope. On 100 pilot seeds per difficulty, the hard-condition success rates were 92% for `contact_belief`, 87% for guarded admittance, 79% without orientation update, 86% without retract, 57% for direct insertion, 57% for vision-only staged insertion, 48% for spiral search, and 100% for the oracle. These values are calibration diagnostics, not formal results.

No controller, threshold, environment constant, endpoint, or hypothesis may change after this freeze without incrementing the protocol version and labeling all subsequent results as a new study.

## Reproduction and audit

The formal command is:

```powershell
wsl -d Ubuntu-24.04 -- env PYTHONPATH=src .venv-dlolab/bin/python scripts/run_insertbot_formal.py --output artifacts/papers/insertbot/formal/contact_v1 --physical-seeds 200 --bootstrap-draws 10000
```

The runner writes episode-level, aggregate, paired-comparison, and JSON report files with SHA-256 hashes. The independent audit must verify row counts and uniqueness, scenario pairing, finite/range-safe values, aggregate recomputation, exact paired tests and Holm adjustment, success invariants, and file hashes before any paper figure or table is generated.
