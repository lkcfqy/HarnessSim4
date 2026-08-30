# RouteBot manuscript workspace

`main.tex` is an evidence-linked manuscript draft, not yet a submission final.
Numerical claims must be copied only from frozen artifacts under
`artifacts/papers/routebot/`; the current pilot runs are development evidence.

Required final inputs:

- 100-seed-per-difficulty paired main table (frozen: 3,600 episodes);
- nine-assignment counterfactual table after an independent teacher-feasibility screen
  (frozen: 13,500 episodes);
- representation and data-size ablations;
- direct MuJoCo closed-loop table;
- external closed-loop WireCraft or DLO-Lab replication;
- compiled PDF and anonymous reproducibility archive.

Frozen external audit already available:

- Berkeley Cable Routing real-robot data pinned at commit `20a7774...`;
- 1,647 complete episodes / 42,328 frames / four camera streams;
- whole-episode 1,153/247/247 train/validation/test split;
- state, action-persistence, and state-plus-action-history ridge controls with
  a 10,000-resample paired episode bootstrap;
- artifacts under `artifacts/papers/routebot/public_real/`.

Frozen direct cross-physics audit also available:

- current MuJoCo poses are converted to graph observations every control step;
- 20 paired physical seeds across all nine semantic assignments;
- 180/180 TopoHarness, 0/180 equal-parameter Geometry, 180/180 teacher;
- exact paired test, scene/checkpoint hashes, per-episode fixture offsets and
  calibration provenance under artifacts/papers/routebot/final/mujoco_direct/;
- a successful high-detail UR10e final frame under
  artifacts/papers/routebot/final/mujoco_direct_visual/.

Frozen paired PBD main table also available:

- 100 unseen physical seeds at each of difficulties 0.2, 0.5, and 0.8;
- 12 policies and 3,600 closed-loop episodes with complete per-episode export;
- TopoHarness 251/300 versus equal-parameter Geometry 10/300;
- paired advantages of 90, 83, and 68 percentage points with Holm-adjusted
  exact McNemar `p <= 2.29e-18`;
- every relation-equipped variant matches 251/300, so the paper attributes the
  signal to process-relation access rather than a unique network architecture;
- report SHA-256 `099bc1ca643fd363d775e4cbf87722edd82727dabfb043877f950382875eadac`.

Frozen nine-assignment counterfactual also available:

- three difficulties × 100 physical seeds × nine assignments × five policies,
  totaling 13,500 episodes with complete pairing;
- TopoHarness and ACT-RelPool 2,340/2,700 each, Geometry 57/2,700,
  ACT-Pointer 268/2,700, and teacher 2,700/2,700;
- all 27 TopoHarness-versus-Geometry cells show 60--100 percentage-point
  advantages with Holm-adjusted `p <= 1.06e-16`;
- TopoHarness and ACT-RelPool have the identical success/failure partition,
  reinforcing the relation-availability rather than architecture claim;
- report SHA-256 `a743ef9685336c31c1904e016eaa5db47638e829679e4a5808995a4e00e6e4f8`.

The public-data block is an offline action-interface audit.  The direct MuJoCo
block is closed-loop simulated evidence but still not a substitute for the
remaining external closed-loop semantic replication gate.

Frozen DLO-Lab execution sanity check also available:

- official source commit `c5026a9...`, prescribed Mushroom-RL fork commit
  `ec33647...`, and target SHA-256 `4e852122...53072`;
- CUDA reset/step smoke test with finite 206-D observations and 30-vertex state;
- one matched four-environment open-loop comparison in official
  `wiring_post`: ordered path native reward 0.9748, wrong order 0.5715,
  endpoint-straight 0.9698;
- the near-tie between ordered and endpoint-straight is retained as negative
  evidence that the native geometric task does not isolate process semantics;
- reports and hashed arrays under
  `artifacts/papers/routebot/final/dlolab_external/`.

The literature positioning is audited in `docs/ROUTEBOT_LITERATURE_AUDIT.md`.
The DLO-Lab code/asset reproducibility gate is audited separately in
`docs/ROUTEBOT_DLOLAB_EXTERNAL_AUDIT.md`. The executable open-loop sanity check
is complete, but it is not a trained RouteBot semantic-policy result. The asset
ZIP contains no separate redistribution license and must be excluded from
public archives.
