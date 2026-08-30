# InsertBot paper draft

This directory contains the evidence-backed English research draft for the second robot project.

## Status

- Primary paired benchmark: 4,800 episodes, 200 physical seeds per difficulty.
- Direct MuJoCo audit: 300 episodes with native collision forces and direct observations.
- Public proxy audit: 50 complete LeRobot ALOHA episodes / 25,000 frames, split 40/10
  by episode and excluded from wire-terminal success/force claims.
- Independent combined audit: 153/153 checks passed.
- Manuscript status: **research draft; not submission-ready until the hardware gate in Section 7 is completed.**

## Build

From the paper directory in WSL:

```bash
../../third_party/tectonic-0.16.9/tectonic --keep-logs --outdir build main.tex
```

The manuscript reads frozen tables, number macros, and vector figures from:

```text
artifacts/papers/insertbot/formal/paper_assets/
```

The formal source reports are:

```text
artifacts/papers/insertbot/formal/contact_v1/insert_paper_report.json
artifacts/papers/insertbot/formal/mujoco_direct_v1/insert_mujoco_direct_report.json
artifacts/papers/insertbot/formal/audit/insertbot_result_audit.json
artifacts/aloha_ridge_v0/report.json
data/public/aloha_sim_insertion_human/harnessbench_manifest.json
```
