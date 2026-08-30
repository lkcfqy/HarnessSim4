# InspectBot research draft

This directory contains the English research manuscript for the first robot
project.  It is intentionally labeled **DRAFT—NOT FOR SUBMISSION**.

The paper reports:

- audited public-data experiments on MVTec AD cable and the FAU/FAPS Stripped
  Wire Dataset;
- a retained adverse-view negative result and a disjoint clean-first paired
  active-inspection benchmark;
- a corrected descriptive oracle with the original bug retained for audit;
- direct MuJoCo execution of 320 policy-budget scan runs; and
- explicit boundaries between real images, synthetic appearance transforms,
  simulated motion, and missing hardware evidence.

Build from WSL at the repository root:

```bash
bash scripts/build_inspectbot_research_draft.sh
```

The release PDF is written to `output/pdf/01_inspectbot_research_draft.pdf`.
The build fails if lint, targeted tests, frozen-result audit, asset rendering, or
LaTeX compilation fails.
