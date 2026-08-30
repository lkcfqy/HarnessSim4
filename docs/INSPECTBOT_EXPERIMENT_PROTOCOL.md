# InspectBot frozen experiment protocol

Protocol frozen on 2026-08-27 before the first active-inspection benchmark run.
The offline perception experiment had already been executed when this active-policy
protocol was written; it is therefore not a prospective preregistration of the
perception result.

## Research question

Can a topology-derived engineering risk map allocate a limited inspection budget
more effectively than order-only, random, geometry-only, and uncertainty-only
policies when the detector scores are grounded in held-out real cable-defect images?

## Public data and split

- Dataset: MVTec AD, `cable` category, CC BY-NC-SA 4.0.
- Training normals: 224 images. A seed-20260827 permutation fixes 180 images for
  fitting and 44 images for normal-only calibration.
- Test: 58 normal and 92 anomalous images across eight defect categories.
- The detector, feature channel subset, calibration quantile, and view transforms
  are frozen before the active benchmark.
- The official dataset's test labels are used only for evaluation and for sampling
  latent scenarios; no anomalous test image is used to fit or calibrate the detector.

## Detector and synthetic-view bridge

The frozen detector uses ImageNet-pretrained ResNet-18 spatial features from layers
1--3, aligned to a 28 x 28 grid, with 64 deterministic feature channels and a local
Gaussian normal model. Its image anomaly score is the maximum spatial Mahalanobis
distance. Four deterministic transforms form a controlled appearance stress test:
clean, dark-plus-blur, rotate-plus-contrast, and deterministic occlusion-plus-dark.
These are not claimed to be physical viewpoints or robot-camera trajectories.

Each transform is calibrated independently using only the same 44 normal calibration
images. The decision boundary is normalized score zero, corresponding to the frozen
99th-percentile normal calibration threshold for that transform.

## Paired active benchmark

- 1,000 deterministic latent scenarios, seeds 77000000--77000999.
- 12 fixed inspection sites on a branched harness topology.
- Independent anomaly probability 0.30 per site, conditioned on at least one normal
  and one anomalous site per scenario.
- Normal and anomalous images are sampled from the held-out MVTec test pool. Defect
  category is sampled uniformly, then an image is sampled within category.
- All policies receive exactly the same latent scenario for a given seed.
- Scan budgets: 4, 6, 8, 10, 12, and 16 observations.
- View order for repeated scans: occluded-dark, dark-blur, rotate-contrast, clean.
- Site predictions average all normalized scores observed at that site and use the
  frozen zero threshold.

Frozen policies are raster order, random scan, nearest-unvisited geometry coverage,
coverage-first uncertainty revisit, topology-risk plus uncertainty, shuffled-risk
negative control, and a label-aware oracle upper bound. The topology-risk policy may
use only the frozen site-risk vector, scan history, detector scores, and geometry. The
oracle is excluded from the confirmatory comparison family.

## Outcomes and confirmatory hypotheses

The primary outcome is criticality-weighted recall, where an anomalous site's weight
is fixed site risk multiplied by a fixed defect-severity coefficient. Secondary
outcomes are ordinary recall, precision, false-positive rate, coverage, travel
distance, revisit fraction, and critical success. Critical success requires at least
0.999 criticality-weighted recall and false-positive rate no greater than 0.20.

For every budget, the one-sided confirmatory hypothesis is that topology-risk has
higher paired criticality-weighted recall than each of five non-oracle baselines.
The 30 Wilcoxon signed-rank p-values are adjusted together with Holm's method.
Two-sided exact McNemar tests on critical-success discordances are reported as a
secondary family and Holm-adjusted separately. Paired bootstrap 95% intervals use
10,000 resamples of scenario-level differences.

## Claim boundary

This experiment can support only a claim about budget allocation in a paired hybrid
benchmark whose detector evidence comes from real cable images. It cannot establish
robotic viewpoint control, automotive-harness generalization, cycle time, reliability,
or factory deployment. Repeated sampling of a finite test pool means intervals measure
scenario resampling uncertainty, not uncertainty over unseen image populations.

No policy constants, risk values, severity coefficients, thresholds, budgets, or
confirmatory tests will be changed after the first formal run. Any later change must
be labeled a new experiment version and reported separately.
