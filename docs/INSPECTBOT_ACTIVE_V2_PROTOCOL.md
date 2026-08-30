# InspectBot active-inspection deployment-v2 protocol

Protocol frozen on 2026-08-27 before the first deployment-v2 run and after retaining
the complete stress-v1 result.

## Why a second version is necessary

Stress-v1 intentionally began every site with the synthetic occluded-dark appearance.
That view retained only 1.1% anomaly recall at the normal-only threshold, so budgets
4--12 mostly measured a perception floor rather than inspection allocation. This is a
valid negative stress result and remains immutable under `active/stress_v1`.

Deployment-v2 changes exactly two design elements:

1. It begins with the canonical clean acquisition, then uses dark-blur,
   rotate-contrast, and occluded-dark for repeat observations.
2. It uses a disjoint deterministic scenario seed range and 2,000 rather than 1,000
   paired scenarios.

No detector, image score, transform, threshold, topology, site risk, defect severity,
policy equation, scan budget, or statistical test is changed.

## Frozen benchmark

- Score evidence: the immutable 600-row MVTec score cache with SHA-256
  `e3ce8b2c9d43dff9c0992a7e0664e614148268e5cf6366d94f84d225707040a5`.
- Scenarios: 2,000 seeds 78000000--78001999, all policies paired per seed.
- Sites: 12 fixed positions on a branched harness with a fixed engineering-risk map.
- Anomaly probability: 0.30 per site, conditioned on at least one normal and anomaly.
- Budgets: 4, 6, 8, 10, 12, and 16 observations.
- View order: clean, dark-plus-blur, rotate-plus-contrast, occluded-plus-dark.
- Prediction: mean normalized score at a site, anomalous at score at least zero.
- Policies: raster, random, nearest-unvisited geometry, coverage-first uncertainty,
  topology-risk plus uncertainty, shuffled-risk negative control, and label-aware oracle.

## Hypotheses and reporting

The primary endpoint remains criticality-weighted recall. The confirmatory deployment
hypothesis applies to under-coverage budgets 4, 6, 8, and 10: topology-risk should have
higher paired criticality-weighted recall than raster, random, geometry, uncertainty,
and shuffled-risk policies. Budget 12 is a prespecified full-coverage equality check;
budget 16 is an exploratory revisit regime. The oracle is descriptive only.

For maximum conservatism, software applies Holm correction over all 30 topology-versus-
baseline Wilcoxon comparisons, including the equality and exploratory budgets. Paired
bootstrap 95% intervals use 10,000 scenario resamples. Critical-success discordances use
exact two-sided McNemar tests and a separate 30-test Holm family.

Secondary outcomes are ordinary recall, precision, F1, false-positive rate, coverage,
revisit fraction, travel distance, and critical success. Critical success requires at
least 0.999 weighted recall and false-positive rate no greater than 0.20.

## Claim boundary

Deployment-v2 remains a hybrid benchmark: real held-out cable-image detector scores are
embedded into simulated harness sites and the four appearances are deterministic image
transforms, not robot-acquired views. It can support a budget-allocation claim but not a
factory, camera-motion, or physical-robot claim. Stress-v1 and deployment-v2 must always
be reported together; v2 may not be presented as if v1 never occurred.
