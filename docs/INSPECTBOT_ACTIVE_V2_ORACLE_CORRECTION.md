# InspectBot deployment-v2 descriptive-oracle correction

Issued on 2026-08-27 after the first deployment-v2 run.

The original `oracle` policy greedily prioritized the anomaly flag before marginal
coverage. It could repeatedly inspect one known anomalous site and was therefore not an
upper bound, despite its label. The implementation error affected only the descriptive
oracle. The oracle is excluded from every paired comparison, p-value, correction family,
and confirmatory claim.

The corrected oracle solves a multiple-choice knapsack problem independently at each
budget. For every site it considers zero through four prefix views, uses the true frozen
image scores, and maximizes detected criticality weight subject to using no more than the
scan budget. Ties prefer more detected anomalies, fewer false positives, and fewer scans.
This is an offline score-aware upper bound under the benchmark's fixed view-prefix and
threshold rules; it is not an executable policy.

The uncorrected run is retained at
`artifacts/papers/inspectbot/final/active/deployment_v2_initial_oracle_bug` with these
primary hashes:

- episodes CSV: `a6aaa1ef8faf163622a880a74549adbfb50e05350d2941f27ed66e8225ff6dc9`
- aggregate CSV: `700cf5d4826465ff4ed0e538be13a3be6e83fe3f251f985afa6299440183e05b`
- paired CSV: `4e9a48764f3f64e49d35aa8659fe4cc99655ec74c6efb02678680d35d017a340`
- report JSON: `4a50a23f0db6effe1a6ac7af1930ff04c2f95d984c890ada41bfcebe414ed7f8`

After rerunning, all 72,000 non-oracle episode rows (2,000 scenarios x 6 budgets x 6
policies) must match the retained run exactly on every field available in both versions,
apart from the newly added `scans_used` reporting field. The paired-comparison CSV must
remain byte-identical. Failure of either audit invalidates the corrected report.
