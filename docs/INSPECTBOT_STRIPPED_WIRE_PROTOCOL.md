# InspectBot Stripped Wire validation protocol

Protocol frozen on 2026-08-27 before the first model run on this dataset.

## Purpose

This experiment is an independent task replication on the FAU/FAPS Stripped Wire
Dataset, not a zero-shot transfer from MVTec. It tests whether the same frozen
ImageNet-feature anomaly-detection recipe works on a second, more production-relevant
wire-end inspection task.

## Immutable data and split

- Zenodo DOI: 10.5281/zenodo.16686806, release v2, CC BY 4.0.
- Archive MD5: `9e1944f1b7c9bc9e7db433f7bdcf2694`.
- Normal training pool: the 200 released `train/PatchCore` images.
- Seed-20260827 permutation: first 160 for fitting and remaining 40 for normal-only
  threshold calibration.
- Untouched released test split: 133 good, 68 cut-strand, and 99 pulled-strand images.
- The optional three-image VLM reference folder is excluded.
- Exact SHA-256 duplicate checks must show no overlap between PatchCore training and
  test images before fitting.

## Frozen preprocessing and methods

Images have variable widths (89--578 pixels) and fixed height 1079. They are resized
with aspect ratio preserved and symmetrically padded to 224 x 224 using the image's
low-resolution median RGB color. No test-label-dependent crop is allowed.

All neural features use the same frozen torchvision ResNet-18 ImageNet-1K V1 weights
and the same seed-selected 64 channels from aligned layers 1--3. Four methods are
evaluated:

1. RGB mean, standard deviation, and gradient statistics with Ledoit-Wolf distance.
2. Global pooled ResNet features with Ledoit-Wolf distance.
3. Position-wise nearest-neighbour distance in the frozen spatial feature grid.
4. Position-wise regularized Gaussian distance, identical in recipe to the MVTec run.

The image score is the mean of the top 1% spatial scores where applicable. Every
method's decision threshold is the 99th percentile of its own 40 normal calibration
scores. No anomalous image may be used for fitting, thresholding, channel selection,
or method selection.

## Outcomes and statistics

Primary outcomes are image AUROC and AUPR. Deployment outcomes are precision, recall,
F1, and false-positive rate at the normal-only threshold. Results are also reported
separately for cut strands and pulled strands against the shared normal test set.

The directional confirmatory expectation is that the spatial Gaussian model exceeds
the RGB-statistics and global-feature baselines in AUROC. The spatial nearest-neighbour
method is a strong exploratory reference; no superiority claim over it is prespecified.
Stratified image bootstrap intervals use 2,000 resamples per method and 5,000 paired
resamples for AUROC differences. Thresholded error discordances use exact two-sided
McNemar tests with Holm adjustment across all three comparisons.

## Claim boundary

The dataset contains controlled single stripped-wire crops rather than complete
automotive harnesses or images acquired by a moving robot. A successful result supports
offline wire-end anomaly perception only. It does not establish transfer across
factories, wire suppliers, cameras, robot viewpoints, production cycle time, or safety
certification.

No preprocessing, split, channel count, covariance regularization, score aggregation,
threshold, endpoint, or test will be changed after the first formal run. Any revised
experiment must receive a new versioned protocol and retain this result.
