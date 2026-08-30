# InspectBot literature and public-data audit

Audit date: 2026-08-27 (Asia/Shanghai).

## Datasets actually used

### MVTec AD, cable category

- Primary record: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Paper: https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html
- License: CC BY-NC-SA 4.0.
- Transport mirror only: https://huggingface.co/datasets/foersben/mvtec-ad/tree/main/cable
- Mirror commit: `c75b39616f84db43677bcc8228caaafaf5096d7f`.
- Audited local tree SHA-256:
  `5ecf3b6b9feeec54eae602cbeb3e1c1241fd3110c9722166c4255a7a725a37c1`.
- Counts: 224 normal train, 58 normal test, 92 anomalous test, and 92 masks.

The official MVTec web form requests personal contact fields before download. No
personal fields were transmitted. The Hugging Face repository was used only as a file
transport mirror; dataset authorship, license, and citation remain MVTec's.

### FAU/FAPS Stripped Wire Dataset

- Official record: https://zenodo.org/records/16686806
- DOI: 10.5281/zenodo.16686806, version v2, published 2025-08-01.
- Associated paper: https://arxiv.org/abs/2508.14504
- License in the Zenodo API: CC BY 4.0.
- Archive MD5: `9e1944f1b7c9bc9e7db433f7bdcf2694`.
- Local archive SHA-256:
  `22754426bae0d15420bd7a10198dadf3dc2925a81e5b2fdb6dad95b5d05d3cbd`.
- Audited extracted tree SHA-256:
  `19d7aafc80aba1d41f851e70d6f794127f5b69b4b0f90b8b5e97e637d6640e2f`.
- Counts: 200 normal PatchCore train, 133 normal test, 68 cut-strand test,
  99 pulled-strand test, and three unused VLM references.

There are no exact duplicates between the 200 normal-model images and the 300 test
images. One PatchCore image is copied into the optional VLM reference folder; that folder
is excluded from every experiment.

## Candidate audited but not used

CableInspect-AD is a NeurIPS 2024 expert-annotated power-line cable dataset with 4,798
high-resolution images, 6,023 annotated anomalies, three cables, and seven defect types:

- Project: https://mila-iqia.github.io/cableinspect-ad/
- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/76d9dd096d9469d6b7e732f0cddb51b3-Paper-Datasets_and_Benchmarks_Track.pdf
- Code: https://github.com/mila-iqia/cableinspect-ad-code
- Audited code commit: `f186021013d1219d56635314785098795663e9df`.

The official Hydro-Quebec download redirected by region and the official Google Drive
link required access approval on the audit date. No access request was sent and no
unverified mirror was substituted. CableInspect-AD is therefore cited as related work,
not reported as experimental data.

## Method references and scope

- PaDiM models a multivariate Gaussian distribution at each pretrained spatial feature
  location: https://arxiv.org/abs/2011.08785
- PatchCore uses a nominal patch-feature memory bank and coreset sampling:
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
- The PB-IAD Stripped Wire evaluation reports PatchCore thresholds selected using 20% of
  the labeled original test set. InspectBot instead fixes every threshold from normal
  training-calibration images only. Thresholded F1/recall values are therefore not
  directly comparable; threshold-independent AUROC/AUPR and within-protocol comparisons
  are the defensible evidence.
- A 2024 systematic review identifies robustness, practicality, and exploitation of
  intrinsic harness structure as open problems in robotized wire-harness vision:
  https://arxiv.org/abs/2309.13744
- A 2024 multi-branch study uses 3-D point clouds and real dual-robot assembly validation:
  https://doi.org/10.1016/j.jmsy.2023.12.002

## Evidence boundary

MVTec cable images are controlled cable cross-sections; Stripped Wire images are
controlled single wire ends. Neither dataset contains full automotive-harness topology,
robot viewpoints, temporal camera trajectories, or factory cycle-time labels. The
active benchmark therefore embeds frozen real-image scores into simulated harness sites
and labels itself hybrid. The direct MuJoCo audit executes scan motions but does not
render detector input. No result is hardware or production-line evidence.
