# HarnessSim4 venue and submission strategy

Verified against the official IEEE Robotics and Automation Society author pages on
2026-08-27. This document chooses a defensible venue for each manuscript and separates
venue-format readiness from scientific submission readiness.

## Venue assignment

| Paper | Target | Why this venue | Current IEEE-format pages |
|---|---|---|---:|
| InspectBot | IEEE T-ASE Regular Paper | Structured inspection automation, quality, reliability, and practical deployment limits | 6 |
| InsertBot | IEEE T-ASE Regular Paper | Structured assembly automation, contact-state diagnosis, force safety, and production reliability | 6 |
| RouteBot | IEEE T-RO Regular Paper | General robotics question about process semantics in long-horizon deformable-object manipulation | 5 |
| BranchBot | IEEE T-RO Regular Paper | General robotics question about semantic identity, typed grounding, and bimanual deformable-object manipulation | 4 |

T-ASE is the stronger fit for InspectBot and InsertBot because its official scope emphasizes
methods and systems that improve efficiency, quality, productivity, and reliability in
structured environments. T-RO is the stronger fit for RouteBot and BranchBot because its
scope covers fundamental robotics, manipulation, force/contact reasoning, cooperative
manipulation, robot learning, and industrial robotics.

## Frozen format requirements

- Every manuscript uses `\documentclass[journal]{IEEEtran}`, US Letter geometry, embedded
  fonts, anonymous authorship, and IEEE two-column layout.
- Every abstract is at most 200 words. The conservative LaTeX-aware audit counts are
  InspectBot 172, InsertBot 152, RouteBot 193, and BranchBot 182.
- InspectBot and InsertBot include a 100--300 word Note to Practitioners immediately after
  the abstract. The note states the practical implication and the digital-twin/hardware
  limitation instead of repeating the abstract.
- All four include two to five IEEE keywords.
- T-RO first submissions remain below 18 pages; both current T-RO manuscripts are far below
  that limit. T-ASE manuscripts remain below the repository's 12-page target, which avoids
  relying on overlength pages.
- Every multimedia folder contains a short summary, a ReadMe, and a project-specific H.264
  digital-twin clip. The summary explicitly says that the clip is simulation evidence, not
  hardware validation or factory-cycle evidence.
- The T-ASE cover-letter drafts include the required methodology and application codes.

The machine-readable source of these requirements and venue choices is
`configs/submission/venue_profiles.json`. Rebuild packages with
`python scripts/build_submission_packages.py` and audit them with
`python scripts/audit_venue_packages.py`.

## Two different readiness states

`venue_format_ready` means that a manuscript and its staging package satisfy the frozen
layout, anonymity, abstract, keyword, Note-to-Practitioners, page, PDF, and multimedia
checks. It does **not** mean that the scientific evidence is sufficient for submission.

`submission_ready` additionally requires the matching preregistered real-robot hardware
gate to pass and the red `DRAFT---NOT FOR SUBMISSION` warning to be removed. No warning may
be removed simply to make an audit green. At present, the four hardware gates are open, so
all generated packages are staging packages and must not be uploaded to a journal.

## Project-specific evidence that still blocks submission

1. **InspectBot:** calibrated real-camera multiview inspection across representative harness
   defects, suppliers, lighting, and fixtures.
2. **InsertBot:** calibrated force/torque insertion trials with real connectors, cables,
   latching verification, damage definitions, and post-insertion pull tests.
3. **RouteBot:** force-controlled semantic clip-routing replication on hardware or a
   genuinely semantic external benchmark.
4. **BranchBot:** vision-based three-to-five-branch identity tests and collision-aware
   dual-arm hardware trials.

## Official sources

- IEEE T-ASE, Information for Authors:
  <https://www.ieee-ras.org/publications/t-ase/information-for-authors-t-ase/>
- IEEE T-ASE, submission checklist:
  <https://www.ieee-ras.org/publications/t-ase/information-for-authors-t-ase/author-checklist-for-papers-submitted-to-ieee-t-ase/>
- IEEE T-ASE, Note to Practitioners:
  <https://www.ieee-ras.org/publications/t-ase/information-for-authors-t-ase/t-ase-note-to-practitioners/>
- IEEE T-ASE, methodology and application codes:
  <https://www.ieee-ras.org/publications/t-ase/information-for-authors-t-ase/t-ase-journal-primary-methodologies-and-applications/>
- IEEE T-RO, Information for Authors:
  <https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/>
