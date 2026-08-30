# BranchBot literature and novelty audit

Updated: 2026-08-27

## Claim that remains defensible

BranchBot does **not** claim to introduce bimanual DLO manipulation, cable
untangling, graph representations, or ACT.  The narrow hypothesis is:

> When branch geometry is held fixed and only observable A/B endpoint--arm--target
> relations are renamed, an explicit relational representation preserves the
> correct dual-arm assignment more reliably than otherwise matched policies that
> omit those relations.

The current controlled task has exactly two branches.  It is a semantic
counterfactual and cross-physics study, not evidence for arbitrary 2--5 branch
harnesses, visual perception under occlusion, force control, or production use.

## Closest primary work

- Sundaresan et al., *Untangling Dense Non-Planar Knots by Learning Manipulation
  Features and Recovery Policies*, RSS 2021.  Real-cable experiments establish
  strong learning and recovery evidence for dense knot untangling, but do not
  study branch-name/target counterfactuals.
  <https://roboticsproceedings.org/rss17/p013.html>
- Viswanath et al., *Autonomously Untangling Long Cables*, RSS 2022.  SGTM uses
  bilateral manipulation, RGB-D perception, and specialized primitives on long
  real cables.  It is stronger than BranchBot on real perception and hardware.
  <https://www.roboticsproceedings.org/rss18/p034.html>
- Viswanath et al., *HANDLOOM: Learned Tracing of One-Dimensional Objects for
  Inspection and Manipulation*, CoRL 2023.  The work supplies an annotated
  public RGB-D cable-tracing dataset and bimanual untangling experiments.  Its
  traces are relevant external perception evidence but do not contain our
  branched-harness semantic relation labels.
  <https://proceedings.mlr.press/v229/viswanath23a.html>
- Yu et al., *A Coarse-to-Fine Framework for Dual-Arm Manipulation of Deformable
  Linear Objects with Whole-Body Obstacle Avoidance*, 2022/2023.  This work
  combines global planning and closed-loop local control for dual-arm DLO shape
  manipulation and obstacle avoidance; BranchBot cannot claim first dual-arm
  closed-loop DLO planning.
  <https://arxiv.org/abs/2209.11145>
- Yu et al., *Generalizable Whole-Body Global Manipulation of Deformable Linear
  Objects by Dual-Arm Robot in 3-D Constrained Environments*, 2023.  This work
  directly addresses 3-D constrained dual-arm DLO motion and real-world
  generalization, stronger physical evidence than BranchBot's present
  task-space MuJoCo audit.
  <https://arxiv.org/abs/2310.09899>
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware*, RSS 2023.  ACT is the temporal imitation baseline; our compact
  implementation is an independent state-based adaptation and is not the
  authors' code or an exact visuomotor reproduction.
  <https://arxiv.org/abs/2304.13705>

## Public-data decision

HANDLOOM is the closest public real-data source for tracing crossings and cable
identity.  It is not suitable for measuring BranchBot's semantic assignment
success because it lacks branched endpoint--target labels and matched A/B
renaming interventions.  Any future use must therefore be reported as an
offline perception/tracing audit, not as real-world completion evidence.

## Evidence still required for a submission-ready claim

1. variable 2--5 branch topology with held-out branch counts and target
   permutations;
2. visual endpoint/branch identity estimation under occlusion and colour/name
   permutations;
3. non-symmetric material, length, and fixture randomization;
4. force-aware dual-arm collision constraints or real-hardware validation;
5. at least three training seeds for learned-method uncertainty, in addition to
   paired evaluation-seed statistics.
