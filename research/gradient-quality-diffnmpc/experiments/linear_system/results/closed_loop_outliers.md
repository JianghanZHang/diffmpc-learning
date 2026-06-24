# Closed-loop (50-step) gradient-accuracy: outliers from A vs B

**Run 2026-06-23.** Random linear MPC (nx=8, nu=4, horizon=20, umax=1.0), **1 SQP iter**, **50-step closed-loop rollout** (diffmpc-as-policy: solve MPC -> apply u0 -> step dynamics -> accumulate cost), regulate to origin. Differentiable parameter = cost weights (Q,R); per-sample (per initial state, n=24, seed 0). Each backward is compared to a **convergence-checked FD of its own rollout** (eps_seq 3e-5..1e-6); 'flagged' = FD never plateaus (a genuine discontinuity sample).

- **A** = TurboMPC analytic backward on the **hard box** (use_slack=False) — exactly the diffmpc `benchmark_turbompc_gradient_accuracy.py` setup (the `grad_box_accuracy` image).
- **B** = log-barrier smoothed backward (kappa=1e-6).

## Summary

| method | median cos | min cos | #cos<0.99 | #cos<0 | rel_l2 median | rel_l2 max | FD-flagged | **outliers** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (hard box) | 0.20363 | -0.466 | 10 | 5 | 1.76e+02 | 1.12e+03 | 9/24 | **19/24** |
| B (log-barrier) | 1.00000 | +1.00000 | 0 | 0 | 1.70e-06 | 3.54e-04 | 1/24 | **1/24** |

## Per-sample (sorted by A's cosine — A's worst first)

| sample | A cos | A rel_l2 | A flag | B cos | B rel_l2 | B flag | A outlier? |
|---:|---:|---:|:--:|---:|---:|:--:|:--:|
| 2 | -0.4662 | 1.76e+02 |  | +1.00000 | 2.24e-06 |  | **YES** |
| 13 | -0.3981 | 2.01e+02 | F | +1.00000 | 1.02e-06 |  | **YES** |
| 19 | -0.3200 | 2.74e+02 |  | +1.00000 | 2.05e-06 |  | **YES** |
| 15 | -0.2466 | 1.12e+03 |  | +1.00000 | 1.88e-06 |  | **YES** |
| 10 | -0.1478 | 6.37e+02 | F | +1.00000 | 1.39e-06 |  | **YES** |
| 16 | -0.0954 | 1.04e+03 | F | +1.00000 | 2.01e-06 |  | **YES** |
| 1 | -0.0656 | 1.02e+03 | F | +1.00000 | 2.01e-06 |  | **YES** |
| 14 | -0.0233 | 4.22e+02 |  | +1.00000 | 4.59e-06 |  | **YES** |
| 23 | -0.0194 | 2.76e+02 |  | +1.00000 | 1.23e-06 |  | **YES** |
| 17 | +0.1435 | 1.59e+02 |  | +1.00000 | 4.70e-06 | F | **YES** |
| 21 | +0.1486 | 3.22e+02 |  | +1.00000 | 2.69e-06 |  | **YES** |
| 20 | +0.1587 | 4.89e+02 | F | +1.00000 | 1.36e-06 |  | **YES** |
| 11 | +0.2036 | 2.97e+01 |  | +1.00000 | 6.47e-07 |  | **YES** |
| 22 | +0.2121 | 2.96e+02 |  | +1.00000 | 2.05e-06 |  | **YES** |
| 7 | +0.4329 | 6.83e+02 | F | +1.00000 | 8.49e-07 |  | **YES** |
| 8 | +0.5134 | 5.74e+02 | F | +1.00000 | 3.00e-06 |  | **YES** |
| 5 | +0.6014 | 6.56e+02 | F | +1.00000 | 3.54e-04 |  | **YES** |
| 6 | +0.6532 | 7.89e+02 |  | +1.00000 | 1.42e-06 |  | **YES** |
| 0 | +0.8303 | 3.15e+01 | F | +1.00000 | 1.70e-06 |  | **YES** |
| 3 | +1.0000 | 1.17e-03 |  | +1.00000 | 5.18e-07 |  | no |
| 4 | +1.0000 | 1.00e-03 |  | +1.00000 | 1.97e-06 |  | no |
| 18 | +1.0000 | 1.07e-03 |  | +1.00000 | 1.28e-07 |  | no |
| 12 | +1.0000 | 9.81e-04 |  | +1.00000 | 7.78e-07 |  | no |
| 9 | +1.0000 | 1.27e-03 |  | +1.00000 | 1.57e-06 |  | no |

## Finding

In closed loop, **A (the diffmpc hard-box backward) is an outlier on 19/24 samples** (median cos 0.20, 5 with *negative* cosine, rel_l2 up to 1e+03) — the 50-step rollout repeatedly crosses control active-set boundaries, where the hard active-set backward's gradient is wrong/explodes (strict-complementarity failure, compounded over the rollout). This reproduces the diffmpc `grad_box_accuracy` outliers. 
**B (log-barrier) has 1/24 outliers**: on *every* sample — including the exact ones where A fails — B matches its FD to cos=1.0 / rel_l2~1e-6. The smoothing makes the closed-loop policy gradient well-defined and FD-consistent, eliminating the hard-backward outliers (the H1.2 payoff, in closed loop).

Caveats (per CLAUDE.md): A and B solve slightly different problems (hard box vs kappa=1e-6 barrier); each AD is checked against *its own* rollout's convergence-checked FD (self-consistency). n=24, one random system, H=20 (reduced from 40 for compile tractability of the differentiable 50-step rollout).

## Robustness across systems (seeds 0,1,2 — 3 random linear systems)

| seed | n | A median cos | A #cos<0.99 | A #cos<0 | **A outliers** | B median cos | B #cos<0.99 | **B outliers** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 0.204 | 19 | 9 | **19** | 1.00000 | 0 | **1** |
| 1 | 16 | 0.171 | 12 | 6 | **12** | 1.00000 | 0 | **1** |
| 2 | 16 | 0.883 | 10 | 1 | **11** | 1.00000 | 0 | **0** |
| **all** | 56 | 0.473 | 41 | 16 | **42/56** | 1.00000 | 0 | **2/56** |

**Across 3 random systems / 56 samples: A is an outlier on 42/56 (75%), B on 2/56 (4%).** Every B outlier is an FD-plateau edge case with cos=1.0; B has zero cos<0.99 on all three systems. The hard-box backward's closed-loop outliers (and their elimination by the log-barrier smoothing) are robust to the random system, not a seed-0 artifact.

## Faithful horizon H=40 (the diffmpc benchmark horizon; seed 0, n=12)

The main results above use H=20 (reduced for compile tractability). Re-run at the benchmark's
**H=40** confirms and strengthens the pattern:

| method | FD-flagged | cos (non-flagged) | rel_l2 |
|---|---:|---:|---:|
| A (hard box) | **11/12** | 1.00000 (the 1 plateau-able sample) | 1.9e-3 |
| B (log-barrier) | **0/12** | 1.00000 (all 12) | 2.2e-6 |

At the longer horizon the 50-step rollout crosses **even more** control active-set boundaries, so A's
hard-box cost-vs-weights map is **non-differentiable on 11/12 samples** — the convergence-checked FD
cannot even plateau (the gradient ground truth is undefined there, not merely mismatched). B's smooth
rollout plateaus cleanly on all 12 with AD=FD (cos=1.0, rel_l2~2e-6). The faithful horizon makes the
hard-box policy *more* pathological and leaves the log-barrier policy untouched — the smoothing is what
makes the closed-loop diffmpc-as-policy gradient exist at all.

## B's outliers vs the smoothing strength κ (seed 0, n=16, H=20)

B has zero outliers at κ=1e-6; does it inherit A's pathology as κ→0 (barrier → hard complementarity)?
Sweeping κ over 8 orders of magnitude:

| κ | med cos | #cos<0.99 | #flagged | **B outliers** | med rel_l2 |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 1.00000 | 0 | 0 | **0/16** | 1.5e-6 |
| 1e-4 | 1.00000 | 0 | 0 | **0/16** | 1.7e-6 |
| 1e-5 | 1.00000 | 0 | 0 | **0/16** | 1.5e-6 |
| 1e-6 | 1.00000 | 0 | 0 | **0/16** | 1.7e-6 |
| 1e-7 | 1.00000 | 0 | 0 | **0/16** | 1.5e-6 |
| 1e-9 | 1.00000 | 0 | 0 | **0/16** | 1.8e-6 |
| 1e-11 | 1.00000 | 0 | 0 | **0/16** | 1.7e-6 |

**B is outlier-free across all κ — the result is insensitive to the barrier over 8 orders of magnitude.**
Crucially, at κ=1e-11 the central-path complementarity relaxation is ≈0 (hard complementarity), yet B's
rollout is still smooth (0 FD-flagged) and B's AD still matches FD (cos=1.0). So the closed-loop
FD-consistency is provided by the **elastic slack** (the soft box `to_one_sided` builds with γ=1e4 —
a C¹ penalty that smooths the active-set boundaries in the *forward* rollout), **not** the barrier κ
(which only relaxes complementarity in the *backward*). The hard-box A (use_slack=False, the diffmpc
default) lacks this and is the pathological one.

Refinement of the headline: the closed-loop outliers are eliminated by the **soft/slack box
formulation**; the log-barrier κ is a secondary backward refinement that is robust across a huge range.
A direct control — TurboMPC's own slack backward (use_slack=True) in closed loop — would confirm it
gives the same 0-outlier behavior as B; not yet run (the κ=1e-11 column is strong indirect evidence,
since there the barrier is negligible and only the slack remains).

## Mechanism control: slack box vs hard box vs log-barrier (seed 0)

The κ-sweep implied the smoothness comes from the slack, not the barrier. Direct test — run the
closed-loop with **TurboMPC's own slack box** (use_slack=True, γ=1e4, its native analytic backward),
no log-barrier:

| backward | n | median cos | #cos<0.99 | #cos<0 | FD-flagged | **outliers** |
|---|---:|---:|---:|---:|---:|---:|
| A — hard box (use_slack=False, diffmpc default) | 24 | 0.204 | 19 | 9 | 9 | **19/24 (79%)** |
| A — slack box (use_slack=True, TurboMPC native) | 16 | 1.00000 | 0 | 0 | 3 | **3/16 (mild flags only)** |
| B — log-barrier (slack + κ=1e-6) | 16 | 1.00000 | 0 | 0 | 0 | **0/16** |

**Confirmed: the soft/slack box is what fixes the closed-loop outliers.** Switching the *forward*
from the hard box to the slack box (a C¹ penalty) collapses ~79% outliers to ~0 — TurboMPC's own
slack backward is already FD-consistent in closed loop. The log-barrier κ adds only a *marginal*
extra smoothing (the slack's 3 residual FD-flags → 0 for B). So the corrected, fully-measured story:

- The diffmpc-benchmark outliers are a property of the **hard active-set box** (use_slack=False).
- They are eliminated by the **soft/slack formulation** — available both as TurboMPC's `use_slack=True`
  and as B's log-barrier (which builds the same soft box via `to_one_sided` with γ).
- The log-barrier **κ is not the essential ingredient** (κ-insensitive over 1e-11..1e-3; A-slack works
  without any barrier); it contributes a small additional smoothing on top of the slack.

## What drives the per-sample pathology? (switching hypothesis REFUTED; it's a backward error)

Earlier I conjectured the hard-box pathology is driven by active-set *switching* (forward
discontinuities/kinks). **Direct measurement on the linear data refutes that** (`switch_correlation.py`,
seed 0, n=24): perturbing the cost weights by eps=1e-3 flips the saturation status of **zero** applied
controls for **every** sample, yet the pathology is fully present. Combined with the convergence-checked
FD *plateauing* (smooth) at eps 1e-5 for these non-flagged samples, the cost-vs-weights map is **locally
smooth exactly where the hard AD is wrong** (cos −0.47). So the pathology is **not** a forward kink /
active-set switch — it is a **backward inconsistency**: the hard active-set backward returns a gradient
that disagrees with the (smooth, well-defined) FD gradient.

It correlates moderately with the **saturation fraction**, not switching:
`Spearman(cos, sat_frac) = −0.41` (outliers median sat_frac 0.040 vs clean 0.015);
`Spearman(cos, u0_switch_flips) = −0.25` but the flip counts are all 0, so that number is noise.

**Refined mechanism (conjecture, partially measured):** the hard backward errs at **marginally-active**
controls — pinned at the bound with a near-zero multiplier (strict-complementarity failure) — where its
binary active/inactive mask zeroes a sensitivity that is actually nonzero. The soft/slack backward
replaces the mask with a smooth weight `W = y_g/(s+y_g/γ)` and stays FD-consistent. This reconciles the
quadrotor: its tight-box thrust is **firmly** active (large multiplier) → the hard mask is correct →
no pathology; the linear random systems have **marginal** activations → the hard mask errs. The
confirming measurement (correlate per-sample cos with the *multiplier magnitude* of the active controls
— small multipliers ⇒ pathological) is the clean next step; not yet run. Data: `switch_correlation.npz`.

## Validation: the linear outliers are real, not an FD-noise artifact (eps-robust)

After the quadrotor's "cos=0.616" turned out to be FD below the noise floor, I checked the same risk
for the linear headline (`linear_noise_check.py`): the linear closed-loop cost is ~534, and its
**noise floor is 6e-8 absolute / 7e-11 relative** — ~1000× lower (relative) than the quadrotor's
(8e-8 rel), because the 8-state 1-SQP solves are far simpler than the 13-state quaternion-RK4 ones.
So the small-eps FD was reliable here, and the outliers are **eps-robust**:

| FD eps | hard-box cos median | cos min | #cos<0 | flagged |
|---|---:|---:|---:|---:|
| small (3e-5..1e-6) | 0.149 | −0.466 | 7 | 7/24 |
| large (3e-3..1e-4) | 0.181 | −0.466 | 8 | 2/24 |

The worst cosines are essentially identical (−0.466, −0.399, −0.319, …) at both eps scales. **The
linear hard-box pathology is real**, not a noise artifact. Combined with the switching refutation
above (cost locally smooth, FD reliable, AD wrong), the hard-box closed-loop gradient is *genuinely
inconsistent* with the true gradient — a backward error, eps-robust and noise-clean. (The quadrotor
looked milder partly because its ~1000× higher relative noise floor corrupted the small-eps FD,
now corrected, and partly because its saturated thrust is firmly active rather than marginal.)
