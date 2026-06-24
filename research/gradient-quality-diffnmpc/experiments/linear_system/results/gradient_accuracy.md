# Closed-loop gradient accuracy vs ONE common ground truth

Redesign of the 4-variant comparison to fix the earlier methodology (each variant had been scored
against its OWN finite-difference ground truth over different flagged subsets → apples-to-oranges).
Now: **one common ground truth** = the FD of the **true hard-constrained** (`|u|≤u_max`) closed-loop
loss, computed **once** on the hard-box forward solved very tightly (GT_TOL=1e-11); the **same 64
samples** for every variant.

Setup: closed-loop, (B, horizon, nx, nu) = (64, 160, 8, 4), SIM_STEPS=50, umax=1, seed=0.
Variants: A = TurboMPC (`ADMM_FUSED_CUDSS` fwd, `DIRECT_CUDSS_FFI` bwd); B = log-barrier central-path
(kappa=1e-6). slack = Moreau penalty γ=1e4; no-slack = hard box (A) / pure barrier γ=1e12 (B).
A's loss (`build_rollout_fn`) and B's loss (manual scan) are **identical** (verified: cos(A-slack,
B-slack)=1.0000). Script: `closed_loop_gradient_accuracy.py`.

## Result — cos median (cos min) vs the true hard-constrained gradient

| AD tol | A no-slack | A slack | B no-slack | B slack |
|---:|:---:|:---:|:---:|:---:|
| 1e-1 | 0.14 (−0.44) | 0.16 (−0.40) | −0.02 (−0.64) | 0.03 (−0.65) |
| 1e-3 | 1.000 (0.03) | 0.205 (−0.36) | 0.47 (−0.59) | 0.206 (−0.28) |
| 1e-5 | 1.000 (0.03) | 0.205 (−0.36) | 1.000 (0.69) | 0.205 (−0.36) |
| 1e-7 | 1.000 (0.03) | 0.205 (−0.36) | 1.000 (1.00) | 0.205 (−0.36) |
| 1e-9 | 1.000 (0.03) | 0.205 (−0.36) | 1.000 (1.00) | 0.205 (−0.36) |

(A no-slack `#cos<0.99 = 1` — the single anomaly sample 45, see below; it is cos≈1.0 on the other
63/64. Slack variants: `#cos<0.99 = 61/64`, `#cos<0 = 16/64`, flat across tol.)

## Findings

1. **The hard box and the log-barrier are FAITHFUL to the true constrained gradient.** A no-slack
   (it *is* the GT) and B no-slack (barrier κ=1e-6) reach **cos 1.0** once solved tightly (A by
   tol≤1e-3, B by tol≤1e-5 — the central-path solve converges slower).

2. **The Moreau slack (γ=1e4) gradient is BIASED from the true hard-constrained gradient, and the bias
   does NOT vanish with a tighter solve.** A-slack and B-slack sit at **cos≈0.205, flat across AD
   tolerance**. It is a **formulation** bias, not a numerical one — confirmed by two *independent*
   solvers (TurboMPC ADMM, central-path ADMM) on the same Moreau formulation agreeing on 0.205 (and
   cos(A-slack,B-slack)=1.0 with each other).

3. **The bias COMPOUNDS over the closed loop** (interpretation; sim_steps sweep pending to confirm): at
   SIM_STEPS=5 the smoke test gave A-slack cos **0.9994**; at SIM_STEPS=50 it is **0.205**. The
   per-step Moreau violation O(1/γ)=1e-4 accumulates over the long rollout and rotates the gradient
   away from the true hard-constrained one.

4. **At loose tol (1e-1) everyone is wrong** (even A no-slack = the GT, at 0.14) — a loose solve biases
   every gradient regardless of formulation.

**Why the redesign mattered:** the earlier per-config ground truths hid finding 2 entirely — each
slack variant looked cos 1.0 against its *own* (equally-biased) forward. Only a common true-gradient
reference + a long horizon reveals that the Moreau relaxation's gradient is not faithful to the
hard-constrained problem.

## Ground-truth FD: convergence and independent validation

- **GT plateau found 64/64** over a FIXED eps grid `[3e-6, 1e-3]`. (First attempt flagged 5 samples;
  diagnostic `fd_diagnose.py` showed those were a **noise-floor artifact**, not discontinuities — the
  FD is stable at large eps and only goes erratic ~1/eps below ~1e-6 because horizon-160 makes the
  cost ~10²–10³ and GT_TOL=1e-11 leaves absolute cost noise ~1e-9. Fix: keep eps ABOVE the noise floor
  and accept convergence with a norm-scaled absolute tolerance so near-zero components pass. The
  "shrink eps" strategy was the wrong direction here. See memory `fd-noise-floor-is-system-size-dependent`.)
- **FD vs an INDEPENDENT analytic gradient** (hard-box DIRECT backward, unrelated to finite
  differences): **cos median = 0.999999, rel-err median = 1.5e-3** → the FD ground truth is correct.

### The one anomaly (sample 45) — the FD is right; the hard-box analytic backward is not

One sample (45) has hard-box analytic-vs-FD cos = 0.034. It is **not** a near-zero gradient
(`‖gFD‖=5.0e-2` > median 2.3e-2), and on it the **log-barrier matches the FD perfectly (cos 1.0)**
while the hard box does not. Ruled out, in order:
- **strict complementarity failure** — REFUTED (`strict_comp_check.py`): the barrier's `min(s_i,y_i)`
  scales as **κ, ratio 100** (a degenerate constraint would scale as √κ, ratio ~10); sample 45 is only
  rank 11 in degeneracy; and the genuinely √κ samples (28: ratio 7.0, 34) have **zero** gradient error.
  `corr(degeneracy, error)=+0.015`. So strict-comp failure is neither necessary nor sufficient here.
- **LICQ failure** — structurally impossible: linear dynamics give a full-rank (triangular identity
  blocks) equality Jacobian, and active control-box constraints are distinct unit vectors independent
  of it. LICQ always holds for this problem class.
- **under-converged forward** — REFUTED (`fwd_conv_check.py`): 0/50 steps hit the ADMM cap; it
  converged like the clean samples.
- **remaining cause: QP ill-conditioning.** Sample 45's forward needed **1942 ADMM iters vs ~400** for
  clean samples (a near-active constraint, multiplier ≈0.04 — strict comp holds, barely), so the
  **DIRECT backward's KKT solve loses accuracy** while the barrier regularizes it.

**Bottom line: the FD ground truth is correct.** On the only sample where it looked suspicious it
agrees with the independent log-barrier gradient (cos 1.0); the discrepancy is in the hard-box
*analytic* backward (QP ill-conditioning), not the finite differences. On the other 63/64, FD vs
analytic agree to 0.15%.

Scripts: `closed_loop_gradient_accuracy.py` (main), `fd_diagnose.py` (noise-floor),
`strict_comp_check.py` (κ/√κ degeneracy), `fwd_conv_check.py` (forward convergence). Data:
`results/gradient_accuracy.npz`, `results/strict_comp_check.npz`.
