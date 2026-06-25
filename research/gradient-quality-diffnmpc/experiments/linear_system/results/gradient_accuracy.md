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

2. **The Moreau slack (γ=1e4) gradient is biased from the true hard-constrained gradient — this is the
   relaxation, NOT a bug and NOT a solve-tolerance issue.** A-slack and B-slack sit at **cos≈0.205, flat
   across AD tolerance** (so it is not solve accuracy), and two *independent* solvers agree
   (cos(A-slack,B-slack)=1.0). Three diagnostics pin the mechanism (`gamma_sweep.py`, `loose_bound_check.py`):
   - **γ→∞**: cos rises **0.205 (γ=1e4) → 0.535 → 0.971 → 1.000 (γ=1e8)** — the slack gradient *converges*
     to the hard gradient (a bug would not).
   - **horizon**: cos degrades **0.9998 (5 steps) → 0.776 (20) → 0.205 (50)** — the error compounds with
     rollout length.
   - **box tightness**: cos rises **0.228 (umax=1) → 0.621 (2) → 1.000, 0/64 outliers (umax=20)** — it
     vanishes once no control is active.

   **Mechanism:** a control on the bound is *pinned* in the hard box (`du/dθ=0`) but *soft* in the slack
   box (`du/dθ ≈ −(dy/dθ)/γ ≠ 0`). This per-step gradient mismatch, present only at active controls,
   **accumulates through the closed-loop rollout**: γ=1e4 over 50 steps compounds to cos 0.2. The total
   relative gradient error is **measured to scale exactly as 1/γ** (`gamma_scaling.py`): rel-err
   22 → 2.3 → 0.24 → 0.024 → 0.0024 for γ = 1e4…1e8, a clean 10×/decade — the Moreau-relaxation order.
   So γ=1e4 is too soft to get an accurate *gradient* over a 50-step horizon (you need γ≈1e7 for <1%
   error), even though the *solution* stays within ~1e-4 — fixed by larger γ, a shorter horizon, or a
   looser box.

3. **At loose tol (1e-1) everyone is wrong** (even A no-slack = the GT, at 0.14) — a loose solve biases
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
- **remaining cause: the DIRECT backward is numerically imprecise at a near-active constraint.** Sample
  45's forward needed **1942 ADMM iters vs ~400** for clean samples (a near-active constraint, multiplier
  ≈0.04 — strict comp holds, barely). Sweeping the barrier **κ→0** (`kappa_sweep45.py`) confirms the
  hard gradient is *well-defined*: the barrier gradient on sample 45 stays **cos=1.0000 down to κ=1e-10**
  (it does NOT degrade toward 0.034). So the limit is not ill-conditioned and the FD is right — it is
  specifically the `DIRECT_CUDSS_FFI` backward's KKT solve that loses accuracy here, while the barrier
  (κ≤1e-6) is both tight enough to match the hard gradient and well-conditioned enough to compute it.

**Bottom line: the FD ground truth is correct.** On the only sample where it looked suspicious it
agrees with the independent log-barrier gradient (cos 1.0); the discrepancy is in the hard-box
*analytic* backward, not the finite differences. On the other 63/64, FD vs analytic agree to 0.15%.

**This DIRECT-backward imprecision is systematic but rare** (`directbwd_recurrence.py`, hard-box DIRECT
vs barrier gradient across 4 seeds): **11/256 = 4.3%** of samples (1–4 per seed, always near-active
constraints, cos down to −0.26), and on *every* one the barrier gradient is correct. So the hard-box
gradient is faithful *in principle* (well-defined — confirmed by the κ→0 sweep) but the
`DIRECT_CUDSS_FFI` backward **fails numerically on ~4% of near-active samples**, whereas the log-barrier
(κ=1e-6) is robust on all of them. Net: the **log-barrier wins on all three counts** — faithful (cos 1.0,
unlike the Moreau slack's 0.2), no horizon-compounding bias, and numerically robust where the hard-box
DIRECT backward is not.

**Root cause of the DIRECT-backward imprecision — a binary active set vs a weakly-active constraint**
(`directbwd_rootcause.py`, `directbwd_threshold.py`, code `backward_kkt_jax.py:144`):
- **Not a solve issue.** `DIRECT_CUDSS_FFI` and `DIRECT_JAX_DENSE` (FFI cuDSS vs pure-JAX x64 dense) give
  the *identical* wrong cos 0.033 — so it is not cuDSS precision, and not ill-conditioning (those would
  diverge between solvers). Both accurately solve the *same* KKT, which encodes the wrong gradient.
- **Not brittle.** sample-45 cos is stable (0.033) across forward tol 1e-7…1e-11 — a *systematic*
  formulation error, not a precision-sensitive active-set flip.
- **The formulation.** The DIRECT backward builds an **active-set-only KKT** (inactive inequality rows
  zeroed, `-eps_reg` on their diagonal): a constraint is fully pinned or dropped, **ignoring the
  multiplier magnitude**. At a **weakly-active** constraint (active `s≈0` but tiny multiplier `y≈0.04` —
  strict comp holds, barely) it pins the control fully, but the true sensitivity lets it move a little.
  The barrier's smooth complementarity weight `W = y/(s + y/κ)` scales with `y`, so it treats `y=0.04` as
  weakly binding and gets the gradient right. So the ~4% imprecision is a fundamental limitation of
  **binary active-set differentiation at weakly-active constraints**, cured by smooth (barrier) weighting.

Scripts: `closed_loop_gradient_accuracy.py` (main), `fd_diagnose.py` (noise-floor),
`strict_comp_check.py` (κ/√κ degeneracy), `fwd_conv_check.py` (forward convergence). Data:
`results/gradient_accuracy.npz`, `results/strict_comp_check.npz`.
