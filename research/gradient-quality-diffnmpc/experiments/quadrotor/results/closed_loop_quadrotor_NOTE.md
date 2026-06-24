# Quadrotor closed-loop (nu=4 nonlinear): tractable, but the FD is convergence-confounded

**Run 2026-06-24.** Extends the closed-loop hard-box-vs-slack gradient study to the **nu=4 nonlinear
quadrotor** (13 states, RK4, hover regulation, control box). Driver: `closed_loop_quadrotor.py`.

## Tractability (correcting an earlier false claim)

An earlier version of this note said the quadrotor closed-loop rollout-grad was *intractable*. **That
was wrong — a GPU-contention artifact** (a killed run left processes/memory occupied, starving the
next runs). **Clean runs are tractable**: sqp=10 n=2 ≈120 s; the full sweep AD ≈82 s. The FD is just
*slow* (~38–44 min/run: 17 cost-weights × 4 eps × a 50-step rollout), not intractable. Lesson: isolate
GPU contention before declaring a compute boundary. See [[isolate-gpu-contention-before-boundary]].

## Results

**umax=1.2 (mild box, sqp=10, n=8):** A-hardbox ≈ A-slack — both cos median ~1.0, min ~0.96, both 8/8
FD-flagged. The box at 1.2× hover barely saturates the controls (few active-set crossings), so the two
formulations are indistinguishable. Analogous to the cartpole umax=2 (mild) case.

**umax=1.05 (tight box, sqp=8, n=6):** per-sample (same x0 across the two):

| sample | hard cos | hard rel | slack cos | slack rel |
|---:|---:|---:|---:|---:|
| 0 | 0.9692 | 2.5e-1 | 0.8749 | 5.0e-1 |
| 1 | 1.0000 | 1.1e-3 | 0.9820 | 1.9e-1 |
| 2 | 1.0000 | 9.9e-3 | 1.0000 | 2.3e-3 |
| 3 | 0.9230 | 3.9e-1 | 0.8909 | 4.5e-1 |
| **4** | **0.6160** | **2.8e0** | **1.0000** | **1.5e-3** |
| 5 | 0.9999 | 1.3e-2 | 0.9999 | 1.1e-2 |

**This is mixed, and all 6/6 are FD-flagged for both.** Sample 4 is a clean instance of the
mechanism (hard box pathological cos=0.62/rel=2.8 → slack cos=1.0/rel=1.5e-3), but on samples 0 and 1
the slack is *worse* than the hard box. The mixed signal + universal FD-flagging indicates **NLP
under-convergence noise dominates the active-set effect**: sqp=8 (reduced for FD speed) does not
converge the harder nonlinear quadrotor NLP tightly enough, so both AD and FD carry solver noise and
the FD never plateaus. (The cartpole used sqp=12–15 and gave a clean A-slack cos=1.0; the quadrotor at
sqp=8 does not.)

## Conclusion

The quadrotor closed-loop is **tractable but convergence-confounded** at computationally-feasible sqp
counts. It gives a *glimpse* of the mechanism (sample 4) but **not a clean nu=4 confirmation** — both
methods show outliers and all samples FD-flag, consistent with NLP under-convergence rather than a
clean active-set comparison. A clean quadrotor result would need much tighter NLP convergence (high
sqp), which the ~40-min FD makes impractical here (would be ~2 h/pair).

**The clean confirmation of the hard-box-vs-slack mechanism stands on the linear (nu=4) and cartpole
(nu=1 nonlinear) systems** (`../linear_system/results/closed_loop_outliers.md`,
`../cartpole/results/closed_loop_cartpole.md`). The quadrotor adds: the approach scales to a 13-state
system, but a clean nu=4 *nonlinear* result is gated by NLP convergence cost, not by tractability.
Data: `closed_loop_quadrotor_{hardbox,slack}_umax{1p2,1p05}.npz`.
