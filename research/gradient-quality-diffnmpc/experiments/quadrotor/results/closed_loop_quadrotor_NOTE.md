# Quadrotor closed-loop — status & correction

**CORRECTION (2026-06-24).** An earlier version of this note claimed the quadrotor closed-loop
differentiable rollout was *intractable*. **That was wrong — a GPU-contention artifact.** The
"timeouts" happened because a previously killed run left processes/GPU memory occupied, slowing the
subsequent runs. **Clean runs are tractable**: with the GPU idle, the n=2 probe compiles+runs in
~70s (sqp=1) to ~120s (sqp=10); the full n=8 study runs (AD ~82s; the FD is the slow part, ~2300s,
because it is 17 cost-weights × 4 eps × a 50-step sqp=10 rollout — slow, not intractable). Lesson:
always isolate GPU contention before declaring a compute boundary.

## Setup

nu=4 nonlinear quadrotor (13 states, RK4, hover regulation), 50-step closed-loop, MPC horizon=12,
sqp=10 (converged: closed-loop cost drops 224→~20 vs sqp=1), per-sample (n=8, seed 0). A-hardbox
(use_slack=False) vs A-slack (use_slack=True, γ=1e4), each AD vs convergence-checked FD of its own
rollout. Driver: `closed_loop_quadrotor.py` (`--slack`, `--umax`, `--sqp_iter`).

## Result so far (umax=1.2 — mild box)

- **A-hardbox**: 8/8 FD-flagged, but **cos 0.976–1.0** (high), rel_l2 ≤ 0.22 — *not* the wild
  negative-cosine pathology seen on the linear/cartpole tight-box cases.

This looks like the **mild-box regime** (umax=1.2 ≈ 1.2× hover thrust, so the controls rarely
saturate hard → few active-set crossings — analogous to the cartpole umax=2 case, which was also
clean). The 8/8 flags may also be partly sqp=10 under-convergence noise (the FD plateau is marginal).
A **tight-box** run (smaller umax, forcing saturation) is the decisive test — pending; the A-slack
control at umax=1.2 is also pending (run in progress).

## Open

Tight-box quadrotor (umax small enough to saturate) to see if the hard-box pathology + slack fix
appear with nu=4 nonlinear (as they did on the cartpole tight box). The FD is slow (~38 min/run at
17 weights × 4 eps), so reduce eps count / n for the tight-box sweep.
