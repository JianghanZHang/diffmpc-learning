# Nonlinear cartpole closed-loop: does the hard-box vs slack outlier mechanism generalize?

**Run 2026-06-24.** The linear-system study found the diffmpc closed-loop gradient outliers are a
property of the **hard active-set box** (use_slack=False), eliminated by the **soft/slack box**.
Does that hold under **nonlinear dynamics**? Cartpole (nx=4, **nu=1**, RK4, dt=0.04), 50-step
closed-loop regulation about upright, MPC horizon=15, per-sample (n=8, seed 0). Each backward's
AD (jax.grad through the rollout) vs convergence-checked FD of its own rollout. Driver:
`closed_loop_cartpole.py`. (Swing-up — θ≈π — needs ~300 SQP iters and is intractable for a
differentiable 50-step rollout grad; regulation needs only ~12–15, so AD/FD reflect converged
solves.)

| regime | backward | median cos | cos range | #cos<0.99 | FD-flagged | verdict |
|---|---|---:|---|---:|---:|---|
| **mild box** umax=2 (rarely binds) | A-hardbox | 1.00000 | — | 0 | 1/8 | clean |
| | A-slack | 1.00000 | — | 0 | 1/8 | clean |
| **tight box** umax=0.5 (saturates) | A-hardbox | — | **−0.81 … +0.88** | (all flagged) | **8/8** | **pathological** |
| | A-slack | 1.00000 | 1.0 … 1.0 | **0** | 7/8 (cos=1.0) | consistent |

## Findings

- **The hard-box pathology is gated by active-set engagement, not by linearity.** With the mild box
  (umax=2) the single control rarely saturates → few active-set crossings → both hard and slack are
  clean. With the tight box (umax=0.5) the control saturates frequently → A-hardbox becomes fully
  pathological: **8/8 FD-flagged (the rollout cost-vs-weights map is non-differentiable) with cosines
  scattered from −0.81 to +0.88.** This is the same pathology seen on the linear nu=4 system, now
  reproduced under **nonlinear** dynamics.
- **The slack box is FD-consistent (cos=1.0) everywhere**, including the tight-box regime where the
  hard box fails. Its 7/8 "flags" are benign: the near-hard soft penalty (γ=1e4) makes the FD plateau
  marginal, but the AD still matches FD to cos=1.0 / rel_l2≤5e-3 — qualitatively different from
  A-hardbox's flags (genuine non-differentiability + wrong/negative cos).
- **Why nu=1 still shows it under saturation**: even a single control that saturates and desaturates
  along a 50-step rollout produces frequent active-set switches; the hard active-set backward is
  inconsistent across them (compounded over the rollout), while the soft penalty is C¹ throughout.

## Conclusion (combined with the linear study)

The diffmpc closed-loop gradient outliers are a property of **frequently-binding hard constraints**,
present under both linear and nonlinear dynamics, and eliminated by the **soft/slack formulation**
(TurboMPC's `use_slack=True`, or B's log-barrier which builds the same soft box). The effect requires
the constraints to actually engage often: when they rarely bind (mild box), even the hard box is fine.

Caveats: A-slack/hard solve slightly different problems (hard vs γ=1e4 soft box); each AD vs its own
rollout's convergence-checked FD; n=8, one seed, regulation regime (swing-up intractable for the
differentiable rollout). B (log-barrier) on the nonlinear cartpole would need a multi-SQP custom_vjp
(forward = jittable SQP loop, backward = relaxed-KKT with the exact Lagrangian Hessian) — not built
this session; the A-slack control already isolates the slack as the mechanism.
