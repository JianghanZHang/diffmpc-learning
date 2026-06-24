# Quadrotor closed-loop: a tractability boundary (not a result)

**Run 2026-06-24.** Attempted to extend the closed-loop hard-box-vs-slack gradient study to the
**nu=4 nonlinear quadrotor** (13 states, RK4, hover regulation, control box) via the same
differentiable-rollout approach (`closed_loop_quadrotor.py`, jax.grad through a 50-step rollout).

**It is intractable with this approach.** Every configuration tried failed to finish *compiling*:
- n=2, H=10, sim=20, sqp=5  → timed out (> 8m40s)
- n=2, H=12, sim=50, sqp=3  → died during compile
- n=2, H=12, sim=50, **sqp=1** → timed out (> 7m)

Since even **sqp=1** (one QP per step) and **n=2** hang, the bottleneck is **not** the SQP-iteration
unroll, the rollout length (build_rollout_fn uses lax.scan, so sim_steps does not unroll), or the
vmap width. It is the **per-step compile of the 13-state quadrotor QP grad**: the RK4 discretization
of the nonlinear quaternion dynamics, differentiated twice (the backward uses the exact Lagrangian
Hessian D + λᵀ∇²f), produces a jaxpr whose XLA compile does not complete in a usable time. The
linear (8-state) and cartpole (4-state RK4) bodies compile in ~10–30s; the quadrotor does not.

**Implication.** The closed-loop diffmpc-as-policy gradient via `jax.grad` through the unrolled
(scanned) rollout scales to small systems but **not** to a 13-state nonlinear system. A drone
closed-loop study needs a different gradient path — e.g. manual per-step adjoint chaining (avoid
re-tracing the full solve under grad), a hand-written rollout VJP that reuses the solver's own
custom_vjp per step without nesting it in a big traced graph, an analytic-Hessian backend that
compiles, or simply a much larger compile budget. That is a focused engineering effort, not an
autonomous-loop continuation.

The tractable closed-loop findings stand on the linear (nu=4) and cartpole (nu=1 nonlinear)
systems — see `../linear_system/results/closed_loop_outliers.md` and
`../cartpole/results/closed_loop_cartpole.md`. The driver `closed_loop_quadrotor.py` is kept for
whoever picks up the drone case with a tractable gradient path.
