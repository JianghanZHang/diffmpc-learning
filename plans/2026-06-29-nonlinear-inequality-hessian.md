# Nonlinear Inequality-Constraint Hessian (μᵀ∇²g) — IMPLEMENTED ✅ (2026-06-29)

**Status:** done and verified. Implemented in `external/turbompc` (canonical solver), mirroring the
dynamics-Hessian pattern, then mirrored into the local central-path backward. Approved plan:
`~/.claude/plans/serialized-juggling-cat.md`.

## What & why

The exact Lagrangian Hessian for the backward/sensitivity needs the inequality-constraint curvature
`Σᵢ μᵢ ∇²gᵢ` (Frey2025 Thm 2 / Remark 3). turbompc had the **dynamics** Hessian `λᵀ∇²f`
(`get_dynamics_lagrangian_hessian`) but **no inequality Hessian** — fine for linear/box constraints
(`∇²g=0`), wrong for nonlinear ones (obstacle). The corridor experiment confirmed linear is fine; this
adds the missing term for nonlinear constraints, **backward-only**, gated by the existing
`use_full_hessian` (faithful mirror: forward stays Gauss-Newton and converges to the same NLP-KKT;
box constraints unaffected).

## Changes

**external/turbompc**
- `turbompc/problems/optimal_control_problem.py` — new `get_inequality_lagrangian_hessian(states,
  controls, params, mus) -> (N+1,n,n)` (per-stage `jax.hessian` of `μᵀg`, reuses
  `_prepare_pointwise_inequality_params`), mirroring `get_dynamics_lagrangian_hessian`.
- `turbompc/solvers/turbompc_solver.py` — new `_augment_D_with_inequality_hessian` (mirrors
  `_augment_D_with_dynamics_hessian`; curvature multiplier = RAW forward inequality dual
  `admm_state.y_g` / `kkt_state.y_ineq`, same convention as the dynamics `y_f_dyn`); called in
  `_build_backward_qp` right after the dynamics augmentation, under `if self._use_full_hessian:`.
- `tests/python/solvers/test_inequality_hessian.py` — NEW. 2-D double integrator + obstacle (linear
  dynamics ⇒ isolates the inequality term). 3 tests pass on cuDSS.

**local (src/diffmpc_learning)**
- `solvers/backward.py` `central_path_nlp_grad` — calls `program.get_inequality_lagrangian_hessian`
  with net multiplier `y_g_net = y_g_stacked[:,:m] − y_g_stacked[:,m:]`, adds to `D_full` alongside
  `dyn_hess`; new `include_ineq_hessian=True` kwarg (ablation toggle).
- `solvers/central_path_admm.py` + `tests/conftest.py` — **switched the `turbompc` path shim from
  `diffmpc2/` → `external/turbompc/`** (CLAUDE.md canonical solver; diffmpc2 lacks the new method and
  has the sign bug). Added the linear-system benchmarking dir to conftest.
- `tests/python/solvers/test_inequality_hessian.py` — NEW. Central-path backward on the obstacle.

## Verification (cuDSS, `.venv-cudss` python, `XLA_PYTHON_CLIENT_PREALLOCATE=false`)

- **turbompc** `test_inequality_hessian.py`: 3 pass. On the obstacle: AD **with** the term vs FD →
  cos 1.0, **rel 9e-5**; **without** (Gauss-Newton) → cos 0.9988, **rel 0.124** (12% off). Term is
  correct (sign = raw `y_g`, verified by FD) and load-bearing. Existing `test_timevaryingbounds_ineqs`
  (linear) still passes (∇²g=0 ⇒ adds exactly 0).
- **local** `test_inequality_hessian.py`: 3 pass — central-path forward converges + obstacle active;
  AD (with term) matches convergence-checked FD; ablation (`include_ineq_hessian=False`) materially worse.
- **Re-validation of the diffmpc2→external switch:** existing local suites pass on external/turbompc —
  `test_central_path_admm.py` (7) and `test_backward_central_path.py` (5).

## Notes / gotchas

- turbompc's **hard active-set DIRECT backward is unstable on the nonconvex obstacle via CPU
  `JAX_DENSE`** (singular KKT at near-tangent activation → `1e240`/heap corruption). Use **cuDSS**
  (per the user's directive); it is stable. The clean turbompc verification uses linear dynamics +
  obstacle (off-axis goal, firmly active) so the active set is non-degenerate.
- Running turbompc's own tests under the `.venv-cudss` python requires `external/turbompc` first on the
  path (`PYTHONPATH=external/turbompc`) so it wins over the pip-shadowed diffmpc2.
