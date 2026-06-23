# Central-Path (Closed-Form Log-Barrier Retraction + Elastic Slack) ADMM — Forward Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a central-path ADMM forward solve in JAX whose inequality update is the **closed-form elastic log-barrier retraction** `s = b_Γ(r)`, `ξ = κ/(γs)` (the formula we derived), solving the linear system with **cuDSS**, and verify its converged primal solution matches the current solver as `κ → 0` — against the **soft-box** solver (`use_slack=True`, matched `γ`) since the elastic slack converges to the quadratic-penalty solution.

**Architecture:** TurboMPC's ADMM handles the box `l ≤ Gx ≤ u` in the `z_g`-update (currently a hard projection). The closed-form retraction is **one-sided**, so we first **stack the box into one-sided rows** `[G;−G]x ≤ [u;−l]` (helper `to_one_sided`). Then the `z_g`-update becomes the closed-form elastic retraction per row: `s = b_Γ(r)`, `Γ = κ(1/ρ+1/γ)`, `ξ = κ/(γs)`, `z_g = h − s + ξ` — one `√`, no iteration. The x-update is unchanged (block-tridiagonal Schur, **cuDSS** backend). Forward solve only. New code is standalone under the research project; `diffmpc2/` stays pristine. As `κ→0`: at inactive rows `ξ→0, z_g→z̃`; at active rows `z_g→h+(z̃−h)·ρ/(γ+ρ)` — the **soft** solution, identical to TurboMPC's `frac`-blend slack. That soft solution (matched `γ`) is the equivalence target; `γ→∞` recovers the hard box.

**Tech Stack:** JAX (x64), `turbompc` (diffmpc2), **cuDSS FFI Schur backend** (`SchurSolverBackend.CUDSS_FFI`, JAX_LOOP family), pytest. GPU required.

## Global Constraints

- **x64 required:** every module/test sets `jax.config.update("jax_enable_x64", True)` before importing `turbompc`.
- **cuDSS / GPU required.** Use `SchurSolverBackend.CUDSS_FFI` with `AdmmBackend.JAX_LOOP` (NOT the fused cuDSS path — it bakes the z-update into the kernel). Needs: RTX 5090 (sm_120), the rebuilt `diffmpc2/build/ffi/libcudss_blktridi_ffi.so` (already built), `LD_LIBRARY_PATH`, and an **eager import** of `turbompc.solvers.linear_systems_solvers.cudss_ffi_backend` to register the FFI handler.
- **Every run command** is prefixed with `export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"` (see `CARTPOLE_COUPLING_HANDOFF.md` §2 if that file is missing).
- **Keep `diffmpc2/` pristine:** all new files under `research/gradient-quality-diffnmpc/experiments/cartpole/`. Reuse `turbompc` by import only.
- **`sys.path` shim:** prepend the `diffmpc2/` checkout root (mirror `benchmark_cartpole_coupling.py:62-70`).
- **One-sided rows:** the closed-form retraction `b_Γ` is one-sided; the two-sided box is stacked into one-sided rows by `to_one_sided`. Fixed `κ` (`target_kappa`); **no κ-annealing**.
- **Equivalence target:** the current `ADMMSolver` with `use_slack=True` and the **same** `slack_penalization_weight = γ` (soft box). As `κ→0` the elastic retraction → this soft solution. (`γ→∞` ⇒ hard box, matching `use_slack=False`; we test the soft case since the slack is the point.)

---

## File Structure

- `research/gradient-quality-diffnmpc/experiments/cartpole/central_path_admm.py` — NEW. `retraction_map` (closed-form `b_γ`), `elastic_retraction` (→ `(z_g, ξ)`), `to_one_sided` (stack box → one-sided rows), `solve_qp_central_path` (forward loop, cuDSS Schur). Eager cuDSS FFI registration + the `turbompc` reuse imports live here.
- `research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py` — NEW. pytest: retraction/slack unit tests (Task 1), loop convergence on a toy OCP (Task 2), cartpole QP fixture + soft reference (Task 3), the equivalence test (Task 4).

---

### Task 1: Closed-form elastic log-barrier retraction

**Files:**
- Create: `research/gradient-quality-diffnmpc/experiments/cartpole/central_path_admm.py`
- Test: `research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py`

**Interfaces:**
- Produces: `retraction_map(v, gamma) -> jnp.ndarray` (`b_γ(v)=(v+√(v²+4γ))/2`, stable, `→max(v,0)` as `γ→0`); `elastic_retraction(z_tilde, h, kappa, rho, slack_weight) -> (z_g, xi)` for one-sided rows `Gx ≤ h` (`z_g`, `xi` same shape as `z_tilde`; `z_g` is the consensus value of `Gx`, `xi ≥ 0`).

- [ ] **Step 1: Write the module header + retraction functions**

Create `central_path_admm.py`:

```python
"""Central-path (closed-form log-barrier retraction + elastic slack) ADMM forward solve.

Reuses TurboMPC's structured Schur x-update (cuDSS); swaps only the inequality
z-update for a closed-form elastic log-barrier retraction over one-sided rows.
Forward solve only.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)  # x64 required (Global Constraints)
import jax.numpy as jnp

# --- sys.path shim: resolve `turbompc` (and `tests`) from the diffmpc2 checkout ---
_DL_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))  # diffmpc-learning/
_SOLVER_ROOT = os.path.join(_DL_ROOT, "diffmpc2")
if _SOLVER_ROOT not in sys.path:
    sys.path.insert(0, _SOLVER_ROOT)

# Eager cuDSS FFI registration (loads libcudss_blktridi_ffi.so; needs LD_LIBRARY_PATH).
import turbompc.solvers.linear_systems_solvers.cudss_ffi_backend  # noqa: E402,F401

from turbompc.solvers.admm.admm import (  # noqa: E402
    compute_S_Phiinv, compute_gamma, _apply_C_parts, _apply_G,
)
from turbompc.solvers.qp_data import QPData, QPInequalityBlocks  # noqa: E402


def _inf_norm(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(a)) if a.size else jnp.asarray(0.0, a.dtype)


def retraction_map(v, gamma):
    """b_gamma(v) = (v + sqrt(v^2 + 4*gamma))/2.  Stable; b_g(v)*b_g(-v)=gamma; -> max(v,0) as gamma->0."""
    gamma = jnp.asarray(gamma, v.dtype)
    sq = jnp.sqrt(v * v + 4.0 * gamma)
    out = jnp.where(v >= 0.0, 0.5 * (v + sq), 2.0 * gamma / (sq - v))  # stable branch for v<0
    return jnp.where(gamma == 0.0, jnp.maximum(v, 0.0), out)


def elastic_retraction(z_tilde, h, kappa, rho, slack_weight):
    """Closed-form elastic log-barrier retraction for one-sided rows  Gx <= h.

    Slack s = h - Gx + xi >= 0 (barrier -kappa*log s), elastic xi >= 0 penalized by
    gamma = slack_weight. Joint (s, xi) minimizer is closed form:
        r = h - z_tilde;   s = b_Gamma(r), Gamma = kappa*(1/rho + 1/gamma);
        xi = kappa/(gamma*s);   z_g = h - s + xi   (consensus value of Gx).
    Returns (z_g, xi). As kappa->0: inactive rows -> z_g=z_tilde, xi=0; active rows
    -> z_g = h + (z_tilde - h)*rho/(gamma+rho) (the soft/quadratic-penalty solution).
    """
    dtype = z_tilde.dtype
    kappa = jnp.asarray(kappa, dtype)
    rho = jnp.asarray(rho, dtype)
    gamma = jnp.asarray(slack_weight, dtype)
    Gamma = kappa * (1.0 / rho + 1.0 / gamma)
    r = h - z_tilde
    sq = jnp.sqrt(r * r + 4.0 * Gamma)
    s = jnp.where(r >= 0.0, 0.5 * (r + sq), 2.0 * Gamma / (sq - r))        # s = b_Gamma(r) > 0
    s_minus_r = jnp.where(r >= 0.0, 2.0 * Gamma / (sq + r), s - r)         # s - r, stable for r>>0
    xi = kappa / (gamma * s)
    z_g = z_tilde - s_minus_r + xi                                        # == h - s + xi (no cancellation)
    return z_g, xi
```

- [ ] **Step 2: Write the failing retraction/slack tests**

Create `test_central_path_admm.py`:

```python
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from central_path_admm import retraction_map, elastic_retraction  # noqa: E402


def test_retraction_map_complementarity_and_relu_limit():
    v = jnp.asarray(np.linspace(-5, 5, 21))
    g = 0.3
    assert float(jnp.max(jnp.abs(retraction_map(v, g) * retraction_map(-v, g) - g))) < 1e-10
    assert float(jnp.max(jnp.abs(retraction_map(v, 1e-9) - jnp.maximum(v, 0.0)))) < 1e-4


def test_elastic_retraction_inactive_row_recovers_target_as_kappa_to_zero():
    h = jnp.array([2.0])
    z_tilde = jnp.array([0.5])          # below the bound -> inactive
    z_g, xi = elastic_retraction(z_tilde, h, kappa=1e-8, rho=0.3, slack_weight=1.0)
    assert float(jnp.abs(z_g[0] - 0.5)) < 1e-4 and float(xi[0]) < 1e-4


def test_elastic_retraction_active_row_matches_soft_blend_and_slack_engages():
    h = jnp.array([2.0])
    z_tilde = jnp.array([7.0])          # above the bound -> active/violated
    rho, gamma, kappa = 0.3, 0.1, 1e-6
    z_g, xi = elastic_retraction(z_tilde, h, kappa, rho, gamma)
    soft = h[0] + (z_tilde[0] - h[0]) * rho / (gamma + rho)   # frac-blend / soft solution
    assert float(jnp.abs(z_g[0] - soft)) < 1e-3
    assert float(xi[0]) > 0.0 and float(z_g[0]) > float(h[0])  # slack engages, constraint relaxed
```

- [ ] **Step 3: Run the tests**

```bash
cd research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
python -m pytest test_central_path_admm.py -k "retraction_map or elastic_retraction" -v
```
Expected: all three PASS. (Importing the module exercises cuDSS FFI registration; "Could not find libcudss_blktridi_ffi.so" → rebuild per `CARTPOLE_COUPLING_HANDOFF.md` §2.)

- [ ] **Step 4: Commit**

```bash
git add research/gradient-quality-diffnmpc/experiments/cartpole/central_path_admm.py \
        research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py
git commit -m "feat(central-path): closed-form elastic log-barrier retraction"
```

---

### Task 2: `to_one_sided` + central-path ADMM forward loop (cuDSS Schur)

**Files:**
- Modify: `research/gradient-quality-diffnmpc/experiments/cartpole/central_path_admm.py`
- Test: `research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py`

**Interfaces:**
- Consumes: `elastic_retraction` (Task 1); `compute_S_Phiinv(qp_data, rho_f, sigma, rho_ineq)`, `compute_gamma(...)`, `_apply_C_parts(qp_data, x) -> (row0 (n0,), rows (N,nx))`, `_apply_G(qp_data, x) -> (N+1,m)` (from `turbompc.solvers.admm.admm`); a cuDSS Schur solver from `make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=...)` exposing `.solve(schur, gammas, zs_guess) -> (x_blocks, debug)`; `QPInequalityBlocks`, `dataclasses.replace`.
- Produces: `to_one_sided(qp_data, slack_weight, big=1e7) -> QPData` (box `l≤Gx≤u` → one-sided `[G;−G]x ≤ [u;−l]`, with `use_slack_variables=True`, `slack_penalization_weight=slack_weight`); `solve_qp_central_path(qp_data, schur_solver, *, target_kappa, slack_weight, rho_bar=0.1, sigma=1e-6, rho_f_factor=1000.0, max_iter=20000, tol=1e-9) -> (x_blocks (N+1, nx+nu), info dict)` with keys `iters`, `delta`, `prim_res`, `xi_max`. **`qp_data` passed to the loop must already be one-sided** (use `to_one_sided`).

- [ ] **Step 1: Append `to_one_sided` and `solve_qp_central_path`**

Add to `central_path_admm.py`:

```python
def to_one_sided(qp_data: QPData, slack_weight: float, big: float = 1.0e7) -> QPData:
    """Stack the two-sided box l<=Gx<=u into one-sided rows [G;-G]x <= [u;-l].

    Upper bounds become u' = [u; -l]; lower bounds are inert (-big). Enables the
    elastic slack on the QP (use_slack_variables=True, slack_penalization_weight).
    """
    G, l, u = qp_data.ineq.G, qp_data.ineq.l, qp_data.ineq.u   # (N+1,m,n), (N+1,m), (N+1,m)
    G1 = jnp.concatenate([G, -G], axis=1)                      # (N+1, 2m, n)
    u1 = jnp.concatenate([u, -l], axis=1)                      # (N+1, 2m) upper bounds
    l1 = jnp.full_like(u1, -float(big))                        # inert lower bounds
    ineq = QPInequalityBlocks(
        G=G1, l=l1, u=u1,
        slack_penalization_weight=jnp.asarray(slack_weight, u1.dtype),
        use_slack_variables=True,
    )
    return dataclasses.replace(qp_data, ineq=ineq)


def solve_qp_central_path(
    qp_data: QPData,
    schur_solver,
    *,
    target_kappa: float,
    slack_weight: float,
    rho_bar: float = 0.1,
    sigma: float = 1.0e-6,
    rho_f_factor: float = 1000.0,
    max_iter: int = 20000,
    tol: float = 1.0e-9,
):
    """ADMM forward solve with the closed-form elastic retraction z-update (fixed kappa).

    qp_data MUST be one-sided (Gx <= u; use to_one_sided). Identical to TurboMPC's
    ADMM except the inequality z_g-update is `elastic_retraction` instead of a box
    projection. Over-relaxation alpha=1, no adaptive rho. Linear system: cuDSS Schur.
    """
    dtype = qp_data.cost.q.dtype
    Np1, n = qp_data.cost.D.shape[0], qp_data.cost.D.shape[1]
    N = Np1 - 1
    nx = qp_data.eq.A_minus.shape[1]
    n0 = qp_data.eq.A0.shape[0]
    m = qp_data.ineq.G.shape[1]
    rho_bar = jnp.asarray(rho_bar, dtype)
    rho_f = rho_bar * rho_f_factor
    h = qp_data.ineq.u  # one-sided upper bounds (N+1, m)

    schur = compute_S_Phiinv(qp_data, rho_f, sigma, rho_ineq=rho_bar)

    x0 = jnp.zeros((Np1, n), dtype)
    y_g0 = jnp.zeros((Np1, m), dtype)
    y_f_00 = jnp.zeros((n0,), dtype)
    y_f_dyn0 = jnp.zeros((N, nx), dtype)
    if m:
        z_g0, _ = elastic_retraction(_apply_G(qp_data, x0), h, target_kappa, rho_bar, slack_weight)
    else:
        z_g0 = jnp.zeros((Np1, 0), dtype)
    inf = jnp.asarray(jnp.inf, dtype)
    zero = jnp.asarray(0.0, dtype)
    init = (jnp.asarray(0, jnp.int32), x0, y_g0, y_f_00, y_f_dyn0, z_g0, inf, zero)

    def cond(s):
        it, _, _, _, _, _, delta, _ = s
        return jnp.logical_and(it < max_iter, delta > tol)

    def body(s):
        it, x, y_g, y_f_0, y_f_dyn, z_g, _, _ = s
        gammas = compute_gamma(qp_data, x, z_g, y_g, y_f_0, y_f_dyn,
                               rho_f=rho_f, rho_ineq=rho_bar, sigma=sigma)
        x_new, _ = schur_solver.solve(schur, gammas, x)
        Cx0, Cx = _apply_C_parts(qp_data, x_new)
        ineq_vals = _apply_G(qp_data, x_new)
        if m:
            z_tilde = ineq_vals + y_g / rho_bar
            z_g_new, xi_g = elastic_retraction(z_tilde, h, target_kappa, rho_bar, slack_weight)
            y_g_new = y_g + rho_bar * (ineq_vals - z_g_new)
            xi_max = _inf_norm(xi_g)
        else:
            z_g_new, y_g_new, xi_max = z_g, y_g, zero
        y_f_0_new = y_f_0 + rho_f * (Cx0 - qp_data.eq.c0)
        y_f_dyn_new = y_f_dyn + rho_f * (Cx - qp_data.eq.c)
        delta = jnp.maximum(_inf_norm(x_new - x), _inf_norm(z_g_new - z_g))
        return (it + 1, x_new, y_g_new, y_f_0_new, y_f_dyn_new, z_g_new, delta, xi_max)

    it, x, _, _, _, z_g, delta, xi_max = jax.lax.while_loop(cond, body, init)
    Cx0, Cx = _apply_C_parts(qp_data, x)
    prim = jnp.maximum(_inf_norm(Cx0 - qp_data.eq.c0), _inf_norm(Cx - qp_data.eq.c))
    if m:
        prim = jnp.maximum(prim, _inf_norm(_apply_G(qp_data, x) - z_g))
    return x, {"iters": it, "delta": delta, "prim_res": prim, "xi_max": xi_max}
```

- [ ] **Step 2: Write the failing loop test (toy OCP, stacked one-sided)**

Add to `test_central_path_admm.py`:

```python
from central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402

from tests.helpers.problem_fixtures import cost_blocks_from_qr  # noqa: E402
from turbompc.solvers.qp_data import qpdata_from_ocp_blocks  # noqa: E402
from turbompc.solvers.qp_utils import ZShape  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend, AdmmBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402

_PCG = {"max_iter": 400, "tol_epsilon": 1.0e-12}  # required kwarg of make_schur_solver (ignored by cuDSS)


def _toy_qp(bound):
    rng = np.random.default_rng(1)
    N, nx, nu = 3, 1, 1
    As_next = jnp.asarray(0.1 * rng.standard_normal((N, nx, nx)))
    Bs_next = jnp.asarray(0.1 * rng.standard_normal((N, nx, nu)))
    As = jnp.asarray(0.1 * rng.standard_normal((N, nx, nx)))
    Bs = jnp.asarray(0.1 * rng.standard_normal((N, nx, nu)))
    Cs = jnp.asarray(0.5 * rng.standard_normal((N + 1, nx)))
    Qm = jnp.tile(jnp.eye(nx)[None], (N + 1, 1, 1)); Rm = jnp.tile(jnp.eye(nu)[None], (N + 1, 1, 1))
    Rd = jnp.tile(jnp.eye(nu)[None], (N, 1, 1)) * 0.05
    qv = jnp.asarray(0.5 * rng.standard_normal((N + 1, nx))); rv = jnp.asarray(0.5 * rng.standard_normal((N + 1, nu)))
    D, E, q = cost_blocks_from_qr(Qm, Rm, Rd, qv, rv)
    A0 = jnp.concatenate([jnp.eye(nx), jnp.zeros((nx, nu))], axis=1)
    lo = jnp.concatenate([-1e7 * jnp.ones((N + 1, nx)), -bound * jnp.ones((N + 1, nu))], -1)
    hi = jnp.concatenate([1e7 * jnp.ones((N + 1, nx)), bound * jnp.ones((N + 1, nu))], -1)
    ineq_blocks = jnp.tile(jnp.eye(nx + nu)[None], (N + 1, 1, 1))
    qp = qpdata_from_ocp_blocks(D=D, E=E, q=q, A0=A0, c0=Cs[0], As_next=As_next, Bs_next=Bs_next,
                                As=As, Bs=Bs, c_dyn=Cs[1:], ineq_blocks=ineq_blocks, ineq_l=lo, ineq_u=hi)
    return qp, N, nx, nu


def test_central_path_loop_converges_on_cudss():
    qp, N, nx, nu = _toy_qp(bound=0.05)
    qp1 = to_one_sided(qp, slack_weight=1e2)
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    x, info = solve_qp_central_path(qp1, schur, target_kappa=1e-4, slack_weight=1e2,
                                    rho_bar=0.1, max_iter=20000, tol=1e-10)
    assert float(info["prim_res"]) < 1e-6        # dynamics + consensus feasible
    assert int(info["iters"]) < 20000            # converged before the cap
    assert jnp.all(jnp.isfinite(x))
```

- [ ] **Step 3: Run the loop test**

```bash
cd research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
python -m pytest test_central_path_admm.py -k central_path_loop_converges_on_cudss -v
```
Expected: PASS. Troubleshooting: `prim_res ≥ 1e-6` → raise `max_iter` to 50000; cuDSS errors on this tiny size → bump the toy to `N=8, nx=2, nu=1`.

- [ ] **Step 4: Commit**

```bash
git add research/gradient-quality-diffnmpc/experiments/cartpole/central_path_admm.py \
        research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py
git commit -m "feat(central-path): one-sided stacking + elastic-retraction ADMM loop (cuDSS)"
```

---

### Task 3: Cartpole QP fixture + soft-box reference (`use_slack=True`, cuDSS)

**Files:**
- Modify: `research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py`

**Interfaces:**
- Consumes: `build_cartpole_problem` (from `benchmark_cartpole_coupling`); `OptimalControlProblem`, `TurboMPCSolver`, `ForwardBackend`, `load_solver_params`, `ADMMSolver`, `ZShape`, `make_schur_solver(SchurSolverBackend.CUDSS_FFI,...)`, `AdmmBackend.JAX_LOOP`; `to_one_sided`.
- Produces (test-local): `_cartpole_one_sided_qp(umax, slack_weight) -> qp_data` (two-sided cartpole QP via `_build_qp_data`, then `to_one_sided`); `_soft_reference(qp_data, N, nx, nu, slack_weight) -> (states, controls, stats)` via the current ADMM with `use_slack=True`.

- [ ] **Step 1: Write the failing cartpole-fixture test**

Add to `test_central_path_admm.py`:

```python
from benchmark_cartpole_coupling import build_cartpole_problem  # noqa: E402 (reuses its path shim)
from turbompc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.solvers.admm import ADMMSolver  # noqa: E402

NX, NU = 4, 1
_POLE_DOWN = jnp.array([0.0, 0.0, jnp.pi, 0.0])
_GAMMA = 1.0e2  # slack penalty (soft box); large => approaches hard box


def _cartpole_one_sided_qp(umax, slack_weight):
    dynamics, pp = build_cartpole_problem(horizon=25, umax=umax, dt=0.04)
    pp["initial_state"] = _POLE_DOWN
    ocp = OptimalControlProblem(dynamics=dynamics, params=pp)
    sp = load_solver_params("turbompc.yaml")
    solver = TurboMPCSolver(
        program=ocp, params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
    )
    ig = solver.initial_guess(pp)
    qp_two_sided = solver._build_qp_data(ig.states, ig.controls, pp)
    return to_one_sided(qp_two_sided, slack_weight=slack_weight)


def _soft_reference(qp, N, nx, nu, slack_weight):
    """Current solver's quadratic-slack soft solution on the same one-sided QP."""
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, nx, nu, pcg_params=_PCG)
    ref = ADMMSolver(
        zshape=ZShape(horizon=N, num_states=nx, num_controls=nu),
        schur_solver=schur, pcg_params=_PCG,
        sigma=1e-6, max_iter=50000, eps_abs=1e-11, eps_rel=1e-9,
        rho_f_factor=1000.0, admm_backend=AdmmBackend.JAX_LOOP, use_slack=True,
    )
    (states, controls), stats, _ = ref.solve(qp, rho_bar=0.1, slack_weight=slack_weight)
    return states, controls, stats


def test_cartpole_soft_reference_converges_and_bound_engages():
    qp = _cartpole_one_sided_qp(umax=2.0, slack_weight=_GAMMA)
    assert qp.ineq.use_slack_variables is True
    N = qp.cost.D.shape[0] - 1
    states, controls, stats = _soft_reference(qp, N, NX, NU, _GAMMA)
    assert int(stats.num_iter) > 0
    # the swing-up pushes the control near/over the soft bound (slack engages)
    assert float(jnp.max(jnp.abs(controls))) >= 0.9 * 2.0
```

- [ ] **Step 2: Run the fixture test**

```bash
cd research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
python -m pytest test_central_path_admm.py -k cartpole_soft_reference -v
```
Expected: PASS. Bounded fixes: `load_solver_params("turbompc.yaml")` arg → copy the exact call from `benchmark_cartpole_coupling.py`; `_build_qp_data` args → match `turbompc_solver.py:1101`; **slack not engaging in the reference** → confirm whether `ADMMSolver.solve` reads `slack_weight=` (passed here) vs `qp.ineq.slack_penalization_weight` (set by `to_one_sided`) — set/keep both consistent; bound not pushed → lower `umax` to `1.0`.

- [ ] **Step 3: Commit**

```bash
git add research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py
git commit -m "test(central-path): cartpole one-sided QP fixture + soft (use_slack) cuDSS reference"
```

---

### Task 4: Equivalence — central-path (κ→0) matches the soft solver

**Files:**
- Modify: `research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py`

**Interfaces:**
- Consumes: `solve_qp_central_path` (Task 2), `_cartpole_one_sided_qp` / `_soft_reference` (Task 3), `make_schur_solver(SchurSolverBackend.CUDSS_FFI,...)`.

- [ ] **Step 1: Write the failing equivalence test**

Add to `test_central_path_admm.py`:

```python
def test_central_path_matches_soft_solver_as_kappa_to_zero():
    qp = _cartpole_one_sided_qp(umax=2.0, slack_weight=_GAMMA)
    N = qp.cost.D.shape[0] - 1
    states_ref, controls_ref, _ = _soft_reference(qp, N, NX, NU, _GAMMA)
    xref = jnp.concatenate([states_ref, controls_ref], axis=-1)   # (N+1, nx+nu)

    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)
    errs = {}
    for kappa in (1e-2, 1e-4, 1e-6):
        x_cp, info = solve_qp_central_path(qp, schur, target_kappa=kappa, slack_weight=_GAMMA,
                                           rho_bar=0.1, max_iter=50000, tol=1e-11)
        assert float(info["prim_res"]) < 1e-6, f"central-path did not converge at kappa={kappa}"
        errs[kappa] = float(jnp.max(jnp.abs(x_cp - xref)))

    # error -> 0 monotonically as kappa -> 0 (central path -> the SAME soft solution)
    assert errs[1e-2] > errs[1e-4] > errs[1e-6]
    assert errs[1e-6] < 1e-4
```

- [ ] **Step 2: Run the equivalence test**

```bash
cd research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
python -m pytest test_central_path_admm.py -k matches_soft_solver -v
```
Expected: PASS — `errs` decreasing in κ, `errs[1e-6] < 1e-4`. Troubleshooting: marginal (2) → tighten both solvers (`eps_abs`/`eps_rel` & `max_iter` for the reference; `tol` & `max_iter` for central-path) below the comparison tol, then (only if a floor is demonstrated) loosen `1e-4`. If the *reference* slack and the *retraction* slack disagree even at tiny κ, re-check that both use the same `γ` (`_GAMMA`) and that the reference's slack is actually active (Task 3 troubleshooting).

- [ ] **Step 3: (Optional) bare-barrier hard-box cross-check**

Add to `test_central_path_admm.py` (cheap sanity check that `γ→∞` recovers the hard box):

```python
def test_bare_barrier_recovers_hard_box():
    qp = _cartpole_one_sided_qp(umax=2.0, slack_weight=1e8)          # gamma huge => ~hard
    N = qp.cost.D.shape[0] - 1
    states_ref, controls_ref, _ = _soft_reference(qp, N, NX, NU, 1e8)  # soft with huge gamma ~ hard
    xref = jnp.concatenate([states_ref, controls_ref], axis=-1)
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params=_PCG)
    x_cp, info = solve_qp_central_path(qp, schur, target_kappa=1e-6, slack_weight=1e8,
                                       rho_bar=0.1, max_iter=50000, tol=1e-11)
    assert float(info["prim_res"]) < 1e-6
    assert float(jnp.max(jnp.abs(x_cp - xref))) < 1e-4
    assert float(info["xi_max"]) < 1e-3                              # slack ~ off at huge gamma
```

- [ ] **Step 4: Run the full file + commit**

```bash
cd research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
python -m pytest test_central_path_admm.py -v
git add research/gradient-quality-diffnmpc/experiments/cartpole/test_central_path_admm.py
git commit -m "test(central-path): elastic-retraction ADMM matches soft solver as kappa->0 on cartpole"
```

---

## Notes for the executor

- **Closed form, not Newton:** the inequality update is `s = b_Γ(r)` (one `√`) + `ξ = κ/(γs)`, applied per one-sided row. This is "the retraction" we derived. The two-sided box is made one-sided by `to_one_sided` (stack `[G;−G]`).
- **Why the soft reference:** with a finite slack penalty `γ`, the elastic retraction's `κ→0` limit is the quadratic-penalty (soft) solution, identical to TurboMPC's `frac`-blend slack — so the equivalence target is `use_slack=True` with the same `γ`. The bare barrier (`γ→∞`, Task 4 Step 3) recovers the hard box. This is the same elastic slack we discussed (`s = h − Gx + ξ`), now one-sided per stacked row — and it is exactly the machinery one-sided **obstacle** constraints will reuse.
- **cuDSS, not fused:** `SchurSolverBackend.CUDSS_FFI` keeps the JAX loop so the custom retraction z-update lives in JAX; the fused cuDSS path can't host it.
- **Barrier doesn't touch the x-update:** `compute_S_Phiinv`/`compute_gamma` reused unchanged (`rho_ineq=rho_bar`); only the z-update line changes.
- **Out of scope (separate follow-up plans):** the κ-relaxed *backward*/VJP, κ-annealing, and integration into `diffmpc2/`'s production `admm.py`.

## Self-review (done while writing)

- **Spec coverage:** central-path iteration in JAX → Tasks 1–2; **cuDSS** (mod 1) → Global Constraints + `SchurSolverBackend.CUDSS_FFI` in Tasks 2–4; **slack** (mod 2) → `elastic_retraction` + `to_one_sided` (Task 1–2), soft reference (Task 3), equivalence (Task 4); "(B) closed-form retraction" → `b_Γ`, no Newton; "same solution as current solver" → Task 4 vs `use_slack=True` (and Step 3 vs hard); "cartpole" → Tasks 3–4. ✓
- **Placeholders:** none — full code + exact commands (with `LD_LIBRARY_PATH`).
- **Type consistency:** `retraction_map(v,gamma)`, `elastic_retraction(z_tilde,h,kappa,rho,slack_weight)->(z_g,xi)`, `to_one_sided(qp,slack_weight)->QPData`, `solve_qp_central_path(qp,schur,*,target_kappa,slack_weight,...)` all referenced consistently across tasks; one-sided rows everywhere downstream of `to_one_sided`; `x_blocks (N+1,nx+nu)` vs reference `(states (N+1,nx), controls (N+1,nu))` concatenated for comparison. ✓
- **Risk flagged inline:** cuDSS toy-size fallback (Task 2); `load_solver_params`/`_build_qp_data`/slack-engaging-in-reference (`slack_weight=` vs `qp.ineq.slack_penalization_weight`) (Task 3); comparison-tolerance tightening (Task 4).
