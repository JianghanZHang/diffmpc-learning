# Central-Path ADMM CUDA Forward Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fused on-device (cuDSS) central-path (log-barrier retraction) ADMM **forward** mode to the diffmpc2 solver, with clean **non-slack** (pure barrier) and **slack** (elastic barrier) inequality modes over **one-sided** rows, whose converged solution **matches the QP solution** and whose **iteration count matches** a JAX reference.

**Architecture:** The central-path solver is OSQP-style ADMM differing from the hard solver in **exactly one kernel — the z-update** (verified three ways; see `notes/CENTRAL_PATH_BACKWARD.md` and the PrismQP report `external/PrismQP/report/main.tex`). All new code is a **new feature branch on the `external/diffmpc2` git repo**, modifying diffmpc2's *own* ADMM kernels **in place**: add two barrier branches to `slackUpdateDevice` (built on the retraction map `b_g(r)=(r+√(r²+4g))/2`), thread a `kappa` runtime buffer + an `ineq_mode` flag (mirroring the existing `slack_weight` threading), expose it through diffmpc2's existing cuDSS FFI, and add a JAX central-path reference + tests inside diffmpc2. The two-sided box `l≤Gx≤u` is converted to one-sided rows `[G;−G]x≤[u;−l]` **upstream** (`to_one_sided`), so the kernel only ever sees `Gx≤h`. Because the new modes are gated by `ineq_mode` (0=HARD), the existing hard-box forward is **unchanged** (`ineq_mode=0`).

**Tech Stack:** CUDA 12 + cuDSS FFI (XLA custom-call), JAX (x64), diffmpc2's existing cmake/`make install` build, pytest. The JAX reference is a port of diffmpc2's own `_solve_jax_loop` with the z-update swapped (the same relationship `src/diffmpc_learning/solvers/central_path_admm.py` has to turbompc's loop — that file is the proven template to copy).

## Global Constraints

- **Location: a NEW branch on `external/diffmpc2`.** Every git command runs with `git -C external/diffmpc2 …`. Create the branch in Task 0; do **not** work on diffmpc2's current branch. All new files live under `external/diffmpc2/`.
- **Branch base (decision, Task 0):** branch from **`inequality-hessian`** (recommended — it already carries the obstacle QP fixtures in `tests/python/solvers/test_inequality_hessian.py` and the backward sign fix) unless the user says otherwise (`main` for a clean forward-only base).
- **Run env (every command):** `export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"; export XLA_PYTHON_CLIENT_PREALLOCATE=false`; python = `/home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python`; `PYTHONPATH=external/diffmpc2` (so `turbompc` resolves to *this* checkout, not the pip-installed `diffmpc`). x64 always on (`jax.config.update("jax_enable_x64", True)`).
- **cuDSS is 0.7.1**; diffmpc2's `.cu` ship the cuDSS-0.8 `cudssMatrixCreateCsr` API. The build needs the **same backport shim diffmpc2 already uses** (the `HANDOFF.md` recipe: `-D` compat macros + the `cudssMatrixCreateCsr` index-arg revert). Do **NOT commit** that `.cu` API-revert or any `.bak` files onto the branch.
- **Backward-compatible:** the new `ineq_mode` defaults to 0 (HARD) → the existing hard-box `slackUpdateDevice` path and all diffmpc2 ADMM forward tests must still pass unchanged (regression gate, Task 2 Step 8).
- **Clean slack semantics (user-directed):** when slack is OFF (`ineq_mode=1`, BARRIER) `γ` (slack_weight) must **never appear** — the pure-barrier branch must not form `1/γ` or `γ·ξ`. Not "γ→∞".
- **Two reference notions of "the QP solution":** (a) the in-repo JAX `solve_qp_central_path` at the same `(κ, γ, ρ, α, tol)` — the κ-relaxed QP solution and the iteration-count oracle; (b) the underlying hard QP, reached by the barrier mode as κ→0, cross-checked against diffmpc2's hard `ADMM_FUSED_CUDSS` forward.

---

## File structure (all under `external/diffmpc2/`)

- `turbompc/solvers/admm/central_path_retraction.py` — *new* — `retraction_map`, `elastic_retraction`, `barrier_retraction` (copied from `src/diffmpc_learning/solvers/retraction.py`, plus the pure barrier). [Task 1]
- `turbompc/solvers/admm/central_path_jax.py` — *new* — `to_one_sided`, `solve_qp_central_path(use_slack=...)` (copied/adapted from `src/diffmpc_learning/solvers/central_path_admm.py`; imports resolve in-repo, drop the sys.path shim). The JAX oracle + clean two-branch reference. [Task 1]
- `turbompc/solvers/admm/csrc/admm_math.cuh` — *modify* `slackUpdateDevice` (add `retraction_b` + BARRIER/ELASTIC branches). [Task 2]
- `turbompc/solvers/admm/csrc/admm_cudss.cu` — *modify* (kernel sig, per-batch `kappa` load, call sites, `cfg.ineq_mode`). [Task 2]
- `turbompc/solvers/admm/csrc/admm_cudss.cuh` — *modify* launcher signature + `cfg`. [Task 2]
- `turbompc/solvers/admm/csrc/admm_cudss_ffi.cc` — *modify* (add `kappa_init` Arg + `ineq_mode64` Attr). [Task 2]
- `turbompc/solvers/admm/admm_cudss_ffi_backend.py` — *modify* — extend the FFI call to pass `kappa_init` + `ineq_mode`; add `central_path_cudss_forward(...)`. [Task 3]
- `tests/python/solvers/test_central_path_forward.py` — *new* — the JAX gates + the CUDA forward gates. [Tasks 1, 4]

**Build artifact:** diffmpc2's existing FFI `.so` (rebuilt by `make install` / its cmake), git-ignored.

---

## Task 0: Create the diffmpc2 feature branch

**Files:** none (git only)

**Interfaces:** Produces a clean working branch `Barrier_QP` on `external/diffmpc2`.

- [ ] **Step 1: Confirm diffmpc2 is clean and pick the base**

Run: `git -C external/diffmpc2 status -sb && git -C external/diffmpc2 branch -a`
Expected: working tree clean (stash/commit anything outstanding first). Note the current branch.

- [ ] **Step 2: Create + check out the branch from the chosen base**

Run (recommended base `inequality-hessian`):
```bash
git -C external/diffmpc2 fetch origin
git -C external/diffmpc2 checkout -b Barrier_QP inequality-hessian
git -C external/diffmpc2 status -sb
```
Expected: `On branch Barrier_QP`.

---

## Task 1: JAX central-path reference (the oracle), in diffmpc2

The pure-barrier branch is the apples-to-apples oracle for the CUDA non-slack mode **and** the clean "γ-never-appears-when-slack-off" reference. Copy the project's *proven* JAX central-path (which is already "diffmpc2's `_solve_jax_loop` with the z-update swapped") into diffmpc2 and add the barrier branch.

**Files:**
- Create: `external/diffmpc2/turbompc/solvers/admm/central_path_retraction.py`
- Create: `external/diffmpc2/turbompc/solvers/admm/central_path_jax.py`
- Test: `external/diffmpc2/tests/python/solvers/test_central_path_forward.py`

**Interfaces:**
- Consumes (in-repo): `turbompc.solvers.admm.admm.{compute_S_Phiinv, compute_gamma, _apply_C_parts, _apply_G, ADMMState, ADMMResiduals, _compute_residuals, _residuals_too_large, _update_rho}`; `turbompc.solvers.qp_data.{QPData, QPInequalityBlocks}`.
- Produces:
  - `retraction_map(v, gamma)`, `elastic_retraction(z_tilde, h, kappa, rho, slack_weight)`, `barrier_retraction(z_tilde, h, kappa, rho)` (pure barrier, **no γ**).
  - `to_one_sided(qp_data, slack_weight, big=1e7, *, use_slack=True)`.
  - `solve_qp_central_path(qp_data, schur_solver, *, target_kappa, slack_weight, use_slack=True, rho_bar=0.1, sigma=1e-6, rho_f_factor=1000.0, alpha=1.6, rho_min=1e-6, rho_max=1e6, adapt_rho_every=25, check_termination_every=25, adaptive_rho_tolerance=5.0, max_iter=20000, tol=1e-9) -> (x_blocks, (y_f_0,y_f_dyn,y_g), info)`; `info` keys `iters, prim_res, dual_res, final_rho, xi_max`.

- [ ] **Step 1: Copy the proven JAX solver into diffmpc2 and add the pure-barrier branch**

```bash
cp src/diffmpc_learning/solvers/retraction.py     external/diffmpc2/turbompc/solvers/admm/central_path_retraction.py
cp src/diffmpc_learning/solvers/central_path_admm.py external/diffmpc2/turbompc/solvers/admm/central_path_jax.py
```
In `central_path_jax.py`: delete the `sys.path` shim block (lines ~17–27 — `turbompc` is in-repo now) and change the imports to `from turbompc.solvers.admm.admm import (...)`, `from turbompc.solvers.qp_data import QPData, QPInequalityBlocks`, and `from .central_path_retraction import retraction_map, elastic_retraction, barrier_retraction, _inf_norm`.

In `central_path_retraction.py`, append the pure barrier:
```python
def barrier_retraction(z_tilde, h, kappa, rho):
    """Pure one-sided log-barrier z-update for rows Gx <= h. NO elastic slack, NO gamma.

    z_g = argmin_z [ -kappa*log(h - z) + (rho/2)(z - z_tilde)^2 ]
        = h - b_{kappa/rho}(h - z_tilde),   b = retraction_map.
    Barrier slack s = h - z_g > 0; implied dual y_g = kappa/s > 0 satisfies s*y_g = kappa.
    -> z_g = min(z_tilde, h) as kappa->0. Inert rows (h = +inf) return z_g = z_tilde.
    """
    r = h - z_tilde
    s = retraction_map(r, kappa / rho)
    z_g = h - s
    return jnp.where(jnp.isinf(h), z_tilde, z_g)
```

- [ ] **Step 2: Add the `use_slack` flag (two clean branches) to the JAX solver**

In `central_path_jax.py`: add `use_slack: bool = True` to `to_one_sided` (`*, use_slack=True`) and set `use_slack_variables=use_slack` in the `QPInequalityBlocks(...)`. Add `use_slack: bool = True` to `solve_qp_central_path` (after `slack_weight`). Replace the z-update block with:
```python
        if m:
            z_tilde = alpha * ineq_vals + (1.0 - alpha) * state.z_g + state.y_g / state.rho_bar
            if use_slack:
                z_g, xi = elastic_retraction(z_tilde, h, target_kappa, state.rho_bar, slack_weight)
                xi_g = -xi
            else:
                z_g = barrier_retraction(z_tilde, h, target_kappa, state.rho_bar)
                xi_g = jnp.zeros_like(z_g)   # non-slack: slack dual-residual term vanishes
        else:
            z_g, xi_g = state.z_g, state.xi_g
```

- [ ] **Step 3: Write the failing test (pure barrier: FOC + κ→0 hard limit)**

In `external/diffmpc2/tests/python/solvers/test_central_path_forward.py`:
```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from turbompc.solvers.admm.central_path_retraction import barrier_retraction

def _rel_linf(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))

def test_barrier_retraction_pure_no_gamma():
    z_tilde = jnp.array([0.3, 1.0, -0.5, 2.0]); h = jnp.ones(4); kappa, rho = 1e-3, 0.1
    z_g = barrier_retraction(z_tilde, h, kappa, rho); s = h - z_g
    assert bool(jnp.all(s > 0.0))
    foc = rho * (z_g - z_tilde) - kappa / s                      # prox stationarity
    assert float(jnp.max(jnp.abs(foc))) < 1e-9
    assert _rel_linf(barrier_retraction(z_tilde, h, 0.0, rho), jnp.minimum(z_tilde, h)) < 1e-12
```

- [ ] **Step 4: Run it; expect pass**

Run: `cd /home/jianghan/Workspace/diffmpc-learning && PYTHONPATH=external/diffmpc2 /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python -m pytest external/diffmpc2/tests/python/solvers/test_central_path_forward.py::test_barrier_retraction_pure_no_gamma -v`
Expected: PASS.

- [ ] **Step 5: Add the QP fixture + Schur-solver helpers and the JAX QP-match test**

Add two helpers near the top of the test file (reuse diffmpc2's existing infra — do NOT invent a QP):
- `_build_box_qp()` — return a `QPData` with an **active two-sided box** `l≤Gx≤u`. Build it from diffmpc2's existing obstacle/box fixture: extract one SQP-linearized `QPData` from `tests/python/solvers/test_inequality_hessian.py::_build`'s solver (its `_build_qp_data`/`build_qp_data` path), or copy that test's QP assembly. Choose weights/bounds so a row is active at the solution (assert it).
- `_make_cudss_schur_solver()` — construct the cuDSS Schur-solver object passed to `solve_qp_central_path`; copy the construction line from diffmpc2's existing central-path / ADMM JAX tests. Do not guess the class name — copy it.

```python
from turbompc.solvers.admm.central_path_jax import to_one_sided, solve_qp_central_path

def test_jax_pure_barrier_matches_hard_qp():
    qp = _build_box_qp()
    qp1 = to_one_sided(qp, slack_weight=1e2, use_slack=False)
    ss = _make_cudss_schur_solver()
    x_b, _, info_b = solve_qp_central_path(qp1, ss, target_kappa=1e-6, slack_weight=1e2, use_slack=False, tol=1e-9)
    x_e, _, _      = solve_qp_central_path(qp1, ss, target_kappa=1e-6, slack_weight=1e6, use_slack=True,  tol=1e-9)
    assert _rel_linf(x_b, x_e) < 1e-3       # barrier == elastic at large gamma
    assert float(info_b["xi_max"]) == 0.0   # non-slack: NO elastic slack
```

- [ ] **Step 6: Run; expect pass**

Run: `... -m pytest external/diffmpc2/tests/python/solvers/test_central_path_forward.py -k barrier -v`
Expected: PASS.

- [ ] **Step 7: Commit on the branch**

```bash
git -C external/diffmpc2 add turbompc/solvers/admm/central_path_retraction.py turbompc/solvers/admm/central_path_jax.py tests/python/solvers/test_central_path_forward.py
git -C external/diffmpc2 commit -m "feat(central-path): JAX reference solver (elastic + pure-barrier z-update, one-sided)"
```

---

## Task 2: CUDA barrier z-update + κ/mode threading (modify diffmpc2 kernels in place)

The single algorithmic change + the plumbing. `γ` is referenced **only** in the elastic branch; `ineq_mode=0` keeps the existing hard path byte-for-byte.

**Files (all `external/diffmpc2/turbompc/solvers/admm/csrc/`):**
- Modify: `admm_math.cuh` (`slackUpdateDevice`, add `retraction_b`)
- Modify: `admm_cudss.cu` (kernel sig, per-batch `kappa` load, call sites, `cfg.ineq_mode`)
- Modify: `admm_cudss.cuh` (launcher sig + `cfg`)
- Modify: `admm_cudss_ffi.cc` (`kappa_init` Arg + `ineq_mode64` Attr)
- Test: `tests/python/solvers/test_central_path_forward.py`

**Interfaces:**
- Produces: the `central_path_admm_cudss` forward honoring `ineq_mode ∈ {0=HARD, 1=BARRIER, 2=ELASTIC}` + a per-batch `kappa` buffer. FFI handler accepts `kappa_init` (Arg, after `slack_weight_init`) + `ineq_mode64` (Attr, alongside `use_slack64`).

- [ ] **Step 1: Add the retraction helper to `admm_math.cuh`**

Above `slackUpdateDevice`:
```cpp
// b_g(r) = (r + sqrt(r^2 + 4g))/2 ; cancellation-safe for r<0 ; -> max(r,0) as g->0. (PrismQP eq.163)
template<typename T>
__device__ inline T retraction_b(T r, T g) {
    T sq = sqrt(r * r + T(4) * g);
    return (r >= T(0)) ? T(0.5) * (r + sq) : T(2) * g / (sq - r);
}
```

- [ ] **Step 2: Replace the per-row body of `slackUpdateDevice`**

Add params `T kappa, int ineq_mode, T big` to the signature (after `T* xi_g`). Replace the per-row body (`admm_math.cuh:183–196`) with:
```cpp
            T z_old = z_g[i];
            T z_tilde = alpha * scratch_Gx[i] + (T(1) - alpha) * z_old + y_g[i] / rho_bar;
            T hb = u_g[i];                       // one-sided upper bound (l_g inert for barrier modes)
            if (ineq_mode == 0) {                // HARD: box projection (UNCHANGED existing path)
                T clamped = z_tilde;
                if (clamped < l_g[i]) clamped = l_g[i];
                if (clamped > hb)     clamped = hb;
                if (use_slack) {
                    T frac = slack_weight / (slack_weight + rho_bar);
                    z_g[i] = (T(1) - frac) * z_tilde + frac * clamped;
                    xi_g[i] = (rho_bar / (slack_weight + rho_bar)) * (clamped - z_tilde);
                } else { z_g[i] = clamped; }
            } else if (hb >= big) {              // inert one-sided row (h = +inf): no constraint
                z_g[i] = z_tilde; xi_g[i] = T(0);
            } else if (ineq_mode == 1) {         // BARRIER (non-slack): NO gamma
                T s = retraction_b(hb - z_tilde, kappa / rho_bar);
                z_g[i] = hb - s; xi_g[i] = T(0);
            } else {                             // ELASTIC: barrier + gamma slack
                T Gam = kappa * (T(1) / rho_bar + T(1) / slack_weight);
                T s = retraction_b(hb - z_tilde, Gam);
                T xi = kappa / (slack_weight * s);
                z_g[i] = hb - s + xi; xi_g[i] = -xi;   // sign: JAX xi_g = -xi (residual gamma*xi_g + y_g)
            }
```

- [ ] **Step 3: Thread `kappa` + `ineq_mode` through `admm_cudss.cu`**

Mirror `slack_weight_global` exactly: add `const T* __restrict__ kappa_global` kernel arg next to `slack_weight_global`; load `T kappa = kappa_global[bid];` next to the `slack_weight` load; add `int ineq_mode` to the `cfg` struct + a kernel param; pass `kappa, ineq_mode, /*big=*/T(1e7)` into the `slackUpdateDevice(...)` and `computeResidualsDevice(...)` calls. In `computeResidualsDevice`, gate the slack-dual-residual block on `ineq_mode == 2` (BARRIER/HARD use the plain dual residual).

- [ ] **Step 4: Update the launcher in `admm_cudss.cuh`**

Add `const T* kappa_init` to the host launcher signature (next to `slack_weight_init`) and `int ineq_mode` to its `cfg`; forward both to the kernel launch.

- [ ] **Step 5: Add `kappa_init` Arg + `ineq_mode64` Attr to `admm_cudss_ffi.cc`**

Add `BufT kappa_init` to `AdmmCudssCudaImpl` args (immediately after `slack_weight_init`); add `int64_t ineq_mode64` to attrs (next to `use_slack64`); set `cfg.ineq_mode = (int)ineq_mode64;`; pass `kappa_init.typed_data()` to the launcher. In the `Ffi::Bind()` chain add `.Arg<BufF64>() // kappa_init` (same position for the F32 + F64 registrations) and `.Attr<int64_t>("ineq_mode")`.

- [ ] **Step 6: Rebuild diffmpc2's FFI**

Apply the cuDSS-0.7.1 backport shim per `HANDOFF.md` (do not commit the `.cu` revert), then:
```bash
cd /home/jianghan/Workspace/diffmpc-learning/external/diffmpc2
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
make install      # or the documented cmake build in turbompc/solvers/csrc (see diffmpc2 README)
```
Expected: builds clean; the FFI `.so` is refreshed.

- [ ] **Step 7: Smoke-run the kernel in BARRIER mode (no oracle yet)**

Add a temporary test calling the barrier forward once (helper added properly in Task 3; here a minimal direct ffi_call is fine) asserting `prim_res < 1e-6`, `iters < max_iter`, all-finite.

Run: `... -m pytest external/diffmpc2/tests/python/solvers/test_central_path_forward.py -k cuda_barrier_runs -v`
Expected: PASS.

- [ ] **Step 8: Regression — the hard path is unchanged**

Run diffmpc2's existing ADMM-forward test(s) (e.g. `tests/python/solvers/` ADMM/cuDSS forward tests) to confirm `ineq_mode=0` behavior is byte-for-byte:
Run: `... -m pytest external/diffmpc2/tests/python/solvers/ -k "admm or cudss or fused" -v`
Expected: all existing forward tests PASS (no regression from the new params).

- [ ] **Step 9: Commit (source only — NOT the .cu backport revert or the .so)**

```bash
git -C external/diffmpc2 add turbompc/solvers/admm/csrc/admm_math.cuh turbompc/solvers/admm/csrc/admm_cudss.cuh turbompc/solvers/admm/csrc/admm_cudss_ffi.cc tests/python/solvers/test_central_path_forward.py
git -C external/diffmpc2 add turbompc/solvers/admm/csrc/admm_cudss.cu   # re-verify NO 0.7.1 cudssMatrixCreateCsr revert lines are staged (git diff first)
git -C external/diffmpc2 commit -m "feat(central-path-cuda): barrier + elastic z-update; thread kappa + ineq_mode (ineq_mode=0 keeps hard path)"
```
> Before staging `admm_cudss.cu`, `git -C external/diffmpc2 diff turbompc/solvers/admm/csrc/admm_cudss.cu` and confirm the only changes are the `kappa`/`ineq_mode` threading — the cuDSS-0.7.1 `cudssMatrixCreateCsr` revert must remain an un-committed build-time edit.

---

## Task 3: Python wrapper `central_path_cudss_forward` (in diffmpc2)

Mirror the JAX `solve_qp_central_path` signature so the gate is a one-line comparison; reuse `to_one_sided`.

**Files:**
- Modify: `external/diffmpc2/turbompc/solvers/admm/admm_cudss_ffi_backend.py`
- Test: `external/diffmpc2/tests/python/solvers/test_central_path_forward.py`

**Interfaces:**
- Consumes: the extended FFI (Task 2), `to_one_sided` (Task 1), `compute_S_Phiinv`.
- Produces: `central_path_cudss_forward(qp_data, *, target_kappa, slack_weight, use_slack=True, rho_bar=0.1, sigma=1e-6, rho_f_factor=1000.0, alpha=1.6, rho_min=1e-6, rho_max=1e6, adapt_rho_every=25, check_termination_every=25, adaptive_rho_tolerance=5.0, max_iter=20000, tol=1e-9) -> (x_blocks, (y_f_0, y_f_dyn, y_g), info)` — **identical keyword names + return shape to the JAX `solve_qp_central_path`** (so one `**_CFG` drives both).

- [ ] **Step 1: Extend the FFI backend + add the wrapper**

In `admm_cudss_ffi_backend.py`: extend the existing `ffi_call` arg list to include `kappa_init` (a `(Nb,)` buffer, after `slack_weight_init`) and the `ineq_mode` attr; then add `central_path_cudss_forward(...)` which (a) `qp1 = to_one_sided(qp_data, slack_weight, use_slack=use_slack)`; (b) `S0 = compute_S_Phiinv(qp1, rho_bar*rho_f_factor, sigma, rho_ineq=rho_bar)`; (c) packs args in the exact `Ffi::Bind()` order ending `…, rho_bar_init, slack_weight_init, kappa_init`; (d) attrs `use_slack=use_slack`, `ineq_mode=(2 if use_slack else 1)`, plus `tol, max_iter, check_termination_every, adapt_rho_every, adaptive_rho_tolerance, alpha, sigma, rho_f_factor`; (e) repackages outputs into `(x_blocks, (y_f_0,y_f_dyn,y_g), info)`. **Copy the arg-packing/output-unpacking from the existing `admm_cudss_ffi_solve_single` in this same file and insert `kappa_init` + `ineq_mode`.**

- [ ] **Step 2: Test the wrapper (barrier mode shapes + convergence)**

```python
from turbompc.solvers.admm.admm_cudss_ffi_backend import central_path_cudss_forward
def test_cuda_wrapper_shapes_and_convergence():
    qp = _build_box_qp()
    x, (y_f_0, y_f_dyn, y_g), info = central_path_cudss_forward(
        qp, target_kappa=1e-4, slack_weight=1e2, use_slack=False, tol=1e-9)
    assert x.shape == (qp.cost.D.shape[0], qp.cost.D.shape[1])
    assert float(info["prim_res"]) < 1e-6
    assert bool(jnp.all(y_g >= -1e-9))    # one-sided duals non-negative
```

- [ ] **Step 3: Run; expect pass**

Run: `... -m pytest external/diffmpc2/tests/python/solvers/test_central_path_forward.py::test_cuda_wrapper_shapes_and_convergence -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git -C external/diffmpc2 add turbompc/solvers/admm/admm_cudss_ffi_backend.py tests/python/solvers/test_central_path_forward.py
git -C external/diffmpc2 commit -m "feat(central-path-cuda): central_path_cudss_forward wrapper (one-sided, kappa+mode), JAX-shaped returns"
```

---

## Task 4: The forward gates — match the QP solution + match the iteration count

The acceptance test. Both modes vs the in-repo JAX oracle (same `(κ,γ,ρ,α,tol,check/adapt)`), plus the hard-QP cross-check.

**Files:**
- Test: `external/diffmpc2/tests/python/solvers/test_central_path_forward.py`

**Interfaces:** Consumes `central_path_cudss_forward` (Task 3), `solve_qp_central_path` (Task 1), diffmpc2's hard `ADMM_FUSED_CUDSS` forward.

- [ ] **Step 1: Elastic — match the JAX oracle solution AND iteration count**

```python
_CFG = dict(rho_bar=0.1, sigma=1e-6, rho_f_factor=1000.0, alpha=1.6,
            tol=1e-9, max_iter=20000, check_termination_every=25, adapt_rho_every=25,
            adaptive_rho_tolerance=5.0)   # keyword names match BOTH solver signatures

def test_cuda_elastic_matches_jax_solution_and_itercount():
    qp = _build_box_qp(); ss = _make_cudss_schur_solver()
    qp1 = to_one_sided(qp, slack_weight=1e2, use_slack=True)
    x_j, (_, _, yg_j), info_j = solve_qp_central_path(qp1, ss, target_kappa=1e-4, slack_weight=1e2, use_slack=True, **_CFG)
    x_c, (_, _, yg_c), info_c = central_path_cudss_forward(qp, target_kappa=1e-4, slack_weight=1e2, use_slack=True, **_CFG)
    assert _rel_linf(x_c, x_j) < 1e-6                                            # MATCH QP SOLUTION (relaxed)
    assert _rel_linf(yg_c, yg_j) < 1e-6                                          # duals match
    assert abs(int(info_c["iters"]) - int(info_j["iters"])) <= _CFG["check_termination_every"]   # MATCH ITER COUNT
```

- [ ] **Step 2: Barrier (non-slack) — match the JAX barrier oracle solution AND iteration count**

```python
def test_cuda_barrier_matches_jax_solution_and_itercount():
    qp = _build_box_qp(); ss = _make_cudss_schur_solver()
    qp1 = to_one_sided(qp, slack_weight=1e2, use_slack=False)
    x_j, _, info_j = solve_qp_central_path(qp1, ss, target_kappa=1e-4, slack_weight=1e2, use_slack=False, **_CFG)
    x_c, _, info_c = central_path_cudss_forward(qp, target_kappa=1e-4, slack_weight=1e2, use_slack=False, **_CFG)
    assert _rel_linf(x_c, x_j) < 1e-6
    assert abs(int(info_c["iters"]) - int(info_j["iters"])) <= _CFG["check_termination_every"]
```

- [ ] **Step 3: Underlying QP — barrier at κ→0 matches diffmpc2's hard fused forward**

```python
def test_cuda_barrier_matches_hard_qp_solution():
    qp = _build_box_qp()
    x_hard, _ = _diffmpc2_fused_cudss_forward(qp)     # diffmpc2's hard ADMM_FUSED_CUDSS forward on the SAME box QP
    x_c, _, _ = central_path_cudss_forward(qp, target_kappa=1e-6, slack_weight=1e6, use_slack=False, **_CFG)
    assert _rel_linf(x_c, x_hard) < 1e-4              # converges to the hard QP as kappa->0
```
> `_diffmpc2_fused_cudss_forward` is a thin helper around diffmpc2's existing `admm_cudss_ffi_solve_single` (hard mode) on the same `to_one_sided` QP — copy the call from this file's existing usage.

- [ ] **Step 4: Run the full forward suite**

Run: `cd /home/jianghan/Workspace/diffmpc-learning && export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH" && export XLA_PYTHON_CLIENT_PREALLOCATE=false && PYTHONPATH=external/diffmpc2 /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python -m pytest external/diffmpc2/tests/python/solvers/test_central_path_forward.py -v`
Expected: ALL PASS — both modes match the JAX QP solution (rel-ℓ∞ < 1e-6), iteration counts match (within one check interval), barrier→hard QP (rel-ℓ∞ < 1e-4).

- [ ] **Step 5: Remove the Task 2 temporary smoke test (subsumed) and commit**

```bash
git -C external/diffmpc2 add tests/python/solvers/test_central_path_forward.py
git -C external/diffmpc2 commit -m "test(central-path-cuda): forward gates — match QP solution (both modes) + match JAX iteration count"
```

---

## Verification (end-to-end)

In the cuDSS env (x64), on branch `Barrier_QP`, `external/diffmpc2/tests/python/solvers/test_central_path_forward.py` passes with:

1. **Matches the QP solution.** `central_path_cudss_forward` vs the in-repo JAX `solve_qp_central_path` at the same `(κ, γ, ρ_bar, α, tol)`: `rel-ℓ∞(x) < 1e-6`, `rel-ℓ∞(y_g) < 1e-6`, for **both** `use_slack=True` and `use_slack=False`. Independently, barrier at `κ=1e-6` matches diffmpc2's hard `ADMM_FUSED_CUDSS` forward (`rel-ℓ∞ < 1e-4`).
2. **Matches the iteration count.** `|iters_cuda − iters_jax| ≤ check_termination_every` (= 25) for both modes — the fused on-device loop reproduces the JAX convergence trajectory, not just the fixed point.
3. **Clean slack semantics.** `use_slack=False` (BARRIER) never forms `1/γ` or `γ·ξ`; `info["xi_max"]==0` (JAX) and `xi_g≡0` (kernel).
4. **No regression.** diffmpc2's existing hard ADMM forward tests pass unchanged (`ineq_mode=0`).
5. **One-sided correctness.** `to_one_sided` stacking + the kernel `h≥big` inert mask; duals `y_g ≥ 0`.

---

## Risks / open questions

- **Branch base.** Default `inequality-hessian` (has the obstacle QP fixtures + backward sign fix). Confirm with the user if `main` is preferred.
- **Build is the main risk.** It's diffmpc2's *own* build, which is known-working with the cuDSS-0.7.1 backport (HANDOFF recipe) — but keep the `.cu` `cudssMatrixCreateCsr` revert as a build-time edit, never committed. Task 2 Step 8 (regression) catches a broken build before the new math is trusted.
- **Iteration-count match depends on identical hyperparameters.** The wrapper-forwarded values (`check_termination_every`, `adapt_rho_every`, `adaptive_rho_tolerance`, `alpha`, `rho_bar`, `rho_f_factor`, `sigma`) and the kernel's residual-termination formula must equal the JAX `solve_qp_central_path` defaults (`_CFG`). If iter counts differ by more than one check interval, diff the two configs first, not the kernel math.
- **`_build_box_qp` / `_make_cudss_schur_solver` must reuse validated diffmpc2 infra.** Build the `QPData` from `test_inequality_hessian.py::_build`'s solver (active box) and copy the Schur-solver construction from diffmpc2's existing ADMM JAX tests — do not synthesize a fresh QP or guess class names, or the active set may never engage.
- **f32 vs f64.** Build + test in **f64** (x64); register/exercise the `_f64` FFI target. The `_f32` target builds but is untested here.
- **Backward unaffected (already analyzed).** The one-sided forward feeds the existing JAX backward with non-negative one-sided `y_g`; no sign change. The only backward follow-up (out of scope) is a γ-free `W = y_g/(h−G1x)` branch to match the non-slack forward — see `notes/CENTRAL_PATH_BACKWARD.md`.
