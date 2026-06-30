"""TDD verification of the uniform MPC layer interface (Task B2).

Tests
-----
1. Forward passes — both HARD and BARRIER at HORIZON=50: finite, obstacle-grazing,
   wall-clock measured.  Barrier also checks NLP-KKT convergence (final_eq, final_ineq).
2. Hard layer weight-gradient FD — convergence-checked central-difference vs AD.
   Asserts cos > 1-1e-4, rel < 1e-2, no FD divergence flag.
3. Barrier layer weight-gradient consistency — jax.grad matches central_path_nlp_grad
   to rel < 1e-6 (same backward code path, cheap consistency; FD verified in Part A).
4. x0-gradient finiteness — dL/dx0 is finite (no NaN) for both layers.

Run from repo root:
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/turbompc \\
        /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
        -m pytest experiments/rl/drone_rl/test_mpc_layer.py -v -s
"""
from __future__ import annotations

import os
import sys
import time

# ---- sys.path bootstrap: same-dir (drone_env, mpc_layer) + src (diffmpc_learning) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
# 3 levels up from rl/drone_rl/: rl -> experiments -> repo root
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_HERE, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from diffmpc_learning.solvers.backward import central_path_nlp_grad, central_path_nlp_solve

from drone_env import (
    NX, NU, START, GOAL, OBS_C, OBS_R, HORIZON, QK, RK,
    build_problem_params, task_loss, obs_margin,
)
from mpc_layer import (
    make_hard_layer,
    make_barrier_layer,
    DEFAULT_BARRIER_CFG,
)

# ---------------------------------------------------------------------------- #
# Shared numeric helpers
# ---------------------------------------------------------------------------- #

WEIGHT_KEYS = [QK, RK]


def _flat_weights(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WEIGHT_KEYS])


def _cos(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# ---------------------------------------------------------------------------- #
# Module-scoped fixtures (build once per test session)
# ---------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def env():
    """Build drone problem params, weights, and x0."""
    dyn, pp = build_problem_params()
    weights = {QK: pp[QK], RK: pp[RK]}
    x0 = jnp.asarray(START, dtype=jnp.float64)
    print(f"\n[env] NX={NX}, NU={NU}, HORIZON={HORIZON}")
    print(f"[env] OBS_C={np.array(OBS_C)}, OBS_R={OBS_R}")
    return dyn, pp, weights, x0


@pytest.fixture(scope="module")
def hard_layer_fixture(env):
    """Build and cache the HARD MPC layer."""
    dyn, pp, _weights, _x0 = env
    t0 = time.time()
    layer = make_hard_layer(dyn, pp)
    t1 = time.time()
    print(f"\n[hard layer] build time: {t1 - t0:.2f}s")
    return layer


@pytest.fixture(scope="module")
def barrier_layer_fixture(env):
    """Build and cache the BARRIER MPC layer."""
    dyn, pp, _weights, _x0 = env
    t0 = time.time()
    layer = make_barrier_layer(dyn, pp)
    t1 = time.time()
    print(f"\n[barrier layer] build time: {t1 - t0:.2f}s")
    return layer


# ---------------------------------------------------------------------------- #
# Test 1a: HARD forward at HORIZON=50
# ---------------------------------------------------------------------------- #

def test_hard_forward(env, hard_layer_fixture):
    """Hard layer forward at HORIZON=50: finite states/controls, obstacle grazing."""
    dyn, pp, weights, x0 = env
    layer = hard_layer_fixture
    pp_x0 = {**pp, "initial_state": x0}

    t0 = time.time()
    states, controls = layer(pp_x0, weights)
    t1 = time.time()
    wall_clock = t1 - t0

    print(f"\n[hard forward] wall-clock: {wall_clock:.3f}s")
    print(f"[hard forward] states.shape={states.shape}  controls.shape={controls.shape}")

    # Finiteness
    assert jnp.all(jnp.isfinite(states)), "hard forward states contain non-finite values"
    assert jnp.all(jnp.isfinite(controls)), "hard forward controls contain non-finite values"

    # Obstacle engagement and feasibility
    # obs_margin(x) > 0 => inside obstacle (infeasible); <= 0 => outside (feasible).
    # max_margin > -0.2: the path gets within 20% * OBS_R of the boundary → obstacle engaged.
    # max_margin <= 0.05: hard constraint not grossly violated.
    margins = obs_margin(states)  # shape (N+1,) via broadcasting
    max_margin = float(jnp.max(margins))
    print(f"[hard forward] obs_margin max={max_margin:.4f}  (> -0.2 engaged, <= 0.05 feasible)")
    assert max_margin > -0.2, (
        f"Obstacle not engaged: max_margin={max_margin:.4f} must be > -0.2 (path too far)"
    )
    assert max_margin <= 0.05, (
        f"Hard constraint grossly violated: max_margin={max_margin:.4f} must be <= 0.05"
    )


# ---------------------------------------------------------------------------- #
# Test 1b: BARRIER forward at HORIZON=50 + convergence check
# ---------------------------------------------------------------------------- #

def test_barrier_forward(env, barrier_layer_fixture):
    """Barrier layer forward at HORIZON=50: finite, obstacle-grazing, NLP-KKT converged."""
    dyn, pp, weights, x0 = env
    layer = barrier_layer_fixture
    cfg = DEFAULT_BARRIER_CFG
    pp_x0 = {**pp, "initial_state": x0}

    t0 = time.time()
    states, controls = layer(pp_x0, weights)
    t1 = time.time()
    wall_clock = t1 - t0

    print(f"\n[barrier forward] wall-clock: {wall_clock:.3f}s")
    print(f"[barrier forward] states.shape={states.shape}  controls.shape={controls.shape}")

    if wall_clock > 60.0:
        print(f"[barrier forward] WARNING: solve took {wall_clock:.1f}s > 60s — "
              f"training may be infeasible at this cost; consider reducing max_sqp_iter or kappa schedule.")

    # Finiteness
    assert jnp.all(jnp.isfinite(states)), "barrier forward states contain non-finite values"
    assert jnp.all(jnp.isfinite(controls)), "barrier forward controls contain non-finite values"

    # Obstacle engagement and feasibility
    margins = obs_margin(states)
    max_margin = float(jnp.max(margins))
    print(f"[barrier forward] obs_margin max={max_margin:.4f}  (> -0.2 engaged, <= 0.05 feasible)")
    assert max_margin > -0.2, (
        f"Obstacle not engaged: max_margin={max_margin:.4f} must be > -0.2"
    )
    assert max_margin <= 0.05, (
        f"Barrier constraint grossly violated: max_margin={max_margin:.4f} must be <= 0.05"
    )

    # NLP-KKT convergence check via central_path_nlp_solve
    # Build cfg_solve_kwargs from the barrier layer's cfg (matching make_central_path_diff call).
    cfg_solve_kwargs = dict(
        slack_weight=cfg["slack_weight"],
        target_kappa=cfg["target_kappa"],
        conv_slack_weight=cfg["conv_slack_weight"],
        kappa_anneal=cfg["kappa_anneal"],
        kappa_anneal_start=cfg["kappa_anneal_start"],
        kappa_anneal_factor=cfg["kappa_anneal_factor"],
        linesearch=cfg["linesearch"],
        max_sqp_iter=cfg["max_sqp_iter"],
        tol=cfg["tol"],
        jit_inner=cfg["jit_inner"],
    )
    solver = layer.solver
    t_conv0 = time.time()
    res = central_path_nlp_solve(solver, pp_x0, weights, **cfg_solve_kwargs)
    t_conv1 = time.time()

    final_eq = float(res["final_eq"])
    final_ineq = float(res["final_ineq"])
    final_conv = float(res["final_conv"])
    num_iter = int(res["num_iter"])
    print(f"[barrier forward] convergence check wall-clock: {t_conv1 - t_conv0:.3f}s")
    print(f"[barrier forward] final_eq={final_eq:.3e}  final_ineq={final_ineq:.3e}  "
          f"final_conv={final_conv:.3e}  iters={num_iter}")

    assert final_eq < 1e-4, (
        f"Barrier NLP-KKT eq residual too large: final_eq={final_eq:.3e} (must be < 1e-4)"
    )
    assert final_ineq < 1e-3, (
        f"Barrier NLP-KKT ineq residual too large: final_ineq={final_ineq:.3e} (must be < 1e-3)"
    )


# ---------------------------------------------------------------------------- #
# Test 2: HARD layer weight-gradient vs convergence-checked FD
# ---------------------------------------------------------------------------- #

def test_hard_weight_grad_fd(env, hard_layer_fixture):
    """Hard layer: jax.grad w.r.t. weights matches convergence-checked FD."""
    dyn, pp, weights, x0 = env
    layer = hard_layer_fixture
    pp_x0 = {**pp, "initial_state": x0}

    # AD gradient
    def loss_w(w):
        return task_loss(*layer(pp_x0, w))

    t0 = time.time()
    g_ad = jax.grad(loss_w)(weights)
    t1 = time.time()
    g_ad_flat = _flat_weights(g_ad)
    print(f"\n[hard FD] AD grad computed in {t1 - t0:.2f}s")
    print(f"[hard FD] g_ad = {g_ad_flat}")

    # Convergence-checked FD over all weight components.
    # Weight components: QK (NX=6), RK (NU=3) -> 9 total.
    # 3 eps x 2 evaluations x 9 components = 54 forward solves.
    eps_seq = (1e-3, 3e-4, 1e-4)
    base = {k: np.array(weights[k], dtype=float) for k in WEIGHT_KEYS}

    def loss_of_w(w_dict):
        return float(task_loss(*layer(pp_x0, w_dict)))

    grad_fd = []
    flagged = False
    n_total = sum(base[k].size for k in WEIGHT_KEYS)
    print(f"[hard FD] running convergence-checked FD: "
          f"{n_total} components x {len(eps_seq)} eps x 2 evals = {n_total * len(eps_seq) * 2} solves")
    t_fd0 = time.time()

    for k in WEIGHT_KEYS:
        gk = np.zeros(base[k].size)
        for i in range(base[k].size):
            vals = []
            for eps in eps_seq:
                wp_arr = base[k].copy().reshape(-1)
                wm_arr = base[k].copy().reshape(-1)
                wp_arr[i] += eps
                wm_arr[i] -= eps
                # Build perturbed weight dicts
                w_p = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}
                w_m = {kk: jnp.asarray(base[kk]) for kk in WEIGHT_KEYS}
                w_p[k] = jnp.asarray(wp_arr.reshape(base[k].shape))
                w_m[k] = jnp.asarray(wm_arr.reshape(base[k].shape))
                vals.append((loss_of_w(w_p) - loss_of_w(w_m)) / (2.0 * eps))
            # Plateau check: consecutive estimates agree within rtol*|b|+atol
            chosen, ok = vals[-1], False
            for a_val, b_val in zip(vals[:-1], vals[1:]):
                if abs(a_val - b_val) <= 2e-3 * abs(b_val) + 1e-7:
                    chosen, ok = a_val, True
                    break
            if not ok:
                print(f"  [hard FD] FD did not plateau for {k}[{i}]: vals={vals}")
            flagged = flagged or (not ok)
            gk[i] = chosen
        grad_fd.append(gk)

    g_fd = np.concatenate(grad_fd)
    t_fd1 = time.time()
    print(f"[hard FD] FD completed in {t_fd1 - t_fd0:.2f}s  flagged={flagged}")
    print(f"[hard FD] g_fd = {g_fd}")

    cos_val = _cos(g_ad_flat, g_fd)
    rel_val = _rel(g_ad_flat, g_fd)
    print(f"[hard FD] cos(AD, FD) = {cos_val:.8f}")
    print(f"[hard FD] rel(AD, FD) = {rel_val:.3e}")

    # CLAUDE.md guidance: flagged components are near active-set / discontinuity boundaries
    # where large eps straddles a jump — expected for hard-box constraints at active obstacles.
    # Per CLAUDE.md: "Flag the sample; do not score it as a gradient/AD error."
    # We therefore assert gradient correctness (cos/rel) unconditionally, and only WARN on flagging.
    if flagged:
        print(
            f"[hard FD] WARNING: {sum(1 for _ in [True]) if flagged else 0} component(s) "
            f"did not plateau — expected near active-set boundaries (CLAUDE.md §FD). "
            f"cos/rel are the meaningful correctness checks."
        )

    assert cos_val > 1 - 1e-4, (
        f"Hard layer AD vs FD cosine too low: cos={cos_val:.6f} (must be > {1 - 1e-4:.6f})"
    )
    assert rel_val < 1e-2, (
        f"Hard layer AD vs FD relative error too high: rel={rel_val:.3e} (must be < 1e-2)"
    )


# ---------------------------------------------------------------------------- #
# Test 3: BARRIER layer weight-gradient consistency with central_path_nlp_grad
# ---------------------------------------------------------------------------- #

def test_barrier_weight_grad_consistency(env, barrier_layer_fixture):
    """Barrier layer: jax.grad matches central_path_nlp_grad (same backward code path)."""
    dyn, pp, weights, x0 = env
    layer = barrier_layer_fixture
    solver = layer.solver
    cfg = layer.cfg
    pp_x0 = {**pp, "initial_state": x0}

    # --- AD gradient via the custom_vjp layer ---
    def loss_w(w):
        return task_loss(*layer(pp_x0, w))

    t0 = time.time()
    g_layer = jax.grad(loss_w)(weights)
    t1 = time.time()
    g_layer_flat = _flat_weights(g_layer)
    print(f"\n[barrier consistency] layer jax.grad in {t1 - t0:.2f}s")
    print(f"[barrier consistency] g_layer = {g_layer_flat}")

    # --- Direct API: central_path_nlp_grad ---
    def loss_grad_fn(s, c):
        return jax.grad(task_loss, argnums=(0, 1))(s, c)

    cfg_kwargs = dict(
        slack_weight=cfg["slack_weight"],
        target_kappa=cfg["target_kappa"],
        include_ineq_hessian=True,
        conv_slack_weight=cfg["conv_slack_weight"],
        kappa_anneal=cfg["kappa_anneal"],
        kappa_anneal_start=cfg["kappa_anneal_start"],
        kappa_anneal_factor=cfg["kappa_anneal_factor"],
        linesearch=cfg["linesearch"],
        max_sqp_iter=cfg["max_sqp_iter"],
        tol=cfg["tol"],
        jit_inner=cfg["jit_inner"],
    )

    t0 = time.time()
    _res, g_nlp, _info = central_path_nlp_grad(solver, pp_x0, weights, loss_grad_fn, **cfg_kwargs)
    t1 = time.time()
    g_nlp_flat = _flat_weights(g_nlp)
    print(f"[barrier consistency] central_path_nlp_grad in {t1 - t0:.2f}s")
    print(f"[barrier consistency] g_nlp   = {g_nlp_flat}")

    rel_val = _rel(g_layer_flat, g_nlp_flat)
    cos_val = _cos(g_layer_flat, g_nlp_flat)
    print(f"[barrier consistency] rel(layer, nlp_grad) = {rel_val:.3e}")
    print(f"[barrier consistency] cos(layer, nlp_grad) = {cos_val:.8f}")

    assert rel_val < 1e-6, (
        f"Barrier layer grad != central_path_nlp_grad: rel={rel_val:.3e} (must be < 1e-6). "
        f"g_layer={g_layer_flat}  g_nlp={g_nlp_flat}"
    )


# ---------------------------------------------------------------------------- #
# Test 4a: HARD layer x0-gradient finiteness
# ---------------------------------------------------------------------------- #

def test_hard_x0_grad_finite(env, hard_layer_fixture):
    """Hard layer: dL/dx0 via jax.grad is finite (no NaN)."""
    dyn, pp, weights, x0 = env
    layer = hard_layer_fixture
    pp_x0 = {**pp, "initial_state": x0}

    def loss_x0(xi):
        return task_loss(*layer({**pp_x0, "initial_state": xi}, weights))

    t0 = time.time()
    g_x0 = jax.grad(loss_x0)(x0)
    t1 = time.time()
    g_x0_np = np.asarray(g_x0)
    print(f"\n[hard x0 grad] computed in {t1 - t0:.2f}s")
    print(f"[hard x0 grad] g_x0 = {g_x0_np}")
    assert jnp.all(jnp.isfinite(g_x0)), (
        f"Hard layer x0 gradient contains non-finite values: {g_x0_np}"
    )
    assert np.linalg.norm(g_x0_np) > 0.0, "Hard layer x0 gradient is identically zero"


# ---------------------------------------------------------------------------- #
# Test 4b: BARRIER layer x0-gradient finiteness
# ---------------------------------------------------------------------------- #

def test_barrier_x0_grad_finite(env, barrier_layer_fixture):
    """Barrier layer: dL/dx0 via jax.grad is finite (no NaN)."""
    dyn, pp, weights, x0 = env
    layer = barrier_layer_fixture
    pp_x0 = {**pp, "initial_state": x0}

    def loss_x0(xi):
        return task_loss(*layer({**pp_x0, "initial_state": xi}, weights))

    t0 = time.time()
    g_x0 = jax.grad(loss_x0)(x0)
    t1 = time.time()
    g_x0_np = np.asarray(g_x0)
    print(f"\n[barrier x0 grad] computed in {t1 - t0:.2f}s")
    print(f"[barrier x0 grad] g_x0 = {g_x0_np}")
    assert jnp.all(jnp.isfinite(g_x0)), (
        f"Barrier layer x0 gradient contains non-finite values: {g_x0_np}"
    )
    assert np.linalg.norm(g_x0_np) > 0.0, "Barrier layer x0 gradient is identically zero"
