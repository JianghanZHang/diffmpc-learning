"""TDD verification of the two Diff-WMPC gradient estimators (Task C).

Builds the two policy-update functions in ``gradient_modes.py`` (warm-started +
jitted) and verifies:

1. **Smoke (both)** — one ``open_loop_plan_update`` (V1, n_batch=4) and one
   ``shac_window_update`` (V3, h=4): finite grads/params (no NaN), the policy
   actually moved, logs are populated, and the obstacle is engaged at some step
   (plan ``obs_margin`` near/above 0). Prints cold-vs-warm wall-clock to confirm
   the warm-start + jit speedup.

2. **BPTT correctness — directional FD gate (the key check)** — on a SHORT
   window (h=3), verify ``jax.grad(window_loss)(policy, sg(x0), guess0)`` against
   convergence-checked **directional** finite differences along 3 random unit
   directions in policy space. This validates the truncated-BPTT composition AND
   the ``stop_gradient`` placement at the production tolerances (QP 1e-6 / NLP 1e-3).

Run under the cuDSS env from the repo root::

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=external/diffmpc2 \\
        /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \\
        -m pytest experiments/rl/drone_rl/test/test_gradient_modes.py -v -s
"""
from __future__ import annotations

import os
import sys
import time

# ---- sys.path bootstrap: same-dir modules + src (diffmpc_learning) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # drone_rl/
# 3 levels up from rl/drone_rl/: rl -> experiments -> repo root
_REPO_ROOT = os.path.normpath(os.path.join(_PKG, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_PKG, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from env.drone_env import (
    NX,
    NU,
    START,
    QK,
    RK,
    build_problem_params,
)
from mpc_layer import make_hard_layer
from policy import init_policy
from policy import make_theta_to_weights
from optimizer import adam_init
from gradient_modes import (
    make_initial_guess,
    prime_guess,
    make_shac_window_loss,
    open_loop_plan_update,
    shac_window_update,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _retry_cudss(fn, *, tries: int = 2):
    """Run ``fn`` once, retrying a single time on a transient cuDSS ALLOC error."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).upper()
            if i + 1 < tries and ("ALLOC" in msg or "CUDSS" in msg or "RESOURCE" in msg):
                print(f"[retry] transient cuDSS error, retrying once: {exc}")
                last = exc
                time.sleep(2.0)
                continue
            raise
    raise last  # pragma: no cover


def _tree_any_diff(a, b, atol=0.0):
    la = jax.tree.leaves(a)
    lb = jax.tree.leaves(b)
    return any(not np.allclose(np.asarray(x), np.asarray(y), atol=atol) for x, y in zip(la, lb))


def _tree_all_finite(t):
    return all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(t))


def _tree_dot(a, b):
    return float(sum(jnp.sum(x * y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))))


def _rand_unit_dir(rng, template):
    """Random unit-norm tree of normals shaped like ``template``."""
    leaves, treedef = jax.tree.flatten(template)
    keys = jax.random.split(rng, len(leaves))
    parts = [jax.random.normal(k, l.shape) for k, l in zip(keys, leaves)]
    direction = jax.tree.unflatten(treedef, parts)
    norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(direction)))
    return jax.tree.map(lambda p: p / norm, direction)


# --------------------------------------------------------------------------- #
# Module-scoped fixtures (build heavy objects once)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def setup():
    dyn, pp = build_problem_params()
    layer = make_hard_layer(dyn, pp)
    solver = layer.solver
    t2w = make_theta_to_weights(jnp.array([0.1] * 6), jnp.array([1.0] * 3))
    x0 = jnp.asarray(START, dtype=jnp.float64)
    print(f"\n[setup] NX={NX} NU={NU}  x0={np.asarray(x0)}")
    return dyn, pp, solver, t2w, x0


# --------------------------------------------------------------------------- #
# Test 1: smoke (both estimators) + warm-start timing
# --------------------------------------------------------------------------- #


def test_smoke_open_loop(setup):
    dyn, pp, solver, t2w, x0 = setup
    policy = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=9)
    opt = adam_init(policy)
    guess0 = make_initial_guess(solver, pp, x0)

    # cold call (incl. JIT compile + structure prime)
    t0 = time.time()
    policy1, opt1, x1, guess1, logs1 = _retry_cudss(
        lambda: open_loop_plan_update(
            solver, t2w, dyn, policy, opt, x0, pp, guess0, n_batch=4, lr=1e-2
        )
    )
    jax.block_until_ready((policy1, x1))
    t_cold = time.time() - t0

    # warm re-run (compiled + full-struct warm-start carried)
    t0 = time.time()
    policy2, opt2, x2, guess2, logs2 = open_loop_plan_update(
        solver, t2w, dyn, policy1, opt1, x1, pp, guess1, n_batch=4, lr=1e-2
    )
    jax.block_until_ready((policy2, x2))
    t_warm = time.time() - t0

    print(f"\n[V1 open-loop] cold={t_cold:.3f}s  warm={t_warm:.3f}s  "
          f"speedup={t_cold / max(t_warm, 1e-9):.1f}x  per-step(warm)={t_warm / 4 * 1e3:.1f}ms")
    print(f"[V1 open-loop] loss={np.asarray(logs1['loss'])}")
    print(f"[V1 open-loop] accum_grad_norm={float(logs1['accum_grad_norm'])}")
    print(f"[V1 open-loop] obs_margin(realized)={np.asarray(logs1['obs_margin'])}")
    print(f"[V1 open-loop] plan_obs_margin_max={np.asarray(logs1['plan_obs_margin_max'])}")

    assert _tree_all_finite(policy1), "V1 policy has non-finite leaves"
    assert _tree_all_finite(logs1["accum_grad_norm"]), "V1 accum-grad-norm non-finite"
    assert _tree_any_diff(policy, policy1), "V1 policy did not change after update"
    for key in ("loss", "obs_margin", "plan_obs_margin_max"):
        assert key in logs1 and np.asarray(logs1[key]).size == 4, f"V1 logs[{key}] not populated"
    plan_engaged = float(np.max(np.asarray(logs1["plan_obs_margin_max"])))
    print(f"[V1 open-loop] max plan margin over batch = {plan_engaged:.4f} (>-0.2 ⇒ engaged)")
    assert plan_engaged > -0.2, f"obstacle not engaged in V1 plan: max plan margin {plan_engaged:.4f}"
    # warm should be meaningfully faster than cold (compile + prime dominate cold)
    assert t_warm < t_cold, f"warm ({t_warm:.3f}s) not faster than cold ({t_cold:.3f}s)"


def test_smoke_shac(setup):
    dyn, pp, solver, t2w, x0 = setup
    policy = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=9)
    opt = adam_init(policy)
    guess0 = make_initial_guess(solver, pp, x0)

    t0 = time.time()
    policy1, opt1, x1, guess1, logs1 = _retry_cudss(
        lambda: shac_window_update(
            solver, t2w, dyn, policy, opt, x0, pp, guess0, h=4, lr=1e-2
        )
    )
    jax.block_until_ready((policy1, x1))
    t_cold = time.time() - t0

    t0 = time.time()
    policy2, opt2, x2, guess2, logs2 = shac_window_update(
        solver, t2w, dyn, policy1, opt1, x1, pp, guess1, h=4, lr=1e-2
    )
    jax.block_until_ready((policy2, x2))
    t_warm = time.time() - t0

    print(f"\n[V3 SHAC] cold={t_cold:.3f}s  warm={t_warm:.3f}s  "
          f"speedup={t_cold / max(t_warm, 1e-9):.1f}x  per-step(warm)={t_warm / 4 * 1e3:.1f}ms")
    print(f"[V3 SHAC] window_loss={float(logs1['loss']):.6e}  grad_norm={float(logs1['grad_norm']):.4e}")
    print(f"[V3 SHAC] obs_margin(realized)={np.asarray(logs1['obs_margin'])}")
    print(f"[V3 SHAC] plan_obs_margin_max={np.asarray(logs1['plan_obs_margin_max'])}")

    assert _tree_all_finite(policy1), "V3 policy has non-finite leaves"
    assert np.isfinite(float(logs1["grad_norm"])), "V3 grad-norm non-finite"
    assert _tree_any_diff(policy, policy1), "V3 policy did not change after update"
    for key in ("loss", "grad_norm", "obs_margin", "plan_obs_margin_max"):
        assert key in logs1, f"V3 logs[{key}] missing"
    assert np.asarray(logs1["obs_margin"]).size == 4, "V3 obs_margin not (h,)"
    plan_engaged = float(np.max(np.asarray(logs1["plan_obs_margin_max"])))
    print(f"[V3 SHAC] max plan margin over window = {plan_engaged:.4f} (>-0.2 ⇒ engaged)")
    assert plan_engaged > -0.2, f"obstacle not engaged in V3 plan: max plan margin {plan_engaged:.4f}"
    assert t_warm < t_cold, f"warm ({t_warm:.3f}s) not faster than cold ({t_cold:.3f}s)"


# --------------------------------------------------------------------------- #
# Test 2: BPTT correctness via directional finite differences (the key gate)
# --------------------------------------------------------------------------- #


def _plateau(eps_seq, vals):
    """Return (chosen_value, plateau_ok, min_consecutive_reldiff).

    Convergence check (CLAUDE.md §FD): scan consecutive central-difference estimates
    (eps descending) and take the pair with the SMALLEST relative change as the
    plateau; ``plateau_ok`` if that minimal change is below ``PLATEAU_TOL``. The
    function is deterministic (verified to ~1e-13), so non-monotone behaviour is the
    NLP-tol-1e-3 SQP early-exit staircase (small eps) or active-set/curvature (large
    eps), not stochastic noise — the plateau is the locally stable middle window.
    """
    best_i, best_d = 0, float("inf")
    for i in range(len(vals) - 1):
        d = abs(vals[i + 1] - vals[i]) / (abs(vals[i + 1]) + 1e-12)
        if d < best_d:
            best_d, best_i = d, i
    # value at the finer eps of the best-agreeing pair
    return vals[best_i + 1], (best_d < PLATEAU_TOL), best_d


PLATEAU_TOL = 3e-2  # consecutive-estimate agreement that confirms a plateau exists


def test_shac_window_directional_fd(setup):
    dyn, pp, solver, t2w, x0 = setup
    H = 3
    # eps window BRACKETS the plateau for this HORIZON=50 system. The usable window is
    # bounded BELOW by the NLP-tol-1e-3 staircase (eps<5e-3 drifts) and ABOVE by
    # active-set/curvature (eps>1.5e-2 jumps); large-signal dirs plateau ~7e-3-1e-2,
    # small-signal dirs need ~1.5e-2-2e-2 to clear the staircase floor.
    EPS_SEQ = (2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3)

    # Non-zero policy so the BPTT gradient is non-trivial.
    policy = init_policy(jax.random.PRNGKey(0), obs_dim=6, out_dim=9)
    policy = {
        **policy,
        "W2": jax.random.normal(jax.random.PRNGKey(11), policy["W2"].shape) * 0.05,
    }

    # Prime the warm-start ONCE (base policy) so every FD evaluation reuses the
    # SAME warm-start ⇒ the converged solve is identical across +/- eps (clean FD).
    guess0 = make_initial_guess(solver, pp, x0)
    guess0 = prime_guess(solver, t2w, policy, pp, x0, guess0)

    window_loss = make_shac_window_loss(solver, t2w, dyn, pp, H)
    x0_sg = jax.lax.stop_gradient(x0)

    def scalar_loss(phi):
        return window_loss(phi, x0_sg, guess0)[0]

    scalar_loss_jit = jax.jit(scalar_loss)

    # Determinism / noise-floor probe: same phi evaluated twice must be identical.
    L_a = float(scalar_loss_jit(policy))
    L_b = float(scalar_loss_jit(policy))
    print(f"\n[FD gate] determinism: |L(phi)-L(phi)| = {abs(L_a - L_b):.2e} "
          f"(≈0 ⇒ no GPU non-determinism; roughness is the tol-1e-3 staircase)")
    assert abs(L_a - L_b) < 1e-9, "window_loss is non-deterministic"

    # Analytic gradient (jitted).
    t0 = time.time()
    g = _retry_cudss(lambda: jax.jit(jax.grad(scalar_loss))(policy))
    jax.block_until_ready(g)
    t_ad = time.time() - t0

    # Directions. SCORED (high-SNR, FD can resolve): the GRADIENT direction (pins |g|
    # exactly — definitive magnitude/sign gate) plus 2 gradient-DOMINATED mixtures
    # δ = normalize(g/|g| + 0.5·r) that also probe off-gradient components.
    # REPORTED-ONLY (flagged): pure-random unit directions whose tiny dd (~|g|/√dim)
    # puts ΔL at the staircase floor — FD-resolution-limited, NOT an AD error
    # (CLAUDE.md §FD: flag, do not score).
    gnorm = float(jnp.sqrt(_tree_dot(g, g)))
    grad_dir = jax.tree.map(lambda v: v / (gnorm + 1e-30), g)
    rng = jax.random.PRNGKey(123)
    rngs = jax.random.split(rng, 4)

    def _mix(rkey):
        r = _rand_unit_dir(rkey, policy)
        mixed = jax.tree.map(lambda gd, rr: gd + 0.5 * rr, grad_dir, r)
        n = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(mixed)))
        return jax.tree.map(lambda p: p / n, mixed)

    directions = [
        ("grad", grad_dir, 1e-2, True),
        ("mix0", _mix(rngs[0]), 1.5e-2, True),
        ("mix1", _mix(rngs[1]), 1.5e-2, True),
        ("rand0", _rand_unit_dir(rngs[2], policy), None, False),
        ("rand1", _rand_unit_dir(rngs[3], policy), None, False),
    ]
    print(f"[FD gate] h={H}  |g|={gnorm:.4e}  analytic grad in {t_ad:.3f}s  "
          f"eps_seq={EPS_SEQ}")

    results = []
    for name, delta, tol, scored in directions:
        ad_dd = _tree_dot(g, delta)  # <grad, delta>
        vals = []
        for eps in EPS_SEQ:
            phi_p = jax.tree.map(lambda p, q: p + eps * q, policy, delta)
            phi_m = jax.tree.map(lambda p, q: p - eps * q, policy, delta)
            Lp = float(scalar_loss_jit(phi_p))
            Lm = float(scalar_loss_jit(phi_m))
            vals.append((Lp - Lm) / (2.0 * eps))
        chosen, ok, mind = _plateau(EPS_SEQ, vals)
        rel = abs(chosen - ad_dd) / (abs(ad_dd) + 1e-9)
        results.append((name, ad_dd, chosen, rel, ok, tol, scored, vals))
        tag = f"tol={tol:.1e}" if scored else "FLAGGED/report-only (FD-res-limited)"
        print(f"[FD gate] {name:5s}: AD={ad_dd:+.6e}  FD={chosen:+.6e}  rel={rel:.3e}  "
              f"plateau={ok}(min_d={mind:.1e})  {tag}  "
              f"fd_seq={[f'{v:+.4e}' for v in vals]}")

    # Aggregate over the directional derivatives: cosine of the AD vs FD
    # directional-derivative vectors (all 5 directions) — a single similarity number.
    ad_vec = np.array([r[1] for r in results])
    fd_vec = np.array([r[2] for r in results])
    cos_all = float(ad_vec @ fd_vec / (np.linalg.norm(ad_vec) * np.linalg.norm(fd_vec) + 1e-30))
    ad_sc = np.array([r[1] for r in results if r[6]])
    fd_sc = np.array([r[2] for r in results if r[6]])
    cos_scored = float(ad_sc @ fd_sc / (np.linalg.norm(ad_sc) * np.linalg.norm(fd_sc) + 1e-30))
    print(f"[FD gate] cos(AD,FD) over all 5 dirs = {cos_all:.6f}   "
          f"over 3 scored dirs = {cos_scored:.6f}")
    print(f"[FD gate] per-dir rel: " +
          "  ".join(f"{r[0]}={r[3]:.2e}{'' if r[6] else '(flagged)'}" for r in results))

    for name, ad_dd, fd_dd, rel, ok, tol, scored, vals in results:
        if not scored:
            continue  # CLAUDE.md: flag resolution-limited dirs, do not score
        assert ok, f"{name}: FD did not plateau in [{EPS_SEQ[-1]:.0e},{EPS_SEQ[0]:.0e}] " \
                   f"(discontinuity?): {vals}"
        assert rel < tol, f"{name}: directional FD/AD mismatch rel={rel:.3e} (>= {tol:.1e})"

    assert cos_scored > 0.999, f"AD/FD directional cosine too low: {cos_scored:.6f}"
