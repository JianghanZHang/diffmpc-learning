"""Diff-WMPC BARRIER gradient estimators (V2/V4) on the diffmpc2 LogBarrier solver.

V2 = plan_barrier : V1's open-loop-plan myopic accumulation, barrier MPC layer.
V4 = bptt_barrier : V3's SHAC truncated-BPTT window, barrier MPC layer
                    (true ∂u*₀/∂x₀ feedback via the layer's initial_state cotangent).

The layer is the CANONICAL external/diffmpc2 log-barrier solve
(``logbarrier_nlp_solve`` forward — eager SQP, cuDSS inner ADMM, outer-slack
FTB+filter globalization — and the κ-relaxed W-fold backward
``_relaxed_nlp_backward``, both FD-verified 2026-07-02). It is EAGER (a Python-loop
SQP), so V2/V4 are eager Python loops rather than jitted ``lax.scan``s: under
eager ``jax.grad`` the ``custom_vjp`` forward runs on CONCRETE primals, which this
module exploits for a tracing-invisible receding-horizon warm start (a mutable
cell holding the previous solution, shifted one step, fed through
``program.initial_guess`` — the exact analog of ``gradient_modes.shift_guess``,
primal-only). Warm-start state is saved/restored around evals so evaluation never
clobbers the training warm start.

Semantics mirror ``gradient_modes`` exactly: V2 detaches the NN input and
``initial_state`` per step (only the weights carry the policy gradient); V4
attaches the live state through ``simulate_step`` + the MPC feedback within an
h-step window; warm starts carry no gradient in either.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "diffmpc2")
for _p in (_HERE, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from turbompc.solvers.backward.logbarrier_backward import (  # noqa: E402
    logbarrier_nlp_solve,
    logbarrier_nlp_solve_jit,
    _relaxed_nlp_backward,
    _build_problem_params_cotangent,
)

from optimizer import adam_step  # noqa: E402

# Verified regime (2026-07-02 gates + globalization probes): elastic barrier
# kappa=1e-4 / gamma=1e2, filter globalization. Training tolerance matches the
# hard arms' convention (NLP-KKT 1e-3 for speed; QP/inner tol stays tight).
# forward="fused" (2026-07-06): WARM solves go through the JITTED fused-CUDA SQP
# (logbarrier_nlp_solve_jit, full steps, fixed kappa); the COLD solve after each
# reset stays on the eager globalized path (filter + restoration). "eager" = the
# pre-07-06 all-eager behavior.
DEFAULT_LB_CFG = dict(
    target_kappa=1.0e-4,
    slack_weight=1.0e2,
    use_slack=True,
    include_ineq_hessian=True,
    max_sqp_iter=15,
    sqp_tol=1.0e-3,
    globalization="filter",
    forward="fused",
    # Fused-loop iteration budget (jit while_loop exits early when converged, so a
    # larger cap only costs time on the hard instances that need it). The fused loop
    # is globalized in-jit (relaxed-barrier merit LS) — no rescue path.
    fused_max_sqp_iter=40,
    # Backward dual/W sourcing. BOTH modes: analytic dual + clearance-W (gated: at
    # sqp_tol=1e-3 crossing states, cos=0.999988 vs the tight reference; the cached
    # dual-W alternative measured WORSE there — cos 0.50 point / 0.988 window,
    # 2026-07-07 w_mode_gate). bwd_w_cap (pure only): clamp the fold weight to
    # [0, cap] — the 1e-30 clearance clip otherwise drives |W| to ~1e26..1e56 on
    # tolerance-level crossed rows (frequent: 8/14 solves), which saturates the same
    # hard-active limit as W~1e6 but risks f64 cancellation in the reduced KKT
    # (leading hypothesis for V4p's intermittent ~2e4 grad spikes).
    bwd_yg_mode="analytic",
    # Pure backward dual/W sourcing. Default = retraction-SMOOTHED reconstruction
    # (gamma_eff=1e8): bounded W and bounded Hessian dual — never produced a
    # converged-solve spike in any run. "Case 1" (y=kappa/s_raw, W=y^2/kappa,
    # bwd_w_from_dual=True) gated clean at single states but produced a 2.8e7 grad
    # spike with all solves converged in the LS-fused run (upd 8) — unbounded-W tail
    # suspect, kept as a non-default option. "Case 2" (cached forward dual) failed
    # its gate outright (cos 0.52; one-anchor-behind staleness).
    bwd_w_from_dual="auto",       # "auto": False (Case-1 demoted after the upd-8 spike)
    bwd_w_cap="auto",             # "auto": off
    bwd_pure_smooth_gamma="auto",  # "auto": 1e8 for pure (bounded W + bounded Hessian dual)
    inner_cfg=dict(
        rho_bar=0.1, sigma=1e-6, rho_f_factor=1000.0, alpha=1.6,
        tol=1e-9, max_iter=5000, check_termination_every=25,
        adapt_rho_every=25, adaptive_rho_tolerance=5.0,
    ),
)


def _shift_primal(states, controls):
    """Receding-horizon shift: roll (states, controls) one step, repeat the tail."""
    st = np.asarray(states)
    co = np.asarray(controls)
    return (jnp.asarray(np.concatenate([st[1:], st[-1:]], axis=0)),
            jnp.asarray(np.concatenate([co[1:], co[-1:]], axis=0)))


def make_barrier_ws_layer(solver, cfg=None):
    """Warm-started differentiable LogBarrier layer over an existing TurboMPCSolver.

    Replicates ``make_logbarrier_diff``'s custom_vjp wiring, with a mutable
    warm-start cell updated INSIDE the (concrete) forward: each solve is
    warm-started from the previous solution shifted one step. ``layer.solve(pp, w)``
    is differentiable w.r.t. ``weights`` and ``pp["initial_state"]``.

    Attributes: ``layer.solver``, ``layer.cfg``, ``layer.cell`` (dict with
    "guess"/"stats"), ``layer.reset(guess_or_none)``, ``layer.snapshot()``,
    ``layer.restore(snap)``.
    """
    cfg = {**DEFAULT_LB_CFG, **(cfg or {})}
    if cfg["bwd_w_from_dual"] == "auto":
        # Case-1 (y=kappa/s_raw, W=y^2/kappa) DEMOTED to non-default 2026-07-07: in the
        # LS-fused production run it produced a grad spike of 2.8e7 at upd 8 with ALL
        # solves converged (n_nonconv=0) — the unbounded W=kappa/s_raw^2 tail on
        # tolerance-level crossing rows is the suspect (exact replay pending better
        # spike checkpoints). The bounded smooth backward has never produced a
        # converged-solve spike in any run.
        cfg["bwd_w_from_dual"] = False
    if cfg["bwd_pure_smooth_gamma"] == "auto":
        cfg["bwd_pure_smooth_gamma"] = 1.0e8 if not cfg["use_slack"] else None
    if cfg["bwd_w_cap"] == "auto":
        cfg["bwd_w_cap"] = None
    cell = {"guess": None, "iters": [], "convs": []}

    # JITTED fused-CUDA fast path (warm solves): one compiled call per solve. Closed
    # over solver/cfg (static); pp/weights/warm-start primals are traced arguments.
    @jax.jit
    def _fused_solve_jit(pp, w, states0, controls0):
        return logbarrier_nlp_solve_jit(
            solver, pp, w, states0, controls0,
            slack_weight=cfg["slack_weight"], target_kappa=cfg["target_kappa"],
            use_slack=cfg["use_slack"], max_sqp_iter=cfg["fused_max_sqp_iter"],
            sqp_tol=cfg["sqp_tol"], inner_cfg=cfg["inner_cfg"])

    def _eager_solve(pp, w, max_sqp_iter=None):
        """Eager globalized solve (filter + restoration) — cold starts + fallback."""
        orig_ig = solver.program.initial_guess
        if cell["guess"] is not None:
            g = cell["guess"]
            solver.program.initial_guess = lambda params=None: g
        try:
            return logbarrier_nlp_solve(
                solver, pp, w,
                slack_weight=cfg["slack_weight"], target_kappa=cfg["target_kappa"],
                use_slack=cfg["use_slack"],
                max_sqp_iter=max_sqp_iter or cfg["max_sqp_iter"],
                sqp_tol=cfg["sqp_tol"], globalization=cfg["globalization"],
                inner_cfg=cfg["inner_cfg"])
        finally:
            solver.program.initial_guess = orig_ig

    def _fwd_solve(pp, w):
        if cfg["forward"] == "fused" and cell["guess"] is not None:
            states0, controls0 = cell["guess"]
            sol = _fused_solve_jit(pp, w, states0, controls0)
            res = {
                "states": sol.states, "controls": sol.controls,
                "num_iter": int(sol.num_iter), "final_conv": float(sol.final_conv),
                "kappas": jnp.asarray([cfg["target_kappa"]]),
                "y_f_dyn": sol.y_f_dyn, "y_g_stacked": sol.y_g_stacked,
            }
            # No rescue (user decision 2026-07-07): the fused loop is globalized
            # in-jit (relaxed-barrier merit LS, cap 40) — non-convergence is
            # expected not to occur; per-solve final_conv stays recorded in the
            # cell stats and surfaced per update by train.py (fwd_conv_max /
            # fwd_n_nonconv), so any residual failure is visible, not silent.
        else:
            res = _eager_solve(pp, w)
        cell["guess"] = _shift_primal(res["states"], res["controls"])
        cell["iters"].append(int(res["num_iter"]))
        cell["convs"].append(float(res["final_conv"]))
        return res

    @jax.custom_vjp
    def solve(problem_params, weights):
        res = _fwd_solve(problem_params, weights)
        return res["states"], res["controls"]

    def solve_fwd(problem_params, weights):
        res = _fwd_solve(problem_params, weights)
        residual = (res["states"], res["controls"], float(res["kappas"][-1]),
                    problem_params, weights, res["y_f_dyn"], res["y_g_stacked"])
        return (res["states"], res["controls"]), residual

    # JITTED backward (needs yg_crosscheck_tol=None — the concrete crosscheck is
    # skipped, trace-safe since the 2026-07-06 solver-side guard). Returns arrays only
    # (info carries a string and is not jit-returnable).
    @jax.jit
    def _bwd_jit(pp, w, states_c, controls_c, final_kappa, d_states, d_controls,
                 y_f_dyn_c, y_g_stacked_c):
        dL_dweights, dL_dx_init, _info = _relaxed_nlp_backward(
            solver, pp, w, states_c, controls_c, final_kappa,
            d_states, d_controls,
            slack_weight=cfg["slack_weight"], use_slack=cfg["use_slack"],
            include_ineq_hessian=cfg["include_ineq_hessian"],
            y_f_dyn_c=y_f_dyn_c, y_g_stacked_c=y_g_stacked_c,
            # Analytic dual + clearance-W (gated best at training tol); pure mode
            # additionally caps the fold weight (see DEFAULT_LB_CFG). Crosscheck
            # disabled at training tolerance (the strict assert would abort mid-run).
            yg_mode=cfg["bwd_yg_mode"], yg_crosscheck_tol=None,
            w_from_dual=cfg["bwd_w_from_dual"], w_cap=cfg["bwd_w_cap"],
            pure_gamma_smooth=cfg["bwd_pure_smooth_gamma"])
        return dL_dweights, dL_dx_init

    def solve_bwd(residual, cot):
        d_states, d_controls = cot
        (states_c, controls_c, final_kappa, problem_params, weights,
         y_f_dyn_c, y_g_stacked_c) = residual
        dL_dweights, dL_dx_init = _bwd_jit(
            problem_params, weights, states_c, controls_c, final_kappa,
            d_states, d_controls, y_f_dyn_c, y_g_stacked_c)
        problem_params_cotangent = _build_problem_params_cotangent(
            solver, problem_params, dL_dx_init)
        return (problem_params_cotangent, dL_dweights)

    solve.defvjp(solve_fwd, solve_bwd)

    class _Layer:
        def __init__(self):
            self.solver = solver
            self.cfg = cfg
            self.cell = cell
            self.solve = solve

        def __call__(self, pp, w):
            return solve(pp, w)

        def reset(self, guess=None):
            cell["guess"] = guess

        def snapshot(self):
            return cell["guess"]

        def restore(self, snap):
            cell["guess"] = snap

        def pop_stats(self):
            iters, convs = cell["iters"], cell["convs"]
            cell["iters"], cell["convs"] = [], []
            return iters, convs

    return _Layer()


def _grad_norm(g):
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree.leaves(g)))


# --------------------------------------------------------------------------- #
# V2 — open-loop plan update (barrier layer, eager)
# --------------------------------------------------------------------------- #

def open_loop_plan_update_barrier(
    layer, theta_to_weights, dyn, policy, opt_state, x, pp_base, *,
    n_batch, lr, simulate_step, task_loss, obs_margin,
):
    """V2 — V1's per-step myopic open-loop-plan accumulation on the barrier layer.

    Mirrors ``gradient_modes.open_loop_plan_update``: per step, detach the state,
    ``value_and_grad`` of the FULL open-loop plan cost w.r.t. the policy (only the
    weights carry gradient), advance with the plan's first control (forward only).
    Returns ``(policy, opt_state, x, logs)`` (warm start lives inside the layer).
    """
    accum = jax.tree.map(jnp.zeros_like, policy)
    Ls, margins, plan_margins = [], [], []
    for _k in range(n_batch):
        xk = jax.lax.stop_gradient(x)

        def step_loss(phi):
            w = theta_to_weights(phi, xk)
            st, co = layer.solve({**pp_base, "initial_state": xk}, w)
            return task_loss(st, co), (st, co)

        (L_k, (st, co)), g_k = jax.value_and_grad(step_loss, has_aux=True)(policy)
        accum = jax.tree.map(lambda a, b: a + b, accum, g_k)
        u0 = jax.lax.stop_gradient(co[0])
        x = jax.lax.stop_gradient(simulate_step(dyn, xk, u0))
        Ls.append(float(L_k))
        margins.append(float(obs_margin(x)))
        plan_margins.append(float(jnp.max(obs_margin(st))))

    accum_mean = jax.tree.map(lambda a: a / n_batch, accum)
    policy, opt_state = adam_step(policy, accum_mean, opt_state, lr=lr)
    logs = {
        "loss": jnp.asarray(Ls),
        "obs_margin": jnp.asarray(margins),
        "plan_obs_margin_max": jnp.asarray(plan_margins),
        "accum_grad_norm": _grad_norm(accum_mean),
    }
    return policy, opt_state, jax.lax.stop_gradient(x), logs


# --------------------------------------------------------------------------- #
# V4 — SHAC truncated-BPTT window update (barrier layer, eager)
# --------------------------------------------------------------------------- #

def shac_window_update_barrier(
    layer, theta_to_weights, dyn, policy, opt_state, x, pp_base, *,
    h, lr, simulate_step, task_loss, obs_margin,
):
    """V4 — V3's SHAC truncated-BPTT window on the barrier layer.

    Mirrors ``gradient_modes.shac_window_update``: within the window the NN reads
    the LIVE state and the OCP ``initial_state`` is attached, so grad flows through
    ``simulate_step`` + the barrier layer's ∂u*₀/∂x₀ + ∂/∂weights. Warm starts are
    internal to the layer (concrete, gradient-free by construction).
    """
    x0 = jax.lax.stop_gradient(x)

    def window_loss(phi):
        xw = x0
        L = jnp.zeros((), dtype=x0.dtype)
        margins, plan_margins = [], []
        for _t in range(h):
            w = theta_to_weights(phi, xw)                    # x ATTACHED
            st, co = layer.solve({**pp_base, "initial_state": xw}, w)
            u0 = co[0]
            xw = simulate_step(dyn, xw, u0)
            L = L + task_loss(xw[None], u0[None])            # REALIZED state cost
            margins.append(obs_margin(xw))
            plan_margins.append(jnp.max(obs_margin(st)))
        return L, (xw, jnp.stack(margins), jnp.stack(plan_margins))

    (L, (x_final, margins, plan_margins)), g = jax.value_and_grad(
        window_loss, has_aux=True)(policy)
    policy, opt_state = adam_step(policy, g, opt_state, lr=lr)
    logs = {
        "loss": L,
        "grad_norm": _grad_norm(g),
        "obs_margin": margins,
        "plan_obs_margin_max": plan_margins,
    }
    return policy, opt_state, jax.lax.stop_gradient(x_final), logs


# --------------------------------------------------------------------------- #
# Closed-loop evaluation with the barrier controller
# --------------------------------------------------------------------------- #

def closed_loop_eval_barrier(layer, theta_to_weights, dyn, policy, pp_base, x0,
                             *, n_steps, env):
    """Realized closed-loop metrics deploying the BARRIER MPC (mirrors
    ``train.closed_loop_eval``'s metric definitions). Saves/restores the layer's
    training warm start; the eval itself starts cold-from-default at ``x0``."""
    simulate_step, task_loss = env.simulate_step, env.task_loss
    obs_margin, goal_dist = env.obs_margin, env.goal_dist

    snap = layer.snapshot()
    layer.reset(None)
    x = jax.lax.stop_gradient(jnp.array(x0, dtype=jnp.float64))
    total_cost, margins = 0.0, []
    for _t in range(n_steps):
        w = theta_to_weights(policy, x)
        st, co = layer.solve({**pp_base, "initial_state": x}, w)
        u0 = jax.lax.stop_gradient(co[0])
        x = jax.lax.stop_gradient(simulate_step(dyn, x, u0))
        total_cost += float(task_loss(x[None], u0[None]))
        margins.append(float(obs_margin(x)))
    layer.restore(snap)

    margins = np.asarray(margins)
    return {
        "cost": float(total_cost),
        "closest_margin": float(np.max(margins)),
        "n_grazing": int(np.sum(margins > -0.1)),
        "n_violations": int(np.sum(margins > 0.0)),
        "final_dist_to_goal": float(goal_dist(x)),
    }
