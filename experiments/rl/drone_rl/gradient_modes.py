"""Two differentiable-MPC policy-gradient estimators for the drone Diff-WMPC task.

A state-input NN (``policy``) emits MPC cost weights (via ``theta_to_weights``); the
hard-box TurboMPC layer (``solver``) solves the OCP and we backprop a task loss into
the policy. This module provides the TWO update functions we compare:

* ``open_loop_plan_update``  — **V1**, the paper's Algorithm 1: per-step *myopic*
  accumulation. The NN input and the OCP ``initial_state`` are DETACHED, so the only
  gradient path is the open-loop-plan sensitivity ``∂L/∂z*·∂z*/∂θ`` (through the cost
  weights). NO backprop through the closed-loop rollout.

* ``shac_window_update``     — **V3**, SHAC-style truncated BPTT (no value bootstrap):
  ``x0`` is detached at window ENTRY only; the realized state stays ATTACHED *within*
  the window, so the gradient flows through ``simulate_step`` AND the MPC feedback
  ``∂u*₀/∂x₀`` AND ``∂/∂weights`` across ``h`` steps.

Warm-start + JIT (essential for speed)
--------------------------------------
The previous solve's solution is threaded as the next solve's ``guess``
(receding-horizon warm-start). ``solver.solve`` ``stop_gradient``'s its ``guess`` arg
(its ``custom_vjp`` backward returns ``None`` for it — see
``turbompc_solver.py:1092``), so the warm-start affects ONLY the forward iteration
count (→ ~1 SQP iter → ms), NOT the gradient. The per-step grad (V1) and the
``h``-step window (V3) are jitted; ``solver.solve`` is jittable & scannable.

Structure priming
------------------
``solver.initial_guess(pp)`` returns a solution with ``admm_state=None`` /
``kkt_state=None``, whereas ``solver.solve(...)`` returns those fields *populated*.
That pytree-structure change would break a ``jax.lax.scan`` carry (V3) and force a
JIT re-trace (V1). ``prime_guess`` runs one ``stop_gradient``'d forward solve to lift a
fresh ``initial_guess`` to the full solve-output structure; both updates call it and
the carried ``guess`` thereafter is already full-structure (so priming is a no-op).
"""
from __future__ import annotations

import os
import sys

# ---- sys.path bootstrap (same-dir modules + src), mirrors mpc_layer.py ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "../../../"))
_SRC = os.path.join(_REPO_ROOT, "src")
_TURBOMPC = os.path.join(_REPO_ROOT, "external", "turbompc")
for _p in (_HERE, _SRC, _TURBOMPC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Default env functions (drone) — used when callers don't inject an env explicitly,
# so the existing drone pipeline/tests keep working unchanged. The nonlinear quadrotor
# (quadrotor_env.py) injects its OWN simulate_step/task_loss/obs_margin via the keyword
# args on the public update functions below.
from drone_env import (
    obs_margin as _default_obs_margin,
    simulate_step as _default_simulate_step,
    task_loss as _default_task_loss,
)
from optimizer import adam_step


def _resolve_env(simulate_step, task_loss, obs_margin):
    """Return (simulate_step, task_loss, obs_margin), falling back to the drone defaults."""
    return (
        _default_simulate_step if simulate_step is None else simulate_step,
        _default_task_loss if task_loss is None else task_loss,
        _default_obs_margin if obs_margin is None else obs_margin,
    )

# Caches so the jitted per-step grad (V1) / window grad (V3) are built ONCE per
# (solver, theta_to_weights, pp_base[, dyn, h]). Keyed by object id — pass a STABLE
# pp_base object across training steps for the cache to hit (the receding-horizon
# update is via the ``initial_state`` override, not a fresh pp_base each step).
_OPEN_LOOP_CACHE: dict = {}
_SHAC_WINDOW_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def _grad_norm(g):
    return jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(g)))


def make_initial_guess(solver, pp_base, x):
    """Initial warm-start guess for the first update call.

    Returns ``solver.initial_guess`` for ``pp_base`` with ``initial_state=x``. This is
    structurally *minimal* (``admm_state``/``kkt_state`` are ``None``); the update
    functions ``prime_guess`` it to the full solve-output structure on first use.
    """
    return solver.initial_guess({**pp_base, "initial_state": x})


def prime_guess(solver, theta_to_weights, policy, pp_base, x, guess):
    """Lift a fresh ``initial_guess`` to the full solve-output pytree structure.

    No-op if ``guess`` is already full-structure (``admm_state`` populated). Runs one
    ``stop_gradient``'d forward solve at the current policy/state, so the returned
    guess (a) matches the structure of every subsequent ``solver.solve`` output and
    (b) carries no gradient.
    """
    if guess.admm_state is not None:
        return guess
    xk = jax.lax.stop_gradient(x)
    w0 = theta_to_weights(policy, xk)
    sol = solver.solve(guess, {**pp_base, "initial_state": xk}, w0)
    return jax.lax.stop_gradient(sol)


def _roll1(a):
    """Roll axis-0 by -1, repeating the last row (one-step horizon shift)."""
    return jnp.concatenate([a[1:], a[-1:]], axis=0)


def shift_guess(sol):
    """Receding-horizon shift of a solve solution for use as the NEXT warm-start guess.

    Rolls the primal trajectory (states, controls) AND the ADMM duals (x_blocks, y_g,
    y_f_dyn, z_g, xi_g) forward by one step, so the guess's trajectory starts at the
    *predicted* next state (= the realized next state here, since the env has no model
    mismatch). This is the standard receding-horizon warm-start: it cuts warm-started
    solves from ~3-7 SQP iters (raw/unshifted, off-by-one guess) to ~1-2 (measured ~4x
    wall-clock). ``y_f_0`` (initial-condition dual) and ``rho_bar`` are kept as-is.

    GRADIENT-SAFE: the warm-start guess is ``stop_gradient``'d by ``solver.solve`` (its
    custom_vjp returns None for the guess cotangent), so shifting changes only the forward
    iteration count, not the gradient. Requires a full-structure guess (``prime_guess``).
    """
    a = sol.admm_state
    a2 = a._replace(
        x_blocks=_roll1(a.x_blocks), y_g=_roll1(a.y_g), y_f_dyn=_roll1(a.y_f_dyn),
        z_g=_roll1(a.z_g), xi_g=_roll1(a.xi_g),
    )
    return sol._replace(states=_roll1(sol.states), controls=_roll1(sol.controls), admm_state=a2)


# --------------------------------------------------------------------------- #
# V1 — open-loop plan update (paper Algorithm 1)
# --------------------------------------------------------------------------- #


def _get_open_loop_grad(solver, theta_to_weights, dyn, pp_base, n_batch,
                        *, simulate_step, task_loss, obs_margin):
    """Return a jitted ``g(policy, x0, guess0) -> (accum_grad, x_final, guess_final, aux)``.

    The paper's per-step myopic accumulation, as ONE jitted ``jax.lax.scan`` (no Python
    loop). The scan BODY computes the per-step open-loop-plan gradient locally
    (``value_and_grad`` of one solve, NN input + ``initial_state`` detached so the only
    ``phi``-path is the cost weights) and accumulates it in the carry. This is exactly
    Algorithm 1's ``sum_k grad_phi L_k`` — but each ``grad`` is local to its step (no
    reverse pass *through* the scan, which would re-materialize the warm-start carry and
    cost ~2x). The rollout state advances with the (detached) plan control; the
    (detached) solution warm-starts the next solve. ``guess0`` must be full-structure
    (``prime_guess``).
    """
    key = (id(solver), id(theta_to_weights), id(dyn), id(pp_base), int(n_batch),
           id(simulate_step), id(task_loss), id(obs_margin))
    fn = _OPEN_LOOP_CACHE.get(key)
    if fn is not None:
        return fn

    def g(policy, x0, guess0):
        accum0 = jax.tree.map(jnp.zeros_like, policy)

        def step(carry, _):
            x, guess, accum = carry
            xk = jax.lax.stop_gradient(x)  # NN input + initial_state DETACHED

            def step_loss(phi):
                w = theta_to_weights(phi, xk)  # only the weights carry the policy grad
                sol = solver.solve(guess, {**pp_base, "initial_state": xk}, w)
                return task_loss(sol.states, sol.controls), sol  # FULL open-loop plan cost z*_k

            (L_k, sol), g_k = jax.value_and_grad(step_loss, has_aux=True)(policy)
            accum = jax.tree.map(lambda a, b: a + b, accum, g_k)
            u0 = jax.lax.stop_gradient(sol.controls[0])
            x_next = jax.lax.stop_gradient(simulate_step(dyn, xk, u0))  # advance (forward only)
            guess_next = jax.lax.stop_gradient(shift_guess(sol))  # receding-horizon warm-start (detached)
            aux = (L_k, obs_margin(x_next), jnp.max(obs_margin(sol.states)))
            return (x_next, guess_next, accum), aux

        (x_f, guess_f, accum), (Ls, margins, plan_margins) = jax.lax.scan(
            step, (x0, guess0, accum0), xs=None, length=n_batch
        )
        return accum, x_f, guess_f, (Ls, margins, plan_margins)

    fn = jax.jit(g)
    _OPEN_LOOP_CACHE[key] = fn
    return fn


def open_loop_plan_update(
    solver,
    theta_to_weights,
    dyn,
    policy,
    opt_state,
    x,
    pp_base,
    guess,
    *,
    n_batch,
    lr,
    simulate_step=None,
    task_loss=None,
    obs_margin=None,
):
    """V1 — per-step myopic open-loop-plan gradient accumulation (Algorithm 1).

    For ``k in range(n_batch)``: detach the state, compute the open-loop-plan loss and
    its policy gradient (NN input + ``initial_state`` detached), accumulate the
    gradient, advance the realized state with the plan's first control (forward only),
    and warm-start the next solve with the (detached) solution. After the batch, apply
    one Adam step with the mean accumulated gradient.

    ``simulate_step`` / ``task_loss`` / ``obs_margin`` default to the drone env; pass the
    quadrotor env's functions to run the nonlinear-quadrotor task.

    Returns ``(policy, opt_state, x, guess, logs)`` where ``x``/``guess`` are carried
    (detached) for the next call and ``logs`` holds per-step ``loss`` / ``grad_norm`` /
    ``obs_margin`` (realized) / ``plan_obs_margin_max`` (over the planned trajectory).
    """
    simulate_step, task_loss, obs_margin = _resolve_env(simulate_step, task_loss, obs_margin)
    guess = prime_guess(solver, theta_to_weights, policy, pp_base, x, guess)
    g_fn = _get_open_loop_grad(solver, theta_to_weights, dyn, pp_base, n_batch,
                               simulate_step=simulate_step, task_loss=task_loss,
                               obs_margin=obs_margin)

    x0 = jax.lax.stop_gradient(x)
    accum, x_f, guess_f, (Ls, margins, plan_margins) = g_fn(policy, x0, guess)
    accum_mean = jax.tree.map(lambda a: a / n_batch, accum)
    policy, opt_state = adam_step(policy, accum_mean, opt_state, lr=lr)

    logs = {
        "loss": Ls,                                # per-step open-loop-plan losses (N_batch,)
        "obs_margin": margins,                     # realized per-step margin (N_batch,)
        "plan_obs_margin_max": plan_margins,       # max planned margin per step (N_batch,)
        "accum_grad_norm": _grad_norm(accum_mean),  # norm of the applied (mean) update
    }
    return policy, opt_state, jax.lax.stop_gradient(x_f), jax.lax.stop_gradient(guess_f), logs


# --------------------------------------------------------------------------- #
# V3 — SHAC truncated-BPTT window update (no value bootstrap)
# --------------------------------------------------------------------------- #


def make_shac_window_loss(solver, theta_to_weights, dyn, pp_base, h,
                          *, simulate_step=None, task_loss=None, obs_margin=None):
    """Build ``window_loss(phi, x0, guess0) -> (L, aux)`` for an ``h``-step BPTT window.

    ``x0`` is treated as a constant leaf (the caller passes it ``stop_gradient``'d), so
    within the window the realized state becomes ``phi``-dependent through
    ``simulate_step``. Each step the NN reads the LIVE (attached) state, the OCP
    ``initial_state`` is the attached state, and the realized state cost is accumulated
    — so ``grad_phi L`` is the truncated-BPTT gradient through ``simulate_step`` + the
    MPC feedback ``∂u*₀/∂x₀`` + ``∂/∂weights``. The warm-start ``guess`` is detached
    every step (carries no gradient). ``guess0`` must be full-structure (see
    ``prime_guess``).

    ``aux = (x_final, guess_final, obs_margin_per_step, plan_obs_margin_max_per_step)``.

    ``simulate_step`` / ``task_loss`` / ``obs_margin`` default to the drone env; pass the
    quadrotor env's functions to run the nonlinear-quadrotor task.
    """
    simulate_step, task_loss, obs_margin = _resolve_env(simulate_step, task_loss, obs_margin)

    def window_loss(phi, x0, guess0):
        def step(carry, _):
            x, guess, L = carry
            w = theta_to_weights(phi, x)  # x ATTACHED
            ppt = {**pp_base, "initial_state": x}  # ATTACHED
            sol = solver.solve(guess, ppt, w)
            u0 = sol.controls[0]
            x_next = simulate_step(dyn, x, u0)
            L = L + task_loss(x_next[None], u0[None])  # REALIZED state cost
            guess = jax.lax.stop_gradient(shift_guess(sol))  # receding-horizon warm-start (detached)
            margin = obs_margin(x_next)
            plan_margin = jnp.max(obs_margin(sol.states))
            return (x_next, guess, L), (margin, plan_margin)

        (x_final, guess_final, L), (margins, plan_margins) = jax.lax.scan(
            step,
            (x0, guess0, jnp.zeros((), dtype=x0.dtype)),
            xs=None,
            length=h,
        )
        return L, (x_final, guess_final, margins, plan_margins)

    return window_loss


def _get_shac_window(solver, theta_to_weights, dyn, pp_base, h,
                     *, simulate_step, task_loss, obs_margin):
    """Return a jitted ``value_and_grad`` (w.r.t. ``phi``) of the ``h``-step window."""
    key = (id(solver), id(theta_to_weights), id(dyn), id(pp_base), int(h),
           id(simulate_step), id(task_loss), id(obs_margin))
    fn = _SHAC_WINDOW_CACHE.get(key)
    if fn is not None:
        return fn
    window_loss = make_shac_window_loss(solver, theta_to_weights, dyn, pp_base, h,
                                        simulate_step=simulate_step, task_loss=task_loss,
                                        obs_margin=obs_margin)
    fn = jax.jit(jax.value_and_grad(window_loss, has_aux=True))
    _SHAC_WINDOW_CACHE[key] = fn
    return fn


def shac_window_update(
    solver,
    theta_to_weights,
    dyn,
    policy,
    opt_state,
    x,
    pp_base,
    guess,
    *,
    h,
    lr,
    simulate_step=None,
    task_loss=None,
    obs_margin=None,
):
    """V3 — SHAC truncated-BPTT update over an ``h``-step window.

    Detach ``x`` at window entry, run the jitted BPTT window (gradient flows through
    the realized rollout), apply one Adam step with the window gradient, and carry the
    (detached) final state and warm-start for the next call.

    ``simulate_step`` / ``task_loss`` / ``obs_margin`` default to the drone env; pass the
    quadrotor env's functions to run the nonlinear-quadrotor task.

    Returns ``(policy, opt_state, x, guess, logs)`` with ``logs`` holding the window
    ``loss`` / ``grad_norm`` plus per-step ``obs_margin`` (realized) and
    ``plan_obs_margin_max``.
    """
    simulate_step, task_loss, obs_margin = _resolve_env(simulate_step, task_loss, obs_margin)
    guess = prime_guess(solver, theta_to_weights, policy, pp_base, x, guess)
    grad_fn = _get_shac_window(solver, theta_to_weights, dyn, pp_base, h,
                               simulate_step=simulate_step, task_loss=task_loss,
                               obs_margin=obs_margin)

    x0 = jax.lax.stop_gradient(x)
    (L, (x_final, guess_final, margins, plan_margins)), g = grad_fn(policy, x0, guess)
    policy, opt_state = adam_step(policy, g, opt_state, lr=lr)

    logs = {
        "loss": L,
        "grad_norm": _grad_norm(g),
        "obs_margin": margins,
        "plan_obs_margin_max": plan_margins,
    }
    return (
        policy,
        opt_state,
        jax.lax.stop_gradient(x_final),
        jax.lax.stop_gradient(guess_final),
        logs,
    )
