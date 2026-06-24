"""Verify A's forward backend doesn't change the result: ADMM_FUSED_CUDSS (production fused kernel)
vs ADMM_JAX_LOOP_CUDSS_FFI, same DIRECT_CUDSS_FFI backward. Per-sample 50-step closed-loop gradient
(hard box and slack), at a loose and a tight tolerance. Reports cos(fused, jaxloop), rel-diff, and the
forward-cost match.

    python research/gradient-quality-diffnmpc/experiments/linear_system/fused_check.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_4way_sweep import (  # noqa: E402
    _problem, _sp, _reward, ad, _flat, _cos, WK, SIM_STEPS, GAMMA_SLACK,
    OptimalControlProblem, OptimalControlProblemSlack, TurboMPCSolver, ForwardBackend, BackwardBackend,
    ProblemConfig, build_rollout_fn)


def cost_fn_A(dyn, pp, w, x0b, tol, slack, fwd):
    sp = _sp(tol)
    if slack:
        ppA = {**pp, "use_slack_variables": True, "slack_penalization_weight": GAMMA_SLACK}
        cls = OptimalControlProblemSlack
    else:
        ppA = dict(pp); cls = OptimalControlProblem
    solver = TurboMPCSolver(program=cls(dynamics=dyn, params=ppA), params=sp, forward_backend=fwd,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init = solver.solve(solver.initial_guess(ppA), problem_params=ppA, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=cls, problem_params=ppA, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=ppA, init_solution=init,
                              warm_start=False, num_sim_steps=SIM_STEPS)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


def main():
    B = 16
    dyn, pp, w, x0b, A_sd, B_m, b_v = _problem(0, B)
    JAX, FUSED = ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, ForwardBackend.ADMM_FUSED_CUDSS
    print(f"A forward-backend match: FUSED_CUDSS vs JAX_LOOP_CUDSS_FFI (same DIRECT backward), batch={B}")
    print(f"\n{'box':>8} {'tol':>6} | {'cos(fused,jax) med':>18} {'cos min':>9} {'relg med':>9} {'cost reldiff':>13}")
    for slack in (False, True):
        for tol in (1e-1, 1e-7):
            cj = cost_fn_A(dyn, pp, w, x0b, tol, slack, JAX)
            cf = cost_fn_A(dyn, pp, w, x0b, tol, slack, FUSED)
            gj = ad(cj, w, x0b); gf = ad(cf, w, x0b)
            costs_j = np.asarray(jax.jit(jax.vmap(cj, in_axes=(None, 0)))(w, x0b))
            costs_f = np.asarray(jax.jit(jax.vmap(cf, in_axes=(None, 0)))(w, x0b))
            cos = np.array([_cos(gf[i], gj[i]) for i in range(B)])
            relg = np.array([np.linalg.norm(gf[i] - gj[i]) / (np.linalg.norm(gj[i]) + 1e-30) for i in range(B)])
            creld = float(np.max(np.abs(costs_f - costs_j) / (np.abs(costs_j) + 1e-30)))
            box = "slack" if slack else "hard"
            print(f"{box:>8} {tol:6.0e} | {np.median(cos):18.6f} {np.min(cos):9.6f} "
                  f"{np.median(relg):9.2e} {creld:13.2e}", flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
