"""Is the binary active-set error in the DIRECT backward BRITTLE (classification flips with forward
precision near the zero-slack threshold) or STABLE (consistently wrong)? Sweep the forward solve
tolerance and watch the bad sample (45) cos vs the barrier (correct reference). If cos jumps with the
forward tol -> brittle active-set classification at the near-zero slack; if it stays ~0.033 ->
the binary active-set is consistently wrong at the weakly-active constraint (formulation, not precision).

    python research/gradient-quality-diffnmpc/experiments/linear_system/directbwd_threshold.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, _cos, _reward, ad, cost_fn_B, WK,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblem,
    ProblemConfig, build_rollout_fn)

FWD_TOLS = [1e-7, 1e-9, 1e-11, 1e-13]
J = 45


def hard_cf(dyn, pp, w, x0b, fwd_tol, sim_steps):
    ppA = dict(pp); sp = _sp(fwd_tol)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=ppA), params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True)
    init = solver.solve(solver.initial_guess(ppA), problem_params=ppA, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblem, problem_params=ppA,
                        solver_params=sp, weight_keys=WK, reward_fn=_reward,
                        update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=ppA, init_solution=init,
                               warm_start=False, num_sim_steps=sim_steps)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


def main():
    n = 64
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, n, 50, 160)
    g_bar = ad(cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, 1e-9, False, 50), w, x0b)
    print(f"hard-box DIRECT gradient vs barrier, sweeping FORWARD tolerance (sample {J}):")
    for ftol in FWD_TOLS:
        t = time.time()
        g = ad(hard_cf(dyn, pp, w, x0b, ftol, 50), w, x0b)
        cos = np.array([_cos(g[i], g_bar[i]) for i in range(n)])
        bad = np.where(cos < 0.99)[0]
        print(f"  fwd_tol={ftol:.0e}: sample{J} cos={cos[J]:+.4f}  |  #<0.99={len(bad)}/{n} bad={list(bad)}  "
              f"({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
