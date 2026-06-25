"""Root-cause the hard-box DIRECT backward imprecision (~4% of near-active samples): is it a cuDSS-FFI
solve precision issue, or fundamental KKT ill-conditioning? Compute the hard-box closed-loop gradient
with DIRECT_CUDSS_FFI vs DIRECT_JAX_DENSE (a different, pure-JAX x64 dense KKT solver), each vs the
barrier gradient (the correct, well-conditioned reference). seed 0.
  - if DIRECT_JAX_DENSE fixes the bad samples -> cuDSS-FFI solve precision
  - if DIRECT_JAX_DENSE fails on the same samples -> fundamental KKT ill-conditioning (solver-independent)

    python research/gradient-quality-diffnmpc/experiments/linear_system/directbwd_rootcause.py
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

BWDS = [BackwardBackend.DIRECT_CUDSS_FFI, BackwardBackend.DIRECT_JAX_DENSE]


def hard_cf(dyn, pp, w, x0b, bwd, sim_steps):
    ppA = dict(pp); sp = _sp(1e-9)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=ppA), params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=bwd, use_full_hessian=True)
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
    g_bar = ad(cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, 1e-9, False, 50), w, x0b)  # correct reference
    print("hard-box gradient vs barrier, two DIRECT backward solvers (seed 0, sim_steps=50):")
    res = {}
    for bwd in BWDS:
        t = time.time()
        g = ad(hard_cf(dyn, pp, w, x0b, bwd, 50), w, x0b)
        cos = np.array([_cos(g[i], g_bar[i]) for i in range(n)]); res[bwd.name] = cos
        bad = np.where(cos < 0.99)[0]
        print(f"  {bwd.name:18s}: #<0.99={len(bad):2d}/{n}  min={np.min(cos):+.4f}  "
              f"bad={list(bad)} cos={[round(float(cos[b]),3) for b in bad]}  ({time.time()-t:.0f}s)", flush=True)
    cf, jd = res["DIRECT_CUDSS_FFI"], res["DIRECT_JAX_DENSE"]
    fixed = np.where((cf < 0.99) & (jd >= 0.99))[0]
    both = np.where((cf < 0.99) & (jd < 0.99))[0]
    print(f"\n  cuDSS-bad fixed by JAX_DENSE: {list(fixed)}  | bad in BOTH: {list(both)}")
    print(f"  verdict: {'cuDSS-FFI solve precision' if len(fixed) and not len(both) else ('KKT ill-conditioning (solver-independent)' if len(both) else 'inconclusive')}")


if __name__ == "__main__":
    main()
