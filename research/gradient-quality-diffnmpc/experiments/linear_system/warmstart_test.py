"""Is the hard-box cos_all discrepancy (diffmpc2 ~0.36 vs external/reference ~1.0) caused by the rollout
warm-start, not the DIRECT backward? diffmpc2's build_rollout_fn with warm_start=False uses
`jax.lax.stop_gradient(solution)` to chain rollout steps -> the AD misses the warm-start's weight-
dependence that the FD includes (SQP iter=1, so the solution depends on its initial guess) -> AD != FD.
With warm_start=True it differentiates through the chain -> AD == FD. Same fused/DIRECT backend both ways.

If warm_start=True gives cos_all ~1.0 while warm_start=False gives ~0.36, the rollout warm-start is the
cause and the earlier "DIRECT-backward outlier" reading was a confound.

    python research/gradient-quality-diffnmpc/experiments/linear_system/warmstart_test.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, _cos, _reward, ad, _fd_at_eps, _flat, _unflat, WK,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblem,
    ProblemConfig, build_rollout_fn)

SEEDS = [0, 1, 2, 3, 4]
TOL, EPS, HZ, SIM = 1e-9, 1e-4, 40, 50


def cost_fn(dyn, pp, w, x0b, warm_start):
    sp = _sp(TOL)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=dict(pp)), params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True)
    init = solver.solve(solver.initial_guess(pp), problem_params=pp, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblem, problem_params=pp, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=pp, init_solution=init,
                               warm_start=warm_start, num_sim_steps=SIM)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


def main():
    print("hard-box cos(AD, FD) by rollout warm_start (fused/DIRECT, tol 1e-9, horizon 40):")
    for warm in (False, True):
        sums, psmed, nout = [], [], []
        for seed in SEEDS:
            dyn, pp, w, x0b, *_ = _problem(seed, 64, SIM, HZ)
            cf = cost_fn(dyn, pp, w, x0b, warm)
            gAD = ad(cf, w, x0b)
            single = jax.jit(jax.vmap(lambda wf, st: cf(_unflat(wf), st)))
            gFD = _fd_at_eps(single, np.asarray(_flat(w)), x0b, EPS)
            cos_ps = np.array([_cos(gAD[i], gFD[i]) for i in range(64)])
            sums.append(_cos(gAD.sum(0), gFD.sum(0))); psmed.append(float(np.median(cos_ps)))
            nout.append(int((cos_ps < 0.99).sum()))
            jax.clear_caches()
        print(f"  warm_start={str(warm):5s}: batch-sum cos_all med={np.median(sums):+.3f} "
              f"(per-seed {[round(s,2) for s in sums]}) | per-sample med={np.median(psmed):.3f} "
              f"| #out/seed={np.mean(nout):.1f}", flush=True)


if __name__ == "__main__":
    main()
