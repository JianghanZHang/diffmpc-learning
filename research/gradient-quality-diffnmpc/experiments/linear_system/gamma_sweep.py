"""Why does the Moreau slack give cos~0.2 vs the hard-box gradient? Sweep the slack weight gamma -> inf.
If the slack gradient converges to the hard gradient (cos -> 1) then it is the relaxation (gamma=1e4 is
too soft for a 50-step gradient; the per-step O(1/gamma) error compounds), not a bug. Also report the
forward solution closeness ||u_slack - u_hard||_inf over the rollout. sim_steps sweep too, at gamma=1e4.

    python research/gradient-quality-diffnmpc/experiments/linear_system/gamma_sweep.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, _cos, _reward, ad, fd_gt_adaptive, cost_fn_A, WK, GT_TOL,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblemSlack,
    ProblemConfig, build_rollout_fn)

GAMMAS = [1e4, 1e5, 1e6, 1e8]
STEPS = [5, 20, 50]


def slack_cf(dyn, pp, w, x0b, tol, gamma, sim_steps):
    ppA = {**pp, "use_slack_variables": True, "slack_penalization_weight": gamma}
    sp = _sp(tol)
    solver = TurboMPCSolver(program=OptimalControlProblemSlack(dynamics=dyn, params=ppA), params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True)
    init = solver.solve(solver.initial_guess(ppA), problem_params=ppA, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblemSlack, problem_params=ppA,
                        solver_params=sp, weight_keys=WK, reward_fn=_reward,
                        update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=ppA, init_solution=init,
                               warm_start=False, num_sim_steps=sim_steps)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


def main():
    n = 64
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, n, 50, 160)
    Z = np.load(os.path.join(_HERE, "results", "gradient_accuracy.npz"))
    gFD = Z["gFD"]; keep = Z["gt_converged"].astype(bool)

    print("[A] slack gradient vs the hard-box FD GT, gamma -> inf (sim_steps=50, tol=1e-9):")
    for gamma in GAMMAS:
        t = time.time()
        gAD = ad(slack_cf(dyn, pp, w, x0b, 1e-9, gamma, 50), w, x0b)
        cos = np.array([_cos(gAD[i], gFD[i]) for i in range(n)])[keep]
        print(f"  gamma={gamma:.0e}: cos med={np.median(cos):.4f} min={np.min(cos):.4f} "
              f"#<0.99={int((cos < 0.99).sum())}  ({time.time()-t:.0f}s)", flush=True)

    print("\n[B] compounding: slack(gamma=1e4) vs a per-SIM_STEPS hard FD GT:")
    for ss in STEPS:
        t = time.time()
        gfd_ss, conv, _, _ = fd_gt_adaptive(cost_fn_A(dyn, pp, w, x0b, GT_TOL, False, ss), w, x0b)
        gAD = ad(slack_cf(dyn, pp, w, x0b, 1e-9, 1e4, ss), w, x0b)
        k = conv
        cos = np.array([_cos(gAD[i], gfd_ss[i]) for i in range(n)])[k]
        print(f"  sim_steps={ss:3d}: cos med={np.median(cos):.4f} min={np.min(cos):.4f} "
              f"(GT conv {int(k.sum())}/{n})  ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
