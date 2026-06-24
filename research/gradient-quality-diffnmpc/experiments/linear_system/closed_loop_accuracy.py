"""Closed-loop (50-step) gradient-accuracy sweep on the linear MPC, per-sample, A vs B vs FD.

Mirrors diffmpc's `benchmark_turbompc_gradient_accuracy.py` (sim_steps=50 closed-loop rollout,
1 SQP iter, box constraints) but: (1) per-sample (per initial state) so outliers are visible,
(2) convergence-checked FD ground truth (CLAUDE.md), (3) compares BOTH backwards:
  A = TurboMPC analytic backward (the existing benchmark)
  B = log-barrier smoothed backward            [added on a later iteration]
Goal: identify the outlier samples (low cos / high rel_l2) for each method.

This file currently implements A (B is added next). Run:
    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/closed_loop_accuracy.py
"""
from __future__ import annotations
import os, sys, time, argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (os.path.join(_DL_ROOT, "src"), os.path.join(_DL_ROOT, "diffmpc2"),
           os.path.join(_DL_ROOT, "diffmpc2", "benchmarking", "linear-system")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from utils import generate_problem_data, N_STATE, N_CTRL  # noqa: E402
from turbompc.problems.optimal_control_problem import OptimalControlProblem  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
SIM_STEPS = 50
HORIZON = 20
UMAX = 1.0
TIGHT = 1e-9   # tight solve so AD/FD reflect the true rollout gradient (image used GT@1e-9)


def _reward(state, control):
    return -(jnp.sum(state ** 2) + jnp.sum(control ** 2))


def _flat(d):
    return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _solver_params(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1
    sp["tol_convergence"] = eps
    sp["warm_start_backward"] = False
    sp["linesearch"] = False
    sp["admm"]["max_iter"] = 4000
    sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = eps
    sp["admm"]["eps_rel"] = eps
    return sp


def run_A(n_samples, seed, verbose=True):
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(n_samples, seed, n_state=NX, n_ctrl=NU)
    pp = dict(pp_t)
    pp["dynamics_state_dot_params"] = {"A": jnp.asarray(A - np.eye(NX)), "B": jnp.asarray(B), "b": jnp.asarray(b)}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    x0_batch = jnp.asarray(x0)
    w = {k: pp[k] for k in WK}

    sp = _solver_params(TIGHT)
    solver = TurboMPCSolver(
        program=OptimalControlProblem(dynamics=dyn, params=pp), params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init_solution = solver.solve(solver.initial_guess(pp), problem_params=pp, weights={**w, "initial_state": x0_batch[0]})

    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblem, problem_params=pp,
                        solver_params=sp, weight_keys=WK, reward_fn=_reward,
                        update_per_seed=lambda s, b_, p: ({}, None))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=pp,
                               init_solution=init_solution, warm_start=False, num_sim_steps=SIM_STEPS)

    def cost_one(ww, x0i):
        return rollout(x0i, {**ww, "initial_state": x0i})[0]
    cost_vec = jax.jit(jax.vmap(cost_one, in_axes=(None, 0)))
    gA_fn = jax.jit(jax.vmap(jax.grad(cost_one, argnums=0), in_axes=(None, 0)))

    t = time.time()
    gA_d = gA_fn(w, x0_batch)
    gA = np.stack([_flat({k: gA_d[k][i] for k in WK}) for i in range(n_samples)])
    print(f"[A AD] {time.time()-t:.0f}s")

    # convergence-checked FD ground truth (per sample)
    t = time.time()
    eps_seq = (3e-5, 1e-5, 3e-6, 1e-6)   # fine enough to resolve closed-loop active-set kinks
    per_eps = []
    for eps in eps_seq:
        cols = []
        for k in WK:
            base = np.asarray(w[k], float)
            for i in range(base.size):
                ap = base.copy(); ap[i] += eps; am = base.copy(); am[i] -= eps
                wp = {**w, k: jnp.asarray(ap)}; wm = {**w, k: jnp.asarray(am)}
                cols.append((np.asarray(cost_vec(wp, x0_batch)) - np.asarray(cost_vec(wm, x0_batch))) / (2 * eps))
        per_eps.append(np.stack(cols, axis=1))
    gFD = np.array(per_eps[-1]); flagged = np.zeros(n_samples, bool)
    for n in range(n_samples):
        ok_all = True
        for j in range(per_eps[0].shape[1]):
            seq = [pe[n, j] for pe in per_eps]; chosen, ok = seq[-1], False
            for a_, b_ in zip(seq[:-1], seq[1:]):
                if abs(a_ - b_) <= 1e-2 * abs(b_) + 1e-7:
                    chosen, ok = a_, True; break
            gFD[n, j] = chosen; ok_all = ok_all and ok
        flagged[n] = not ok_all
    print(f"[FD] {time.time()-t:.0f}s  flagged {int(flagged.sum())}/{n_samples}")

    cosA = np.array([_cos(gA[n], gFD[n]) for n in range(n_samples)])
    relA = np.array([_rel(gA[n], gFD[n]) for n in range(n_samples)])
    return cosA, relA, flagged, gA, gFD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(f"Closed-loop (sim_steps={SIM_STEPS}) gradient accuracy, A vs FD: nx={NX} nu={NU} H={HORIZON} "
          f"umax={UMAX} tol={TIGHT} n_samples={args.n_samples}")
    cosA, relA, flagged, gA, gFD = run_A(args.n_samples, args.seed)
    keep = ~flagged
    print(f"\n=== A (TurboMPC) vs convergence-checked FD, {int(keep.sum())}/{args.n_samples} non-flagged ===")
    print(f"  cos:    median={np.median(cosA[keep]):.5f}  min={np.min(cosA[keep]):.5f}  "
          f"#<0.99={int((cosA[keep]<0.99).sum())}  #<0={int((cosA[keep]<0).sum())}")
    print(f"  rel_l2: median={np.median(relA[keep]):.2e}  max={np.max(relA[keep]):.2e}")
    order = np.argsort(cosA)
    print("  worst-A samples (idx, cos, rel, flagged):")
    for i in order[:10]:
        print(f"    sample {i:3d}: cos={cosA[i]:+.5f} rel={relA[i]:.2e} flagged={bool(flagged[i])}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "closed_loop_A.npz"),
             cosA=cosA, relA=relA, flagged=flagged, gA=gA, gFD=gFD)
    print(f"\nsaved results/closed_loop_A.npz")


if __name__ == "__main__":
    main()
