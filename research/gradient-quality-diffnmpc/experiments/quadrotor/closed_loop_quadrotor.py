"""Closed-loop (50-step) gradient accuracy on the NONLINEAR quadrotor (13 states, nu=4) — does
the hard-box vs slack-box outlier mechanism hold with MANY controls under nonlinear dynamics?

Hover regulation: perturb about hover, MPC drives back to hover; control box ±umax (hover thrust
~0.981, so umax just above it makes the thrust saturate during recovery). Compares A-hardbox
(use_slack=False, diffmpc default) vs A-slack (use_slack=True, gamma) — each AD (grad through the
50-step rollout) vs convergence-checked FD of its own rollout. (B needs a multi-SQP custom_vjp;
the A-slack control already isolates the slack as the mechanism.)

    python research/gradient-quality-diffnmpc/experiments/quadrotor/closed_loop_quadrotor.py --smoke
    python .../closed_loop_quadrotor.py [--slack] --n_samples 6 --umax 1.2 --sqp_iter 10
"""
from __future__ import annotations
import os, sys, time, argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_CART = os.path.join(_DL_ROOT, "research", "gradient-quality-diffnmpc", "experiments", "cartpole")
for _p in (os.path.join(_DL_ROOT, "src"), os.path.join(_DL_ROOT, "diffmpc2"), _CART):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_cartpole_coupling import (  # noqa: E402
    build_quadrotor_problem, generate_quadrotor_x0, QR_NX, QR_NU, QR_Q, QR_R,
    QR_REF_STATE, QR_REF_CONTROL,
)
from turbompc.problems.optimal_control_problem import OptimalControlProblem, OptimalControlProblemSlack  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402

QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
SIM_STEPS = 50
HORIZON = 12
DT = 0.04
GAMMA = 1.0e4
_REF_S = QR_REF_STATE
_REF_C = QR_REF_CONTROL


def _reward(state, control):
    return -(jnp.sum((state - _REF_S) ** 2) + jnp.sum((control - _REF_C) ** 2))


def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _solver_params(sqp_iter, eps=1e-9):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = sqp_iter
    sp["tol_convergence"] = eps
    sp["warm_start_backward"] = False
    sp["linesearch"] = True
    sp["admm"]["max_iter"] = 4000
    sp["admm"]["check_termination_every"] = 25
    sp["admm"]["eps_abs"] = eps
    sp["admm"]["eps_rel"] = eps
    return sp


def run(n_samples, seed, use_slack, sqp_iter, umax, x0_scale=1.0):
    dyn, pp = build_quadrotor_problem(horizon=HORIZON, umax=umax, dt=DT)
    pp = dict(pp); pp[QK] = QR_Q; pp[RK] = QR_R
    x0 = generate_quadrotor_x0(n_samples, seed, x0_scale); x0_batch = jnp.asarray(x0)
    w = {k: pp[k] for k in WK}
    prob_cls = OptimalControlProblemSlack if use_slack else OptimalControlProblem
    if use_slack:
        pp = {**pp, "use_slack_variables": True, "slack_penalization_weight": GAMMA}

    sp = _solver_params(sqp_iter)
    solver = TurboMPCSolver(
        program=prob_cls(dynamics=dyn, params=pp), params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init_solution = solver.solve(solver.initial_guess(pp), problem_params=pp,
                                 weights={**w, "initial_state": x0_batch[0]})

    cfg = ProblemConfig(dynamics=dyn, problem_class=prob_cls, problem_params=pp,
                        solver_params=sp, weight_keys=WK, reward_fn=_reward,
                        update_per_seed=lambda s, b_, p: ({}, None))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=pp,
                               init_solution=init_solution, warm_start=False, num_sim_steps=SIM_STEPS)

    def cost_one(ww, x0i):
        return rollout(x0i, {**ww, "initial_state": x0i})[0]
    cost_vec = jax.jit(jax.vmap(cost_one, in_axes=(None, 0)))
    g_fn = jax.jit(jax.vmap(jax.grad(cost_one, argnums=0), in_axes=(None, 0)))

    t = time.time(); g_d = g_fn(w, x0_batch)
    gAD = np.stack([_flat({k: g_d[k][i] for k in WK}) for i in range(n_samples)])
    print(f"[AD] {time.time()-t:.0f}s")

    t = time.time(); eps_seq = (3e-5, 1e-5, 3e-6, 1e-6); per_eps = []
    for eps in eps_seq:
        cols = []
        for k in WK:
            base = np.asarray(w[k], float)
            for i in range(base.size):
                ap = base.copy(); ap[i] += eps; am = base.copy(); am[i] -= eps
                cols.append((np.asarray(cost_vec({**w, k: jnp.asarray(ap)}, x0_batch))
                             - np.asarray(cost_vec({**w, k: jnp.asarray(am)}, x0_batch))) / (2 * eps))
        per_eps.append(np.stack(cols, axis=1))
    gFD = np.array(per_eps[-1]); flagged = np.zeros(n_samples, bool)
    for nn in range(n_samples):
        ok_all = True
        for j in range(per_eps[0].shape[1]):
            seq = [pe[nn, j] for pe in per_eps]; chosen, ok = seq[-1], False
            for a_, b_ in zip(seq[:-1], seq[1:]):
                if abs(a_ - b_) <= 1e-2 * abs(b_) + 1e-7: chosen, ok = a_, True; break
            gFD[nn, j] = chosen; ok_all = ok_all and ok
        flagged[nn] = not ok_all
    print(f"[FD] {time.time()-t:.0f}s  flagged {int(flagged.sum())}/{n_samples}")

    cos = np.array([_cos(gAD[n], gFD[n]) for n in range(n_samples)])
    rel = np.array([_rel(gAD[n], gFD[n]) for n in range(n_samples)])
    keep = ~flagged; tag = "A-slack" if use_slack else "A-hardbox"
    print(f"\n=== {tag} (quadrotor, umax={umax}, sqp={sqp_iter}) vs FD, {int(keep.sum())}/{n_samples} non-flagged ===")
    if keep.any():
        print(f"  cos: median={np.median(cos[keep]):.5f} min={np.min(cos[keep]):.5f} "
              f"#<0.99={int((cos[keep]<0.99).sum())} #<0={int((cos[keep]<0).sum())}  "
              f"rel_l2 median={np.median(rel[keep]):.2e}")
    for i in np.argsort(cos)[:8]:
        print(f"    sample {i:3d}: cos={cos[i]:+.5f} rel={rel[i]:.2e} flagged={bool(flagged[i])}")
    out = int(((cos < 0.99) | flagged).sum())
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", f"closed_loop_quadrotor_{'slack' if use_slack else 'hardbox'}.npz"),
             cos=cos, rel=rel, flagged=flagged, gAD=gAD, gFD=gFD, umax=umax)
    print(f"  outliers (cos<0.99 or flagged): {out}/{n_samples}")
    return cos, rel, flagged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slack", action="store_true")
    p.add_argument("--sqp_iter", type=int, default=10)
    p.add_argument("--umax", type=float, default=1.2)
    p.add_argument("--x0_scale", type=float, default=1.0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    global SIM_STEPS, HORIZON
    n, sqp = args.n_samples, args.sqp_iter
    if args.smoke:
        n, sqp, SIM_STEPS, HORIZON = 2, 5, 20, 10
    print(f"Quadrotor closed-loop (sim_steps={SIM_STEPS}, H={HORIZON}, umax={args.umax}, dt={DT}) "
          f"n={n} seed={args.seed} sqp={sqp} scale={args.x0_scale} slack={args.slack}")
    run(n, args.seed, args.slack, sqp, args.umax, args.x0_scale)


if __name__ == "__main__":
    main()
