"""50-step CLOSED-LOOP gradient accuracy vs ADMM tolerance, A and B — structured after
diffmpc2/benchmarking/linear-system/benchmark_turbompc_gradient_accuracy.py.

Per ADMM tolerance, per seed (a random linear MPC + a batch of initial states): the BATCH-SUMMED
gradient of the 50-step closed-loop rollout cost w.r.t. the cost weights (Q,R), compared to a
single-eps finite-difference of the same rollout (`cosine`, `rel_l2`), aggregated over seeds —
exactly the benchmark's protocol. Two backwards:
  A = TurboMPC analytic backward, SLACK box (use_slack=True, Moreau/quadratic penalty γ)
  B = log-barrier central-path backward (central-path ADMM stopped at the same tolerance)
Both A and B are soft constraints (γ=1e4), so this isolates the ADMM-tolerance effect.

Deliberate deviation from the benchmark: it caps admm_max_iter=50 (a timing choice that caps tight
tolerances mid-convergence); here admm_max_iter is uncapped so each tolerance is actually REACHED —
a clean accuracy-vs-achieved-tolerance sweep. Reduced size (batch/seeds/horizon) for tractability
(B's per-step custom_vjp rollout is the bottleneck); noted in the output.

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/closed_loop_admm_sweep.py
"""
from __future__ import annotations
import os, sys, time, argparse, gc

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
from turbompc.problems.optimal_control_problem import OptimalControlProblem, OptimalControlProblemSlack  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402

from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402
import cl_b  # noqa: E402

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
SIM_STEPS, HORIZON, UMAX = 50, 20, 1.0
GAMMA, KAPPA = 1e4, 1e-6
ADMM_MAX_ITER = 1000   # cap; the sweep VERIFIES no trial reaches it (else the tol wasn't reached)


def _reward(s, c): return -(jnp.sum(s ** 2) + jnp.sum(c ** 2))
def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _sp(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = eps; sp["warm_start_backward"] = False
    sp["linesearch"] = False; sp["admm"]["max_iter"] = ADMM_MAX_ITER
    sp["admm"]["check_termination_every"] = 1; sp["admm"]["eps_abs"] = eps; sp["admm"]["eps_rel"] = eps
    return sp


def _problem(seed, batch):
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(batch, seed, n_state=NX, n_ctrl=NU)
    A_sd = jnp.asarray(A - np.eye(NX)); B_m = jnp.asarray(B); b_v = jnp.asarray(b)
    pp = dict(pp_t); pp["dynamics_state_dot_params"] = {"A": A_sd, "B": B_m, "b": b_v}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    return dyn, pp, {k: pp[k] for k in WK}, jnp.asarray(x0), A_sd, B_m, b_v


def _fd(cost_sum, w, eps):
    g = {}
    for k in WK:
        base = np.asarray(w[k], float); gk = np.zeros_like(base)
        for i in range(base.size):
            ap = base.copy(); ap[i] += eps; am = base.copy(); am[i] -= eps
            gk[i] = (float(cost_sum({**w, k: jnp.asarray(ap)})) - float(cost_sum({**w, k: jnp.asarray(am)}))) / (2 * eps)
        g[k] = gk
    return g


def _A(dyn, pp, w, x0b, tol, fd_eps):
    sp = _sp(tol)
    # A = TurboMPC SLACK box (Moreau/quadratic soft constraint, gamma matched to B), use_slack=True
    pp = {**pp, "use_slack_variables": True, "slack_penalization_weight": GAMMA}
    solver = TurboMPCSolver(program=OptimalControlProblemSlack(dynamics=dyn, params=pp), params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True)
    init = solver.solve(solver.initial_guess(pp), problem_params=pp, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblemSlack, problem_params=pp, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=pp, init_solution=init,
                              warm_start=False, num_sim_steps=SIM_STEPS)
    rb = jax.jit(jax.vmap(lambda st, ww: rollout(st, {**ww, "initial_state": st})[0], in_axes=(0, None)))
    rb_it = jax.jit(jax.vmap(lambda st, ww: rollout(st, {**ww, "initial_state": st})[1], in_axes=(0, None)))
    cost_sum = lambda ww: float(jnp.sum(rb(x0b, ww)))
    g_ad = jax.jit(jax.grad(lambda ww: jnp.sum(rb(x0b, ww))))(w)
    g_fd = _fd(cost_sum, w, fd_eps)
    max_it = int(np.max(np.asarray(rb_it(x0b, w))))   # max per-solve ADMM iters over batch x 50 steps
    return _cos(_flat(g_ad), _flat(g_fd)), _rel(_flat(g_ad), _flat(g_fd)), max_it


def _B(dyn, pp, w, x0b, A_sd, B_m, b_v, tol, fd_eps):
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(1e-9),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp)
    N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU, pcg_params={"max_iter": 400, "tol_epsilon": 1e-12})
    b_solve = cl_b.make_b_mpc_solve(solver, pp, schur, ig, NX, NU, kappa=KAPPA, cp_tol=tol)

    def rollout_cost(weights, state):
        def step(carry, _):
            st, cost = carry
            s, c = b_solve(weights, st); u0 = c[0]
            new = st + 1.0 * (A_sd @ st + B_m @ u0 + b_v)
            return (new, cost + jnp.sum(new ** 2) + jnp.sum(u0 ** 2)), None
        (_, tot), _ = jax.lax.scan(step, (state, jnp.asarray(0.0)), None, length=SIM_STEPS)
        return tot
    rb = jax.jit(jax.vmap(rollout_cost, in_axes=(None, 0)))
    cost_sum = lambda ww: float(jnp.sum(rb(ww, x0b)))
    g_ad = jax.jit(jax.grad(lambda ww: jnp.sum(rb(ww, x0b))))(w)
    g_fd = _fd(cost_sum, w, fd_eps)

    # max per-solve central-path ADMM iters over batch x 50 steps (forward-only, for the cap check)
    def _qp1(weights, state):
        pp_i = {**pp, "initial_state": state, QK: weights[QK], RK: weights[RK]}
        return to_one_sided(solver._build_qp_data(ig.states, ig.controls, pp_i), GAMMA)
    def iters_traj(weights, state):
        def step(st, _):
            x, _, info = solve_qp_central_path(_qp1(weights, st), schur, target_kappa=KAPPA,
                                               slack_weight=GAMMA, rho_bar=0.1, max_iter=ADMM_MAX_ITER, tol=tol)
            u0 = x[0, NX:]
            return st + 1.0 * (A_sd @ st + B_m @ u0 + b_v), info["iters"]
        _, its = jax.lax.scan(step, state, None, length=SIM_STEPS)
        return jnp.max(its)
    max_it = int(np.max(np.asarray(jax.jit(jax.vmap(iters_traj, in_axes=(None, 0)))(w, x0b))))
    return _cos(_flat(g_ad), _flat(g_fd)), _rel(_flat(g_ad), _flat(g_fd)), max_it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerances", type=float, nargs="+", default=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9])
    p.add_argument("--n_seeds", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fd_eps", type=float, default=1e-5)
    a = p.parse_args()
    print(f"Closed-loop (sim_steps={SIM_STEPS}) ADMM-tolerance gradient accuracy, A & B vs FD")
    print(f"  nx={NX} nu={NU} H={HORIZON} umax={UMAX} batch={a.batch} n_seeds={a.n_seeds} "
          f"gamma={GAMMA:g} kappa={KAPPA:g} fd_eps={a.fd_eps:g} admm_max_iter={ADMM_MAX_ITER} (uncapped)")
    rows = []
    for tol in a.tolerances:
        cA, rA, cB, rB, iA, iB = [], [], [], [], [], []
        for seed in range(a.n_seeds):
            dyn, pp, w, x0b, A_sd, B_m, b_v = _problem(seed, a.batch)
            t = time.time()
            ca, ra, ia = _A(dyn, pp, w, x0b, tol, a.fd_eps)
            cb, rb_, ib = _B(dyn, pp, w, x0b, A_sd, B_m, b_v, tol, a.fd_eps)
            cA.append(ca); rA.append(ra); cB.append(cb); rB.append(rb_); iA.append(ia); iB.append(ib)
            cap = "  <-- HIT CAP" if (ia >= ADMM_MAX_ITER or ib >= ADMM_MAX_ITER) else ""
            print(f"  tol={tol:.0e} seed={seed}: A cos={ca:+.4f} rel={ra:.2e} it={ia:4d} | "
                  f"B cos={cb:+.4f} rel={rb_:.2e} it={ib:4d}  ({time.time()-t:.0f}s){cap}", flush=True)
            jax.clear_caches(); gc.collect()
        cA, rA, cB, rB = map(np.array, (cA, rA, cB, rB))
        rows.append((tol, np.mean(cA), np.median(cA), np.min(cA), np.mean(rA),
                     np.mean(cB), np.median(cB), np.min(cB), np.mean(rB), max(iA), max(iB)))
    print(f"\n{'tol':>7} | {'A cosμ':>8} {'A cosmed':>9} {'A cosmin':>9} {'A relμ':>9} {'A maxit':>7} | "
          f"{'B cosμ':>8} {'B cosmed':>9} {'B cosmin':>9} {'B relμ':>9} {'B maxit':>7}")
    for (tol, acm, acmd, acmn, arm, bcm, bcmd, bcmn, brm, mia, mib) in rows:
        print(f"{tol:7.0e} | {acm:8.4f} {acmd:9.4f} {acmn:9.4f} {arm:9.2e} {mia:7d} | "
              f"{bcm:8.4f} {bcmd:9.4f} {bcmn:9.4f} {brm:9.2e} {mib:7d}")
    wA = max(r[9] for r in rows); wB = max(r[10] for r in rows)
    hit = (wA >= ADMM_MAX_ITER) or (wB >= ADMM_MAX_ITER)
    print(f"\nADMM_MAX_ITER={ADMM_MAX_ITER}: worst A iters={wA}, worst B iters={wB} -> "
          f"{'*** A TRIAL HIT THE CAP; raise max_iter ***' if hit else 'OK: no trial reached the cap'}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "closed_loop_admm_sweep.npz"),
             tolerances=np.array([r[0] for r in rows]), rows=np.array([r[1:] for r in rows]),
             n_seeds=a.n_seeds, batch=a.batch, admm_max_iter=ADMM_MAX_ITER)
    print("\nsaved results/closed_loop_admm_sweep.npz")


if __name__ == "__main__":
    main()
