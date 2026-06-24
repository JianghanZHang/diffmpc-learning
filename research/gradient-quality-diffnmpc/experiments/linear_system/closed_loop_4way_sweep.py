"""50-step CLOSED-LOOP per-sample gradient accuracy vs QP (ADMM) tolerance — 4-way comparison,
PARALLELIZED (wide batch + fully-vectorized FD).

  1. A NO-slack   : TurboMPC hard box (OptimalControlProblem, DIRECT backward)
  2. A slack       : TurboMPC slack box (OptimalControlProblemSlack, gamma=1e4, DIRECT backward)
  3. B NO-slack    : log-barrier central-path, gamma->1e12 (pure barrier, = acados tau_min), kappa=1e-6
  4. B slack       : log-barrier central-path, gamma=1e4 (barrier + Moreau slack), kappa=1e-6

Per (config, tol): per-sample dL/d(Q,R) of the 50-step rollout (vmap-grad over the batch) vs the
convergence-checked FD of the SAME forward (CLAUDE.md protocol). The FD evaluates ALL perturbations
(n_eps x n_w x 2) x ALL samples in ONE chunked vmap (thousands of parallel rollouts) — no Python-serial
per-perturbation calls / device syncs. Reports per-sample cos median/min, flagged fraction, and the
OUTLIER counts (#cos<0.99, #cos<0).

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/closed_loop_4way_sweep.py
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
           os.path.join(_DL_ROOT, "diffmpc2", "benchmarking", "linear-system"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from utils import generate_problem_data, N_STATE, N_CTRL  # noqa: E402
from turbompc.problems.optimal_control_problem import OptimalControlProblem, OptimalControlProblemSlack  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402
import cl_b  # noqa: E402

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
NW = NX + NU
SIM_STEPS, HORIZON, UMAX = 50, 20, 1.0
GAMMA_SLACK, GAMMA_NOSLACK, KAPPA = 1e4, 1e12, 1e-6
ADMM_MAX_ITER = 2000
FD_EPS = (1e-4, 3e-5, 1e-5, 3e-6)
LANE_CAP = 8192          # max parallel rollout lanes per vmap call (chunk the FD grid to bound memory)
# A forward backend: ADMM_FUSED_CUDSS computes the CORRECT hard-box gradient (cos 1.0 vs FD); the
# ADMM_JAX_LOOP_CUDSS_FFI forward returns duals that make the DIRECT backward wrong for the hard box
# (cos 0.21 vs FD) -- see backend_vs_fd.py. Slack box agrees on both.
A_FORWARD_BACKEND = ForwardBackend.ADMM_FUSED_CUDSS


def _reward(s, c): return -(jnp.sum(s ** 2) + jnp.sum(c ** 2))
def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _unflat(wf): return {QK: wf[:NX], RK: wf[NX:NW]}
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


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


def cost_fn_A(dyn, pp, w, x0b, tol, slack):
    """Return cost_fn(w_dict, state) -> scalar 50-step closed-loop cost (single sample)."""
    sp = _sp(tol)
    if slack:
        ppA = {**pp, "use_slack_variables": True, "slack_penalization_weight": GAMMA_SLACK}
        cls = OptimalControlProblemSlack
    else:
        ppA = dict(pp); cls = OptimalControlProblem
    solver = TurboMPCSolver(program=cls(dynamics=dyn, params=ppA), params=sp,
        forward_backend=A_FORWARD_BACKEND,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init = solver.solve(solver.initial_guess(ppA), problem_params=ppA, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=cls, problem_params=ppA, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=ppA, init_solution=init,
                               warm_start=False, num_sim_steps=SIM_STEPS)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


def cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, tol, slack):
    cl_b.GAMMA = GAMMA_SLACK if slack else GAMMA_NOSLACK     # monkeypatch slack weight (1e12 = no slack)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(1e-9),
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp); N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU,
                              pcg_params={"max_iter": 400, "tol_epsilon": 1e-12})
    b_solve = cl_b.make_b_mpc_solve(solver, pp, schur, ig, NX, NU, kappa=KAPPA, cp_tol=tol)

    def cost(weights, state):
        def step(carry, _):
            st, c = carry
            s, ctrl = b_solve(weights, st); u0 = ctrl[0]
            new = st + 1.0 * (A_sd @ st + B_m @ u0 + b_v)
            return (new, c + jnp.sum(new ** 2) + jnp.sum(u0 ** 2)), None
        (_, tot), _ = jax.lax.scan(step, (state, jnp.asarray(0.0)), None, length=SIM_STEPS)
        return tot
    return cost


def ad(cost_fn, w, x0b):
    """Per-sample analytic gradient (vmap-grad over the batch) -> [B, n_w]."""
    B = x0b.shape[0]
    gd = jax.jit(jax.vmap(jax.grad(cost_fn, argnums=0), in_axes=(None, 0)))(w, x0b)
    return np.stack([_flat({k: gd[k][i] for k in WK}) for i in range(B)])


def fd_gt(cost_fn, w, x0b):
    """Convergence-checked FD of the rollout (all perturbations x all samples in one chunked vmap).
    cost_fn is the forward solved at the GT (tightest) tolerance -> the converged 'true' gradient.
    Returns (gFD [B, n_w], flagged [B])."""
    B = x0b.shape[0]
    # --- FD grid: W_pert [n_eps, n_w, 2, n_w] -> flatten to [P, n_w] ---
    w_flat = jnp.asarray(_flat(w))
    E = jnp.eye(NW); eps_arr = jnp.asarray(FD_EPS); signs = jnp.asarray([1.0, -1.0])
    delta = eps_arr[:, None, None, None] * signs[None, None, :, None] * E[None, :, None, :]
    W_pert = (w_flat + delta).reshape(-1, NW)            # [P, n_w], P = n_eps*n_w*2
    P = W_pert.shape[0]
    single = jax.jit(jax.vmap(lambda wf, st: cost_fn(_unflat(wf), st)))   # one lane = (weight, state)
    cP = max(1, LANE_CAP // B)                           # perturbations per chunk
    costs = np.zeros((P, B))
    for s in range(0, P, cP):
        Wc = W_pert[s:s + cP]; c = Wc.shape[0]
        costs[s:s + c] = np.asarray(single(jnp.repeat(Wc, B, axis=0),
                                           jnp.tile(x0b, (c, 1)))).reshape(c, B)
    costs = costs.reshape(len(FD_EPS), NW, 2, B)
    fd_all = (costs[:, :, 0, :] - costs[:, :, 1, :]) / (2 * np.asarray(eps_arr)[:, None, None])  # [n_eps,n_w,B]
    # --- per-sample convergence check over eps ---
    per = [fd_all[ei].T for ei in range(len(FD_EPS))]   # list of [B, n_w]
    gFD = np.array(per[-1]); flagged = np.zeros(B, bool)
    for nn in range(B):
        ok = True
        for j in range(NW):
            seq = [pe[nn, j] for pe in per]; ch, o = seq[-1], False
            for a, bb in zip(seq[:-1], seq[1:]):
                if abs(a - bb) <= 1e-2 * abs(bb) + 1e-7: ch, o = a, True; break
            gFD[nn, j] = ch; ok = ok and o
        flagged[nn] = not ok
    return gFD, flagged


CONFIGS = ["A noslack", "A slack", "B noslack", "B slack"]


def build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, tol):
    if name == "A noslack":   return cost_fn_A(dyn, pp, w, x0b, tol, slack=False)
    elif name == "A slack":   return cost_fn_A(dyn, pp, w, x0b, tol, slack=True)
    elif name == "B noslack": return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, tol, slack=False)
    else:                     return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, tol, slack=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tolerances", type=float, nargs="+", default=[1e-1, 1e-3, 1e-5, 1e-7, 1e-9])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol_gt", type=float, default=None, help="GT FD tolerance (default = tightest sweep tol)")
    p.add_argument("--configs", type=str, nargs="+", default=None, help="subset of configs (default all 4)")
    a = p.parse_args()
    n = a.batch
    configs = a.configs if a.configs else CONFIGS
    assert all(c in CONFIGS for c in configs), f"--configs must be subset of {CONFIGS}"
    TOL_GT = a.tol_gt if a.tol_gt is not None else min(a.tolerances)
    print(f"Closed-loop 4-way (sim_steps={SIM_STEPS}) gradient ACCURACY vs QP tolerance [parallel]")
    print(f"  nx={NX} nu={NU} H={HORIZON} batch={n} seed={a.seed} kappa={KAPPA:g} "
          f"gamma_slack={GAMMA_SLACK:g} gamma_noslack={GAMMA_NOSLACK:g} lane_cap={LANE_CAP}")
    print(f"  GT = convergence-checked FD of each config's forward at the TIGHTEST tol={TOL_GT:.0e} "
          f"(the converged 'true' gradient); per-config fixed reference for all tolerances.")
    dyn, pp, w, x0b, A_sd, B_m, b_v = _problem(a.seed, n)

    # --- Phase 1: per-config ground truth = FD at the tightest tolerance ---
    print("\n[Phase 1] ground-truth FD at tol_gt:")
    GT = {}
    for name in configs:
        t = time.time()
        cf = build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, TOL_GT)
        gFD, fl = fd_gt(cf, w, x0b)
        GT[name] = (gFD, fl)
        print(f"    {name:>10}: GT-flagged {int(fl.sum()):3d}/{n}  ({time.time()-t:.0f}s)", flush=True)
        jax.clear_caches(); gc.collect()

    # --- Phase 2: analytic gradient at each tol vs the fixed GT ---
    results = {}
    for tol in a.tolerances:
        print(f"\n--- tol={tol:.0e} ---  {'config':>10} | {'cos med':>8} {'cos min':>8} {'GTflag':>8} "
              f"{'#cos<.99':>9} {'#cos<0':>7}")
        for name in configs:
            t = time.time()
            cf = build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, tol)
            gAD = ad(cf, w, x0b)
            gFD, fl = GT[name]; keep = ~fl
            cos = np.array([_cos(gAD[i], gFD[i]) for i in range(n)])
            cm = np.median(cos[keep]) if keep.any() else float("nan")
            cmin = np.min(cos[keep]) if keep.any() else float("nan")
            n99 = int((cos[keep] < 0.99).sum()); n0 = int((cos[keep] < 0).sum())
            results[(tol, name)] = dict(cos=cos)
            print(f"{'':>17}{name:>10} | {cm:8.4f} {cmin:8.4f} {int(fl.sum()):3d}/{n:<4d} "
                  f"{n99:9d} {n0:7d}   ({time.time()-t:.0f}s)", flush=True)
            jax.clear_caches(); gc.collect()

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "closed_loop_4way_sweep.npz"),
             tolerances=np.array(a.tolerances), batch=n, seed=a.seed, tol_gt=TOL_GT,
             **{f"GTflag|{name}": GT[name][1] for name in configs},
             **{f"{name}|{tol:.0e}|cos": results[(tol, name)]["cos"]
                for (tol, name) in results})
    print("\nsaved results/closed_loop_4way_sweep.npz")


if __name__ == "__main__":
    main()
