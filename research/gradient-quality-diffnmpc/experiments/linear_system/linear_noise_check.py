"""CRITICAL VALIDATION: is the linear closed-loop hard-box 'pathology' real, or (like the quadrotor's
cos=0.616) an FD-below-noise-floor artifact? The linear rollout cost is large (~1e4), so its FD at the
small eps I used (3e-5..1e-6) may be noise-dominated. Measure the cost noise floor and re-compute the
hard-box cos vs FD at LARGER eps; if the outliers vanish, the headline was compromised.

    python research/gradient-quality-diffnmpc/experiments/linear_system/linear_noise_check.py
"""
from __future__ import annotations
import os, sys
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

from benchmark_problem_setup import build_turbompc_linear_problem
from utils import generate_problem_data, N_STATE, N_CTRL
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend
from turbompc.utils.load_params import load_solver_params
from turbompc.utils.timing import ProblemConfig, build_rollout_fn

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
SIM_STEPS, HORIZON, UMAX, TIGHT = 50, 20, 1.0, 1e-9


def _reward(s, c): return -(jnp.sum(s ** 2) + jnp.sum(c ** 2))
def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _sp(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = eps; sp["warm_start_backward"] = False
    sp["linesearch"] = False; sp["admm"]["max_iter"] = 4000; sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = eps; sp["admm"]["eps_rel"] = eps
    return sp


def main():
    n = 24
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(n, 0, n_state=NX, n_ctrl=NU)
    pp = dict(pp_t); pp["dynamics_state_dot_params"] = {"A": jnp.asarray(A - np.eye(NX)), "B": jnp.asarray(B), "b": jnp.asarray(b)}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    x0b = jnp.asarray(x0); w = {k: pp[k] for k in WK}
    sp = _sp(TIGHT)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init = solver.solve(solver.initial_guess(pp), problem_params=pp, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=OptimalControlProblem, problem_params=pp, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, None))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=pp, init_solution=init, warm_start=False, num_sim_steps=SIM_STEPS)
    def cost_one(ww, xi): return rollout(xi, {**ww, "initial_state": xi})[0]
    cv = jax.jit(jax.vmap(cost_one, in_axes=(None, 0)))
    gfn = jax.jit(jax.vmap(jax.grad(cost_one, argnums=0), in_axes=(None, 0)))

    c1 = np.asarray(cv(w, x0b)); c2 = np.asarray(cv(w, x0b))
    print(f"cost magnitude: median={np.median(np.abs(c1)):.1f} max={np.max(np.abs(c1)):.1f}")
    print(f"NOISE FLOOR: max|c1-c2|={np.max(np.abs(c1-c2)):.2e}  rel={np.max(np.abs((c1-c2)/c1)):.2e}")

    gA_d = gfn(w, x0b)
    gA = np.stack([_flat({k: gA_d[k][i] for k in WK}) for i in range(n)])

    def fd_cos(eps_seq):
        per = []
        for eps in eps_seq:
            cols = []
            for k in WK:
                base = np.asarray(w[k], float)
                for i in range(base.size):
                    ap = base.copy(); ap[i] += eps; am = base.copy(); am[i] -= eps
                    cols.append((np.asarray(cv({**w, k: jnp.asarray(ap)}, x0b)) - np.asarray(cv({**w, k: jnp.asarray(am)}, x0b))) / (2 * eps))
            per.append(np.stack(cols, axis=1))
        g = np.array(per[-1]); flag = np.zeros(n, bool)
        for nn in range(n):
            ok = True
            for j in range(per[0].shape[1]):
                seq = [pe[nn, j] for pe in per]; ch, okj = seq[-1], False
                for a_, b_ in zip(seq[:-1], seq[1:]):
                    if abs(a_ - b_) <= 1e-2 * abs(b_) + 1e-7: ch, okj = a_, True; break
                g[nn, j] = ch; ok = ok and okj
            flag[nn] = not ok
        cos = np.array([_cos(gA[nn], g[nn]) for nn in range(n)])
        return cos, flag

    for label, eps_seq in [("SMALL eps (original)", (3e-5, 1e-5, 3e-6, 1e-6)),
                           ("LARGE eps (above floor)", (3e-3, 1e-3, 3e-4, 1e-4))]:
        cos, flag = fd_cos(eps_seq)
        keep = ~flag
        print(f"\n{label}: cos median={np.median(cos[keep]):.4f} min={np.min(cos[keep]):.4f} "
              f"#<0.99={int((cos[keep]<0.99).sum())} #<0={int((cos[keep]<0).sum())} flagged={int(flag.sum())}/{n}")
        print(f"   worst cos: {np.round(np.sort(cos),3)[:6]}")


if __name__ == "__main__":
    main()
