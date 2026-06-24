"""Test the active-set-SWITCHING hypothesis on the linear closed-loop (where the hard-box gradient
pathology IS dramatic). Hypothesis: a sample's hard-box outlier severity (low cos vs FD) is driven by
how much the applied-control active set SWITCHES under small weight perturbations (= how many kinks the
rollout-cost-vs-weights map has near that sample), not by mere constraint binding.

Per sample, count saturation-status flips of the 50 applied controls u0[k] across small +/-eps weight
perturbations, then correlate that switch-count with the per-sample hard-box cos from
results/closed_loop_A_s0.npz. A strong negative correlation confirms the hypothesis.

    python research/gradient-quality-diffnmpc/experiments/linear_system/switch_correlation.py
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

NX, NU = N_STATE, N_CTRL
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
SIM_STEPS, HORIZON, UMAX, TIGHT = 50, 20, 1.0, 1e-9


def _solver_params(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = eps; sp["warm_start_backward"] = False
    sp["linesearch"] = False; sp["admm"]["max_iter"] = 4000; sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = eps; sp["admm"]["eps_rel"] = eps
    return sp


def main():
    n = 24
    dyn, pp_t = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(n, 0, n_state=NX, n_ctrl=NU)
    A_sd = jnp.asarray(A - np.eye(NX)); B_m = jnp.asarray(B); b_v = jnp.asarray(b)
    pp = dict(pp_t); pp["dynamics_state_dot_params"] = {"A": A_sd, "B": B_m, "b": b_v}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    x0_batch = jnp.asarray(x0); w = {k: pp[k] for k in WK}
    sp = _solver_params(TIGHT)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=sp,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp)

    def u0_traj(weights, state):
        # closed-loop rollout returning the 50 applied controls u0[k]
        def step(carry, _):
            st = carry
            sol = solver.solve(ig, problem_params=pp, weights={**weights, "initial_state": st})
            u0 = sol.controls[0]
            new_st = st + 1.0 * (A_sd @ st + B_m @ u0 + b_v)
            return new_st, u0
        _, us = jax.lax.scan(step, state, None, length=SIM_STEPS)
        return us  # (SIM_STEPS, NU)
    u0_vec = jax.jit(jax.vmap(u0_traj, in_axes=(None, 0)))

    def sat_mask(weights):
        us = np.asarray(u0_vec(weights, x0_batch))      # (n, 50, NU)
        return (np.abs(us) > 0.99 * UMAX)               # saturated decisions

    base = sat_mask(w)
    eps = 1e-3
    flips = np.zeros(n, int)
    for k in WK:
        v = np.asarray(w[k], float)
        for i in range(v.size):
            for sgn in (+1, -1):
                vp = v.copy(); vp[i] += sgn * eps
                m = sat_mask({**w, k: jnp.asarray(vp)})
                flips += (m != base).reshape(n, -1).sum(axis=1)   # per-sample flipped decisions

    A = np.load(os.path.join(_HERE, "results", "closed_loop_A_s0.npz"))
    cos = A["cosA"]; flagged = A["flagged"].astype(bool)
    sat_frac = base.reshape(n, -1).mean(axis=1)   # fraction of (step,ctrl) saturated at nominal

    print(f"Active-set SWITCHING vs hard-box cos (linear, seed 0, n={n}, eps={eps:g}):")
    print(f"{'i':>2} {'cos':>8} {'flagged':>7} {'switch_flips':>12} {'sat_frac':>9}")
    order = np.argsort(cos)
    for i in order:
        print(f"{i:>2} {cos[i]:>8.4f} {str(bool(flagged[i])):>7} {flips[i]:>12} {sat_frac[i]:>9.3f}")
    # correlations (Spearman via rank, since the relationship may be monotone-nonlinear)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    out = (cos < 0.99) | flagged
    print(f"\nSpearman(cos, switch_flips)   = {spearman(cos, flips):+.3f}  (expect NEGATIVE: more switching -> lower cos)")
    print(f"Spearman(cos, sat_frac)       = {spearman(cos, sat_frac):+.3f}  (binding alone, expect weaker)")
    print(f"median switch_flips: outliers={np.median(flips[out]):.0f} vs clean={np.median(flips[~out]):.0f}")
    print(f"median sat_frac:     outliers={np.median(sat_frac[out]):.3f} vs clean={np.median(sat_frac[~out]):.3f}")
    np.savez(os.path.join(_HERE, "results", "switch_correlation.npz"),
             cos=cos, flagged=flagged, switch_flips=flips, sat_frac=sat_frac)
    print("saved results/switch_correlation.npz")


if __name__ == "__main__":
    main()
