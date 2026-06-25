"""Sample-45 anomaly: is the hard-box gradient genuinely ill-conditioned, or is the DIRECT backward just
imprecise there? Sweep the barrier kappa -> 0 (barrier -> hard box) and watch sample 45's closed-loop
gradient vs the hard-box FD GT. If cos stays ~1.0 as kappa->0, the true hard gradient is well-defined
(= the FD) and the DIRECT backward is the imprecise one; if it degrades toward the DIRECT's ~0.034, the
hard limit itself is ill-conditioned.

    python research/gradient-quality-diffnmpc/experiments/linear_system/kappa_sweep45.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, _cos, ad, make_b_mpc_solve, GAMMA_NOSLACK, NX, NU, QK, RK,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblem,
    make_schur_solver, SchurSolverBackend)

KAPPAS = [1e-4, 1e-6, 1e-8, 1e-10]
J = 45


def b_cost_fn(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, kappa, sim_steps):
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(1e-9),
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp); N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU,
                              pcg_params={"max_iter": 800, "tol_epsilon": 1e-12})
    b_solve = make_b_mpc_solve(solver, pp, schur, ig, NX, NU, gamma=GAMMA_NOSLACK, kappa=kappa, cp_tol=1e-9)

    def cost(weights, state):
        def step(carry, _):
            st, c = carry
            s, ctrl = b_solve(weights, st); u0 = ctrl[0]
            new = st + dt * (A_sd @ st + B_m @ u0 + b_v)
            return (new, c + jnp.sum(new ** 2) + jnp.sum(u0 ** 2)), None
        (_, tot), _ = jax.lax.scan(jax.checkpoint(step), (state, jnp.asarray(0.0)), None, length=sim_steps)
        return tot
    return cost


def main():
    n = 64
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, n, 50, 160)
    Z = np.load(os.path.join(_HERE, "results", "gradient_accuracy.npz")); gFD = Z["gFD"]
    print(f"barrier (B no-slack) closed-loop gradient vs hard-box FD GT, kappa -> 0  (sample {J}):")
    print(f"  (hard-box DIRECT backward gave cos[{J}] = 0.034)")
    for kappa in KAPPAS:
        t = time.time()
        gAD = ad(b_cost_fn(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, kappa, 50), w, x0b)
        cos = np.array([_cos(gAD[i], gFD[i]) for i in range(n)])
        print(f"  kappa={kappa:.0e}: sample{J} cos={cos[J]:+.4f}  |  all: med={np.median(cos):.4f} "
              f"min={np.min(cos):.4f} #<0.99={int((cos < 0.99).sum())}  ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
