"""Is the hard-box gradient anomaly (sample 45) a STRICT-COMPLEMENTARITY failure?

For each sample's closed-loop (pure-barrier forward, gamma=1e12), at each step solve the QP at two
kappas and read every box constraint's slack s_i (=u-Gx) and multiplier y_i (s_i*y_i=kappa). Define
D = max over (steps, constraints) of min(s_i, y_i):
  - strict complementarity holds  -> D ~ kappa   (drops ~100x for 100x smaller kappa)
  - strict complementarity FAILS  -> D ~ sqrt(kappa) (drops ~10x) at a weakly-active constraint.
Then correlate D with the per-sample hard-box-vs-FD discrepancy (1 - fd_check_cos): if sample 45 has
the largest D, D~sqrt(kappa), and D tracks the discrepancy, the anomaly IS strict-comp failure.

    python research/gradient-quality-diffnmpc/experiments/linear_system/strict_comp_check.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, QK, RK, NX, NU, GAMMA_NOSLACK, CP_MAX_ITER,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblem,
    make_schur_solver, SchurSolverBackend, to_one_sided, solve_qp_central_path)

KAPPAS = [1e-5, 1e-7]


def main():
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, 64, 50, 160)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(1e-9),
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp); N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU,
                              pcg_params={"max_iter": 800, "tol_epsilon": 1e-12})
    gamma = GAMMA_NOSLACK

    def step_sy(state, kappa):
        pp_i = {**pp, "initial_state": state, QK: w[QK], RK: w[RK]}
        qp1 = to_one_sided(solver._build_qp_data(ig.states, ig.controls, pp_i), gamma)
        x, duals, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=gamma,
                                            rho_bar=0.1, max_iter=CP_MAX_ITER, tol=1e-9)
        s = qp1.ineq.u - jnp.einsum("tij,tj->ti", qp1.ineq.G, x)          # slacks  >= 0
        y = duals[2]                                                       # multipliers >= 0
        u0 = x[0, NX:]
        return x, s, y, u0

    def rollout_D(x0, kappa):
        def step(carry, _):
            st, Dmax = carry
            _, s, y, u0 = step_sy(st, kappa)
            D = jnp.max(jnp.minimum(jnp.abs(s), jnp.abs(y)))              # max-min(s,y) over all constraints
            return (st + dt * (A_sd @ st + B_m @ u0 + b_v), jnp.maximum(Dmax, D)), None
        (_, Dmax), _ = jax.lax.scan(step, (x0, jnp.asarray(0.0)), None, length=50)
        return Dmax

    Z = np.load(os.path.join(_HERE, "results", "gradient_accuracy.npz"))
    disc = 1.0 - Z["fd_check_cos"]                                        # hard-box-vs-FD discrepancy
    Dk = {}
    for kappa in KAPPAS:
        t = time.time(); Dk[kappa] = np.asarray(jax.jit(jax.vmap(lambda x0: rollout_D(x0, kappa)))(x0b))
        print(f"  D at kappa={kappa:.0e} ({time.time()-t:.0f}s)", flush=True)
    D5, D7 = Dk[1e-5], Dk[1e-7]
    scaling = D5 / (D7 + 1e-30)                                           # ~10 if sqrt(kappa), ~100 if kappa

    j = int(np.argmin(Z["fd_check_cos"]))
    print(f"\nsample {j} (the anomaly): fd_check_cos={Z['fd_check_cos'][j]:.4f}")
    print(f"  D(1e-5)={D5[j]:.2e}  D(1e-7)={D7[j]:.2e}  ratio={scaling[j]:.1f}   "
          f"(~10 => sqrt(kappa) => STRICT COMP FAILS;  ~100 => kappa => holds)")
    print(f"  sqrt(1e-5)={np.sqrt(1e-5):.2e}, sqrt(1e-7)={np.sqrt(1e-7):.2e}  (compare to D above)")
    order = np.argsort(-D5)
    print(f"\n  top-5 samples by D(1e-5) [degeneracy]:  (sample: D5  ratio  discrepancy 1-cos)")
    for k in order[:5]:
        print(f"    sample {int(k):2d}: D5={D5[k]:.2e}  ratio={scaling[k]:5.1f}  disc={disc[k]:.3f}")
    print(f"\n  is sample {j} the most degenerate?  rank by D5 = {int(np.where(order==j)[0][0])} (0=most)")
    rho = float(np.corrcoef(D5, disc)[0, 1])
    print(f"  correlation(D5, discrepancy) over 64 samples = {rho:+.3f}")
    np.savez(os.path.join(_HERE, "results", "strict_comp_check.npz"), D5=D5, D7=D7, disc=disc)


if __name__ == "__main__":
    main()
