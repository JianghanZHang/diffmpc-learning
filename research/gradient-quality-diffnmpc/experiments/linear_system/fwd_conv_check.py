"""Is sample 45's hard-box anomaly an under-converged FORWARD (DIRECT backward assumes optimality)?
Run the hard-box closed loop (fused fwd, GT_TOL) for the anomaly sample + a clean sample, and report
the per-step ADMM convergence_error and iteration count. A step with a large residual / hitting
ADMM_MAX_ITER means the forward isn't a KKT point there -> the DIRECT backward is wrong while the FD
(of the actual map) and the barrier are right.

    python research/gradient-quality-diffnmpc/experiments/linear_system/fwd_conv_check.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, _sp, QK, RK, NX, NU, GT_TOL, ADMM_MAX_ITER,
    TurboMPCSolver, ForwardBackend, BackwardBackend, OptimalControlProblem)


def main():
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, 64, 50, 160)
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(GT_TOL),
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI,
        use_full_hessian=True)
    ig = solver.initial_guess(pp)

    def rollout_conv(x0):
        def step(st, _):
            sol = solver.solve(ig, problem_params={**pp, "initial_state": st, QK: w[QK], RK: w[RK]},
                               weights={**w, "initial_state": st})
            u0 = sol.controls[0]
            return st + dt * (A_sd @ st + B_m @ u0 + b_v), (sol.convergence_error, jnp.max(sol.admm_iters))
        _, (errs, iters) = jax.lax.scan(step, x0, None, length=50)
        return errs, iters

    f = jax.jit(rollout_conv)
    print(f"hard-box forward convergence (GT_TOL={GT_TOL:.0e}, ADMM_MAX_ITER={ADMM_MAX_ITER}):")
    for samp, tag in [(45, "ANOMALY"), (0, "clean"), (28, "strict-comp-fail/clean-grad")]:
        errs, iters = f(x0b[samp]); errs = np.asarray(errs); iters = np.asarray(iters)
        ws = int(np.argmax(errs))
        print(f"  sample {samp:2d} [{tag:>26}]: max conv_err={errs.max():.2e}  max iters={int(iters.max())}  "
              f"hit-cap steps={int((iters >= ADMM_MAX_ITER - 50).sum())}/50  | worst step {ws}: "
              f"err={errs[ws]:.2e} iters={int(iters[ws])}")


if __name__ == "__main__":
    main()
