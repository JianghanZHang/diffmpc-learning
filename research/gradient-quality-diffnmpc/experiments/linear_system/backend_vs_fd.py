"""Which backend (if either) is RIGHT for the hard box? Compare each forward backend's gradient to the
FD ground truth (convergence-checked FD of the forward at a tight tol = the true gradient). If BOTH
backends are far from FD -> the hard-box gradient is genuinely ill-defined (not a one-backend bug).

For A (hard box and slack) at tol=1e-7: g_jax, g_fused, and FD truth; reports cos(g_jax,FD),
cos(g_fused,FD), cos(g_jax,g_fused).

    python research/gradient-quality-diffnmpc/experiments/linear_system/backend_vs_fd.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_4way_sweep import _problem, fd_gt, ad, _cos, ForwardBackend  # noqa: E402
from fused_check import cost_fn_A  # noqa: E402

TOL = 1e-7


def main():
    B = 24
    dyn, pp, w, x0b, A_sd, B_m, b_v = _problem(0, B)
    JAX, FUSED = ForwardBackend.ADMM_JAX_LOOP_CUDSS_FFI, ForwardBackend.ADMM_FUSED_CUDSS
    print(f"A backend vs FD truth (tol={TOL:.0e}, batch={B}): is the hard-box gradient backend-independent?")
    print(f"\n{'box':>6} | {'cos(jax,FD)':>12} {'cos(fused,FD)':>14} {'cos(jax,fused)':>15} {'FD-flagged':>11}")
    for slack in (False, True):
        cj = cost_fn_A(dyn, pp, w, x0b, TOL, slack, JAX)
        cf = cost_fn_A(dyn, pp, w, x0b, TOL, slack, FUSED)
        gj = ad(cj, w, x0b); gf = ad(cf, w, x0b)
        gFD, fl = fd_gt(cj, w, x0b)                 # truth = FD of the (converged) forward
        keep = ~fl
        jF = np.array([_cos(gj[i], gFD[i]) for i in range(B)])[keep]
        fF = np.array([_cos(gf[i], gFD[i]) for i in range(B)])[keep]
        jf = np.array([_cos(gj[i], gf[i]) for i in range(B)])[keep]
        box = "slack" if slack else "hard"
        print(f"{box:>6} | {np.median(jF):12.4f} {np.median(fF):14.4f} {np.median(jf):15.4f} "
              f"{int(fl.sum()):3d}/{B}", flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
