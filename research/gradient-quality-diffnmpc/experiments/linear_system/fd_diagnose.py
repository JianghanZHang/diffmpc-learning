"""Why don't samples {28,43,45,48,60} converge? Look at the hard-box FD gradient vs eps for the
non-converged samples (+ one converged sample for contrast). For each sample, find the weight
component with the largest consecutive-eps change and print its FD-vs-eps trajectory, so we can tell:
 - settling toward a value (changes shrinking) -> kink is within ~eps; smaller eps would resolve it
 - jumping between two levels                  -> w sits exactly on a kink (two-sided limits differ)
 - growing ~1/eps                              -> genuine cost jump (discontinuity)
 - flat then erratic at small eps              -> hit the cost noise floor (need a tighter forward)

    python research/gradient-quality-diffnmpc/experiments/linear_system/fd_diagnose.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, cost_fn_A, _fd_at_eps, _flat, _unflat, NW, GT_TOL)

BAD = [28, 43, 45, 48, 60]
GOOD = [0, 1]
EPS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 3e-9, 1e-9]
WNAMES = [f"Q{i}" for i in range(8)] + [f"R{i}" for i in range(4)]


def main():
    samples = BAD + GOOD
    dyn, pp, w, x0b_full, A_sd, B_m, b_v, dt = _problem(0, 64, 50, 160)
    x0b = x0b_full[jnp.asarray(samples)]
    cf = cost_fn_A(dyn, pp, w, x0b, GT_TOL, slack=False, sim_steps=50)
    single = jax.jit(jax.vmap(lambda wf, st: cf(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w))
    G = {}
    for eps in EPS:
        t = time.time(); G[eps] = _fd_at_eps(single, w_flat, x0b, eps)   # [n_samples, NW]
        print(f"  eps={eps:.0e} ({time.time()-t:.0f}s)", flush=True)
    arrs = [G[e] for e in EPS]
    print(f"\neps grid: {'  '.join(f'{e:.0e}' for e in EPS)}")
    for si, samp in enumerate(samples):
        tag = "BAD " if samp in BAD else "good"
        # worst weight component = largest median consecutive relative change
        rc = np.array([[abs(arrs[k][si, j] - arrs[k + 1][si, j]) /
                        (abs(arrs[k + 1][si, j]) + 1e-30) for k in range(len(EPS) - 1)] for j in range(NW)])
        j = int(np.argmax(rc.max(axis=1)))
        traj = [arrs[k][si, j] for k in range(len(EPS))]
        chg = "  ".join(f"{rc[j, k]:5.0e}" for k in range(len(EPS) - 1))
        print(f"\n[{tag} sample {samp:2d}]  worst weight={WNAMES[j]}")
        print(f"  FD value : " + "  ".join(f"{v:+8.4f}" for v in traj))
        print(f"  rel-chg  :       " + chg)


if __name__ == "__main__":
    main()
