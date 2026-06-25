"""Direct mechanism check: the slack-vs-hard gradient error should scale as O(1/gamma). Compute the
relative gradient error ||g_slack(gamma) - g_hard|| / ||g_hard|| (g_hard = hard-box analytic) across
gamma; a ~10x drop per decade of gamma confirms the per-step O(1/gamma) Moreau-relaxation order.

    python research/gradient-quality-diffnmpc/experiments/linear_system/gamma_scaling.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import _problem, ad, cost_fn_A  # noqa: E402
from gamma_sweep import slack_cf  # noqa: E402

GAMMAS = [1e4, 1e5, 1e6, 1e7, 1e8]


def main():
    n = 64
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(0, n, 50, 160)
    g_hard = ad(cost_fn_A(dyn, pp, w, x0b, 1e-9, False, 50), w, x0b)
    hn = np.linalg.norm(g_hard, axis=1)
    print("relative gradient error ||g_slack - g_hard|| / ||g_hard|| vs gamma (sim_steps=50):")
    print("  (a ~10x drop per decade of gamma => O(1/gamma) Moreau-relaxation order)")
    prev = None
    for gamma in GAMMAS:
        t = time.time()
        g_slk = ad(slack_cf(dyn, pp, w, x0b, 1e-9, gamma, 50), w, x0b)
        rel = np.array([np.linalg.norm(g_slk[i] - g_hard[i]) / (hn[i] + 1e-30) for i in range(n)])
        med = float(np.median(rel))
        drop = (prev / med) if prev else float("nan")
        print(f"  gamma={gamma:.0e}: rel-err med={med:.3e}  drop-vs-prev-decade={drop:5.1f}x  ({time.time()-t:.0f}s)",
              flush=True)
        prev = med


if __name__ == "__main__":
    main()
