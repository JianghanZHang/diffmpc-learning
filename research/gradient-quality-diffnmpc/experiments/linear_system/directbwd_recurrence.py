"""Is the hard-box DIRECT_CUDSS_FFI backward's imprecision (sample 45) systematic or a one-off? Across
several seeds, compare the hard-box DIRECT gradient to the barrier gradient (B no-slack, kappa=1e-6 --
the known-correct, well-conditioned reference) per sample, and count how many samples DIRECT gets wrong
(cos < 0.99). No FD needed. ~1/64 per seed => a systematic-but-rare DIRECT-backward limitation at
near-active constraints; 0 => seed-0-specific.

    python research/gradient-quality-diffnmpc/experiments/linear_system/directbwd_recurrence.py
"""
import os, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import _problem, _cos, ad, cost_fn_A, cost_fn_B  # noqa: E402

SEEDS = [0, 1, 2, 3]


def main():
    n = 64
    print("hard-box DIRECT gradient vs barrier gradient (B no-slack, kappa=1e-6), per sample, tight tol:")
    print("  #<0.99 = samples where the DIRECT backward is wrong while the barrier is right")
    tot = 0
    for seed in SEEDS:
        t = time.time()
        dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(seed, n, 50, 160)
        g_hard = ad(cost_fn_A(dyn, pp, w, x0b, 1e-9, False, 50), w, x0b)
        g_bar = ad(cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, 1e-9, False, 50), w, x0b)
        cos = np.array([_cos(g_hard[i], g_bar[i]) for i in range(n)])
        bad = np.where(cos < 0.99)[0]; tot += len(bad)
        print(f"  seed={seed}: #<0.99={len(bad):2d}/{n}  min={np.min(cos):+.4f}  "
              f"bad samples={list(bad)} cos={[round(float(cos[b]),3) for b in bad]}  ({time.time()-t:.0f}s)",
              flush=True)
    print(f"\ntotal DIRECT-wrong samples over {len(SEEDS)} seeds: {tot}/{len(SEEDS)*n} "
          f"({100*tot/(len(SEEDS)*n):.1f}%)  -> {'systematic but rare' if tot else 'seed-0 only'}")


if __name__ == "__main__":
    main()
