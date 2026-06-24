"""B's closed-loop outliers vs the smoothing strength kappa.

B (log-barrier) is outlier-free at kappa=1e-6. As kappa->0 the barrier -> hard box (= A), so B
must inherit A's closed-loop pathology. This sweep finds the threshold: for each kappa, run the
50-step closed-loop B sweep (seed 0) and report how many samples become outliers / FD-flagged.

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/cl_b_kappa_sweep.py
"""
import os, sys, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import cl_b  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kappas", type=float, nargs="+",
                   default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9, 1e-11])
    args = p.parse_args()
    print(f"B closed-loop outliers vs kappa: H={cl_b.HORIZON} sim_steps={cl_b.SIM_STEPS} "
          f"gamma={cl_b.GAMMA:g} n_samples={args.n_samples} seed={args.seed}")
    rows = []
    for kp in args.kappas:
        cosB, relB, flagged = cl_b.run_B(args.n_samples, args.seed, kappa=kp)
        keep = ~flagged.astype(bool)
        out = int(((cosB < 0.99) | flagged.astype(bool)).sum())
        rows.append((kp, float(np.median(cosB[keep])) if keep.any() else float("nan"),
                     int((cosB[keep] < 0.99).sum()), int((cosB[keep] < 0).sum()),
                     int(flagged.sum()), out, args.n_samples,
                     float(np.median(relB[keep])) if keep.any() else float("nan")))
    print("\n========== B outliers vs kappa ==========")
    print(f"{'kappa':>8} {'med cos':>9} {'#cos<0.99':>10} {'#cos<0':>7} {'#flagged':>9} {'#outliers':>10} {'med rel':>9}")
    for (kp, mc, c99, c0, fl, out, n, mr) in rows:
        print(f"{kp:8.0e} {mc:9.5f} {c99:10d} {c0:7d} {fl:9d} {out:>4d}/{n:<4d} {mr:9.1e}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "closed_loop_B_kappa_sweep.npz"),
             kappas=np.array([r[0] for r in rows]),
             med_cos=np.array([r[1] for r in rows]),
             n_cos_lt99=np.array([r[2] for r in rows]),
             n_flagged=np.array([r[4] for r in rows]),
             n_outliers=np.array([r[5] for r in rows]), n=args.n_samples)
    print("\nsaved results/closed_loop_B_kappa_sweep.npz")


if __name__ == "__main__":
    main()
