"""Reproduce the TRI gradient-accuracy benchmark (benchmarking/linear-system/run_sweeps_gradient_accuracy.sh)
with a CONVERGENCE-VALIDATED finite-difference ground truth instead of the benchmark's single fd_eps=1e-5.

Benchmark setup reproduced faithfully: horizon 40, batch 64, ADMM_FUSED_CUDSS fwd / DIRECT_CUDSS_FFI bwd,
admm_max_iter 1000, sim_steps 50, umax 1.0, weights = Q/R diagonals, AD tolerances [1e-1,1e-3,1e-5,1e-7,1e-9],
FD-reference forward at tol 1e-9. The benchmark finite-differences the BATCH-SUMMED cost (sum_i cost_i) and
reports one cosine per seed over 100 seeds; here we report BOTH that batch-summed cos AND the per-sample cos
(free: the batch-summed FD is just the sum of the per-sample FDs).

STEP 1 (this script, --smoke): confirm the FD plateau and check whether the benchmark's fd_eps=1e-5 sits
inside it. Run once on a few seeds; the validated eps is then reused for the full sweep (no per-seed repeat).

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/benchmark_repro.py --smoke
"""
from __future__ import annotations
import os, sys, time, argparse, gc
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_gradient_accuracy import (  # noqa: E402
    _problem, cost_fn_A, ad, _fd_at_eps, _check_conv, _flat, _unflat, _cos, NW)

HORIZON_B = 40            # benchmark horizon
SIM_STEPS = 50
BENCH_FD_TOL = 1e-9      # benchmark's FD-reference forward tolerance
BENCH_EPS = 1e-5         # the benchmark's single FD step
VALID_EPS = 1e-4         # convergence-validated step (inside BOTH per-sample and batch-sum plateaus)
AD_TIGHT = 1e-9
AD_TOLS = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]   # benchmark's solver-tolerance sweep
# eps grid bracketing the benchmark eps, extended below 3e-6 to locate the horizon-40 noise floor
EPS_GRID = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


def _relerr(a, b):  # per-row ||a-b||/||b||
    return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-30)


def diagnose_seed(seed, batch, sim_steps):
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(seed, batch, sim_steps, HORIZON_B)
    cf = cost_fn_A(dyn, pp, w, x0b, BENCH_FD_TOL, slack=False, sim_steps=sim_steps)
    single = jax.jit(jax.vmap(lambda wf, st: cf(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w))
    G = {}
    for e in EPS_GRID:
        t = time.time(); G[e] = _fd_at_eps(single, w_flat, x0b, e)            # [B, NW]
        print(f"    FD eps={e:.0e}  ({time.time()-t:.0f}s)", flush=True)
    # per-sample plateau, and batch-summed (sum of per-sample FDs == FD of sum_i cost_i)
    gFD, conv, eps_used = _check_conv(G)
    Gsum = {e: G[e].sum(axis=0, keepdims=True) for e in EPS_GRID}
    gFDsum, convsum, eps_used_sum = _check_conv(Gsum)
    # FD@eps vs the converged plateau, and the benchmark eps's standing
    print(f"  eps-sweep: rel-err of FD(eps) vs converged plateau   (<-- benchmark eps=1e-5)")
    print(f"    {'eps':>8} | {'per-sample relerr med':>22} | {'batch-sum relerr':>17}")
    for e in EPS_GRID:
        re_ps = float(np.median(_relerr(G[e], gFD)))
        re_sum = float(_relerr(Gsum[e], gFDsum)[0])
        mark = "  <-- benchmark eps" if e == BENCH_EPS else ""
        print(f"    {e:8.0e} | {re_ps:22.2e} | {re_sum:17.2e}{mark}")
    # AD at tight tol: does FD@plateau match AD, and does FD@benchmark-eps match AD?
    gAD = ad(cf, w, x0b)
    gADsum = gAD.sum(axis=0, keepdims=True)
    g_eps = G[BENCH_EPS]; g_eps_sum = Gsum[BENCH_EPS]
    cos_ps = lambda gx: np.array([_cos(gAD[i], gx[i]) for i in range(batch)])
    print(f"  per-sample cos(AD@1e-9, FD):  plateau med={np.median(cos_ps(gFD)):.5f} min={np.min(cos_ps(gFD)):.5f}"
          f"  |  eps=1e-5 med={np.median(cos_ps(g_eps)):.5f} min={np.min(cos_ps(g_eps)):.5f}")
    print(f"  batch-sum  cos(AD@1e-9, FD):  plateau={_cos(gADsum[0], gFDsum[0]):.5f}"
          f"  |  eps=1e-5={_cos(gADsum[0], g_eps_sum[0]):.5f}")
    nps = int(conv.sum())
    print(f"  per-sample plateau: {nps}/{batch} converged; batch-sum converged={bool(convsum[0])} "
          f"(eps_used={eps_used_sum[0]:.0e}); benchmark eps=1e-5 "
          f"{'INSIDE' if np.median(_relerr(g_eps, gFD)) < 1e-2 else 'OUTSIDE'} the plateau")
    jax.clear_caches(); gc.collect()
    return dict(seed=seed, conv=conv, eps_used=eps_used, eps_used_sum=eps_used_sum,
                relerr_eps_ps=float(np.median(_relerr(g_eps, gFD))),
                relerr_eps_sum=float(_relerr(g_eps_sum, gFDsum)[0]))


def full_seed(seed, batch, sim_steps, eps_list, ad_tols):
    """Benchmark reproduction for ONE seed at the VALIDATED eps (no per-seed convergence sweep): single-eps
    FD GT at each eps in eps_list (tight forward), AD swept over ad_tols, cos per-sample + batch-summed +
    batch-summed with per-sample outliers (cos<0.99) removed. Returns {(tol,eps): (cos_ps[B], cos_sum,
    cos_sum_clean, n_out)}."""
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(seed, batch, sim_steps, HORIZON_B)
    cf_gt = cost_fn_A(dyn, pp, w, x0b, BENCH_FD_TOL, slack=False, sim_steps=sim_steps)
    single = jax.jit(jax.vmap(lambda wf, st: cf_gt(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w))
    gFD = {e: _fd_at_eps(single, w_flat, x0b, e) for e in eps_list}             # [B, NW] per eps
    jax.clear_caches(); gc.collect()
    out = {}
    for tol in ad_tols:
        cf = cost_fn_A(dyn, pp, w, x0b, tol, slack=False, sim_steps=sim_steps)
        gAD = ad(cf, w, x0b)
        for e in eps_list:
            gf = gFD[e]
            cos_ps = np.array([_cos(gAD[i], gf[i]) for i in range(batch)])
            cos_sum = _cos(gAD.sum(0), gf.sum(0))
            keep = cos_ps >= 0.99
            cos_sum_clean = _cos(gAD[keep].sum(0), gf[keep].sum(0)) if keep.any() else float("nan")
            out[(tol, e)] = (cos_ps, float(cos_sum), float(cos_sum_clean), int((~keep).sum()))
        jax.clear_caches(); gc.collect()
    return out


def run_full(seeds, batch, sim_steps, eps_list, ad_tols):
    agg = {(tol, e): {"ps": [], "sum": [], "sum_clean": [], "nout": []} for tol in ad_tols for e in eps_list}
    for si, seed in enumerate(seeds):
        t = time.time()
        res = full_seed(seed, batch, sim_steps, eps_list, ad_tols)
        for key, (cos_ps, cos_sum, cos_sum_clean, n_out) in res.items():
            agg[key]["ps"].append(cos_ps); agg[key]["sum"].append(cos_sum)
            agg[key]["sum_clean"].append(cos_sum_clean); agg[key]["nout"].append(n_out)
        print(f"  seed {seed} ({si+1}/{len(seeds)}) done ({time.time()-t:.0f}s)", flush=True)
    # report
    print(f"\n=== reproduction: cos(AD, FD) vs solver tolerance ({len(seeds)} seeds, horizon {HORIZON_B}) ===")
    for e in eps_list:
        tag = "VALIDATED" if e == VALID_EPS else ("BENCHMARK" if e == BENCH_EPS else "")
        print(f"\n  -- FD eps={e:.0e} {tag} --")
        print(f"    {'tol':>6} | {'batch-sum cos med(min)':>24} | {'per-sample cos med(min)':>24} | "
              f"{'batch-sum cos, outliers removed':>30} | {'#out/seed':>9}")
        for tol in ad_tols:
            d = agg[(tol, e)]
            ssum = np.array(d["sum"]); sclean = np.array(d["sum_clean"]); sps = np.concatenate(d["ps"])
            nout = np.mean(d["nout"])
            print(f"    {tol:6.0e} | {np.median(ssum):8.3f} ({np.min(ssum):6.3f})        | "
                  f"{np.median(sps):8.3f} ({np.min(sps):6.3f})        | "
                  f"{np.median(sclean):8.3f} ({np.min(sclean):6.3f})              | {nout:9.1f}")
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    save = {"seeds": np.array(seeds), "batch": batch, "horizon": HORIZON_B, "eps_list": np.array(eps_list),
            "ad_tols": np.array(ad_tols)}
    for (tol, e), d in agg.items():
        k = f"{tol:.0e}|{e:.0e}"
        save[f"ps|{k}"] = np.array(d["ps"]); save[f"sum|{k}"] = np.array(d["sum"])
        save[f"sumclean|{k}"] = np.array(d["sum_clean"]); save[f"nout|{k}"] = np.array(d["nout"])
    np.savez(os.path.join(_HERE, "results", "benchmark_repro.npz"), **save)
    print("\nsaved results/benchmark_repro.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="FD-convergence diagnosis only")
    p.add_argument("--full", action="store_true", help="full reproduction at validated + benchmark eps")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--nseeds", type=int, default=None, help="use seeds 0..nseeds-1")
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    seeds = a.seeds if a.seeds is not None else (list(range(a.nseeds)) if a.nseeds else [0, 1, 2, 3])

    if a.full:
        print(f"FULL reproduction: horizon={HORIZON_B}, batch={a.batch}, {len(seeds)} seeds, "
              f"FD eps={{validated {VALID_EPS:.0e}, benchmark {BENCH_EPS:.0e}}}, fwd_tol={BENCH_FD_TOL:.0e}")
        run_full(seeds, a.batch, SIM_STEPS, [VALID_EPS, BENCH_EPS], AD_TOLS)
        return

    print(f"FD-convergence diagnosis at the BENCHMARK setup: horizon={HORIZON_B}, batch={a.batch}, "
          f"fwd_tol={BENCH_FD_TOL:.0e}, sim_steps={SIM_STEPS}")
    print(f"  question: is the benchmark's single fd_eps={BENCH_EPS:.0e} inside the FD plateau?")
    out = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        out.append(diagnose_seed(seed, a.batch, SIM_STEPS))
    print("\n=== summary ===")
    for r in out:
        print(f"  seed {r['seed']}: per-sample {int(r['conv'].sum())}/{a.batch} converged; "
              f"batch-sum eps_used={r['eps_used_sum'][0]:.0e}; "
              f"eps=1e-5 rel-err ps={r['relerr_eps_ps']:.1e} sum={r['relerr_eps_sum']:.1e}")


if __name__ == "__main__":
    main()
