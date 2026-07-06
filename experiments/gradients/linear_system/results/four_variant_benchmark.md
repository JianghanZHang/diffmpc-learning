# 4-variant closed-loop gradient accuracy on the CORRECT solver (external/turbompc)

Compares the closed-loop gradient of 4 differentiable-NMPC formulations against **one common ground
truth** — the convergence-checked FD of the *true hard-constrained* (`|u|≤u_max`) closed-loop loss
(external's hard-box forward at GT_TOL=1e-11, the tightest solve; same 64 samples for all variants).
**Solver = `external/turbompc`** (the verified-correct one — diffmpc2 release-cleanup has the
inequality-multiplier sign bug). Setup: horizon 40, batch 64, **10 seeds** (640 per-sample points),
sim_steps 50, umax 1, SQP iter = 1. Reports **both** per-sample cos and batch-summed `cos_all`
(Σ-over-batch per seed). Script: `four_variant_benchmark.py`; figure `plot/four_variant_benchmark.png`
(per-sample box + batch-sum-median red diamond).

## Result — cos(variant AD, common hard-box GT), 10 seeds (GT non-converged = 0/640)

per-sample median (min) [#cos<0.99] · batch-sum median:

| AD tol | 1 turbompc hard box | 2 turbompc Moreau | 3 log-barrier no-slack | 4 log-barrier Moreau |
|---:|:---:|:---:|:---:|:---:|
| 1e-1 | 0.26 (−0.97) · 0.26 | 0.23 (−0.95) · 0.35 | 0.12 (−0.89) · −0.03 | 0.13 (−0.93) · −0.04 |
| 1e-3 | 1.000 [84] · 0.98 | 0.45 [536] · 0.38 | 0.996 [288] · 0.70 | 0.46 [531] · 0.39 |
| 1e-5 | **1.000 [6] · 1.000** | 0.45 [533] · 0.38 | **1.000 [43] · 1.000** | 0.45 [533] · 0.38 |
| 1e-7 | **1.000 [0] · 1.000** | 0.45 [533] · 0.38 | **1.000 [0] · 1.000** | 0.45 [533] · 0.38 |
| 1e-9 | **1.000 [0] · 1.000** | 0.450 (−0.76) [533] · 0.378 | **1.000 [0] · 1.000** | 0.450 (−0.76) [533] · 0.378 |

## Findings

1. **The hard box (external) and the pure log-barrier are FAITHFUL — cos 1.0 with ZERO outliers** over
   640 samples once solved tightly (hard box by tol≤1e-7, barrier by tol≤1e-5; the barrier's central path
   converges slightly slower). This is the **clincher that external/turbompc is correct**: on diffmpc2
   (release-cleanup) the hard box has ~4–8 outliers/seed (`cos_all` ~0.36) from the multiplier-sign bug;
   external has **none** (`#cos<0.99 = 0`, `cos_all = 1.0`).

2. **Both Moreau-slack variants are biased — cos ≈ 0.45**, *identically* (turbompc-Moreau and
   barrier+Moreau give the same 0.450 per-sample / 0.378 batch-sum, flat across tol≥1e-5). This is the
   **Moreau-relaxation bias** from the true hard-constrained gradient (not a solve issue — flat across
   tolerance; not the formulation — same whether on turbompc or the barrier). 116/640 samples even go
   cos<0 (the bias flips direction at active controls).

3. **Per-sample vs batch-summed agree for every variant here** — 1.0/1.0 for the faithful pair, 0.45/0.38
   for the Moreau pair. So the Moreau bias is **systematic** (every sample biased), in contrast to
   diffmpc2's hard box where per-sample stayed ~1.0 but `cos_all` collapsed to ~0.1 because a few
   *outlier* samples dominated the sum. The two metrics together distinguish **systematic bias**
   (per-sample ≈ batch-sum, both low) from **outlier corruption** (per-sample high, batch-sum low).

4. **At the loosest tol (1e-1) everything is wrong** (all four ~0.1–0.26) — a loose solve corrupts every
   gradient regardless of formulation.

**Takeaway:** with the correct solver, the *faithful* gradient comes from the **hard box** or the **pure
log-barrier**; adding a **Moreau slack** (whether to turbompc or the barrier) trades a smooth solution for
a gradient biased ~0.45 from the hard-constrained one. The log-barrier (no slack) gives both a smooth
forward *and* a faithful gradient.
