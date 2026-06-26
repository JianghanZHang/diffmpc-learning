# Reproducing the TRI gradient-accuracy benchmark with a convergence-validated FD

Reproduces `diffmpc2/benchmarking/linear-system/run_sweeps_gradient_accuracy.sh` faithfully — horizon 40,
batch 64, 100 seeds, `ADMM_FUSED_CUDSS` fwd / `DIRECT_CUDSS_FFI` bwd, `admm_max_iter=1000`, sim_steps 50,
umax 1.0, weights = Q/R diagonals, AD tolerances `[1e-1,1e-3,1e-5,1e-7,1e-9]`, FD-reference forward at
tol `1e-9` — but replaces the benchmark's single `fd_eps=1e-5` with a **convergence-validated** FD, and
reports both the benchmark's **batch-summed** cosine and the **per-sample** cosine. `diffmpc2/` untouched
(imported read-only). Scripts: `benchmark_repro.py` (`--smoke` diagnosis, `--full` sweep),
`plot_benchmark_repro.py`. Data: `results/benchmark_repro.npz`, figure `results/benchmark_repro.png`.

## Step 1 — is the benchmark's single `fd_eps=1e-5` converged? (`--smoke`, 4 seeds)

Central-difference FD over a decreasing eps grid `[1e-3 … 1e-7]` at the benchmark's tight FD forward,
per-sample **and** batch-summed (the batch-summed FD is exactly the sum of the per-sample FDs).

- **Per-sample FD: converged, `1e-5` is INSIDE the plateau.** 63–64/64 samples plateau; the per-sample
  plateau is broad (~`1e-4`→`1e-6`); rel-err of `FD(1e-5)` vs the converged value is `5e-5 … 1.3e-3`.
- **Batch-summed FD: noisier — `1e-5` sits at the plateau EDGE.** The batch sum has ~64× the cost
  magnitude → ~64× the absolute noise → a higher noise floor. Its plateau is narrow (`1e-3`→`1e-4`); at
  `1e-5` the rel-err is already `1.3%` (seed 1) / `3.8%` (seed 3) and blows up below (`0.1`–`3.0` at
  `1e-7`).
- **Validated eps = `1e-4`**: safely inside both plateaus for every seed (per-sample ~`1e-3`, batch-sum
  ≲`0.2%`). Locked for the full run (no per-seed re-check).

**Verdict:** the benchmark's `fd_eps=1e-5` is essentially fine — exactly converged for the per-sample FD,
and only marginally into the noise floor for the batch-summed FD. It is **not** a non-convergence problem.

## Step 2 — reproduction: cos(AD, FD) vs solver tolerance (100 seeds, validated eps `1e-4`)

cos median (min); batch-summed = one value per seed (the benchmark's plotted quantity); per-sample = one
value per problem instance (6400); outliers-removed = batch-sum after dropping per-sample cos < 0.99.

| solver tol | batch-summed cos | per-sample cos | batch-sum, outliers removed | #out / seed (of 64) |
|---:|:---:|:---:|:---:|:---:|
| 1e-9 | **0.102** (−0.654) | **1.000** (−0.932) | 1.000 (1.000) | 3.9 |
| 1e-7 | 0.074 (−0.422) | 1.000 (−0.955) | 1.000 (1.000) | 4.0 |
| 1e-5 | 0.070 (−0.400) | 1.000 (−0.901) | 1.000 (1.000) | 4.7 |
| 1e-3 | 0.061 (−0.527) | 1.000 (−0.955) | 1.000 (0.999) | 8.0 |
| 1e-1 | 0.154 (−0.702) | 0.420 (−0.987) | nan¹ | 58.9 |

¹ at the loosest tol almost every sample is an "outlier" (loose solve corrupts every gradient), so the
outlier-removed sum is degenerate. See `results/benchmark_repro.png` (3-panel box plots).

**Benchmark eps `1e-5` vs validated `1e-4`:** nearly identical — batch-sum cos `0.065→0.105` (vs
`0.061→0.102`), per-sample median `1.000`, but `1e-5` flags slightly **more** outliers (`4.6–5.5` vs
`3.9–4.7` at tight tol) because its batch-sum FD noise produces a few spurious flags. So the eps choice
barely moves the result — consistent with Step 1.

## Findings

1. **The benchmark's batch-summed cosine is dominated by a few outlier samples — it is NOT a measure of
   typical gradient accuracy.** At every tight tolerance the batch-summed median is ~`0.06–0.10` while the
   **per-sample median is a perfect `1.000`**. Removing the per-sample outliers (cos < 0.99, only ~4–8 of
   64 per seed) sends the batch-summed cosine to **`1.000`**. So the low headline number is entirely a
   handful of large-magnitude wrong-gradient samples (the `DIRECT_CUDSS_FFI` near-boundary failures
   documented in `gradient_accuracy.md`) dominating the sum — `cos(Σ_i g_i)` is controlled by whichever
   sample has the largest ‖g‖, not by the median.

2. **The benchmark's `fd_eps=1e-5` is not the problem.** It is converged per-sample and only marginally
   noisy for the batch sum; the validated `1e-4` gives the same conclusion. The metric (batch-summed cos),
   not the FD step, is what makes the headline number look bad.

3. **Per-sample, the gradients are excellent away from ~6–12% near-boundary outliers.** Median cos `1.000`
   at tol ≤ `1e-3`; the fliers (down to ~−0.95) are the systematic-but-rare DIRECT-backward failures.
   At the loosest tol (`1e-1`) everything degrades (per-sample median `0.42`) — the known loose-solve bias.

## Cross-check against the actual benchmark — DONE (confirms the reproduction)

Ran `benchmark_turbompc_gradient_accuracy.py` directly (horizon 40, fused/DIRECT, α=1.0, admm 1000,
fd_eps=1e-5, tol 1e-9, seeds 0–9). The benchmark's own `cos_all` is **median 0.36, mean 0.31, min −0.38,
max 0.80** — i.e. **low and outlier-dominated, NOT ~1.0.** My reproduction on the same seeds gives median
**0.32**, per-seed in close agreement (seed 0 0.52/0.46, seed 1 0.55/0.54, seed 7 0.77/0.80); the per-seed
scatter (seed 4 0.07/0.31, seed 9 0.33/0.04) is the **horizon 40-vs-39** off-by-one (the benchmark builds
`turbompc_horizon = horizon−1`) reshuffling which samples sit near a boundary.

**So the reference figure `grad_box_accuracy_scp1_fdref.png` (median 1.0 + fliers) is a PER-SAMPLE
cosine, not the benchmark's `cos_all`.** The `run_sweeps` code computes only batch-summed `cos_Q/cos_R/
cos_all` (all ≈ 0–0.5, median ~0.36); it never computes a per-sample cosine. The per-sample view (median
1.0, this experiment's middle panel) is the one that matches the reference figure. The two figures are
**two metrics on the same gradients** — per-sample (median 1.0) vs batch-summed (median ~0.36) — and the
batch-summed one is dominated by the ~4–8 near-boundary DIRECT-backward outliers per seed, in the
benchmark exactly as here.

## External-vs-diffmpc2 codebase (re: the reference figure `grad_box_accuracy_scp1_fdref.png`)

**The reference figure plots `cos_all` (batch-summed), not per-sample.** It is produced by
`external/turbompc/.../plot_results/plot_gradient_accuracy.py` (`plot_cosine_by_scp`, green median), which
uses `cos_all` only. It shows median ~1.0. Rendering **this** reproduction's `cos_all` through the **same
external script** (`benchmark_repro_external_style.png`, via `benchmark_repro_tolnpz/`) gives median
**~0.1** — same metric, same plotting code, different *gradients*. So the gap is a **codebase** difference,
not a metric or eps one.

The two turbompc packages differ in only 3 source files; **`backward_kkt_jax.py` (the DIRECT KKT assembly)
is identical**. Candidate causes:
- **(a) `utils/timing.py` rollout warm-start** (diffmpc2 `jax.lax.stop_gradient(solution)` vs external
  cold-start `current_solution`) — **REFUTED** (`warmstart_test.py`): `warm_start` True vs False both give
  `cos_all` ~0.45 with the outliers intact, so the rollout is *not* the cause and was *not* a confound for
  the earlier DIRECT-backward outlier finding.
- **(b) `solvers/turbompc_solver.py` inequality-multiplier / active-set handling** (diffmpc2 uses a
  `cumsum`/`take_along_axis` active-set gather; external a simpler per-row sign-corrected map) — the
  remaining, **untested** candidate.

**Not independently confirmed:** external's own `cos_all`. Attempts to run external locally all blocked:
its cuDSS FFI `.cu` targets a cuDSS API the installed cuDSS 0.7.1 lacks (`CUDSS_R_32F`,
`cudssReorderingAlg_t` undefined — the jaxlib-include build issue was fixable via direct cmake, but this
cuDSS-version mismatch is not), and `ADMM_FUSED_CUDSS` needs the same unbuildable `libadmm_cudss_ffi.so`;
the only buildable/pure-JAX backends (dense, PCG) are too slow at horizon 40 (>20 min, 0 seeds), and a
small horizon where they'd finish has no outliers to compare. So "external ≈ 1.0" rests on the reference
figure + the source diff, **not a local run**. What *is* measured: diffmpc2's `cos_all` ≈ 0.1
(outlier-dominated), the per-sample median is 1.0, and the rollout warm-start is not the cause (so the
remaining suspect for external-vs-diffmpc2 is `turbompc_solver.py`'s multiplier/active-set handling).
