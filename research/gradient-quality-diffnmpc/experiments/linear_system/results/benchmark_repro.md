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

## Open cross-check (not yet measured)

My **per-sample** panel (median 1.0 with low fliers) matches the appearance of the reference figure
`grad_box_accuracy_scp1_fdref.png`, whereas the benchmark's **code** computes the batch-summed cosine
(my panel 1, median ~0.1). Either the reference figure is a per-sample plot, or the benchmark's published
numbers differ from this reproduction. Since my reproduction uses the benchmark's *exact* solver, backends,
and problem setup, running `run_sweeps_gradient_accuracy.sh` itself should land on panel 1 (~0.1) — worth
confirming directly rather than inferring.
