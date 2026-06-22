# Handoff — Cartpole Coupling / NLP-Tolerance Gradient-Quality Benchmark

**Date:** 2026-06-18 · **Branch:** `release-gradient-checkpoint` (worktree
`/home/jianghan/Workspace/diffmpc-learning/diffmpc2-gradckpt`) · **Package:** `turbompc`

This is part of the **"Gradient Quality of Differentiable NMPC"** project
(`/home/jianghan/Workspace/diffmpc-learning/research/gradient-quality-diffnmpc/`, see its
`NOTE.md`). This doc covers ONE experiment: the **cartpole closed-loop coupling × NLP-tolerance
gradient-accuracy benchmark**.

Main file: `research/gradient-quality-diffnmpc/experiments/cartpole/benchmark_cartpole_coupling.py`
Plot: `research/gradient-quality-diffnmpc/experiments/cartpole/plot_cartpole_coupling.py`
(both moved here from `diffmpc2-gradckpt/benchmarking/linear-system/`; they still resolve `turbompc`
from the gradckpt worktree at runtime — see the `_WORKTREE_ROOT` block at the top of the main file)

---

## 1. What this experiment asks

When a **diffmpc-as-policy** (NN/weights → MPC problem) is trained by BPTT over a **K-step
closed-loop episode**, the gradient threads through K chained MPC solves coupled by the
nonlinear closed-loop dynamics. Two axes:

- **Coupling length K** ∈ {1, 5, 10, 20} (closed-loop rollout length = `num_sim_steps`).
- **Set NLP-KKT tolerance** `tol_convergence` ∈ {1e-1, 1e-3, 1e-5, 1e-9} (how loosely the
  forward SQP is solved).

We differentiate w.r.t. the MPC cost weights **Q (4) + R (1)** (`WEIGHT_KEYS`), the
diffmpc-as-policy gradient. Metric: cosine / rel-ℓ₂ of the **AD (implicit/backward) gradient**
vs a **finite-difference ground truth**.

System: **cartpole swing-UP** — state `[x, ẋ, θ, θ̇]`, `θ=0` upright (unstable), `θ=π` hanging
down (stable). x0 is the **pole hanging DOWN** (`θ₀≈π`), MPC reference is upright → a maximal,
highly-nonlinear perturbation. **DECISION (user): keep the full pole-down swing-up.**

---

## 2. Environment gotchas (CRITICAL — read before running)

> **UPDATE 2026-06-22 — workspace consolidated; cuDSS rebuilt in `diffmpc2/`.** The
> `diffmpc2-gradckpt` worktree was removed; `turbompc` now resolves from the single **`diffmpc2/`**
> checkout (on `release-cleanup`, which carries the `cccba81` backward-Hessian fix). The benchmark's
> path shim points there (`_SOLVER_ROOT`, not the old `_WORKTREE_ROOT`). cuDSS works again, BUT:
> installed cuDSS is **0.7.1.6** while `release-cleanup` ships the cuDSS-**0.8** API (commit
> `e19a687`, broken on sm_120). The 3 `.cu` files were reverted to the 0.7.1 (13-arg) API and the FFI
> rebuilt — this revert is an **uncommitted working-tree change in `diffmpc2/`** (left uncommitted so
> the vendored repo's history keeps matching origin). To restore it after a hard reset of `diffmpc2/`:
> ```bash
> cd diffmpc2 && git checkout e19a687^ -- \
>   turbompc/solvers/admm/csrc/admm_cudss.cu \
>   turbompc/solvers/backward/csrc/cudss_sparse_kkt.cu \
>   turbompc/solvers/linear_systems_solvers/csrc/cudss_blktridi.cu
> export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
> cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release && cmake --build build/ffi -j
> ```
> The benchmark also now **auto-falls-back to pure-JAX backends** (`admm_jax_loop_pcg` /
> `direct_jax_dense`) with a printed warning if the cuDSS `.so` is absent, so it never hard-crashes.
> The numbered items below describe the original gradckpt setup and are retained for history.

The benchmark uses the **cuDSS FFI backends** (`fwd=admm_fused_cudss`, `bwd=direct_cudss_ffi`).
Getting them to load is fiddly:

1. **cuDSS 0.7.1, not 0.8.x.** The `.cu` files were reverted to the 13-arg
   `cudssMatrixCreateCsr` (cuDSS 0.7.1 API) and rebuilt. The working `.so` files are in **this
   worktree's** `build/ffi/` (built 2026-06-17, e.g. `libadmm_cudss_ffi.so`). The sibling
   `/home/jianghan/Workspace/diffmpc2/build/ffi/` has an **older (Jun-3) `.so` that silently
   fails** → `NOT_FOUND: No FFI handler registered for admm_cudss_cuda_f64`.

2. **`turbompc` MUST resolve to THIS worktree**, not the pip-editable
   `/home/jianghan/Workspace/diffmpc2`. The benchmark now does
   `sys.path.insert(0, _WORKTREE_ROOT)` (top of file) to force this. If you import `turbompc`
   elsewhere, set `PYTHONPATH=/home/jianghan/Workspace/diffmpc-learning/diffmpc2-gradckpt`.

3. **`LD_LIBRARY_PATH` must include the miniconda cuDSS/CUDA libs.** Saved in
   `/tmp/cudss071_ldpath.txt` (the miniconda `nvidia/*/lib` dirs incl. `libcudss.so.0`). Always:
   ```bash
   export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
   ```
   (If `/tmp/cudss071_ldpath.txt` is gone, it's the set of
   `/home/jianghan/miniconda3/lib/python3.13/site-packages/nvidia/*/lib` paths.)

4. The benchmark **eagerly imports** `turbompc.solvers.admm.admm_cudss_ffi_backend` at module
   top to register the FFI handler before any solve is lowered (defensive; the real fix was #1/#2).

5. **x64 is required** (`jax_enable_x64`, set at top) for trustworthy finite differences.
   Run on **GPU** (project standing preference: GPU + slack; here slack isn't used, but GPU is).

6. **jax/jaxlib 0.8.2** (miniconda). Filter noisy `hwloc`/openmpi warnings from output.

### Run commands
```bash
cd /home/jianghan/Workspace/diffmpc-learning/research/gradient-quality-diffnmpc/experiments/cartpole
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
# calibrate (single K, 2 seeds, verbose); --umax 1e7 => NO bound (smooth NLP)
python benchmark_cartpole_coupling.py --system cartpole --umax 1e7 --calibrate --calibrate_K 1 --nlp_tol 1e-9
# full grid (background, unbuffered) -> writes to ../cartpole/results/
nohup python -u benchmark_cartpole_coupling.py --system cartpole --umax 1e7 --coupling 1 5 20 --n_seeds 2 \
  --nlp_tol 1e1 1e0 1e-1 1e-3 1e-5 1e-9 --gt_tol 1e-9 --save_results > /tmp/cartpole_grid.log 2>&1 &
# (--system quadrotor runs the 13-state quadrotor analog; writes to ../quadrotor/results/)
```
NOTE: `grep | tee` **block-buffers** — use `python -u ... > log 2>&1` and `tail -f` the log, or
the Monitor tool, to see per-row output live.

---

## 3. Design decisions already settled (do NOT relitigate)

- **Poor initialization every MPC cycle.** `warm_start=False` in `build_rollout_fn`
  (timing.py:104): every rollout step cold-starts the SQP from the FIXED `init_solution` (the
  OCP solved at the upright origin ≈ all-zeros trajectory), never the previous step. For
  pole-down this is a deliberately bad guess. The straight-line `initial_guess`
  (optimal_control_problem.py:819) is only used to seed `init_solution`. **Keep this.**

- **Do NOT pin SQP iterations.** `--sqp_iter` is a **generous non-binding CEILING** (default
  150). SQP's `cond_fun` is `it < max_iter AND conv > tol_convergence`
  (turbompc_solver.py:1678), so SQP terminates **at the set `tol_convergence`**, which is the
  swept operating point. Verified (tol-bind diagnostic): median achieved KKT tracks the set tol
  (1e-1→5e-2, 1e-3→7e-4, 1e-5→6e-6, 1e-9→9e-10) with median SQP iters 25→29→40→72.

- **Ground truth is ALWAYS at `gt_tol=1e-9`** (converged), never the swept tol. Compare the
  swept-tolerance **AD** gradient against this fixed converged **FD** reference. (Earlier bug:
  GT was FD at the *same swept tol* → AD and FD drifted together, cos stayed ~1, rel-ℓ₂ was pure
  noise. Fixed: loop is now **K-outer**, GT computed once per K and cached; see run().)

- **Differentiate Q + R cost weights** (`WEIGHT_KEYS`). `umax=20` = "loose" control bounds
  (rarely active); a `umax=2` "tight" pass is planned but not yet run.

- **ADMM inner QP held tight** (`--admm_eps 1e-9`, `--admm_max_iter 1000`,
  `check_termination_every=25`) to isolate the NLP-KKT effect. ⚠️ At K≥5, `max_admm` hit **1000**
  (the ADMM cap) — the inner QP may not be fully converging for the longer rollouts; investigate.

---

## 4. Key findings so far

1. **cuDSS works** on RTX 5090 (sm_120) from this worktree's Jun-17 0.7.1 `.so`. The 0.8
   migration (commit e19a687) is what broke it.

2. **Under-converged forward kills the gradient — but only when stopped at an iteration CAP.**
   Earlier "cos→0" episodes came from cutting SQP off at a fixed iteration count (a
   non-stationary, weight-sensitive stopping point → non-smooth forward map → FD sees noise).
   Stopping at a convergence TOLERANCE is a consistent criterion → forward map stays smooth →
   AD and FD agree in direction even when loosely converged.

3. **A removed measurement bug:** the old "achieved_KKT" probe re-solved each state from a fresh
   straight-line `initial_guess` (a GOOD warm-start that converges fast), NOT the rollout's fixed
   `init_solution` (the bad warm-start the gradient actually differentiates). So it reported a
   converged KKT while the gradient's solves were at ~1e-3. **The probe was deleted.** The honest
   per-state convergence comes from solving with `init_solution` as the warm-start.

4. **The batch-sum cos metric is OUTLIER-HIJACKED (this is why we're refactoring).** The
   benchmark currently differentiates the **batch-summed** cost → one gradient vector per seed.
   A few non-converging swing-up states have blown-up gradients that dominate the sum. At K=1:

   | set tol | cos mean | cos median | rel-ℓ₂ |
   |---|---|---|---|
   | 1e-1 | 0.9983 | 0.9988 | 0.47 |
   | 1e-3 | 0.9488 | 0.9988 | 0.52 |
   | 1e-5 | 0.9360 | 0.9988 | 0.52 |
   | **1e-9 (floor)** | **0.9381** | **0.9988** | **0.52** |

   **Median is dead flat at 0.9988 across all tolerances; the mean is noise.**

5. **Low per-state cos is driven by GRADIENT MAGNITUDE, not convergence** (verified with a
   per-state probe at K=1, tol=1e-9, `/tmp/diag_perstate.py` — recreate it; see §5). Evidence:
   - **corr(cos, log‖g_GT‖) = +0.995** — cos tracks gradient magnitude almost perfectly.
   - corr(cos, log achieved-KKT) = +0.595 (weaker, confounded).
   - The 8 **lowest-cos** states are **well-converged** (KKT 1e-9–1e-10) but have **‖g‖~1e-6**
     (≈ zero); the 8 **highest-cos** (cos=1.0) have large ‖g‖ (12–336). The single **unconverged**
     state (KKT=6.9e-2) has **cos=1.000**.
   - **~72% of the 64 states have a near-zero gradient** w.r.t. Q/R (the closed-loop cost is
     insensitive to the weights for that x0); cosine of two ~noise vectors is meaningless there.
   ⚠️ **Therefore an UNWEIGHTED per-state cos is misleading** (its median ≈ 0 just measures the
   zero-gradient noise floor). The earlier "swing-up tail / AD≠FD at convergence" hypothesis was
   WRONG. The batch-sum cos was actually reasonable *because* it is implicitly magnitude-weighted.

---

## 5. ⏳ IMMEDIATE NEXT TASK — refactor to PER-STATE gradient quality

**DECISION (user): switch from batch-sum to per-state measurement.** Treat each of the 64 batch
initial states as its own sample — BUT the metric must be **magnitude-aware** (per finding §4.5:
cos is meaningless for the ~72% of states with near-zero gradient). Goal RQ1 curve: **per-state
gradient cos vs that state's achieved KKT, for the states whose gradient is non-negligible** (and
a magnitude-weighted aggregate). Do NOT report an unweighted per-state cos median — it just
measures the zero-gradient noise floor (~0).

### What to change in `run()` (currently batch-sum at ~lines 181–220)

1. **AD per-state gradient** — replace `jax.grad(jnp.sum(rollout_batch(x,w)[0]))` with a
   per-state vmap. Build the **single-state** rollout (`build_rollout_fn` returns the per-env
   `rollout` before vmap — `make_rollout_batch` currently wraps it in `jit(vmap(...))`). Then:
   ```python
   # rollout_single(x, w) -> scalar episode cost
   per_state_grad = jit(vmap(jax.grad(checkpoint(lambda w, x: rollout_single(x, w)),
                                      argnums=0), in_axes=(None, 0)))
   g_ad = per_state_grad(weights, x0)   # pytree leaves shape (batch, wdim)
   ```
   (grad of the batch SUM = sum of per-state grads, so this is the consistent decomposition.)

2. **FD ground truth per-state** — `gradient_finite_diff` (utils/gradient_finitediff.py) does
   `.sum()` over the batch → batch-summed. Write a per-state version that **keeps the batch
   dim**: for each weight scalar component j, central-difference `rollout_batch(x0, w±eps·e_j)[0]`
   (a length-`batch` cost vector) → `(cost_plus - cost_minus)/(2eps)` gives `∂cost_i/∂w_j` per
   state i. Assemble `{key: (batch, wdim)}`. Cost = 2·(nQ+nR) = 10 forward batch evals at
   `gt_tol=1e-9` (same as before, just un-summed). Cache per K.

3. **Per-state achieved KKT** — `vmap(lambda x:
   solver.solve(init_solution, {**pp,"initial_state":x}, weights).convergence_error)(x0)` →
   `(batch,)`. This is the honest per-state convergence (rollout's fixed `init_solution`
   warm-start). For K>1 the rollout has K solves/state; using the **first (initial-state) solve's
   KKT** as the per-state convergence proxy is reasonable (the pole-down initial state is the
   hardest). Document the choice. (Optionally extend `build_rollout_fn` to return per-step
   `convergence_error` — it already has `solution.convergence_error` in scope at timing.py:90,
   currently only returns `solution.admm_iters`.)

4. **Per-state metrics & storage (MAGNITUDE-AWARE)** — per state i store: `cos_i`, `rel_l2_i`,
   `kkt_i`, AND **`gnorm_i = ‖g_gt_i‖`** (essential — cos is only meaningful where gnorm is not
   ~0). Store full per-state arrays (batch × seeds) per (K, tol). Aggregate metrics that matter:
   - **magnitude-weighted cos** `Σ_i gnorm_i·cos_i / Σ_i gnorm_i` (the headline scalar), and/or
   - per-state cos **filtered to gnorm_i > τ** (e.g. τ = 1e-2·max gnorm, or top-quartile), then
     median/quantiles of cos over that subset.
   - the batch-sum cos (implicitly magnitude-weighted) as a cross-check.
   The headline RQ1 plot: **per-state cos vs achieved-KKT scatter, with points sized/colored by
   gnorm** — so the eye discounts the zero-gradient noise. Do NOT report an unweighted cos median.

5. **Plot** — extend `plot_cartpole_coupling.py` (or add a script) for: (a) per-state cos
   distribution (box/violin) vs K, per tol; (b) the headline **scatter: per-state cos vs
   per-state achieved KKT** (colored by set tol) — the RQ1 "gradient quality vs convergence"
   curve.

### Sanity checks after refactor
- K=1, tol=1e-9: per-state cos for *well-converged* states should be ~1.0; the hard tail
  (kkt ≳ 1e-2) should be the low-cos outliers → cos-vs-KKT scatter should show the cliff.
- Confirm per-state AD summed over the batch ≈ the old batch-sum AD (consistency).
- Then run the full K × tol grid with `--save_results`.

---

## 6. State of the benchmark file (edits already made this session)

- `sys.path` forces this worktree's `turbompc`; eager cuDSS FFI import (top of file).
- `generate_cartpole_x0`: pole-DOWN base `[0,0,π,0]` + `scale·std`, `--x0_scale` knob.
- `solver_params`: `tol_convergence=nlp_tol`, `check_termination_every=25`, linesearch on.
- `run()`: **K-outer**, GT = FD at `--gt_tol` (default 1e-9) cached per K, swept-tol inner,
  **currently still batch-sum cos vs GT** ← this is what §5 replaces.
- Args: `--gt_tol 1e-9`, `--sqp_iter 150` (ceiling), `--nlp_tol` (sweep), `--coupling`,
  `--x0_scale`, `--admm_eps 1e-9`, `--admm_max_iter 1000`, `--fd_eps 1e-5`, `--save_results`,
  `--calibrate`.
- The probe block was **removed**; CSV no longer has `achieved_kkt`.

## 7. Open items / future
- Run the **tight** control-bounds pass (`--umax 2`, active box constraints) after loose.
- Investigate **ADMM hitting max_iter=1000 at K≥5** (inner QP convergence for long rollouts).
- The `max KKT ~7e-2` hard-tail states (swing-up perturbations the upright warm-start can't
  recover even at 150 SQP iters) — quantify the tail fraction vs `x0_scale`.
- Feed results into `research/gradient-quality-diffnmpc/NOTE.md` (RQ1/RQ2).

## 8. UPDATE (2026-06-18) — per-state refactor done, no-bound grid, the lone K=20 outlier

**Status:** the §5 per-state refactor is DONE. The benchmark now computes **per-sample** gradients
(`vmap(grad(single))`, plus a per-state FD that keeps the batch dim; GT = FD at `--gt_tol`).
Methodology (user): **each initial state = one sample** (roll out K on it); report **median +
outliers over per-sample cos**, never a batch-summed gradient. The control bound was removed
(`--umax 1e7`); the earlier `umax=20` "loose" setting actually **saturated u₀ for 72% of swing-up
states** (active bound → du₀/dθ=0 → zero gradient → cos meaningless — a separate red herring,
resolved).

**No-bound K-sweep** (`--umax 1e7 --coupling 1 5 10 20 --n_seeds 2 --nlp_tol 1e-1 1e-3 1e-5 1e-9
--gt_tol 1e-9`): **per-sample cos = 1.0000 at every (K, tol)** — the gradient is accurate,
**tolerance-insensitive** (cos=1.0 even at tol=1e-1, forward at KKT~5e-2), and **coupling-robust**
through K=20 — **except exactly one sample** (idx 114, K=20, all tols): cos(AD, FD@1e-5) = −0.56.
(Removing the bound also fixed ADMM-hits-1000-at-K≥5 — no active set for the inner QP.)

### The lone K=20 outlier — fully diagnosed (ALL MEASURED; scripts in `/tmp/diag_*`, `/tmp/test*`, `/tmp/repro_jump12.py`)

It is an **FD-ground-truth artifact at a nonconvex local-minimum basin boundary** — NOT an AD
error, NOT under-convergence, NOT a discontinuity in the problem. Measured chain:

1. **Not under-convergence:** SQP ceiling 150→600 converges all 20 rollout solves to KKT≈1e−9, yet
   the cost jump and cos=−0.48 **persist unchanged** → property of the fully-converged forward map.
2. **AD is correct:** ‖g_AD‖≈4.05e4, stable across SQP ceilings, and matches FD when the FD step
   stays on one branch (eps=1e−7 → cos 0.85). ‖g_FD‖≈2.15e8 (~5000× larger) → **FD is the artifact**.
3. **Localized via per-step decomposition** (`test3_perstep.py`): `cos(ΣAD,ΣFD)=−0.48` reproduces
   the aggregate; steps 0–11 clean (FD_k≈AD_k), steps **12–19 all blow up** (|FD_k|=1e7–1.5e8 vs
   |AD_k|=10–1e4). So it's the whole tail, originating at step 12 — not one step in isolation.
4. **Mechanism — a basin straddle driven by the propagated STATE, not the weight:**
   - FD perturbs the **weights** over the full 20-step rollout (only input is x₀) → the **entering
     state₁₂ differs by ~2e−4** between the w±eps rollouts (`test_split.py`).
   - Step-12's MPC has a **local-min basin boundary**; the ~2e−4 *state* difference straddles it →
     the two solves land in **two distinct fully-converged minima** (u₀ ≈ −1.2 vs −33, both KKT~1e−9).
   - The **weight step alone (±1e−5) does NOT** straddle it: at fixed state₁₂ the weight-watershed
     is at ≈R₀−1.43e−5, **outside** the ±1e−5 FD window (`test_isolate12.py`). Reproduced + pinned
     to a 5e−8-wide interval, deterministic across repeats, both sides converged (`repro_jump12.py`).
   - The control flip **cascades** (step 13's entering state already ~2 apart) → every downstream
     stage cost differs → steps 12–19 FD_k blow up. AD follows one branch → smooth.
5. **Net:** FD differences **two different minima** → `‖cost_B−cost_A‖/(2eps)`≈1e8, meaningless;
   AD takes the infinitesimal limit, never jumps basins, returns the true within-branch gradient.
   Appears only at K=20 because a long enough chain is needed to propagate a state difference into a
   mid-rollout solve that happens to sit on a basin boundary. **1/128 samples.**

## 9. Prevention — it's an FD *step-size* problem near a real discontinuity, not an AD error

**Framing (corrected):** FD (in the limit eps→0) IS the ground truth. The implicit **AD gradient
equals it** — AD = dF/dw = FD(eps→0) — verified here: FD at eps=1e−7 → 4.6e4 ≈ ‖g_AD‖=4.05e4 (the
residual is the noise floor). So the −0.56 is **not** "AD vs FD"; it's the **fixed eps=1e−5 being
too large for this sample**, stepping across a discontinuity 8e−7 away. (Earlier "treat AD as
primary / FD fallible" was wrong — scratch it.)

The real object is that, for a **nonconvex** MPC, the closed-loop cost F(w) is **piecewise-smooth
with jumps**: the solver's *selected* local minimum is discontinuous in w (the argmin jumps between
minima). Away from a jump dF/dw exists and AD = FD(eps→0); a jump is measure-zero and a finite-eps
FD whose interval contains it reports jump/2eps. So the gradient is fine almost everywhere; the
failure is purely FD overshooting a nearby jump.

What actually prevents misreading/mishandling it:

1. **Adaptive, per-sample FD eps (the real fix).** Shrink eps until FD stabilizes (Richardson): the
   right eps is small enough its interval misses the nearest jump, large enough to beat the cost-noise
   floor (~0.01 GPU non-determinism → FD floor ≈ 0.01/2eps). A *fixed* eps=1e−5 fails wherever a jump
   is closer than that (here 8e−7).
2. **Lower the noise floor** (deterministic reductions / tighter solve) → smaller eps usable → wider
   valid-eps window → fewer samples with no good eps.
3. **Detect & label, don't average.** If FD won't stabilize before the noise floor (or ‖g_FD‖≫‖g_AD‖,
   here ~5000×), the point is effectively **at a policy discontinuity** — report it as
   "gradient ill-defined here," not as a gradient/AD error. Per-sample + explicit outlier inspection
   (already adopted) is what surfaces these; never a batch-sum or bare mean.

NOT a fix: **`warm_start=True`** — both branches already satisfy the FONC, so the issue is the
*multiplicity* of minima and the discontinuous *selection*, not convergence. Tracking warm-start
relocates/thins boundaries but the nonconvex problem still has multiple FONC points and the
selection still jumps at folds. (Use it for deployment realism, not as a cure for this.)

Research takeaway (RQ1): a nonconvex diffmpc-as-policy has **genuine cost discontinuities in
parameter space** (argmin jumps between local minima). That's a real property — FD validation needs
adaptive eps, training can occasionally hit an undefined gradient at a boundary — but **AD is
reliable**: away from the measure-zero jumps, AD = FD(eps→0).

Meta-lesson (why CLAUDE.md's "report only what you measured" matters operationally): the correct
diagnosis only emerged after overturning several *unmeasured* inferences — it was **step 12, not 13**;
**state-mediated, not weight-direct**; "discontinuity in the problem" was wrong (the cost is smooth
on each branch; the *solver's local-min selection* is what's discontinuous); and "AD primary, FD
fallible" was wrong (AD = FD in the limit). Each was corrected only by measuring.
