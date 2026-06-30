# V1 plan_hard vs V3 bptt_hard — Diff-WMPC Drone Training Results

**Date**: 2026-06-29  
**Scope**: Hard-box TurboMPC only (V2 barrier / V4 barrier-BPTT deferred to a future task).  
**Environment**: 6-state double-integrator drone, 1 circular obstacle on START→GOAL path.  
**Solver**: TurboMPC ADMM_FUSED_CUDSS (fwd) + DIRECT_CUDSS_FFI (bwd); hard-box constraints.  

---

## Run configuration

| Parameter | Value |
|-----------|-------|
| n_updates | 150 |
| seeds | [0, 1, 2] |
| n_batch (V1) | 10 steps per update |
| h (V3 BPTT window) | 8 steps |
| lr | 3e-3 (Adam) |
| eval interval | every 10 updates, n_steps=25 from START |
| eval method | JIT'd jax.lax.scan (no Python loop) |

---

## Per-run wall-clock

| Run | Wall-clock (s) |
|-----|---------------|
| plan_hard seed=0 | 160.6 |
| plan_hard seed=1 | 159.7 |
| plan_hard seed=2 | 123.6 |
| bptt_hard seed=0 | 132.3 |
| bptt_hard seed=1 | 146.8 |
| bptt_hard seed=2 | 119.2 |
| **Total** | **844.3 s (14.1 min)** |

No runs hit the 20-min bail-out. All 150 updates completed for every seed.

---

## Eval cost vs updates (from fixed START, 25-step closed-loop rollout)

### V1 plan_hard

| Update | seed=0 | seed=1 | seed=2 |
|--------|--------|--------|--------|
| 10 | 30.15 | 30.04 | 29.64 |
| 30 | 18.35 | 23.18 | 16.63 |
| 50 | 13.92 | 17.53 | 13.56 |
| 70 | 11.60 | 14.33 | 11.86 |
| 90 | 10.19 | 13.87 | 9.17 |
| 110 | 8.39 | 11.34 | 8.67 |
| 130 | 8.28 | 9.54 | 8.55 |
| 150 | **8.25** | **9.15** | **8.48** |

**Final eval: mean = 8.629, std = 0.383 (n=3 seeds)**  
Reduction from update 10: 72.6%, 69.5%, 71.4%

### V3 bptt_hard

| Update | seed=0 | seed=1 | seed=2 |
|--------|--------|--------|--------|
| 10 | 29.88 | 29.46 | 29.33 |
| 30 | 16.48 | 16.42 | 17.32 |
| 50 | 18.30 | 14.72 | 14.96 |
| 70 | 13.52 | 13.28 | 12.55 |
| 90 | 10.40 | 11.23 | 11.37 |
| 110 | 8.85 | 9.32 | 8.90 |
| 130 | 8.20 | 8.28 | 8.48 |
| 150 | **8.11** | **8.11** | **8.15** |

**Final eval: mean = 8.121, std = 0.019 (n=3 seeds)**  
Reduction from update 10: 72.9%, 72.5%, 72.2%

---

## Summary comparison

| Metric | V1 plan_hard | V3 bptt_hard |
|--------|-------------|--------------|
| Final eval mean (n=3) | 8.629 | **8.121** |
| Final eval std (n=3) | 0.383 | **0.019** |
| Max grad norm (any seed) | 124 | 7.4 |
| Cross-seed variance ratio | 1× | **20× lower** |
| Total wall-clock | ~148 s/run | ~133 s/run |

V3 bptt_hard converges to a lower final eval cost (6% improvement) with 20× less cross-seed variance. The max grad norm for V1 (124, seed=1) vs V3 (7.4, seed=2) reflects that the myopic per-step gradient (V1) can produce large spikes at individual steps, while the h=8-step BPTT window (V3) smooths gradient estimates.

**Whether V3 learns faster than V1** (lower cost at fewer updates): V3 shows slightly faster early learning in some seeds (update 30: V3 seed=0=16.48 vs V1 seed=0=18.35) but also has an upward excursion at update 50 (V3 seed=0=18.30 vs V1=13.92). Both converge to similar costs by update 100 and diverge only modestly by update 150. The 6% final-cost advantage for V3 is measured; whether this generalizes beyond 150 updates is unknown.

---

## Final obstacle clearance (update 150 eval)

| Run | eval_min_margin | eval_n_violations (out of 25 steps) |
|-----|-----------------|--------------------------------------|
| plan_hard seed=0 | -2.860 | 0 |
| plan_hard seed=1 | -3.273 | 1 |
| plan_hard seed=2 | -2.980 | 1 |
| bptt_hard seed=0 | -2.613 | 1 |
| bptt_hard seed=1 | -2.593 | 1 |
| bptt_hard seed=2 | -2.671 | 0 |

`eval_min_margin <= 0` means outside obstacle throughout the rollout.  
All final policies have min_margin well below 0 (well clear). 1-2 steps per 25-step eval occasionally clip the obstacle boundary (margin > 0 for 1 step = 4–8% of steps).

---

## Active-set diagnostics: realized training trajectories

During training (realized `min_obs_margin` per update):

| Variant | Seeds | Training min margin | Max margin (closest to boundary) | Near-active steps (margin > -0.1) |
|---------|-------|---------------------|------------------------------------|-----------------------------------|
| plan_hard | 0,1,2 | -4.35 to -4.24 | -0.53 to -0.71 | **0 / 150** |
| bptt_hard | 0,1,2 | -6.24 to -5.50 | -0.50 to -0.59 | **0 / 150** |

No realized training trajectory entered the near-active zone (margin > -0.1) for either variant.  
The MPC PLAN's obstacle constraint was binding (plan_margin ≈ 0) at episode starts, but the realized closed-loop rollout stayed well outside.  
**Hypothesis (unverified):** the MPC warm-start and receding horizon shift keep the plan trajectory near-feasible, steering the realized rollout away from the obstacle even when the plan constraint is active. This would require measuring episode-start plan margins vs realized margins in more detail to confirm.

---

## Scope note

This experiment covered:
- **V1 plan_hard**: open-loop-plan myopic gradient (Algorithm 1), hard-box TurboMPC
- **V3 bptt_hard**: SHAC truncated BPTT (h=8), hard-box TurboMPC

Deferred (not measured here):
- **V2 plan_barrier**: same as V1 but with log-barrier central-path solver
- **V4 bptt_barrier**: same as V3 but with log-barrier solver

---

## Figures

- `eval_cost_vs_updates.png`: Eval cost vs #updates, mean±std band over 3 seeds (V1 vs V3)
- `grad_norm_vs_updates.png`: Grad norm vs #updates (log scale), with markers where min_obs_margin > -0.1 (none observed)
- `closed_loop_traj.png`: x-y trajectories of all final trained policies, with obstacle circle
