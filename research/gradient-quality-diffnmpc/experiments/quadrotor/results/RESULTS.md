# Experiment E1.2 + E2.1 — Gradient quality vs. constraint activity and NLP tolerance

**Date:** 2026-06-16 · **Testbed:** drone obstacle avoidance (`diffmpc2/examples/drone_obstacles/`)
**Driver:** `gradient_quality_sweep.py` (this dir) ·
**Plots:** `plot_gradient_quality.py` · **Data:** `outputs/grad_quality_20260616_164624.csv`
(+ `.png`).

## Setup

- 2×2 constraint grid, **slack always on**, on GPU (RTX 5090), x64, pure-JAX backends
  (`ADMM_JAX_LOOP_PCG` forward, `DIRECT_JAX_DENSE` backward — same backend for reference and
  swept runs so no FFI/JAX numerical confound).
- Horizon 50, `n_seeds=4` (initial states = `DRONE_X0_BASE + N(0,0.05)`), batched with `vmap`.
- **Differentiated target** (the diffmpc-as-policy gradient): cost weights
  `weights_penalization_reference_state_trajectory` (6) + `slack_penalization_weight` (1).
  Objective `J = Σ‖x‖² + Σ‖u‖²` for one open-loop OCP solve.
- **Reference** ("ground truth"): tightest affordable solve, `tol 1e-9, ADMM 1500, SQP 30`
  (no line search — line search empirically *slowed* obstacle convergence here). Compared two
  ways: autodiff at the reference (`g_AD*`) and central finite differences (`g_FD*`).
- **Sweeps:** (A) NLP tolerance `{1e-1…1e-6}` at ADMM 800 / SQP 15; (B) ADMM `max_iter`
  `{5…800}` at tol 1e-10 / SQP 15. Metrics vs `g_AD*`: cosine, rel-ℓ₂, sign agreement,
  descent test; plus achieved `convergence_error` and #active constraints.

`umax` calibration: unconstrained-optimal controls have max|u|≈0.82, p90≈0.12, so the "tight"
cells use `umax=0.1` (saturates the upper ~10–15%) and "loose" uses `umax=1e4` (never binds).
The first run used `umax=2.0`, which never bound — a degenerate axis caught by the
activity print; corrected here.

## The activity ladder

| cell | control | obstacles | #active (box / obs) | reference converges to |
|------|---------|-----------|----------------------|------------------------|
| C1 loose/off | `1e4` | off | 0.0 (0 / 0) | conv ≈ 5e-7 |
| C2 tight/off | `0.1` | off | 26.0 (26 / 0) | conv ≈ 5e-7 |
| C3 loose/on | `1e4` | on | 5.8 (0 / 5.8) | conv ≈ 2.4e-3 (plateau) |
| C4 tight/on | `0.1` | on | 43.8 (37 / 6.8) | conv ≈ 7.9e-3 (plateau) |

## Findings

**F1 — Linear (box) active constraints keep the implicit gradient exact; nonlinear (obstacle)
active constraints degrade it.** `cos(g_AD*, g_FD*)` at the reference:

| C1 (0 active) | C2 (26 box) | C3 (5.8 obs) | C4 (37 box + 6.8 obs) |
|---|---|---|---|
| **1.0000** | **1.0000** | **0.9977** | **0.9956** |

26 active *box* constraints leave the converged implicit gradient in perfect agreement with
finite differences (C2). Once *obstacle* (nonlinear, state-dependent) constraints are active,
the converged gradient no longer matches FD (C3, C4). This is the active-set / general-inequality
gradient pathology the literature predicts (`[AmosKolter2017][Magoon2024][Frey2025]`), here
isolated as a **box-vs-nonlinear** distinction.

**F2 — The obstacle NLP does not converge tightly, which compounds F1.** Box-only cells solve to
conv ≈ 5e-7; obstacle cells plateau at conv ≈ 2.4e-3 (C3) / 7.9e-3 (C4) even at SQP=30. Part of
the F1 gap is this non-convergence (differentiating a non-fixed-point, `[Amos2018]`): tightening
the reference from SQP=18→30 moved `cos(AD,FD)` for C3 from 0.9912→0.9977. **A residual gap
persists**, so it is not *only* non-convergence — but the two effects are entangled for general
inequalities, itself an RQ2 finding (general inequalities make the NLP hard to converge).

**F3 — Tolerance: direction is robust, magnitude is not; sensitivity grows with activity.** At
the loosest tolerance (1e-1), rel-ℓ₂ error ≈ 0.8 in *all* cells (magnitude badly off), but the
**cosine** stays ≈ 1.0 for C1/C2/C3 and only drops to **0.899 for C4**. Tightening recovers
rel-ℓ₂ to ~1e-5 for box cells but it **plateaus** for obstacle cells (≈3e-3 C3, ≈0.13 C4) because
the reference itself is imperfect there.

**F4 — ADMM-iteration dependence is monotone for box, non-monotone for obstacles.** Box cells:
rel-ℓ₂ decreases monotonically with ADMM `max_iter` (C2: 25→0.05, 50→5e-7). Obstacle cells are
**non-monotone** (C3: ADMM 50→4e-4 but 100→0.14; C4 similar) — the active set identified flips
between ADMM budgets, destabilizing the gradient. Direct evidence of active-set-change instability
for general inequalities.

**F5 — The descent direction survives even very loose solves.** `⟨g, g_AD*⟩ > 0` in **every**
cell and **every** swept config (descent = 1.0 throughout), including tol=0.1 and ADMM=5. So even
a badly magnitude-biased, direction-degraded gradient still points downhill — consistent with the
"loose solve still trains" line (`[Fung2022jfb][Geng2021phantom]`) and supportive of RQ2's premise
that lower tolerance is viable, with the caveat that obstacle cells need enough ADMM iters to avoid
the non-monotone regime.

## Caveats / limits of this run
- Obstacle reference is not a true fixed point (conv ~1e-3); obstacle rel-ℓ₂/cos numbers are
  entangled with non-convergence (F2). A genuinely tight obstacle reference would need a
  better-converging SQP (more iters, trust region, or a different globalization) — follow-up.
- `n_seeds=4`; single objective (`Σx²+Σu²`); single backward backend (`DIRECT_JAX_DENSE`).
  ADMM-vs-IPM backward comparison and downstream-training impact are future work (NOTE.md E2.2/E2.4).
- "Descent preserved" is measured against `g_AD*`, not the true gradient; for obstacle cells the
  reference is imperfect, so F5 is suggestive, not conclusive, there.

## Reproduce
From the workspace root (`diffmpc-learning/`); needs the `diffmpc2` deps + a GPU:
```bash
python research/gradient-quality-diffnmpc/experiments/quadrotor/gradient_quality_sweep.py --seeds 4  # ~25 min, GPU
python research/gradient-quality-diffnmpc/experiments/quadrotor/plot_gradient_quality.py    # newest CSV -> PNG
# quick check:
python research/gradient-quality-diffnmpc/experiments/quadrotor/gradient_quality_sweep.py --smoke    # C1 only, ~90s
```
