# Closed-loop (50-step) gradient-accuracy: outliers from A vs B

**Run 2026-06-23.** Random linear MPC (nx=8, nu=4, horizon=20, umax=1.0), **1 SQP iter**, **50-step closed-loop rollout** (diffmpc-as-policy: solve MPC -> apply u0 -> step dynamics -> accumulate cost), regulate to origin. Differentiable parameter = cost weights (Q,R); per-sample (per initial state, n=24, seed 0). Each backward is compared to a **convergence-checked FD of its own rollout** (eps_seq 3e-5..1e-6); 'flagged' = FD never plateaus (a genuine discontinuity sample).

- **A** = TurboMPC analytic backward on the **hard box** (use_slack=False) — exactly the diffmpc `benchmark_turbompc_gradient_accuracy.py` setup (the `grad_box_accuracy` image).
- **B** = log-barrier smoothed backward (kappa=1e-6).

## Summary

| method | median cos | min cos | #cos<0.99 | #cos<0 | rel_l2 median | rel_l2 max | FD-flagged | **outliers** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (hard box) | 0.20363 | -0.466 | 10 | 5 | 1.76e+02 | 1.12e+03 | 9/24 | **19/24** |
| B (log-barrier) | 1.00000 | +1.00000 | 0 | 0 | 1.70e-06 | 3.54e-04 | 1/24 | **1/24** |

## Per-sample (sorted by A's cosine — A's worst first)

| sample | A cos | A rel_l2 | A flag | B cos | B rel_l2 | B flag | A outlier? |
|---:|---:|---:|:--:|---:|---:|:--:|:--:|
| 2 | -0.4662 | 1.76e+02 |  | +1.00000 | 2.24e-06 |  | **YES** |
| 13 | -0.3981 | 2.01e+02 | F | +1.00000 | 1.02e-06 |  | **YES** |
| 19 | -0.3200 | 2.74e+02 |  | +1.00000 | 2.05e-06 |  | **YES** |
| 15 | -0.2466 | 1.12e+03 |  | +1.00000 | 1.88e-06 |  | **YES** |
| 10 | -0.1478 | 6.37e+02 | F | +1.00000 | 1.39e-06 |  | **YES** |
| 16 | -0.0954 | 1.04e+03 | F | +1.00000 | 2.01e-06 |  | **YES** |
| 1 | -0.0656 | 1.02e+03 | F | +1.00000 | 2.01e-06 |  | **YES** |
| 14 | -0.0233 | 4.22e+02 |  | +1.00000 | 4.59e-06 |  | **YES** |
| 23 | -0.0194 | 2.76e+02 |  | +1.00000 | 1.23e-06 |  | **YES** |
| 17 | +0.1435 | 1.59e+02 |  | +1.00000 | 4.70e-06 | F | **YES** |
| 21 | +0.1486 | 3.22e+02 |  | +1.00000 | 2.69e-06 |  | **YES** |
| 20 | +0.1587 | 4.89e+02 | F | +1.00000 | 1.36e-06 |  | **YES** |
| 11 | +0.2036 | 2.97e+01 |  | +1.00000 | 6.47e-07 |  | **YES** |
| 22 | +0.2121 | 2.96e+02 |  | +1.00000 | 2.05e-06 |  | **YES** |
| 7 | +0.4329 | 6.83e+02 | F | +1.00000 | 8.49e-07 |  | **YES** |
| 8 | +0.5134 | 5.74e+02 | F | +1.00000 | 3.00e-06 |  | **YES** |
| 5 | +0.6014 | 6.56e+02 | F | +1.00000 | 3.54e-04 |  | **YES** |
| 6 | +0.6532 | 7.89e+02 |  | +1.00000 | 1.42e-06 |  | **YES** |
| 0 | +0.8303 | 3.15e+01 | F | +1.00000 | 1.70e-06 |  | **YES** |
| 3 | +1.0000 | 1.17e-03 |  | +1.00000 | 5.18e-07 |  | no |
| 4 | +1.0000 | 1.00e-03 |  | +1.00000 | 1.97e-06 |  | no |
| 18 | +1.0000 | 1.07e-03 |  | +1.00000 | 1.28e-07 |  | no |
| 12 | +1.0000 | 9.81e-04 |  | +1.00000 | 7.78e-07 |  | no |
| 9 | +1.0000 | 1.27e-03 |  | +1.00000 | 1.57e-06 |  | no |

## Finding

In closed loop, **A (the diffmpc hard-box backward) is an outlier on 19/24 samples** (median cos 0.20, 5 with *negative* cosine, rel_l2 up to 1e+03) — the 50-step rollout repeatedly crosses control active-set boundaries, where the hard active-set backward's gradient is wrong/explodes (strict-complementarity failure, compounded over the rollout). This reproduces the diffmpc `grad_box_accuracy` outliers. 
**B (log-barrier) has 1/24 outliers**: on *every* sample — including the exact ones where A fails — B matches its FD to cos=1.0 / rel_l2~1e-6. The smoothing makes the closed-loop policy gradient well-defined and FD-consistent, eliminating the hard-backward outliers (the H1.2 payoff, in closed loop).

Caveats (per CLAUDE.md): A and B solve slightly different problems (hard box vs kappa=1e-6 barrier); each AD is checked against *its own* rollout's convergence-checked FD (self-consistency). n=24, one random system, H=20 (reduced from 40 for compile tractability of the differentiable 50-step rollout).

## Robustness across systems (seeds 0,1,2 — 3 random linear systems)

| seed | n | A median cos | A #cos<0.99 | A #cos<0 | **A outliers** | B median cos | B #cos<0.99 | **B outliers** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 0.204 | 19 | 9 | **19** | 1.00000 | 0 | **1** |
| 1 | 16 | 0.171 | 12 | 6 | **12** | 1.00000 | 0 | **1** |
| 2 | 16 | 0.883 | 10 | 1 | **11** | 1.00000 | 0 | **0** |
| **all** | 56 | 0.473 | 41 | 16 | **42/56** | 1.00000 | 0 | **2/56** |

**Across 3 random systems / 56 samples: A is an outlier on 42/56 (75%), B on 2/56 (4%).** Every B outlier is an FD-plateau edge case with cos=1.0; B has zero cos<0.99 on all three systems. The hard-box backward's closed-loop outliers (and their elimination by the log-barrier smoothing) are robust to the random system, not a seed-0 artifact.
