# CLAUDE.md

Guidance for Claude Code working in this **project workspace**
(`/home/jianghan/Workspace/diffmpc-learning/`).

## What this workspace is

This is the **"Gradient Quality of Differentiable NMPC"** research project. It *uses* the
TurboMPC solver but is kept separate from it.

> **⚠️ 2026-06-26 — Use `external/turbompc` (GitHub `main`) as the canonical solver, NOT `diffmpc2/`
> (`release-cleanup`).** diffmpc2 produces **wrong hard-box backward gradients**: its
> `turbompc/solvers/turbompc_solver.py` is missing the inequality-multiplier **sign-correction**
> (`y_ineq = −sign·y_g`, mapping ADMM bound duals → active-constraint multipliers; `sign =
> lower_active − upper_active`) and the **lower/upper dual-sign disambiguation** that external has.
> `backward_kkt_jax.py` (the DIRECT KKT assembly) is **identical** between them, so it is fed
> wrong-signed / mis-classified active multipliers at near-active constraints → outlier gradients
> (benchmark `cos_all` median **~0.36**, negative cosines). External, same config/seeds/cuDSS 0.7.1 →
> `cos_all` median **1.0**. **Re-validate every gradient-quality result that used `diffmpc2/` against
> `external/turbompc`** (the earlier "DIRECT-backward outlier / weakly-active" findings in
> `gradient_accuracy.md` are this diffmpc2 bug, not fundamental — the rollout warm-start was *refuted*).
> Measured: `experiments/gradients/linear_system/results/benchmark_repro.md`; memory
> `diffmpc2-hardbox-outliers-are-release-cleanup-specific` (incl. the cuDSS-0.7.1 build recipe — external's
> `.cu` ship the cuDSS-0.8 API and need the same compat-shim/backport diffmpc2 used).

```
diffmpc-learning/
├── CLAUDE.md                              # this file
├── diffmpc2/                             # the SOLVER (vendored dependency — keep pristine)
└──    # THE PROJECT
    ├── NOTE.md                           # living research log — start here
    ├── REFERENCES.md                     # citation-grounded bibliography
    └── experiments/                      # split into gradients/ (gradient-quality) and rl/ (policy learning)
        ├── gradients/                    # gradient-quality experiments (one folder per system: <system>/*.py + results/)
        │   ├── cartpole/                 # cartpole coupling × NLP-tol sweep (results/cartpole_sweep.md)
        │   ├── linear_system/            # 4-variant gradient-accuracy benchmark
        │   ├── active_set_smoothing/     # Diehl Example-1 + corridor surrogate (jump-vs-C¹)
        │   └── quadrotor/                # drone/quadrotor obstacle-avoidance sweep
        │       ├── gradient_quality_sweep.py # experiment driver
        │       ├── plot_gradient_quality.py  # plots + summary
        │       └── results/              # RESULTS.md + CSVs + figures
        └── rl/                           # diffmpc-as-policy / RL experiments
            └── drone_rl/                 # Diff-WMPC 2×2 (gradient-mode × hard/barrier) on drone obstacle avoidance
```

- **`diffmpc2/`** is the released solver (arXiv:2510.06179): SQP + ADMM, GPU linear solvers,
  gradients via implicit differentiation of the KKT system (`jax.custom_vjp`). It is a separate
  git repo (currently on branch `release-cleanup`). **Treat it as stable infrastructure and keep
  it pristine** — do not add project files into it or commit research there.
- **``** is the active work. Start at `NOTE.md`.

## The research questions (see NOTE.md for full treatment)

1. **RQ1** — When are diffmpc2's backward gradients reliable enough to train RL policies,
   *especially with general (nonlinear, state-dependent) inequality constraints* that
   activate/deactivate?
2. **RQ2** — Can we use a *lower NLP tolerance* / fewer iterations? Can ADMM (first-order)
   substitute for an interior-point solver, *for gradient quality*?
3. **RQ3** — Which RL paradigm trains a *diffmpc-as-policy* (NN outputs the MPC problem) most
   efficiently? (MPC-as-Q/value, pure model-free baseline, and multiple RL methods for MPC-as-policy.)
4. **RQ4** — Is real-time / deploy-time gradient-based auto-tuning of MPC feasible?

## Working conventions for this project

- **No speculation.** Every nontrivial technical claim in project docs must be cited
  (`[Key]` → `REFERENCES.md`) or explicitly flagged as a conjecture/hypothesis. Several RQs are
  *partially* answered — know what is established before claiming novelty.
- **Report only what you measured — do not infer beyond the experiment.** When analyzing results,
  state only quantities the experiment directly produced. Any *mechanism, cause, or explanation*
  that was not directly measured (e.g. "local-min switch", "basin boundary", "FD noise floor") is a
  hypothesis: label it as such, and either run the measurement that would confirm/refute it or say
  it is unverified. Never present an interpretation as a fact, and do not let one analysis's
  inference become the premise of the next.
- **Ground claims in code.** When describing solver behavior, cite `file:line` (paths relative to
  `diffmpc2/`). The mechanics are precise; don't paraphrase from memory.
- **Update the note, don't fork it.** `NOTE.md` is a living log with a changelog — append findings;
  mark hypotheses confirmed/refuted with evidence.
- **Run experiments on GPU, slack always on** (standing preference): batch over seeds with `vmap`;
  use a slack problem class for the MPC problems.
- **FD ground truth: convergence check, never a single fixed eps.** When using finite differences
  as a gradient ground truth, do NOT trust one `eps`. Compute the central-difference gradient over a
  *decreasing* sequence of steps (e.g. `1e-3, 1e-4, 1e-5, 1e-6, 1e-7`) **per sample**, and:
  - if the estimates **converge to a stable plateau** (consecutive steps agree within tol) before
    hitting the solver's cost-noise floor → that plateau is the trustworthy GT (it will match AD);
  - if they **do NOT converge** (e.g. grow ~`1/eps`) → the point is at/near a **discontinuity** of
    the (nonconvex) policy map — an active-set boundary or a local-min switch — where `dF/dw` is
    ill-defined. **Flag the sample; do not score it as a gradient/AD error.** (FD(eps→0) IS the
    ground truth and equals AD away from these measure-zero jumps; a fixed eps can straddle a jump
    and produce a spurious huge FD — that's an FD step-size artifact, not an AD failure.)
  Report per-sample (never a batch-summed gradient or bare mean) plus the flagged fraction. The
  usable-eps window is bounded below by the noise floor (GPU non-determinism; keep x64); if no
  plateau exists in `[noise-floor eps, jump-distance eps]`, the sample sits on a discontinuity.
  Rationale + worked example: `CARTPOLE_COUPLING_HANDOFF.md` §8–9.

## Key solver entry points (in `diffmpc2/`)

- **Solver + differentiation:** `diffmpc2/diffmpc/solvers/sqp_admm.py`
  - `SQPADMMSolver.solve` (`:1372`) — `problem_params` is `jax.lax.stop_gradient`'d (`:1378`);
    gradients flow only through the `weights` arg (so the differentiable MPC parameter is the cost
    weight vector — the diffmpc-as-policy gradient).
  - `jax.custom_vjp` (`:432`); `_solve_bwd` (`:1078`) → `_solve_bwd_direct` (`:830`, full KKT) or
    `_solve_bwd_admm` (`:613`). `ForwardBackend`/`BackwardBackend` (`:46`/`:55`); `SQPADMMSolution`
    (`:159`) exposes `convergence_error`, `num_iter`, `admm_iters`, `solver_stats`, `kkt_state`.
- **General inequalities:** `diffmpc2/diffmpc/problems/optimal_control_problem.py` —
  `inequality_constraints` (`:375`), obstacle `‖p−cᵢ‖≥rᵢ` (`:1305`), `SlackProblemAdapter` (`:1414`).
- **Solver tolerances (RQ2 knobs):** `diffmpc2/diffmpc/solvers/params/sqp_admm.yaml`.
- **Drone testbed:** `diffmpc2/examples/drone_obstacles/` (`make_drone_config` in `timing_drone.py`,
  constants in `benchmark_drone_params.py`). FD ground truth: `diffmpc2/diffmpc/utils/gradient_finitediff.py`.
- **diffmpc-as-policy (APG) reference:** `diffmpc2/notebooks/pointmass_rl/`,
  `diffmpc2/examples/reinforcement_learning/spacecraft/apg/`.

## Running the experiments

```bash
# from this workspace root (diffmpc-learning/); needs the diffmpc2 deps + a GPU
# drone/quadrotor obstacle-avoidance gradient-quality sweep (experiments/gradients/quadrotor/):
python3 experiments/gradients/quadrotor/gradient_quality_sweep.py --smoke   # ~90s, C1 only
python3 experiments/gradients/quadrotor/gradient_quality_sweep.py --seeds 4 # full, ~25 min
python3 experiments/gradients/quadrotor/plot_gradient_quality.py            # newest CSV -> PNG
```

- The scripts resolve `diffmpc2/` as a sibling (`../../../../../diffmpc2` from `experiments/gradients/quadrotor/`)
  and put it on `sys.path` ahead of site-packages — needed because a *different* `diffmpc` v1.0.0 is
  pip-installed at `/home/jianghan/Workspace/diffmpc2` and would otherwise shadow it.
- Pure-JAX backends (`ADMM_JAX_LOOP_PCG` fwd, `DIRECT_JAX_DENSE` bwd) run on GPU without the FFI
  build; x64 is required for trustworthy finite differences.
- To build the solver's FFI backends or run its tests: `cd diffmpc2 && make install` (see
  `diffmpc2/README.md`).

## Pointers

- Project log: `NOTE.md` · bibliography: `…/REFERENCES.md` ·
  results: `…/experiments/gradients/quadrotor/results/RESULTS.md` (drone/quadrotor),
  `…/experiments/gradients/cartpole/results/cartpole_sweep.md` (cartpole).
- Solver: `diffmpc2/README.md`, `diffmpc2/docs/`.
