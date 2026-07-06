# CLAUDE.md

Guidance for Claude Code working in this **project workspace**
(`/home/jianghan/Workspace/diffmpc-learning/`).

## What this workspace is

This is the **"Gradient Quality of Differentiable NMPC"** research project. It *uses* the
TurboMPC solver but is kept separate from it.

> **⚠️ 2026-07-02 — Canonical solver: `external/diffmpc2` (branch `LogBarrier-ADMM-QP`).** It is a
> verified strict superset of the previously-canonical `external/turbompc` (`main`): the `turbompc/`
> trees are identical except that diffmpc2 *adds* the log-barrier ADMM QP backend (forward
> `ADMM_LOGBARRIER_CUDSS` + κ-relaxed differentiable backward, see HANDOFF.md 2026-07-02) and proper
> `#if CUDSS_VERSION` guards (its FFI builds on cuDSS 0.7.1 **and** 0.8 from clean source — no manual
> backport, unlike external/turbompc which still needs the sed-revert recipe). It carries the same
> inequality-multiplier **sign-correction** (`y_ineq = −sign·y_g`, `sign = lower_active −
> upper_active`), lower/upper dual-sign disambiguation, and inequality Lagrangian Hessian
> (`get_inequality_lagrangian_hessian`). All project shims (`tests/conftest.py`,
> `src/diffmpc_learning/solvers/central_path_admm.py`, `experiments/rl/drone_rl/*`) point at it.
>
> **Do NOT use the vendored `diffmpc2/` at repo root (`release-cleanup`)** — it produces **wrong
> hard-box backward gradients**: its `turbompc/solvers/turbompc_solver.py` is missing the
> sign-correction and dual-sign disambiguation above, so the (identical) DIRECT KKT assembly is fed
> wrong-signed / mis-classified active multipliers at near-active constraints → outlier gradients
> (benchmark `cos_all` median **~0.36**, negative cosines; correct solver, same config/seeds/cuDSS
> 0.7.1 → **1.0**). Any gradient-quality result produced on it must be re-validated. Measured:
> `experiments/gradients/linear_system/results/benchmark_repro.md`; memory
> `diffmpc2-hardbox-outliers-are-release-cleanup-specific`.

```
diffmpc-learning/
├── CLAUDE.md                              # this file
├── external/
│   ├── diffmpc2/                         # the SOLVER (CANONICAL, branch LogBarrier-ADMM-QP — keep pristine)
│   ├── turbompc/                         # previous canonical (main); solver tree ⊂ external/diffmpc2
│   └── PrismQP/                          # dense-QP barrier-retraction reference (own repo)
├── src/diffmpc_learning/                 # project package: central-path (log-barrier) ADMM prototype
├── tests/                                # project tests (conftest puts external/diffmpc2 on sys.path)
└──    # THE PROJECT
    ├── notes/NOTE.md                           # living research log — start here
    ├── notes/REFERENCES.md                     # citation-grounded bibliography
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

- **`external/diffmpc2/`** is the solver (arXiv:2510.06179 lineage, package name `turbompc`):
  SQP + ADMM, GPU/cuDSS linear solvers, gradients via implicit differentiation of the KKT system
  (`jax.custom_vjp`), plus the log-barrier ADMM QP backend. It is a separate git repo (branch
  `LogBarrier-ADMM-QP`, pushed). **Treat it as stable infrastructure and keep it pristine** — do
  not add project files into it; solver commits happen there deliberately, never as a side effect
  of project work. (The formerly-vendored `diffmpc2/` at repo root has been removed.)
- **``** is the active work. Start at `notes/NOTE.md`.

## The research questions (see notes/NOTE.md for full treatment)

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
  (`[Key]` → `notes/REFERENCES.md`) or explicitly flagged as a conjecture/hypothesis. Several RQs are
  *partially* answered — know what is established before claiming novelty.
- **Report only what you measured — do not infer beyond the experiment.** When analyzing results,
  state only quantities the experiment directly produced. Any *mechanism, cause, or explanation*
  that was not directly measured (e.g. "local-min switch", "basin boundary", "FD noise floor") is a
  hypothesis: label it as such, and either run the measurement that would confirm/refute it or say
  it is unverified. Never present an interpretation as a fact, and do not let one analysis's
  inference become the premise of the next.
- **Ground claims in code.** When describing solver behavior, cite `file:line` (paths relative to
  `external/diffmpc2/`). The mechanics are precise; don't paraphrase from memory.
- **Update the note, don't fork it.** `notes/NOTE.md` is a living log with a changelog — append findings;
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
  Rationale + worked example: `notes/CARTPOLE_COUPLING_HANDOFF.md` §8–9.

## Key solver entry points (in `external/diffmpc2/`)

- **Solver + differentiation:** `turbompc/solvers/turbompc_solver.py` — `TurboMPCSolver` (`:301`),
  `jax.custom_vjp` solve (`:1045`); the inequality-multiplier **sign-correction** feeding the
  backward (`:898`, `sign = lower_active − upper_active`, `y_ineq = −sign·y_g`); the inequality
  Lagrangian Hessian fold `_augment_D_with_inequality_hessian` (`:1362`).
  `ForwardBackend`/`BackwardBackend` enums select the backends (gradient-correctness work
  **requires the fused-cuDSS forward**: pure-JAX backends fail AD=FD on the obstacle).
- **Log-barrier (central-path) QP backend:** forward `turbompc/solvers/admm/logbarrier_admm_qp.py`
  (`to_one_sided`, `solve_logbarrier_admm_qp`) + `logbarrier_admm_cudss_ffi_backend.py`
  (`ForwardBackend.ADMM_LOGBARRIER_CUDSS`); κ-relaxed differentiable backward
  `turbompc/solvers/backward/logbarrier_backward.py` (`make_logbarrier_diff`, W-fold reduced KKT).
  Suite: `tests/python/solvers/test_logbarrier_admm_qp.py`.
- **General inequalities:** `turbompc/problems/optimal_control_problem.py` — obstacle
  `‖p−cᵢ‖≥rᵢ` (`obstacle_avoidance.py`), `get_inequality_lagrangian_hessian` (`:1444`).
- **Solver tolerances (RQ2 knobs):** `turbompc/solvers/params/turbompc.yaml` (+ `sqp.yaml`).
  Speed needs jit + warm-start + QP-KKT 1e-6 / NLP-KKT 1e-3 (see memory
  `turbompc-solve-speed-tolerances-warmstart`).
- **FD ground truth:** `turbompc/utils/gradient_finitediff.py` (wrap in the convergence-checked
  eps-sweep — see the FD rule above).
- **diffmpc-as-policy (APG) reference:** `examples/pointmass_rl/`, `examples/RL_quadrotor.ipynb`.

## Running the experiments

```bash
# from this workspace root (diffmpc-learning/); needs a GPU
# environment for cuDSS runs (fused backends, project tests):
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python

$PY -m pytest tests -v                                                    # project tests
# drone/quadrotor obstacle-avoidance gradient-quality sweep (experiments/gradients/quadrotor/):
$PY experiments/gradients/quadrotor/gradient_quality_sweep.py --smoke     # ~90s, C1 only
$PY experiments/gradients/quadrotor/gradient_quality_sweep.py --seeds 4   # full, ~25 min
$PY experiments/gradients/quadrotor/plot_gradient_quality.py              # newest CSV -> PNG
```

- The scripts and `tests/conftest.py` resolve `external/diffmpc2/` (repo-relative) and put it on
  `sys.path` ahead of site-packages — needed because a *different* `diffmpc` v1.0.0 is
  pip-installed at `/home/jianghan/Workspace/diffmpc2` and would otherwise shadow it.
- Pure-JAX backends (`ADMM_JAX_LOOP_PCG` fwd, `DIRECT_JAX_DENSE` bwd) run on GPU without the FFI
  build; x64 is required for trustworthy finite differences. **But gradient-correctness tests need
  the fused-cuDSS FFI** (already built at `external/diffmpc2/build/ffi/`; rebuilds work on cuDSS
  0.7.1 and 0.8 from clean source thanks to the branch's `#if CUDSS_VERSION` guards).
- To rebuild the FFI: `cmake -S turbompc/solvers/csrc -B build/ffi -DPython3_EXECUTABLE=$PY
  -DPython3_FIND_VIRTUALENV=ONLY && cmake --build build/ffi -j` from `external/diffmpc2/`.

## Pointers

- Project log: `notes/NOTE.md` · bibliography: `…/notes/REFERENCES.md` ·
  results: `…/experiments/gradients/quadrotor/results/RESULTS.md` (drone/quadrotor),
  `…/experiments/gradients/cartpole/results/cartpole_sweep.md` (cartpole).
- Solver: `external/diffmpc2/README.md`, `external/diffmpc2/docs/`.
