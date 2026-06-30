# Session Handoff — Central-Path (Barrier/Retraction) ADMM for Differentiable NMPC

**Date:** 2026-06-23 · **Repo:** `JianghanZHang/diffmpc-learning` (private) · **Active branch:** `central-path-admm` (8 commits ahead of `main`, **unmerged**)

This is a session-level handoff: what the project is, what this session built, the current state, how to run it, and what's next. For the research log see `NOTE.md`; for the cuDSS/cartpole environment specifics see `CARTPOLE_COUPLING_HANDOFF.md`.

---

## ⚠️ UPDATE 2026-06-26 — `diffmpc2` (`release-cleanup`) gives WRONG hard-box gradients; use `external/turbompc`

The gradient-quality benchmark discrepancy — "diffmpc2 hard-box DIRECT-backward outliers" (`cos_all`
median ~0.36 with negative cosines; the sample-45 / weakly-active investigation in
`research/.../experiments/linear_system/results/gradient_accuracy.md`) — is a **diffmpc2 release-cleanup
bug, not fundamental.** The canonical **`external/turbompc` (GitHub `main`) gives correct gradients**
(`cos_all` median **1.0**, same config/seeds/cuDSS 0.7.1 — measured in `benchmark_repro.md`; reference
figure `grad_box_accuracy_scp1_fdref.png` is external's `cos_all`).

**Cause** (`turbompc/solvers/turbompc_solver.py`; `backward_kkt_jax.py` is **byte-identical** between the
two, so the DIRECT backward is fed wrong *inputs*): diffmpc2 is missing external's two inequality-multiplier
sign refinements — (1) the **sign-correction** `y_ineq = −sign·y_g` (`sign = lower_active − upper_active`)
mapping ADMM bound duals → non-negative active-constraint multipliers, and (2) the **lower/upper dual-sign
disambiguation** (`dual_lower = y_ineq < −tol`, `dual_upper = y_ineq > tol`, else proximity). Without them,
active constraints — especially **near-active**, where lower/upper is delicate — get wrong-signed /
mis-classified multipliers → wrong-direction backward gradients. The rollout warm-start (`timing.py`
`stop_gradient`) was **refuted** as the cause (`warmstart_test.py`).

**Action — `external/turbompc` is the canonical solver now.** It won't build out-of-the-box (its `.cu`
ship the cuDSS-0.8 API; installed cuDSS is 0.7.1 — same situation diffmpc2 had). Backport: compat-shim
`-D` macros (`CUDSS_R_32F=CUDA_R_32F`, `CUDSS_R_64F=CUDA_R_64F`, `CUDSS_R_32I=CUDA_R_32I`,
`cudssReorderingAlg_t=cudssAlgType_t`, `CUDSS_REORDERING_ALG_DEFAULT=CUDSS_ALG_DEFAULT`,
`cudssDataType_t=cudaDataType`) + `sed 's/CUDSS_R_32I, CUDSS_R_32I/CUDSS_R_32I/g'` the three `.cu` (the
0.8 `cudssMatrixCreateCsr` has an extra index-type arg), then `cmake -S turbompc/solvers/csrc -B build/ffi
-DPython3_EXECUTABLE=…/diffmpc2/.venv-cudss/bin/python -DPython3_FIND_VIRTUALENV=ONLY
-DCMAKE_CUDA_FLAGS="<shim>" -DCMAKE_CXX_FLAGS="<shim>"; cmake --build build/ffi -j`; run with that venv
python. `.bak` backups of the patched `.cu` are alongside. **Re-validate all gradient-quality results
that used diffmpc2.** Memory: `diffmpc2-hardbox-outliers-are-release-cleanup-specific`.

### 4-variant benchmark (the current result on the correct solver)
`experiments/linear_system/four_variant_benchmark.{py,md,png,npz}` (external/turbompc, ONE common hard-box
GT, horizon 40, 10 seeds, per-sample **and** batch-summed). At tight tol (640 samples):
- **hard box** & **pure log-barrier** (κ=1e-6): FAITHFUL — cos 1.0, ZERO outliers;
- both **Moreau-slack** variants (turbompc & barrier): biased **~0.45** (the relaxation; flat across tol).

The log-barrier **converges ~1 decade slower** than the hard box: at tol 1e-3 it is *worse* (batch-sum
0.70 vs 0.98; 288 vs 84 per-sample stragglers of 640) — it ties by 1e-5. Per-sample ≈ batch-sum for every
variant (systematic), unlike diffmpc2's outlier-driven hard box (per-sample 1.0 but batch-sum ~0.1).

## ✅ UPDATE 2026-06-29 — inequality-constraint Hessian (μᵀ∇²g) implemented; local solver switched to `external/turbompc`

**The missing exact-Hessian term for NONLINEAR inequality constraints is now implemented** in
`external/turbompc` and mirrored into the local central-path backward. turbompc had the *dynamics*
Lagrangian Hessian (`λᵀ∇²f`) but **no inequality Hessian** — fine for linear/box constraints (`∇²g=0`),
wrong for nonlinear ones (obstacle). Without `Σᵢ μᵢ ∇²gᵢ` the obstacle gradient is the Gauss-Newton
approximation (**12% off FD**); with it, it matches FD (**rel 9e-5**). (Frey2025 Thm 2 / Remark 3.)

- **turbompc:** `OptimalControlProblem.get_inequality_lagrangian_hessian` (mirrors
  `get_dynamics_lagrangian_hessian`) + `_augment_D_with_inequality_hessian` (`turbompc_solver.py`),
  called in `_build_backward_qp` under the existing `use_full_hessian` gate (**backward-only**; forward
  stays Gauss-Newton, converges to the same NLP-KKT). Curvature multiplier = raw forward inequality
  dual (`admm_state.y_g`), same convention as the dynamics `y_f_dyn`.
- **local (`src/diffmpc_learning`):** `central_path_nlp_grad` (`backward.py`) calls the new method with
  net multiplier `y_g_net = y_g_stacked[:,:m] − y_g_stacked[:,m:]`; new `include_ineq_hessian` toggle.
  **`central_path_admm.py` + `tests/conftest.py` switched the `turbompc` path shim `diffmpc2/` →
  `external/turbompc/`** (diffmpc2 lacks the method and has the sign bug).
- **Verified (cuDSS):** turbompc `tests/.../test_inequality_hessian.py` (3 pass: obstacle active, AD=FD
  with term, ablation worse) + local `tests/python/solvers/test_inequality_hessian.py` (3 pass). Existing
  local suites re-validated on external/turbompc (`test_central_path_admm` 7, `test_backward_central_path`
  5). **cuDSS required:** the hard active-set backward is numerically unstable on the nonconvex obstacle
  via CPU JAX_DENSE (near-singular KKT → garbage/heap-corruption); cuDSS is stable. Run with
  `external/turbompc` first on `PYTHONPATH`, `.venv-cudss` python, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Not committed yet** (two repos; **exclude the cuDSS-0.7.1 `.cu` patches** — 4 one-line
  `cudssMatrixCreateCsr` reverts — from any turbompc commit).
- Plan: `~/.claude/plans/serialized-juggling-cat.md`; record:
  `plans/2026-06-29-nonlinear-inequality-hessian.md`.

## ✅ UPDATE 2026-06-30 — Diff-WMPC training pipeline built; hard-box V1-vs-V3 done on the LINEAR drone

**Reproduced the Diff-WMPC training pipeline** (Jahncke et al., RA-L 2026, `reference_papers/Differentiable_Weights-Varying_...pdf`): a state-input NN outputs MPC cost weights; gradients backprop through the differentiable MPC solver. Built + verified hard-box on the **6D LINEAR drone** (double-integrator + quadratic drag, single near-grazing obstacle). New dir `experiments/rl/drone_rl/` (experiments reorganized into `experiments/{gradients, rl}/` — gradient scripts' `../../../../diffmpc2` shims bumped one level).

- **Architecture:** a unified differentiable MPC layer (`mpc_layer.py`: `make_hard_layer` turbompc / `make_barrier_layer` central-path) differentiable w.r.t. `weights` **and** `initial_state`; state-input MLP (`policy.py`, zero-init ⇒ starts at default weights, `weights = default·exp(NN)`); pure-JAX Adam (`optimizer.py`); two gradient estimators (`gradient_modes.py`): **V1** open-loop-plan myopic accumulation (paper's Algorithm 1, jitted `lax.scan` over `N_batch` steps) and **V3** SHAC truncated-BPTT (`h=8`, true `∂u*₀/∂x₀` feedback). Harness `train.py`, plots `plot.py`.
- **Solver work (this session):** verified `external/turbompc`'s `solve` **already returns the `initial_state` cotangent** (`turbompc_solver.py:1011`) ⇒ hard BPTT is true-feedback with no solver change. Built `make_central_path_diff` (`backward.py`) — a `jax.custom_vjp` for the log-barrier solver diff w.r.t. `weights` + `initial_state` (FD-gated, sign positive). Added the **receding-horizon warm-start shift** (`gradient_modes.shift_guess`: roll primal + ADMM duals one step) — cuts warm-started solves from ~3–7 SQP iters to ~1–2.
- **Key perf facts:** turbompc is **ms-scale only when jitted + warm-started** with **QP-KKT tol 1e-6 / NLP-KKT tol 1e-3** (cold/eager/tol-1e-9 = ~2 s; warm+jit ≈ 5–18 ms/solve). The **BPTT reverse pass is ~free** — forward solves dominate, so **V1 and V3 cost the same per step** (the earlier "V3 10× V1" was a degenerate same-state-baseline artifact). Memory: `turbompc-solve-speed-tolerances-warmstart`.
- **Result (hard-box, V1 vs V3, 3 seeds × 150 updates):** both learn — realized closed-loop eval cost ~30 → ~8. **V3 (BPTT) 8.12 ± 0.02 vs V1 (open-loop-plan) 8.63 ± 0.38** (V3 ~6% lower, ~20× tighter cross-seed; grad-norm bounded ~7.4 vs ~124). Files: `experiments/rl/drone_rl/results/{eval_cost_vs_updates,grad_norm_vs_updates,closed_loop_traj}.png`, `RESULTS.md`, 6 CSVs.
- **⚠️ CAVEAT:** the **active-set-switch phenomenon never fired** — the obstacle bound only the MPC's *plan*; the *realized* rollouts stayed outside it (0–1 grazing eval steps), so this is a **baseline** V1-vs-V3 comparison, **not** a test of the jump-vs-continuity thesis. Hard-box only; **barrier V2/V4 deferred** (the `custom_vjp` is ready, just not run — eager central-path is ~18 s/solve).
- **Not committed.** Stray git worktree `.claude/worktrees/agent-a387d1e88688ba179` (incomplete dup of an agent run; real outputs in main tree) — remove. Plan: `~/.claude/plans/serialized-juggling-cat.md`; ledger: `.superpowers/sdd/progress.md`.

## ✅ UPDATE 2026-06-30 (cont.) — Diff-WMPC on the REAL (nonlinear) quadrotor: V1-vs-V3 is a TRUNCATION result (NOT active-set); inequality-Hessian sign corrected + tested

**Did the NEXT-TASK below (nonlinear quadrotor) — and it reframed the V1-vs-V3 story.** New env
`experiments/rl/drone_rl/quadrotor_env.py`: `QuadrotorDynamics` (nx=13 `[pos,vel,quat,ω]`, nu=4
`[thrust,τ]`), RK4 (scheme=2), dt=0.05, H=25; RK4 `simulate_step` == MPC step (no model mismatch);
**quaternion-aware** task loss `w_p‖p‖²+w_v‖v‖²+w_att(1−q₀²/‖q‖²)+w_ω‖ω‖²+w_u‖u−hover‖²` (sign-invariant
SO(3) metric, NOT raw-quaternion diff); ONE **2-D xy-cylinder** obstacle (`1−‖p[:2]−c‖/(r+ε)≤0`,
altitude-independent ⇒ cannot be escaped by climbing ⇒ forces an xy detour). Pipeline made
**env-agnostic** (`gradient_modes`/`train` inject `simulate_step`/`task_loss`/`obs_margin`/`goal_dist`/
`sample_x0`; drone path unchanged — drone FD-gate regression 3/3, cos 0.99999).

- **Learnable regime (key tuning, MEASURED):** with the *well-tuned* default weights the realized
  eval is weight-INSENSITIVE (≈33.6 over a broad weight range ⇒ flat learning). Fix = a **POOR default**
  (low position weight, Q_pos=0.5 ⇒ eval≈85, doesn't reach) so the NN has an 85→33 gap — the drone-analog
  (drone.yaml Q=[0.1]×6 was likewise a poor prior). Verified by `diag_weight_sensitivity` (lowpos 85 vs
  balanced 33.6, H=25).
- **Verified (cuDSS):** fwd conv 4.7e-4 (<1e-3); obstacle engaged; the **realized rollout GRAZES the
  boundary** (closest margin ≈0, ~9–10 grazing steps) — the active-set regime the LINEAR drone could NOT
  produce; BPTT directional-FD gate cos=0.9998; quaternion-norm drift 2.8e-7.
- **V1 vs V3 (hard-box, lr=1e-2, 100 upd, 3 seeds):** V1 (open-loop-plan) learns **85→32.7±0.03** (reaches
  goal); V3 (SHAC trunc-BPTT, h=8) **diverges to ~80** (camps at the obstacle, grazes 34–37). Files:
  `results/{quadrotor_training_summary.json, quadrotor_rollout_v1_v3.png}`, `trained_policies/`.
- **Mechanism = TRUNCATION, NOT active-set (corrected mid-session).** Away from strict-complementarity
  failure (measure-zero) the MPC solution map is differentiable and the BPTT gradient is **exact a.e.**
  (FD cos 0.9998) — active-set switches do NOT inject gradient error. Decisive test = **h-sweep** (V3,
  lr=3e-3): final eval h=8→**64**, h=16→**35**, h=24/32→**33** (=V1). V3→V1 monotonically as `h·dt` covers
  the ~1.7 s task ⇒ the h=8 failure is the *truncated objective* (the 8-step window can't credit the
  detour-to-goal payoff; loitering is cheaper over 8 steps than detouring — supported by "window loss
  minimized yet eval diverges"). NOT an lr artifact (1e-2 diverge / 3e-3 dips to 50 then oscillates→64 /
  1e-4 slow-monotone 85→81). Figs: `results/quadrotor_v3_h_sweep.png`; rollout fig shows V1 & V3@h=24
  BOTH reach the goal (goal-dist 0.011/0.009).
- **Inequality-Hessian multiplier sign** (`turbompc_solver.py` `_augment_D_with_inequality_hessian`,
  both `external/turbompc` & `external/diffmpc2`): `get_inequality_lagrangian_hessian` computes
  `∇²(μᵀg_raw)`, so μ = the **SIGNED net dual** `ν_u−ν_l = admm_state.y_g` (active-masked) — NOT `−sign·y_g`
  (that is the one-sided active multiplier the FIRST-ORDER term `_forward_multipliers` needs; the two
  differ for lower-active, coincide for the upper-active obstacle). Two Copilot suggestions adjudicated by
  FD: `−y_g` (wrong, breaks the obstacle) rejected; signed-`y_g` (correct) applied. Memory:
  `inequality-hessian-multiplier-sign`.
- **Tests** (`external/turbompc/tests/python/solvers/test_inequality_hessian.py`, now **9, all pass on
  ADMM_FUSED_CUDSS**): 3 upper-active (linear dyn), 3 **lower-active** (a negated-obstacle subclass,
  `∇²g≠0`, lower bound binds; incl. a non-vacuous lower-vs-upper cross-check that would FAIL under
  `−sign·y_g`), 3 **drone** (nonlinear `DroneDynamics` drag + obstacle ⇒ exercises BOTH `λᵀ∇²f` and
  `μᵀ∇²g`). FD now routed through `turbompc.utils.gradient_finite_diff` (wrapped in the convergence-checked
  eps-sweep). ⚠️ The pure-JAX backends (`ADMM_JAX_LOOP_PCG`/`DIRECT_JAX_DENSE`) **FAIL** the AD=FD check on
  the obstacle ⇒ gradient-correctness tests REQUIRE the fused-cuDSS backend.
- **Repos:** `external/turbompc` & `external/diffmpc2` solver+test **byte-identical**; diffmpc2
  `inequality-hessian` branch **rebased** onto the remote Copilot autofix `778f4e4` (its JAX-backend swap
  overridden back to fused cuDSS), now **`ahead 2, behind 0`** ⇒ fast-forward push ready (NOT pushed). Main
  repo: nothing committed (verify-only). New files: `experiments/rl/drone_rl/{quadrotor_env, run_experiment,
  verify_quadrotor, plot_h_sweep, plot_rollout}.py` + `trained_policies/` (NN weights kept separate from
  `results/`).

## ⏭️ NEXT TASK — barrier V2/V4 on the GRAZING quadrotor + write-up

1. **Hard-vs-barrier (V2/V4) — now testable.** The quadrotor's realized trajectory GRAZES (real
   active-set switches), so this is finally the regime for the hard-vs-smoothed-complementarity comparison.
   Bring in `make_barrier_layer` (central-path `make_central_path_diff`, diff w.r.t. weights+`x₀`).
   **But** per the corrected finding: the hard gradient is exact a.e., so the barrier's payoff is expected
   in the SMOOTHNESS of the weight→cost map / optimization, NOT in fixing a "wrong" hard gradient. Compare
   {V1,V3} × {hard,barrier} on the same grazing task (eval, grad/update stability across switches, κ bias).
2. **Multi-seed V3@h=24** for error bars (h=24 matches V1; only seed 0 shown so far).
3. **Write up** the V1-vs-V3 / truncation finding into `NOTE.md` (RQ1/RQ3) + a quadrotor `RESULTS.md`. The
   central message: with the EXACT (sign-corrected, full-Hessian) gradient, the V1-vs-V3 outcome is
   governed by the estimator's effective HORIZON (truncation), NOT by active-set non-smoothness — the
   *opposite* of the linear-drone-era "hard jumps destabilize BPTT" framing.

Run env: `external/turbompc` on PYTHONPATH, `.venv-cudss` python, cuDSS backends, QP 1e-6 / NLP 1e-3 (hard);
training lr=1e-2 (V1) / lr=3e-3 (V3 — needed for h≥16 to converge). The local central-path backward
(`src/diffmpc_learning`) is correct & immune to the diffmpc2 multiplier-sign bug (one-sided rows + smooth
complementarity weight `W = y_g/(s + y_g/γ)`, never a two-sided hard active-set).

---

## 1. Project context

**"Gradient Quality of Differentiable NMPC."** We study the quality of gradients obtained by differentiating through an NMPC solver (the **TurboMPC / diffmpc2** solver — SQP + ADMM, GPU linear solvers, KKT-implicit `custom_vjp` gradients), and how to improve it. The differentiable parameter is the MPC cost-weight vector (diffmpc-as-policy). Four research questions (RQ1 gradient reliability under general inequalities; RQ2 cheaper solves / ADMM-vs-IPM; RQ3 RL paradigm; RQ4 deploy-time tuning) — see `NOTE.md`.

**The thread this session pursued (RQ1/RQ2):** inequality **active-set changes** make the diffmpc gradient non-smooth/jumpy (strict-complementarity failure → singular KKT). The fix in the literature (Frey/Diehl, `reference_papers/Diehl_DiffNMPC.pdf`, = `[Frey2025]`) is **interior-point central-path smoothing**: relax complementarity `s·z = 0 → s·z = τ_min`, giving a C¹ solution map with O(τ) bias. The sibling project `external/PrismQP/` does the *same* smoothing but with **ADMM instead of Newton**, via a log-barrier **retraction** in the slack/dual update. **This session built and verified that same central-path/retraction ADMM for the OCP-structured (TurboMPC) setting**, as the foundation for getting smooth gradients through inequality-constrained NMPC.

**Repo layout (root = `/home/jianghan/Workspace/diffmpc-learning/`):**
```
├── HANDOFF.md                  ← this file
├── CLAUDE.md                   ← workspace instructions (read for conventions)
├── pyproject.toml, Makefile    ← NEW: build for the src/ package (TurboMPC-style)
├── src/diffmpc_learning/       ← NEW: the project's solver package (see §3)
├── tests/{python,cuda}/        ← NEW: package tests (11 pass)
├── 
│   ├── NOTE.md, REFERENCES.md          ← research log + cited bibliography
│   ├── SLACK_PENALTY_ADMM.md           ← NEW: ADMM slack-penalty derivation note
│   ├── CARTPOLE_COUPLING_HANDOFF.md    ← cuDSS/cartpole env + earlier findings
│   ├── plans/                          ← NEW: the two implementation plans executed this session
│   └── experiments/{cartpole,quadrotor}/  ← benchmarks + results
├── diffmpc2/                   ← the TurboMPC solver (GITIGNORED — vendored dependency; see §6 gotcha)
├── external/PrismQP/           ← dense-QP barrier-retraction solver (GITIGNORED, own repo)
└── reference_papers/           ← Diehl_DiffNMPC.pdf (untracked)
```

---

## 2. What this session did (overview)

1. **Workspace consolidation & repo init.** Removed the `diffmpc2-gradckpt` git worktree; consolidated the solver to a single `diffmpc2/` on `release-cleanup` (carries the backward-Hessian fix `cccba81`). Repaired the cartpole benchmark (path → `diffmpc2/`, optional-cuDSS-import). Created the **private** GitHub repo `JianghanZHang/diffmpc-learning` and pushed `main`.
2. **Research-log update.** Folded the 2026-06-18 cartpole/quadrotor coupling findings into `NOTE.md`: per-sample convergence-checked-FD methodology, the **backward dynamics-Hessian magnitude bug** (`dt·λᵀ∇²f` Euler hardcode) found via an **acados exact-Hessian cross-check** and fixed in `cccba81` (rel-ℓ₂ ~0.16→~5e-5), and the nonconvex-discontinuity diagnosis.
3. **Theory grounding.** Read `[Frey2025]` and PrismQP; wrote `SLACK_PENALTY_ADMM.md` (how TurboMPC's quadratic slack relates to the log-barrier; the indicator → Moreau-envelope → log-barrier picture; closed-form elastic retraction `s = b_Γ(r)`, `Γ = κ(1/ρ+1/γ)`, `ξ = κ/(γs)`).
4. **Central-path forward solve** (plan: `plans/2026-06-22-central-path-admm-forward.md`, executed subagent-driven). Verified the new barrier/retraction ADMM matches the current TurboMPC solver as `κ→0`.
5. **Packaging.** Migrated the prototype into a proper `src/diffmpc_learning/` package (TurboMPC convention, CUDA-ready).
6. **NLP-KKT convergence** (plan: `plans/2026-06-23-nlp-kkt-sqp-central-path.md`). Built an SQP loop using the central-path solver as the inner QP solver; verified it converges the **nonlinear** KKT on the cartpole swing-up, hard and soft.

All implementation was done **subagent-driven** (implementer → review → fix loop) with whole-branch review; results below were independently re-verified.

---

## 3. The `src/diffmpc_learning/` package (what to build CUDA on)

src-layout, mirrors TurboMPC internally so CUDA can be added the same way:
```
src/diffmpc_learning/solvers/
├── retraction.py        # retraction_map (b_γ, the √); elastic_retraction → (z_g, ξ).  Pure JAX, no turbompc.
├── central_path_admm.py # to_one_sided (box → one-sided rows [G;−G]x≤[u;−l]);
│                        # solve_qp_central_path(qp, schur, *, target_kappa, slack_weight, …) → (x_blocks, (y_f_0,y_f_dyn,y_g), info)
│                        # x-update = TurboMPC's block-tridiag Schur via cuDSS; z-update = elastic_retraction
└── csrc/CMakeLists.txt  # SCAFFOLD for future CUDA (toolchain + cuDSS auto-detect; no targets yet)
tests/python/solvers/{test_central_path_admm.py, test_sqp_central_path.py}   # 11 tests
src/diffmpc_learning/solvers/sqp.py   # sqp_central_path: SQP outer loop w/ central-path inner solver + NLP-KKT residual
```
- **Dependency on TurboMPC:** the package imports `turbompc` (from the `diffmpc2/` sibling) via a `sys.path` shim in `central_path_admm.py`; tests add the paths in `tests/conftest.py`. `turbompc` is the vendored sibling, **not** a pip dependency.
- **Linear system:** cuDSS Schur backend (`SchurSolverBackend.CUDSS_FFI`, JAX-loop family — NOT the fused path, which bakes in the z-update). The barrier only changes the z-update; the Schur x-update is reused unchanged.
- **CUDA later:** drop `.cu`/`.cuh` kernels + a `.cc` FFI glue + a `*_ffi_backend.py` into `solvers/csrc/`, populate `CMakeLists.txt`, build via `make cuda` (`cmake -S src/diffmpc_learning/solvers/csrc -B build/ffi`). Same pattern as `diffmpc2/turbompc/solvers/*/csrc/`.

---

## 4. Key results (verified, 11/11 tests pass)

**Forward solve (RQ1/RQ2 core):** the central-path elastic-retraction ADMM converges to the **same** solution as the current TurboMPC solver as `κ→0`, with the expected **O(κ)** barrier bias:

| κ | 1e-2 | 1e-4 | 1e-6 |
|---|---|---|---|
| ‖x_cp − x_ref‖ | 4.56e-3 | 4.57e-5 | 4.56e-7 |

(both solvers converged to `prim_res < 3e-10`; bare barrier `γ→∞` recovers the hard box.)

**NLP-KKT convergence:** SQP with the central-path inner solver drives the **nonlinear** KKT residual to convergence on the cartpole swing-up (pole-down `[0,0,π,0]`, active `±2` bounds), matching `TurboMPCSolver.solve`:

| | final stationarity | eq | ineq | rel-ℓ∞ vs ref | SQP iters |
|---|---|---|---|---|---|
| **hard** (γ→∞) | 7.7e-8 | 1.3e-11 | 6.1e-8 | ~1.7e-6 | 6 |
| **soft** (γ=1e2) | 6.6e-8 | 1.3e-11 | 0 | ~4.1e-7 | 6 |

A whole-branch (opus) review independently reproduced the per-term residual and confirmed the convergence is **genuine** (driven by real stationarity/equality terms from the true primal + correctly-mapped dual). Design notes that matter: full Newton steps + a short **κ-anneal** are used (TurboMPC's line search stalls the soft case because its merit penalizes the intended soft-bound relaxation); the soft conv-check slack is `s = −y_g/γ` (zeroes the slack-stationarity term, which is *not* load-bearing — the test now asserts the load-bearing terms directly).

---

## 5. Current state

- **Branch `central-path-admm`**, 8 commits ahead of `main`, working tree clean (except untracked `reference_papers/`):
  ```
  ec38516 docs: forward-solve plan + slack-penalty note
  8676b18 feat: closed-form elastic log-barrier retraction
  406575e feat: one-sided stacking + elastic-retraction ADMM loop (cuDSS)
  51c07ca test: cartpole one-sided QP fixture + soft cuDSS reference
  6ca6b6a test: elastic-retraction ADMM matches soft solver as kappa->0
  d14d633 feat(pkg): migrate prototype to src/diffmpc_learning
  3f7d2e6 test: SQP NLP-KKT converges with central-path inner QP (hard+soft)
  8e1a700 test(sqp): assert load-bearing NLP-KKT terms
  ```
- **11 tests pass** (~40s on GPU/cuDSS): 3 retraction unit + 1 toy-loop + 1 cartpole-ref + 1 equivalence(κ→0) + 1 bare-barrier + 2 NLP-KKT(hard/soft) + the surfaced-terms assertions.
- **Not merged.** The branch-completion choice (merge / PR / keep / discard) was deferred while building. If merging, note the earlier whole-branch review predates the `d14d633` migration + SQP commits — re-review those.

---

## 6. How to run (and the cuDSS gotcha)

```bash
cd /home/jianghan/Workspace/diffmpc-learning
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"   # cuDSS 0.7.1 libs (miniconda nvidia/*/lib)
python -m pytest tests -v          # all 11
# or:  make test
```
Requirements: **GPU (RTX 5090, sm_120), x64** (set in the modules), and the cuDSS FFI `.so` at `diffmpc2/build/ffi/`.

**⚠️ Critical gotcha — the cuDSS build depends on an UNCOMMITTED revert in `diffmpc2/`.** Installed cuDSS is **0.7.1.6**, but `diffmpc2/`'s `release-cleanup` ships the cuDSS-**0.8** API (commit `e19a687`, broken on sm_120). The 3 `.cu` files are reverted to the 0.7.1 API as an **uncommitted working-tree change** in `diffmpc2/` (kept uncommitted so the vendored repo stays at origin) and the FFI is rebuilt into `diffmpc2/build/ffi/`. A `git reset --hard` / `checkout` in `diffmpc2/` will wipe this and break cuDSS. To restore (full recipe in `CARTPOLE_COUPLING_HANDOFF.md` §2):
```bash
cd diffmpc2 && git checkout e19a687^ -- \
  turbompc/solvers/admm/csrc/admm_cudss.cu \
  turbompc/solvers/backward/csrc/cudss_sparse_kkt.cu \
  turbompc/solvers/linear_systems_solvers/csrc/cudss_blktridi.cu
export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
cmake -S turbompc/solvers/csrc -B build/ffi -DCMAKE_BUILD_TYPE=Release && cmake --build build/ffi -j
```

---

## 7. Conventions & subtle points (carry forward)

- **diffmpc2/ is vendored and gitignored** — treat as stable infra; do not commit project files into it. It currently has the local cuDSS-0.7.1 `.cu` revert (above) + the `cccba81` backward fix.
- **Two-sided box → one-sided rows.** The closed-form retraction is one-sided, so `to_one_sided` stacks `l ≤ Gx ≤ u` into `[G;−G]x ≤ [u;−l]`; the net two-sided multiplier is `y_g = ν_upper − ν_lower`.
- **Hard vs soft.** With a finite slack penalty `γ`, the elastic retraction's `κ→0` limit is the **soft** (quadratic-penalty) solution (matches TurboMPC `use_slack=True`); `γ→∞` gives the **hard** box (matches `use_slack=False`). Pick the reference to match.
- **FD ground truth = convergence-checked, per-sample** (CLAUDE.md rule): shrink eps to a plateau, flag non-plateau samples as discontinuities; never a batch-summed/bare-mean gradient.
- **No speculation in docs**: cite (`[Key]` → `REFERENCES.md`) or flag as conjecture; report only what was measured.

---

## 8. Open items / next steps (none started)

1. **κ-relaxed backward / VJP** — the actual gradient-quality payoff: differentiate the κ-relaxed retraction-KKT (PrismQP-style `diff_qp`) so the diffmpc-as-policy gradient is *smooth across active-set changes*. The forward solve (done) is the prerequisite; this is the next milestone.
2. **CUDA kernels** — implement the central-path forward (and/or backward) as cuDSS-fused CUDA in `src/diffmpc_learning/solvers/csrc/` (scaffold ready).
3. **Obstacle (one-sided, nonlinear) constraints** — the elastic retraction is already one-sided per row, so it maps directly to `1 − ‖p−c‖/r ≤ 0`; test gradient smoothness across obstacle activation (the original RQ1 motivation).
4. **Finish the branch** — merge / PR `central-path-admm` when ready (re-review the migration + SQP commits).
5. **Fold the NLP-KKT result into `NOTE.md`** (RQ2: ADMM central-path converges the NLP-KKT, matches TurboMPC) — not yet logged there.
