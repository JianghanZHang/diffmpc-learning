# Session Handoff — Central-Path (Barrier/Retraction) ADMM for Differentiable NMPC

**Date:** 2026-06-23 · **Repo:** `JianghanZHang/diffmpc-learning` (private) · **Active branch:** `central-path-admm` (8 commits ahead of `main`, **unmerged**)

This is a session-level handoff: what the project is, what this session built, the current state, how to run it, and what's next. For the research log see `notes/NOTE.md`; for the cuDSS/cartpole environment specifics see `notes/CARTPOLE_COUPLING_HANDOFF.md`.

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

## ✅ UPDATE 2026-07-02 — LogBarrier_ADMM_QP: decoupled CUDA log-barrier ADMM solver + κ-relaxed differentiable backward (`external/diffmpc2`, branch `LogBarrier-ADMM-QP`, **pushed to origin**)

Built + validated a **standalone log-barrier (central-path) ADMM QP solver in `external/diffmpc2`** — CUDA forward **and** differentiable backward — as a *decoupled* turbompc backend (NOT a modification of the hard-box path; no new `inequality_mode`). This is the CUDA realization of the `src/diffmpc_learning` central-path prototype (§3), now with the κ-relaxed backward (open item #1 in §8 — **done**). 5 commits on `origin/LogBarrier-ADMM-QP`.

- **Forward** (decoupled, cuDSS): `turbompc/solvers/admm/logbarrier_admm_qp.py` (`to_one_sided`, `solve_logbarrier_admm_qp`), `logbarrier_admm_cudss_ffi_backend.py` (`logbarrier_admm_qp_forward`), `ForwardBackend.ADMM_LOGBARRIER_CUDSS`. Only the ADMM **z-update** differs from hard-box (closed-form log-barrier retraction `b_γ`, elastic + γ-free pure); shared kernels extracted to `admm_cudss_shared.cuh`. Matches the JAX oracle + hard QP as κ→0 (gates: elastic 1.8e-9, barrier 2.9e-9, vs-hard 6.3e-8).
- **Backward** (`turbompc/solvers/backward/logbarrier_backward.py`): the **κ-relaxed W-fold reduced KKT** — `W = y_g/(s+y_g/γ)` [elastic] or `y_g/s` [pure], fold `G1ᵀdiag(W)G1` + exact Lagrangian Hessian `λᵀ∇²f + y_g_net·∇²g` into D, solve the reduced equality-KKT via `solve_backward_kkt_cudss_ffi`. `make_logbarrier_diff` = `jax.custom_vjp` (w.r.t. weights + x0). **Ported from the FD-verified `src/diffmpc_learning/solvers/backward.py`** + the new pure-barrier branch. **KEY design:** `_bwd` uses the forward SQP's **cached converged duals** (`logbarrier_nlp_solve` returns `(y_f_0,y_f_dyn,y_g)`), NOT a backward re-solve — a cold re-solve drifts off `states_c` (esp. `y_f` init 0) → wrong duals. Cross-check: cached `y_g` vs closed-form barrier dual `y_g_eq=κ/s(states_c)` = **2.5e-10** (confirms convergence; the analytical dual is exact — barrier complementarity is separable per row).
- **cuDSS build-skew ELIMINATED on this branch** (commit `973655c`): replaced the fragile build-time `sed`-revert (0.7.1 vs 0.8 `cudssMatrixCreateCsr`) with proper `#if CUDSS_VERSION>=800` guards + a `<0.8` symbol-name shim across all 3 `.cu`/`.cuh` sites — the FFI now builds on **both** cuDSS versions from clean source, **no manual backport**. (Supersedes the §6 / 06-26 sed recipe *for `external/diffmpc2` on this branch only*; `diffmpc2/` release-cleanup + `external/turbompc` still need theirs.)
- **Gradient accuracy** (AD vs convergence-checked `gradient_finite_diff`, all at small κ): **∇²f (dynamics Hessian): interior drone κ=1e-6, cos=1.0, rel=5.5e-5** (exact at the near-hard regime); ∇²g+W-fold: obstacle-elastic κ=1e-4 cos=1.0; pure-W: box-pure cos=1.0; x0 sign cos=1.0; reduced-KKT cuDSS==dense 7.6e-8; ablation (Hessian on/off) ratio 17.8×. Suite: `tests/python/solvers/test_logbarrier_admm_qp.py`. Whole-branch opus review: math faithful to the reference, hygiene clean (no `.so`/`.bak`/backport committed).

**⚠️ CORRECTION — FD converges for hard projection; being AT an active constraint is NOT "unscorable".** An earlier framing (mine) called the pure-barrier + active-constraint + small-κ case "inherently unscorable" because the trajectory sits on the boundary — **that was wrong.** The parametric solution map `w↦x*(w)` is differentiable wherever the **active set is locally constant** (IFT on the KKT); hard projection onto the active face is a smooth operation with a well-defined gradient — exactly what the implicit-KKT backward computes, and FD converges there. Non-differentiability arises ONLY at active-set **changes** in *parameter* space (measure-zero), NOT at active constraints per se. The drone-pure gate's FD **flagging** at small κ was a **numerical artifact**: at small κ the pure barrier is ill-conditioned (`s=κ/y_g→0`), the solve is noisy, and at the ~1 mm constraint margin the FD plateau window (above the noise floor, below an active-set-change distance) was too narrow — the convergence-check correctly flagged the noise instead of scoring it. Fixable by FD tuning (eps above the noise floor, larger margin, tighter solver); the gradient exists and AD computes it (confirmed exact in the interior W=0 case). The **κ-sweep** (`scratchpad/kappa_sweep.py`, κ=1→1e-6) further confirmed the ~1.37% seen at κ=1 was **entirely** the W-fold SQP-linearization on the active obstacle (`W=κ/s²≈6`), not a ∇²f error. **Open follow-up:** a *direct* pure-barrier + active-constraint AD-vs-FD demonstration at small κ (FD tuned above the noise floor) — not yet run as a gate; the current ∇²f gate uses an interior (W=0) fixture to isolate the dynamics Hessian.

**Commits** (`origin/LogBarrier-ADMM-QP`): `cd7e9a1` W-fold backward + `make_logbarrier_diff` · `973655c` cuDSS version-guard · `af0db65` gradient gates + analytical `y_g` · `8bf271d` cache forward duals / drop re-solve · `0cff888` κ-sweep + re-anchor gate to κ=1e-6.

## ✅ UPDATE 2026-07-02 (cont.) — `external/diffmpc2` designated CANONICAL solver; both suites re-run green on it

**Canonical solver is now `external/diffmpc2` (branch `LogBarrier-ADMM-QP`), superseding
`external/turbompc`** (user decision). Verified safe before switching: the two `turbompc/` package
trees are **identical** except diffmpc2 *adds* the logbarrier backend files/registrations and the
`#if CUDSS_VERSION` guards (external/turbompc still carries the manual 0.7.1 sed-revert + `.bak`s);
`benchmarking/` identical. The sign-correction (`turbompc_solver.py:898`), dual disambiguation, and
inequality Hessian (`:1362`) are all present. CLAUDE.md callout + memory rewritten accordingly.

- **Logbarrier QP suite (the ask): 21/21 PASS** (`tests/python/solvers/test_logbarrier_admm_qp.py`,
  fused cuDSS, 46:53). FD-vs-AD gates: box-elastic cos=1.000000/rel=1.7e-7, box-pure 1.000000/5.8e-6,
  obstacle-elastic 1.000000/1.7e-3, drone-pure-interior (κ=1e-6) 1.000000/5.5e-5, x0 1.000000/5.6e-8 —
  all FD-converged (flagged=False). **FD did NOT plateau (flagged=True) in 3 gates** — obstacle-pure,
  drone-elastic, drone-pure (all nonconvex + trajectory at/near the obstacle, min_dist 0.28–0.40):
  per the FD rule these are NOT scored on cos/rel; the tests assert flag-attribution + AD finite &
  nonzero (printed cos ≥0.999846 but unreliable when flagged). ⇒ the pure-barrier+active-constraint
  case still has NO FD-verified accuracy number — the tuned-FD direct gate remains the open follow-up
  above. Also: reduced-KKT cuDSS==dense 2.1e-7, ineq-Hessian ablation 17.8×.
- **Local project suite: 23/23 PASS on the new shim** (34:45; includes the 5 backward-central-path,
  5 x0-sensitivity/BPTT, 3 inequality-Hessian, 2 NLP-KKT gates).
- **Switch mechanics:** all `sys.path` shims + `PYTHONPATH=` usage lines now resolve
  `external/diffmpc2` — `tests/conftest.py`, `src/diffmpc_learning/solvers/central_path_admm.py`,
  all of `experiments/rl/drone_rl/`, and the gradient-experiment scripts. Two pre-existing breakages
  found & fixed: (1) **the vendored `diffmpc2/` at repo root NO LONGER EXISTS** — the legacy
  cartpole/quadrotor scripts were shimming a dead path (⇒ §6's uncommitted-`.cu`-revert gotcha and
  the old `diffmpc2/build/ffi` instructions are OBSOLETE); (2) `tests/conftest.py` still added the
  pre-flatten `research/gradient-quality-diffnmpc/experiments/cartpole` path — collection was broken;
  now `experiments/gradients/cartpole`. Not committed (this repo); `external/diffmpc2` untouched
  (only its untracked `test_turbompc_x0_sensitivity.py` from earlier remains).

## ✅ UPDATE 2026-07-02 (cont. 2) — FD false-flags root-caused (termination-error floor) + elastic α-throttle mechanism CONFIRMED

**(1) The 3 FD-flagged logbarrier gates were false flags — FD converges (user was right).** Root
cause (measured, `scratchpad/fd_flag_diagnosis.py` + log): the fine eps ladder (1e-5,3e-6,1e-6) sat
entirely below the **SQP termination-error floor** — solves stopping at `sqp_tol=1e-6` carry a
deterministic ~5e-8 cost error (repeat-spread ≤2e-14 ⇒ NOT GPU noise; tol=1e-8 re-solve moves cost
~6e-10 and lands FD on AD), amplified by 1/(2ε) ⇒ ~0.25–2.5% FD error ≫ the 2e-3 plateau rtol.
NO local-min switch ever fired (obstacle margin/side bit-stable across every ± solve, ε≤1e-3).
On (1e-3,3e-4,1e-4) FD plateaus and matches AD: **obstacle-pure rel=1.12e-4, drone-pure 6.30e-5,
drone-elastic 7.64e-4, obstacle-elastic 1.93e-3 — all cos=1.000000, all flagged=False** (5 gates
re-run PASS, incl. ablation 19.7×). Fix: `_NLP_EPS_SEQ_R=(1e-3,3e-4,1e-4)` in
`test_logbarrier_admm_qp.py` (uncommitted, external/diffmpc2). **This closes the "open follow-up"
above: the pure-barrier + ACTIVE-constraint case (margin≈1e-3=κ/y_g) is now FD-verified.**

**(2) drone-elastic one-sided "stall" mechanism CONFIRMED (`scratchpad/fd_stall_diagnosis.py`):
merit/formulation mismatch, elastic-only — not a broken line search, and NOT κγ numerology (refuted:
κ=1e-2 gives the identical trace).** α-history: stalling directions (R[0]+ε / R[1]−ε) pin **α=0.1
(grid min) for all 60 iters**; `linesearch=False` ⇒ same solves converge in **6–7 full-Newton iters**
(identical conv trace to the good directions). Two elastic-path defects in `logbarrier_nlp_solve`:
(a) the backtracking merit is evaluated with the program's `use_slack_variables=False` ⇒ it penalizes
the RAW inequality violation that the elastic optimum *intentionally* carries (ξ≈1.6e-2 on this
fixture) ⇒ rejects good steps one-sidedly (directions that deepen the sag); (b) the reported KKT≈1e-2
at the cap is the **slack-stationarity transient** `‖γ·slacks+y_g‖ = 0.9^k·‖y_g‖∞` (slacks init 0,
α-blended while duals are NOT) — 5.5·0.9⁵⁹ = 9.9e-3 exactly; the primal sits at stat≈5e-6 the whole
time (decomposition: stat 5.5e-6, eq 1.6e-12, ineq 8.1e-5 ⇒ conv=9.9e-3 is the unprinted 4th term).
The pure barrier is immune (interior, no slack state); cold-start converges (α ramps 0.1→1, 16 it).
The local `src/diffmpc_learning` prototype avoids both by design (full Newton + analytic conv-check
slack `s=−y_g/γ` — §4). **Solver-fix options (NOT applied; decide deliberately):** default
`linesearch=False` for the elastic path, and/or stop α-blending the slack state (adopt `−y_g/γ`
post-linesearch), and/or make the merit elastic-aware (SlackProblemAdapter semantics). ⚠️ Relevant
to V2/V4 training: elastic forward solves can be α-throttled directionally ⇒ budget-capped solves
return slightly-off primals + inconsistent cached duals for the backward. (The asymmetry's
sign-selectivity is strongly indicated by the merit's violation term but was not itself
instrumented — merit values along the step not printed.)

## ✅ UPDATE 2026-07-02 (cont. 3) — elastic α-throttle FIXED (barrier merit + analytic slack); pure-mode ∞-merit collapse found & FIXED (relaxed-barrier merit); hard-box iteration parity

**Implemented in `external/diffmpc2` `turbompc/solvers/backward/logbarrier_backward.py` (uncommitted):**
1. **`barrier_merit_linesearch`** — the ℓ1 exact penalty of the BARRIER formulation
   (`f + Σψ_κ,γ(r) + μ‖eq‖₁`; inequalities live inside ψ, only eq penalized; same α-grid/η/μ
   machinery as `turbompc.solvers.linesearch`). ψ closed forms: `elastic_psi` (via the retraction
   root, globally smooth, `dψ/dr = γξ* =` row dual — FD-verified to machine precision) and
   `pure_psi` (reference only). Replaces the hard-violation merit that was minimized O(ξ)=1.6e-2
   off the elastic fixed point (⇒ merit-formulation mismatch ⇒ α pinned 0.1).
2. **Analytic slack** — `slacks = −y_g/γ` adopted post-step; the slack is no longer a
   zero-initialized, α-blended linesearch state (that lag manufactured the phantom
   `slack_stationarity = 0.9^k·‖y_g‖∞` residual, e.g. 5.5·0.9⁵⁹ = the "9.9e-3 stall").
3. **Relaxed-barrier merit for the PURE mode** (`_PURE_MERIT_RELAX_GAMMA=1e8`; Feller/Ebenbauer-
   style relaxed log barrier — `elastic_psi` IS the smooth relaxation, merit-only, the inner solve
   still targets the true pure barrier). Root cause it fixes (measured, `pure_merit_breakdown.py`):
   the pure 60-cap warmup ended **silently non-converged and +1.8e-3 obstacle-INFEASIBLE**, where
   `pure_psi=+∞` ⇒ `merit(current)=∞` ⇒ Armijo finite-mask collapse ⇒ unconditional α=0.1 fallback
   (this, not merit-offset, was the pure crawl; it also means every prior "pure warm-started" probe
   ran from a slightly-off point). Interior merit bias of the relaxation: O(κ/(γs²)) ≈ 1e-6 rel.

**Measured (probes, drone N=8 / obstacle N=5, tol 1e-6):** elastic warm ±1e-4: 60-cap crawl →
**7/7 iters α=1** (ls=False trace exactly recovered); pure: drone 51→**5-6**, obstacle 44→**8-10**,
all α≈1, ± symmetric; cold elastic 17 iters (α ramps 0.7→0.1→1 — globalization intact); κ∈{1e-6,1e-2},
γ=1e3 variants all 7-10 iters. **Hard-box comparison (same fixtures/protocol/tol,
`hardbox_iters_probe.py`): hard warm 5-6 (drone) / 9-10 (obstacle), cold 11/17 — the fixed
logbarrier is at hard-box ITERATION PARITY warm (+0-1 iter), and NOTE `turbompc.yaml` ships
`linesearch: False` — the canonical hard solver globalizes by full Newton, no merit LS at all.**
New tests: `test_barrier_merit_psi_identities`, `test_logbarrier_linesearch_not_throttled`
(parametrized elastic+pure; also asserts the COLD warmup converges — would have caught the silent
pure non-convergence). Diagnostics: `scratchpad/{fd_stall_diagnosis, pure_merit_breakdown,
row_id_probe, hardbox_iters_probe, pure_asymmetry_probe}.py`.

⚠️ Follow-up: the problem-level `equality_constraints` ℓ1 defect read 3.5e-3 at the
(non-converged) pure warm start — most likely just that non-convergence, but re-measure at a truly
converged point to rule out an integrator-convention mismatch (it takes `controls[:horizon],
controls[1:]`). (The full-suite follow-up is resolved — see cont. 4: 26/26 on the final code.)

## ✅ UPDATE 2026-07-02 (cont. 4) — OUTER-SLACK globalization: fraction-to-boundary + Wächter-Biegler FILTER + full restoration; now the DEFAULT for both modes

**Implemented (user-directed) the standard nonlinear-IPM globalization in `logbarrier_backward.py`
(~200 lines, NO inner-solver/kernel changes):** `globalization="filter"|"merit"|"none"` on
`logbarrier_nlp_solve`/`make_logbarrier_diff` (auto = filter; `linesearch=False` = none).
- **Outer slack `s_outer`** — an NLP-LEVEL iterate (one per one-sided row), persistent across SQP
  iterations, initialized interior (`max(−r(x₀), 1e-8)`) and allowed to DISAGREE with `r(x)`:
  the disagreement `‖r(x)+s‖₁` lives in the filter's θ as a violable-equality residual (the IPM
  device that parks infeasibility FINITELY — the pure barrier φ only ever sees `s>0`, never
  `g(x)`, so the ∞-collapse is structurally impossible). Slack step target after each inner solve
  = the converged barrier complementarity **`s_target = κ/y`** (exact per row) ⇒ the inner QP,
  kernels, and backward are untouched.
- **Fraction-to-boundary** (τ=0.99, closed-form on `s`) caps every trial; **filter** (θ,φ) pairs
  with sufficient-decrease + domination + f-type/θ-type switching + Armijo-on-φ + filter reset on
  κ-anneal; **full restoration phase** (dedicated θ-minimizing sub-loop: slack refresh + inner-QP
  steps + Armijo-on-θ under FTB) when the filter rejects all trials.
- **A/B vs the relaxed-barrier merit** (`filter_ab_probe.py`, drone N=8, both modes × cold/warm ±ε/
  infeasible-start(3e-3 into the obstacle)): warm 6-7 iters BOTH arms both modes (ties);
  **elastic COLD: filter 12 iters all-α=1 vs merit 17 (α-ramp) — filter wins ~30%**; pure cold 11
  both (filter exercised restoration once, correctly); infeasible-start 1-2 iters both arms
  (merit's stiff quadratic and the filter's restoration both recover in one step). Per the staged-
  adoption decision, **filter is now the default for BOTH modes**; merit retained as fallback.
  Hard-box parity now holds cold AND warm (hard: 11 cold / 5-6 warm).
- Regression test extended: `test_logbarrier_linesearch_not_throttled` now parametrized
  (use_slack × globalization) — 4 combos, each asserting cold-warmup convergence + ≤20-iter warm
  solves. **Definitive full-suite run on the final code: 26/26 PASS (48:17)** — all QP/CUDA
  oracle gates, all FD-vs-AD gradient gates (unflagged, values matching the verified eps-ladder
  numbers), ψ identities, and all 4 no-throttle combos (cold 11–17 it, warm 5–7 it, α≈1).
- Caveat: `globalization="filter"` auto-falls-back to "merit" when
  `rescale_optimization_variables=True` (unscaled `r(x)` vs scaled QP duals not reconciled).

## ✅ UPDATE 2026-07-05 — V2/V4 (barrier arms) TRAINED on the grazing quadrotor: both learn; the ELASTIC RELAXATION IS EXPLOITABLE by the weight-learning outer loop

**Infrastructure** (new `experiments/rl/drone_rl/barrier_modes.py` + `train.py` 4-variant dispatch):
V2=plan_barrier / V4=bptt_barrier on the CANONICAL diffmpc2 logbarrier layer (elastic κ=1e-4,
γ=1e2, filter globalization, sqp_tol=1e-3 training tolerance to match the hard arms' NLP 1e-3),
eager estimators mirroring V1/V3 semantics exactly; receding-horizon warm start held in a concrete
cell inside the custom_vjp forward (tracing-invisible; analog of the detached `shift_guess`).
**Backward at training tolerance uses the ANALYTIC barrier dual** (`yg_mode="analytic"`:
`y_g_eq=κ/s(states_c)`, exact per row) — the cached forward dual is O(10–26%) stale on grazing
steps at tol 1e-3 and tripped the strict crosscheck assert (now parameterized; gates unchanged).
Verified: loose-solve analytic-dual gradient vs tight FD-verified reference **cos=1.000000,
rel=2.1e-5**. Also: `--time_budget` (was a hidden hardcoded 1200s bail-out that truncated the
first attempts), incremental CSV flush, `--barrier_glob` (NOTE: earlier "filter vs none identical"
observation was an artifact — the flag wasn't plumbed; never actually A/B'd in training).
~127 s/update (V2) / ~350 s (V4@h=24): the eager SQP fwd+bwd dominates ⇒ the fused-CUDA NLP loop
(§8 item 2) is the speed lever.

**Results (seed 0; V2: lr=1e-2, 100 upd; V4: h=24, lr=3e-3, 95 upd — 12h budget):**
- **V4: stable monotone learning 45.8→29.6 eval**, grads bounded (median 3.4, max 48) across
  persistent grazing — consistent with the exact-a.e.-gradient finding (no active-set instability).
- **V2: learned to 29.0 by upd 70, then DESTABILIZED** (eval 56–59 for upd 80–100, grad spikes to
  183) at lr=1e-2 — the V1-family lr on the barrier arm is not late-run stable (seed 0).
- **⚠️ KEY FINDING — the elastic relaxation is exploitable:** BOTH arms progressively deepen
  constraint penetration during training: eval closest-margin +0.08→**+0.49**, 7–9/35 eval steps in
  violation (task loss has no obstacle term; enforcement is the MPC's job). Mechanism (consistent
  with elastic KKT, labeled hypothesis for the write-up): the NN raises cost weights → constraint
  duals y grow → elastic sag ξ=y/γ grows → the "better" eval cost (29.6 vs hard V1's 32.7) is
  bought by cutting through the obstacle. Hard arms structurally cannot (0 violations). ⇒ raw
  eval-cost comparison hard-vs-elastic is NOT apples-to-apples; report cost+violations jointly.
- Follow-ups: γ-sweep (1e3/1e4 shrinks the exploit ~linearly), pure-barrier arms (no sag by
  construction), re-eval barrier-trained weights under the HARD deploy controller, V2@lr=3e-3,
  multi-seed. CSVs: `results/train_quadrotor_{plan,bptt}_barrier_seed0.csv`; policies saved.

## ✅ UPDATE 2026-07-06 — fused CUDA logbarrier WIRED into training (jitted SQP driver); gates PASSED

**Discovery.** Barrier training was ~50–100× slower than the hard arms purely from EAGER dispatch,
not math: `barrier_modes` ran the eager `logbarrier_nlp_solve` Python SQP (~7 s/warm solve vs the
jitted hard solver's 0.9 s at the SAME 5–7 SQP iters ⇒ V2 ≈ 127 s/update, V4@h24 ≈ 353 s/update).
The fused inner solve exists (`AdmmBackend.LOGBARRIER_CUDSS`, one `jax.ffi` call/QP), **but
`ForwardBackend.ADMM_LOGBARRIER_CUDSS` through `TurboMPCSolver.solve` does NOT work**: the fused
branch (`admm.py:774-848`) returns the ONE-SIDED `(N+1, 2m)` ADMM state while `_solve_impl`'s
while_loop carry starts from the two-sided `initial_state` `(N+1, m)` — pytree shape mismatch at
trace time (the suite only ever exercised the QP-level wrapper with backend 6).

**What was built (this session):**
- **`logbarrier_nlp_solve_jit`** (`turbompc/solvers/backward/logbarrier_backward.py`): a
  `lax.while_loop` port of the eager loop's hot path — full Newton steps (no globalization),
  fixed κ, fused-CUDA inner solve via an `ADMMSolver(admm_backend=LOGBARRIER_CUDSS, kappa=κ)`,
  IDENTICAL NLP-KKT conv check (analytic slack `−y/γ`, `_conv_check_qp`). Returns the fixed-shape
  `LogBarrierJitSolution` incl. the cached duals (`y_f_dyn`, one-sided `y_g_stacked`) the backward
  needs. Docstring: cold/infeasible starts belong to the eager globalized solve.
- **Trace-safety guard** in `_relaxed_nlp_backward`: the concrete `float()`/assert/global y_g
  crosscheck runs only when `yg_crosscheck_tol is not None` → the whole backward is now
  `jax.jit`-able (with `yg_mode="analytic"`, tol=None).
- **Layer** (`experiments/rl/drone_rl/barrier_modes.py`): `DEFAULT_LB_CFG["forward"]="fused"` —
  COLD solve (cell empty, once per reset/eval) → eager filter+restoration; WARM solves → jitted
  `logbarrier_nlp_solve_jit`; backward → jitted `_relaxed_nlp_backward` wrapper. `train.py` gained
  `--barrier_forward {fused,eager}`.
- **Suite gates** (`test_logbarrier_admm_qp.py`, now 29 tests): `test_jit_forward_matches_eager_warm`
  (elastic+pure: jit 6 iters conv 2.2e-07, states rel-linf 3e-09 vs eager filter) and
  `test_jit_backward_matches_eager` (jit≡eager to 5.5e-17).

**Gates on the QUADROTOR training problem (measured, scratchpad/fused_gates.py):**
- **(a) forward parity/convergence, 12-step warm training stream:** identical SQP iteration counts
  and conv values vs the eager filter path at BOTH tols — tol 1e-3: 3–5 iters, max states rel-linf
  1.0e-05; tol 1e-6: 5–8 iters, max rel-linf 6.1e-08; jit conv < tol every step. Warm fused solve
  **0.095 s** (steady) vs eager 8.4 s at tol 1e-3 (~90×/solve).
- **(b) AD-vs-FD through the fused layer (eps ladder 1e-3/3e-4/1e-4, sqp_tol=1e-6):**
  dL/dweights over ALL 17 entries: 0 flagged, **cos=1.000000, rel=2.0e-04**; dL/dx0 (positions):
  **cos=1.000000, rel=2.1e-05**.
- **Smoke train (V2-style, n_batch=10): 2.0–2.7 s/update vs 127 s eager (~55×)**, first update
  ~23 s (cold eager solve + jit compile).

**Ops note / data incident:** the env/ package reorg changed `env.__name__` →
`train.py`'s `env_tag` became `env.quadrotor`, so the eager lr=3e-3 retrain wrote
`train_env.quadrotor_*.csv`, which a fused smoke run then overwrote. Recovered all 59 eager
updates from the run log → `results/data/train_quadrotor_plan_barrier_seed0_lr3e-3_eager_partial.csv`;
`env_tag` fixed to the module basename. (lr=1e-2 archive `..._lr1e-2.csv` intact.)

The eager V2 lr=3e-3 retrain was killed at update 59 (user decision) and relaunched on the fused
path (100 updates, same seed/config).

## ⏭️ NEXT TASK — barrier V2/V4 on the GRAZING quadrotor + write-up

1. **Hard-vs-barrier (V2/V4) — now testable.** The quadrotor's realized trajectory GRAZES (real
   active-set switches), so this is finally the regime for the hard-vs-smoothed-complementarity comparison.
   Bring in `make_barrier_layer` (central-path `make_central_path_diff`, diff w.r.t. weights+`x₀`).
   **But** per the corrected finding: the hard gradient is exact a.e., so the barrier's payoff is expected
   in the SMOOTHNESS of the weight→cost map / optimization, NOT in fixing a "wrong" hard gradient. Compare
   {V1,V3} × {hard,barrier} on the same grazing task (eval, grad/update stability across switches, κ bias).
2. **Multi-seed V3@h=24** for error bars (h=24 matches V1; only seed 0 shown so far).
3. **Write up** the V1-vs-V3 / truncation finding into `notes/NOTE.md` (RQ1/RQ3) + a quadrotor `RESULTS.md`. The
   central message: with the EXACT (sign-corrected, full-Hessian) gradient, the V1-vs-V3 outcome is
   governed by the estimator's effective HORIZON (truncation), NOT by active-set non-smoothness — the
   *opposite* of the linear-drone-era "hard jumps destabilize BPTT" framing.

Run env: `external/diffmpc2` on PYTHONPATH (CANONICAL — see 07-02 update), `.venv-cudss` python, cuDSS backends, QP 1e-6 / NLP 1e-3 (hard);
training lr=1e-2 (V1) / lr=3e-3 (V3 — needed for h≥16 to converge). The local central-path backward
(`src/diffmpc_learning`) is correct & immune to the diffmpc2 multiplier-sign bug (one-sided rows + smooth
complementarity weight `W = y_g/(s + y_g/γ)`, never a two-sided hard active-set).

---

## 1. Project context

**"Gradient Quality of Differentiable NMPC."** We study the quality of gradients obtained by differentiating through an NMPC solver (the **TurboMPC / diffmpc2** solver — SQP + ADMM, GPU linear solvers, KKT-implicit `custom_vjp` gradients), and how to improve it. The differentiable parameter is the MPC cost-weight vector (diffmpc-as-policy). Four research questions (RQ1 gradient reliability under general inequalities; RQ2 cheaper solves / ADMM-vs-IPM; RQ3 RL paradigm; RQ4 deploy-time tuning) — see `notes/NOTE.md`.

**The thread this session pursued (RQ1/RQ2):** inequality **active-set changes** make the diffmpc gradient non-smooth/jumpy (strict-complementarity failure → singular KKT). The fix in the literature (Frey/Diehl, `reference_papers/Diehl_DiffNMPC.pdf`, = `[Frey2025]`) is **interior-point central-path smoothing**: relax complementarity `s·z = 0 → s·z = τ_min`, giving a C¹ solution map with O(τ) bias. The sibling project `external/PrismQP/` does the *same* smoothing but with **ADMM instead of Newton**, via a log-barrier **retraction** in the slack/dual update. **This session built and verified that same central-path/retraction ADMM for the OCP-structured (TurboMPC) setting**, as the foundation for getting smooth gradients through inequality-constrained NMPC.

**Repo layout (root = `/home/jianghan/Workspace/diffmpc-learning/`):**
```
├── HANDOFF.md                  ← this file
├── CLAUDE.md                   ← workspace instructions (read for conventions)
├── pyproject.toml, Makefile    ← NEW: build for the src/ package (TurboMPC-style)
├── src/diffmpc_learning/       ← NEW: the project's solver package (see §3)
├── tests/{python,cuda}/        ← NEW: package tests (11 pass)
├── 
│   ├── notes/NOTE.md, notes/REFERENCES.md          ← research log + cited bibliography
│   ├── notes/SLACK_PENALTY_ADMM.md           ← NEW: ADMM slack-penalty derivation note
│   ├── notes/CARTPOLE_COUPLING_HANDOFF.md    ← cuDSS/cartpole env + earlier findings
│   ├── plans/                          ← NEW: the two implementation plans executed this session
│   └── experiments/{cartpole,quadrotor}/  ← benchmarks + results
├── diffmpc2/                   ← the TurboMPC solver (GITIGNORED — vendored dependency; see §6 gotcha)
├── external/PrismQP/           ← dense-QP barrier-retraction solver (GITIGNORED, own repo)
└── reference_papers/           ← Diehl_DiffNMPC.pdf (untracked)
```

---

## 2. What this session did (overview)

1. **Workspace consolidation & repo init.** Removed the `diffmpc2-gradckpt` git worktree; consolidated the solver to a single `diffmpc2/` on `release-cleanup` (carries the backward-Hessian fix `cccba81`). Repaired the cartpole benchmark (path → `diffmpc2/`, optional-cuDSS-import). Created the **private** GitHub repo `JianghanZHang/diffmpc-learning` and pushed `main`.
2. **Research-log update.** Folded the 2026-06-18 cartpole/quadrotor coupling findings into `notes/NOTE.md`: per-sample convergence-checked-FD methodology, the **backward dynamics-Hessian magnitude bug** (`dt·λᵀ∇²f` Euler hardcode) found via an **acados exact-Hessian cross-check** and fixed in `cccba81` (rel-ℓ₂ ~0.16→~5e-5), and the nonconvex-discontinuity diagnosis.
3. **Theory grounding.** Read `[Frey2025]` and PrismQP; wrote `notes/SLACK_PENALTY_ADMM.md` (how TurboMPC's quadratic slack relates to the log-barrier; the indicator → Moreau-envelope → log-barrier picture; closed-form elastic retraction `s = b_Γ(r)`, `Γ = κ(1/ρ+1/γ)`, `ξ = κ/(γs)`).
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

**⚠️ Critical gotcha — the cuDSS build depends on an UNCOMMITTED revert in `diffmpc2/`.** Installed cuDSS is **0.7.1.6**, but `diffmpc2/`'s `release-cleanup` ships the cuDSS-**0.8** API (commit `e19a687`, broken on sm_120). The 3 `.cu` files are reverted to the 0.7.1 API as an **uncommitted working-tree change** in `diffmpc2/` (kept uncommitted so the vendored repo stays at origin) and the FFI is rebuilt into `diffmpc2/build/ffi/`. A `git reset --hard` / `checkout` in `diffmpc2/` will wipe this and break cuDSS. To restore (full recipe in `notes/CARTPOLE_COUPLING_HANDOFF.md` §2):
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
- **No speculation in docs**: cite (`[Key]` → `notes/REFERENCES.md`) or flag as conjecture; report only what was measured.

---

## 8. Open items / next steps (none started)

1. **κ-relaxed backward / VJP** — the actual gradient-quality payoff: differentiate the κ-relaxed retraction-KKT (PrismQP-style `diff_qp`) so the diffmpc-as-policy gradient is *smooth across active-set changes*. The forward solve (done) is the prerequisite; this is the next milestone.
2. **CUDA kernels** — implement the central-path forward (and/or backward) as cuDSS-fused CUDA in `src/diffmpc_learning/solvers/csrc/` (scaffold ready).
3. **Obstacle (one-sided, nonlinear) constraints** — the elastic retraction is already one-sided per row, so it maps directly to `1 − ‖p−c‖/r ≤ 0`; test gradient smoothness across obstacle activation (the original RQ1 motivation).
4. **Finish the branch** — merge / PR `central-path-admm` when ready (re-review the migration + SQP commits).
5. **Fold the NLP-KKT result into `notes/NOTE.md`** (RQ2: ADMM central-path converges the NLP-KKT, matches TurboMPC) — not yet logged there.
