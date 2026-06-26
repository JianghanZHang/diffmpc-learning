# Gradient Quality of Differentiable NMPC — Research Note

**Status:** living research log · **Started:** 2026-06-15 · **Solver:** diffmpc2 / TurboMPC
(arXiv:2510.06179) · **Primary testbed:** drone obstacle avoidance (general inequalities).

This is an internal, evolving note. Each research question (RQ) section is structured as
**Setup → Established → Partially answered → Open → Hypotheses → Experiments**, where
*Established / Partial / Open* are anchored to the literature in `REFERENCES.md` (citation
keys like `[Suh2022]`). The standing rule for this project: **claims are cited or flagged as
conjecture — no speculation dressed as fact.** Where the literature is silent, we say so and
mark it as a candidate contribution.

> Reproducibility of the survey: the literature base was built on 2026-06-15 by five parallel
> research agents, each verifying claims against primary sources; per-source caveats are in
> `REFERENCES.md`. Re-run / extend before relying on any single load-bearing claim.
>
> **Experiments:** runnable code, results, and figures live in `experiments/`
> (`experiments/quadrotor/results/RESULTS.md` is the per-run record).
>
> **Path convention:** project paths (`NOTE.md`, `REFERENCES.md`, `experiments/…`) are relative to
> this directory (`research/gradient-quality-diffnmpc/`); solver code and example paths
> (`diffmpc/…`, `examples/…`) are relative to the **`diffmpc2/`** checkout, a sibling of `research/`
> in the workspace root.

---

## 0. Project framing

**Object of study.** A control policy of the form

```
obs ──[ NN θ ]──► MPC problem parameters p  ──[ diffmpc2 solver ]──► action u*(p)
                  (cost weights, references,        SQP + ADMM,
                   constraint margins, …)           KKT-implicit gradients
```

The MPC solver is *inside* the differentiable computation graph. We can obtain
`∂u*/∂p` (hence `∂L/∂θ`) by differentiating through the solver. The central concern of this
project is **the quality of that gradient** — its bias, variance, and usefulness — as a
function of (i) the presence and activity of **general inequality constraints**, (ii) the
**solver tolerance / solver type**, and how that gradient quality determines (iii) which **RL
paradigm** trains a diffmpc-as-policy best, and (iv) whether **deploy-time gradient-based
tuning** is feasible.

**Why diffmpc2 is the right vehicle.** It already implements the exact machinery the four RQs
need (see §1), and its GPU SQP+ADMM throughput is precisely the lever that several open
problems in the literature are bottlenecked on (per-step solve+backward cost; large-batch RL
through a real solver). The solver's existence turns several "open frontiers" into things we
can actually measure.

**The four research questions.**

1. **RQ1 — Reliability of backward gradients with general inequalities.** Under which
   conditions does the diffmpc2 backward pass yield gradients that are *correct enough* to
   train policies with RL, particularly when general (nonlinear, state-dependent) inequality
   constraints activate and deactivate?
2. **RQ2 — Cheaper solves.** Can we get away with a *lower NLP convergence tolerance*? Can a
   first-order solver (ADMM) substitute for an interior-point (IPM) solver *for the purpose of
   gradient quality*?
3. **RQ3 — Most efficient RL paradigm for diffmpc-as-policy.** Among (a) model-free RL tuning
   of MPC, (b) MPC-as-Q/value approximator, (c) end-to-end first-order analytic gradients, and
   the (d) pure-model-free no-MPC baseline, which trains a diffmpc-as-policy most efficiently?
4. **RQ4 — Real-time deploy-time auto-tuning.** Can we adapt MPC parameters online (while
   running, ideally on hardware) using gradients from the differentiable solver?

---

## 1. How diffmpc2 computes gradients (grounded in code)

A precise read of the solver, because every RQ depends on these mechanics.

- **Forward.** `SQPADMMSolver.solve` (`diffmpc/solvers/sqp_admm.py:1372`) runs SQP (outer,
  default `num_scp_iteration_max: 15`), and each QP subproblem is solved by **ADMM**
  (`diffmpc/solvers/admm/`), with the ADMM linear system handled by a Schur/PCG/cuDSS backend
  (`ForwardBackend` enum, `sqp_admm.py:46`). Default tolerances
  (`diffmpc/solvers/params/sqp_admm.yaml`): `tol_convergence 1e-6`, ADMM `eps_abs/eps_rel 1e-9`,
  ADMM `max_iter 100`. **This is a tight solve by default** — RQ2 is about deliberately loosening
  these.
- **Backward.** The solve is wrapped in **`jax.custom_vjp`** (`sqp_admm.py:432`). The VJP
  (`_solve_bwd`, `sqp_admm.py:1078`) is **implicit differentiation of the converged KKT system**,
  with two backends (`BackwardBackend`, `sqp_admm.py:55`):
  - `_solve_bwd_direct` (`sqp_admm.py:830`) — assembles and solves the full saddle-point KKT
    system directly (`backward/backward_kkt_*.py`):
    `[[P, Cᵀ],[C, 0]] [dx; dy] = [-q; 0]`.
  - `_solve_bwd_admm` (`sqp_admm.py:613`) — re-uses an ADMM solve for the backward linear system.
  This is exactly the **OptNet/diff-MPC fixed-point construction** `[AmosKolter2017][Amos2018]`:
  the backward solves a linear system in the KKT matrix of the converged QP.
- **What is differentiable.** `solve` does `problem_params = jax.lax.stop_gradient(problem_params)`
  (`sqp_admm.py:1378`); gradients flow **only through the `weights` argument** (merged into the
  params after the stop-gradient). So in diffmpc2 the differentiable MPC parameter *is the cost
  weight vector* — which is exactly the diffmpc-as-policy gradient (RQ3) and what the gradient-quality
  study differentiates.
- **General inequalities.** `OptimalControlProblem` (`optimal_control_problem.py`) exposes
  `inequality_constraints` (`:375`), linearized per SQP iteration (`get_inequalities_linearized_matrices`).
  The obstacle constraint is `1 − ‖p−cᵢ‖/(rᵢ+ε) ≤ 0` (`:1305`) — **nonconvex, state-dependent**.
  General nonlinear inequalities can be reformulated with **slack variables** via
  `SlackProblemAdapter` (`:1414`, `use_slack_variables`).
- **Testbed.** Drone obstacle avoidance (`examples/drone_obstacles/`); knobs exposed in
  `make_drone_config` / `benchmark_naming.py`: `use_slack`, `admm_tol`, `scp_iter`, `admm_max_iter`,
  `umax`, obstacles on/off.
- **diffmpc-as-policy already exists.** `notebooks/pointmass_rl/policy.py`: an MLP maps `obs` to
  **per-stage log-weight multipliers** that scale the MPC cost; training (`apg/`) is **Analytic
  Policy Gradient** — `jax.value_and_grad` through the differentiable MPC rollout + Adam.

**Key implication.** diffmpc2's gradient is an **implicit KKT gradient at the (approximately)
converged solution**, computed with a **first-order (ADMM)** forward solver. RQ1 is about that KKT
gradient under inequality activity; RQ2 is about how the ADMM tolerance and the direct-vs-ADMM
backward backend affect it; the solver has **no IPM forward**, so any "ADMM vs IPM" comparison (RQ2)
requires an external IPM baseline (e.g. acados/`[Frey2025]`, OptNet).

---

## RQ1 — Reliable backward gradients with general inequalities

### Setup
Obstacle constraints activate and deactivate along the trajectory. At each such switch the MPC
solution map is non-smooth; the question is whether the implicit gradient remains trustworthy and
useful for RL training.

### Established (literature is clean)
- **Smooth-case conditions are exact and known.** Under **LICQ + strict complementarity + SOSC**,
  the primal-dual solution is a **C¹ function** of the parameters and the implicit-KKT Jacobian is
  the true derivative `[Fiacco1983]`. Dropping strict complementarity (keeping LICQ + SSOSC) gives a
  **Lipschitz**, directionally-differentiable map (Robinson strong regularity) `[Robinson1980]`,
  with the directional derivative computable from an auxiliary QP `[Pacaud2025]`.
- **The failure mode is precisely characterized.** At a **weakly active** constraint (λᵢ = 0 *and*
  constraint tight), strict complementarity fails, the **KKT matrix used for the backward pass is
  singular**, and the gradient is non-unique / one-sided `[AmosKolter2017][Barratt2018][Magoon2024]`.
  These bad points form a **measure-zero set**; the map is differentiable almost everywhere
  `[AmosKolter2017]`.
- **Structure is piecewise-affine (for QP subproblems).** The parametric-QP optimizer is continuous
  PWA over polyhedral critical regions, with kinks exactly where the active set changes; crossing a
  region facet flips **one** constraint `[Bemporad2002][Tondel2003]`. For NMPC this is the per-SQP
  local picture.
- **Nonsmooth autodiff has a rigorous theory.** Conservative Jacobians / path differentiability:
  backprop yields a conservative field, and SGD with such gradients converges to Clarke-critical
  points `[BoltePauwels2021]`; the iterative-solver version covers **ADMM** `[Bolte2022iter]`.
- **Inexact inner solves corrupt the implicit gradient.** `[Amos2018]`: differentiating an iterate
  that is *not a fixed point* "will usually give the wrong gradients." (Bridges to RQ2.)
- **For MPC specifically, the modern reference is `[Frey2025]`:** non-differentiable / jumping
  solution map at active-set changes; interior-point smoothing as the fix.

### Partially answered
- **What a backprop layer *should return* at an active-set change** is not settled (heuristic in
  cvxpylayers `[Agrawal2019]`; IP smoothing in `[Frey2025]`; barrier in `[Jin2021safepdp]`).
- **Gradients through general inequalities in MPC** are handled by *different* devices, each with
  its own bias/accuracy trade-off — **no unified theory of gradient quality across these choices.**

### Open (candidate contributions)
1. **A controlled empirical curve of gradient error vs. constraint activity** — the agents found
   **no paper presenting this** for MPC/QP layers. *(Addressed in part by E1.2 below.)*
2. **Gradient quality of the slack-variable reformulation** (the diffmpc2 path) — **no paper analyzes
   this.**
3. **Joint effect of inexact SQP+ADMM solves *and* active-set-change non-smoothness** — **no combined
   bound** for a GPU SQP+ADMM differentiable-MPC pipeline. *(E1.2/E2.1 begin to probe the interaction.)*
4. **Near-degeneracy** (tiny multipliers / nearly-active constraints) — acknowledged but not
   quantitatively mapped.

### Working hypotheses
- **H1.1** Implicit gradient matches a tight-solve / finite-difference reference except near active-set
  switches and near-degeneracy, where error spikes. *(Predicted by `[AmosKolter2017][Magoon2024]`.)*
- **H1.2** Slack reformulation smooths the gradient near activation (barrier-like), reducing the spike at
  the cost of controllable bias. *(Conjecture; candidate contribution #2.)*
- **H1.3** Because the bad set is measure-zero and SGD avoids it w.p. 1 `[BoltePauwels2021]`, *training*
  is robust to the kinks; the practical risk is variance/conditioning near degeneracy.

### Experiments (E1.x)
> **Run 2026-06-16 — E1.2 (+ E2.1):** `experiments/quadrotor/gradient_quality_sweep.py`;
> full record in `experiments/quadrotor/results/RESULTS.md` (data: `experiments/quadrotor/results/grad_quality_20260616_164624.{csv,png}`).
> 2×2 grid (control loose/tight × obstacles off/on), slack always on, GPU, n_seeds=4; reference =
> tightest affordable solve; gradient compared to autodiff-at-reference and finite differences.
>
> - **H1.1 confirmed, with a sharper distinction.** `cos(g_AD*, g_FD*)` at the reference: C1 (0 active) =
>   **1.0000**, C2 (**26 active linear box** constraints) = **1.0000**, C3 (5.8 active **obstacle**) =
>   **0.9977**, C4 (37 box + 6.8 obstacle) = **0.9956**. → Many active *linear box* constraints keep the
>   converged implicit gradient *exact*; the degradation is specific to **general/nonlinear (obstacle)
>   inequalities**, not active constraints per se. This is the box-vs-nonlinear refinement of the
>   literature's active-set pathology.
> - **H1.3 supported.** The gradient remained a **descent direction** (⟨g, g_ref⟩ > 0) in *every* cell and
>   *every* swept config, including the loosest solves (tol 0.1, ADMM 5).
> - **New finding (RQ1↔RQ2 coupling).** The obstacle NLP only converges to ~1e-3 (vs ~5e-7 for box-only)
>   within a practical SQP budget, so obstacle gradient error is **entangled with non-convergence**;
>   tightening the reference SQP 18→30 moved C3 `cos(AD,FD)` 0.9912→0.9977, but a **residual gap persists**.
> - **H1.2 (slack smoothing) not yet isolated** — slack was on in all cells; a slack-on/off ablation is the
>   next step.

> **Run 2026-06-18 — E1.6 (+E2.5): per-sample gradient accuracy on *smooth* systems, closed-loop coupling
> K × NLP tolerance.** Cartpole swing-up (nx=4) and quadrotor-about-hover (nx=13), **no inequality
> constraints** (`umax=1e7`), K∈{1,5,20}, set-tol∈{1e1…1e-9}, cuDSS FFI backends (`admm_fused_cudss` fwd /
> `direct_cudss_ffi` bwd), x64, GPU. diffmpc-as-policy gradient ∂(K-step closed-loop cost)/∂(Q,R).
> **Methodology change (per CLAUDE.md FD rule):** each of 128 initial states = **one sample**; ground truth
> = **convergence-checked per-sample FD** (shrink eps over a decreasing sequence until the estimate plateaus;
> samples with no plateau are **flagged as discontinuities and excluded from the cos score**, never
> batch-summed). Records: `experiments/cartpole/results/cartpole_sweep.md`,
> `experiments/quadrotor/results/quadrotor_sweep.md`, `experiments/cartpole/results/acados_comparison.md`;
> full diagnosis in `CARTPOLE_COUPLING_HANDOFF.md` §8–9.
>
> - **H1.1 confirmed sharply: AD = FD(eps→0) away from measure-zero jumps.** On both smooth systems, per-sample
>   `cos(AD, FD_GT)` median = **1.0000 at every (K, tol)** with a converged forward, **tolerance-insensitive**
>   (cos≈1.0 even at set-tol 1e-1, forward KKT~5e-2) and **coupling-robust through K=20**. Direction is correct
>   across the whole grid.
> - **Backward *magnitude* bug found and fixed (infrastructure correction; affects all prior AD-vs-FD magnitude
>   numbers).** Pre-fix, AD direction was right but **magnitude was systematically low**, with rel-ℓ₂ floored at
>   **~0.16 (cartpole) / ~0.05 (quadrotor)** at *every* converged tolerance — a tolerance-independent floor, so
>   a *backward* error, not forward non-convergence. An **independent acados exact-Hessian parametric adjoint**
>   on the 108/128 same-solution samples pinned it: acados-exact vs converged-FD agree to **relmag = 1.0000**,
>   while turbompc-AD vs FD was **relmag = 0.837** (≈16% low) → a turbompc backward magnitude error
>   (`acados_comparison.md`). **Cause:** the backward dynamics-Lagrangian Hessian had hardcoded the **Euler**
>   form `D_t += dt·λᵀ∇²f` for *all* explicit integrators, dropping RK4 stage curvature. **Fix** (commit
>   `cccba81`, on the solver's `release-cleanup` branch, present in `diffmpc2/`): the explicit branch now differentiates
>   `λᵀ∇²(predict_next_state)` of the *configured* integrator (`diffmpc2/turbompc/problems/optimal_control_problem.py:1418`,
>   `single_hessian_discrete`; the Euler-only `:1343–1345` docstring is stale). Post-fix, rel-ℓ₂ median collapses
>   to **~5e-5 (cartpole) / ~5e-4 (quadrotor)** at every converged tol, and it also resolves the only two
>   quadrotor K=1 direction misses (cos_min 0.984→0.9999999, #cos<0.99 2→0 — those were the largest-gradient
>   samples where the ~5% magnitude error had tipped cos below 0.99).
> - **Nonconvex argmin-jumps are real, measure-zero, and *not* an inequality effect (RQ1 refinement).** Exactly
>   **1/128** cartpole samples (idx 114, K=20, all tols) gave cos(AD, FD@1e-5) = **−0.56**. Fully diagnosed (all
>   steps measured, `CARTPOLE_COUPLING_HANDOFF.md` §8): **not** an AD error and **not** under-convergence
>   (persists with SQP ceiling 150→600, every solve KKT~1e-9) — an **FD step-size artifact at a nonconvex
>   local-min basin boundary**. A ~2e-4 *state* difference propagated by the K=20 rollout straddles a basin
>   boundary at a mid-rollout solve, so the w±eps rollouts land in **distinct converged minima** (u₀≈−1.2 vs
>   −33) and `‖cost_B−cost_A‖/2eps`≈1e8 is meaningless; AD takes the eps→0 limit (FD@1e-7→4.6e4 ≈ ‖g_AD‖=4.05e4)
>   and returns the true within-branch gradient. → The diffmpc-as-policy cost map is **piecewise-smooth with
>   measure-zero jumps** — the solver's *selected* local minimum is discontinuous in the weights **even with no
>   inequality constraints** (a distinct mechanism from the active-set switches H1.1 names). AD is reliable a.e.;
>   FD validation needs adaptive per-sample eps; such points are **labelled, not scored** (consistent with the
>   measure-zero set of `[BoltePauwels2021]`, H1.3).
> - **Caveat — this re-opens E1.2's obstacle *magnitude* numbers.** E1.2's rel-ℓ₂ compared AD-at-reference to
>   AD-swept (the backward bias cancels) and its only AD-vs-FD metric was the *cosine* (direction), so the ~16%
>   magnitude bias was **invisible to E1.2**; E1.2 also ran **pre-`cccba81`**. The drone obstacle AD-vs-FD
>   *magnitude* is therefore **unmeasured** — re-run E1.2 with per-sample adaptive FD on the fixed backward
>   before trusting it. (E1.2's direction finding F1 and the obstacle non-convergence entanglement F2 are
>   independent of this fix and stand.)

> **Built 2026-06-23 — κ-relaxed (central-path) backward / VJP for the inequality-constrained NMPC
> layer (RQ1 candidate-contribution #2; H1.2 enabler).** New solver package `src/diffmpc_learning/`
> (branch `central-path-admm`): a central-path ADMM forward (closed-form elastic log-barrier
> retraction, cuDSS Schur) + SQP outer loop, and now a **κ-relaxed retraction-KKT backward**
> (`solvers/backward.py`). It differentiates the *same* relaxed fixed point the forward converges to:
> per one-sided row, relaxed complementarity `s·y_g = κ` with barrier slack `s = h − Gx + y_g/γ`,
> folded into the reduced symmetric KKT via the smooth per-row weight `W = y_g/(s + y_g/γ)` — the C¹
> analogue of TurboMPC's hard active-set mask (inactive `W→0`, soft-active `W→γ`, hard-active `W→∞`).
> Exact Lagrangian Hessian reused (`D + λᵀ∇²f` via `get_dynamics_lagrangian_hessian`, the post-`cccba81`
> term). Tests: `tests/python/solvers/test_backward_central_path.py` (5, GPU/cuDSS, x64); a whole-branch
> adversarial review re-derived `W` from the IFT and confirmed every sign (sign-flip/zeroing probes
> rejected by the FD tests).
>
> - **Correctness verified against two independent ground truths at a converged NLP-KKT (cartpole swing-up).**
>   (i) QP-level VJP vs convergence-checked FD on the bounded one-sided QP (active control bounds → `W`
>   exercised): rel-ℓ∞ < 1e-4 on `dL/dq`, `dL/dD`, `dL/dE`. (ii) **Interior** NLP (`umax=50`, bounds present
>   but inactive → `W≈9e-11`): the relaxed backward matches the **unrelaxed TurboMPC** backward to `cos=1.0`,
>   `rel-ℓ₂≈2e-8`, and convergence-checked FD (`rel-ℓ₂<1e-3`) — pinning the exact-Hessian construction and
>   all multiplier signs. (iii) **Bounded** NLP (`umax=2`, control bound active): `dL/dweights` matches
>   convergence-checked FD of the *same* relaxed solver (`cos>1−1e-5`, `rel-ℓ₂<2e-3`) at **both** κ regimes
>   (annealed→1e-6 and fixed 1e-4); a per-eps sweep confirms **FD(eps→0) = AD to 7 digits** on the
>   active-bound `dL/dR` (the apparent gap at coarse eps was an FD truncation artifact, not a backward bias
>   — CLAUDE.md FD caveat). [Differentiable parameter = cost weights (Q,R); loss = pole-up tracking.]
> - **What this establishes vs. what it does not (per CLAUDE.md, report-only-what-was-measured).**
>   *Established:* the κ-relaxed backward is a **correct** gradient of the relaxed solver (= FD where the map
>   is C¹) and **reduces to the exact unrelaxed gradient in the interior** (= TurboMPC). *Not yet measured:*
>   that the relaxed gradient is **smoother / better-conditioned across an active-set boundary** than the hard
>   backward — that is the design rationale (H1.2; IP-smoothing `[Frey2025]`) and the **next experiment**.
>   No RL-training, variance, or obstacle-constraint claim is made yet.

Remaining E1.x: **E1.3** slack on/off ablation (test H1.2; the central-path backward is now the tool for it);
**E1.4** near-degeneracy probe (KKT conditioning); **E1.5** training-time robustness (do active-set crossings
spike loss/grad-norm?); **E1.7** across-active-set smoothness of the κ-relaxed backward (relaxed-AD vs
hard-AD vs FD continuity through an activation) — the not-yet-measured H1.2 payoff above.

---

## RQ2 — Lower NLP tolerance? ADMM vs IPM?

### Setup
Tight solves are expensive. We want the loosest solve that still yields gradients good enough to
*train*. Separately: does the choice of solver class (ADMM vs IPM) matter for gradient quality?

### Established
- **Implicit fixed-point gradients are invariant to inner-iteration count once converged**
  `[ButlerKwon2021][Agrawal2019]`.
- **Gradient bias from an inexact solve is controlled and small:** linear in the residual
  `[Blondel2022]` (= O(tolerance) `[Pedregosa2016]`, geometric in iterations `[Grazzi2020]`); iterate
  suboptimality ≠ gradient suboptimality `[Scieur2022]`.
- **A loosely-converged solve still trains:** descent direction preserved `[Fung2022jfb][Geng2021phantom]`;
  warm-started loose solves match exact-oracle complexity `[ArbelMairal2022]`; Alt-Diff residual bound
  `[Sun2023altdiff]`.
- **The IPM barrier *smooths* the gradient** at active constraints, decoupled from solve tolerance
  `[TracyManchester2024][Howell2022dojo][Jin2021safepdp][Frey2025]`.
- **RTI = deliberately inexact** (one SQP iteration/sample) `[Diehl2005rti][Verschueren2021acados]`.

### Open (candidate contributions)
1. **The ADMM-vs-IPM gradient-quality comparison *for MPC* is essentially unanswered** — existing
   comparisons measure speed/duality-gap, not gradient bias, in generic-ML settings
   `[Sun2023altdiff][ButlerKwon2021][Magoon2024][BPQP2024]`. **The project's clearest white space.**
2. **No residual/tolerance-explicit gradient-error bound for the *implicit ADMM* QP layer** inside
   SQP-MPC.
3. **Differentiating *through* an RTI / single-SQP solve as a learning layer is unaddressed.**
4. **Descent guarantees are existence results, not actionable thresholds** — we can produce an empirical
   "minimum solve for usable training gradient" law.

### Working hypotheses
- **H2.1** Gradient cosine to the reference stays > 0.99 down to a surprisingly loose tolerance, then
  degrades — a usable "knee." *(Predicted shape `[Blondel2022][Grazzi2020]`; magnitude unknown.)*
- **H2.2** Direct-KKT and ADMM backward agree when the forward solve is converged; ADMM backward degrades
  faster as the forward solve loosens. *(Untested — single backward backend so far.)*
- **H2.3** Loosening tolerance biases magnitude but preserves the descent direction, so training reaches
  comparable quality at far less solve cost `[Fung2022jfb][Geng2021phantom]`.
- **H2.4** Against an external IPM baseline, ADMM-implicit gradients are as accurate once converged but
  less smooth across active-set changes. *(Requires an IPM baseline; not yet run.)*

### Experiments (E2.x)
> **Run 2026-06-16 — E2.1 (tolerance + ADMM-iteration sweeps), same driver/data as E1.2.**
>
> - **H2.1 / H2.3 supported, with constraint-type dependence.** For constraint-free and **box-only** cells,
>   the gradient **direction** (cosine) stays ≈ 1.0 down to tol 1e-1 while only the **magnitude** (rel-ℓ₂)
>   degrades (≈0.8 at tol 0.1 → ~1e-5 tight), and the **descent direction is preserved everywhere** — i.e.
>   a loose solve gives a usable training gradient. For the worst cell (C4, box+obstacle) cosine drops to
>   **0.899** at tol 0.1, so the safe-tolerance margin *shrinks with constraint activity*.
> - **New finding — ADMM-iteration dependence is monotone for box, non-monotone for obstacles.** Box-only
>   rel-ℓ₂ decreases monotonically with ADMM `max_iter`; obstacle cells are **non-monotone** (C3:
>   ADMM 50 → 4e-4 but ADMM 100 → 0.14), i.e. the identified active set flips between ADMM budgets —
>   direct evidence of active-set-change instability under a first-order solver. *(Relevant to "can we
>   get away with ADMM?": yes for box, with care for general inequalities.)*
> - **Caveat.** Obstacle rel-ℓ₂ *plateaus* (can't beat the reference, which itself sits at conv ~1e-3),
>   so obstacle tolerance curves are entangled with non-convergence (see RQ1 E1.2 finding).

> **Run 2026-06-18 — E2.5 (smooth-system tolerance × coupling; same data as E1.6).** On the *no-inequality*
> cartpole and quadrotor (per-sample, convergence-checked FD GT, post-`cccba81` backward), the tolerance
> picture is clean and confirms **H2.1/H2.3** without the obstacle entanglement:
> - **Direction is tolerance-insensitive.** Per-sample cos median = 1.0000 from set-tol 1e-1 down to 1e-9 (and
>   at 1e0 cos median is still 1.0000). Only at **tol 1e1** — where the forward barely starts (achieved
>   KKT~7–8) — does direction break (quadrotor K=20: cos median **−0.006**, rel-ℓ₂ max ~2e4). So the usable
>   "knee" for a *smooth* NLP is ~tol 1e0–1e-1, looser than the obstacle cells tolerate.
> - **Residual magnitude error at loose-but-converged tol is *forward* under-convergence, not backward.** Post-fix,
>   rel-ℓ₂ at tol 1e0 ≈ 8e-3 (cartpole K=1) / 7e-3 (quad K=20), shrinking to the ~5e-5 / ~5e-4 FD floor by tol
>   1e-3 — i.e. differentiating a not-yet-fixed-point `[Amos2018]`, decaying with the residual `[Blondel2022]`;
>   the backward itself is now exact (E1.6 fix).
> - **Coupling-robust.** The same picture holds at K=1, 5, 20 — chaining up to 20 closed-loop solves does not
>   degrade direction on these smooth systems (the `[Metz2021]` chaos regime is not triggered here).
> Caveat: smooth systems only (no general inequalities); the obstacle ADMM/tolerance entanglement (E2.1 F2/F4)
> is separate and unchanged.

Remaining E2.x: **E2.2** direct-KKT vs ADMM backward (H2.2); **E2.3** downstream training under loose
solves; **E2.4** ADMM-vs-IPM head-to-head against an external IPM baseline (H2.4).

---

## RQ3 — Most efficient RL paradigm for diffmpc-as-policy

### Setup
The policy is "NN → MPC problem → action." We compare four families (per project scope): **(A)** model-free
RL that tunes the MPC (Gros–Zanon), **(B)** MPC as a Q/value function approximator, **(C)** end-to-end
first-order analytic gradients through the solver (APG/SHAC; diffmpc2's native mode), and the **(D)** pure
model-free baseline with **no MPC**.

### Established
- **Representability.** A parameterized MPC can exactly represent **π⋆, V⋆, Q⋆** of the true MDP *even with
  a wrong model*, by adjusting the cost `[GrosZanon2020]`.
- **The differentiation machinery is exact** (implicit KKT) `[AmosKolter2017][Amos2018][Agrawal2019]`.
- **A working actor-critic-with-MPC-actor instance exists:** `[Romero2024acmpc]` — *slightly behind* the
  model-free baseline on sample efficiency but **far better on robustness/OOD**; bottlenecked by MPC solve
  cost (which a GPU solver attacks). Only baseline is PPO (no SAC/TD3).
- **Structured MPC policy > generic NN on data efficiency** in imitation `[Amos2018]`.
- **First-order analytic gradients are more sample-efficient than zeroth-order model-free on smooth
  problems** `[Xu2022shac][Wiedemann2023apg]`; model-based RL ~order-of-magnitude gains `[Janner2019mbpo]`.
- **The zeroth-vs-first-order trade-off is characterized** `[Suh2022]`: first-order wins on smooth
  landscapes but is biased at discontinuities and high-variance under stiffness/chaos — the regimes MPC's
  active-set switches create. Interpolation (α-order) is the hedge `[Suh2022][Parmas2018]`.

### Open (the crux of RQ3 is genuinely unsettled)
1. **No published head-to-head establishes a *differentiable*-MPC policy is more sample-efficient than
   SAC/TD3.**
2. **First-order (through-the-solver) vs zeroth-order/model-free for diffmpc-as-policy specifically is
   unresolved.** `[Suh2022]` predicts MPC's constraints are where first-order degrades — **our E1.2/E2.1
   results give the first concrete evidence** that the gradient degrades exactly in the obstacle (general
   inequality) regime, which is where a zeroth-order/α-order hedge should matter most.
3. **No consensus "most sample-efficient paradigm."**
4. **GPU SQP+ADMM in an RL loop is novel ground.**

### Working hypotheses
- **H3.1** On smooth/box regimes, first-order APG is most sample-efficient; on the **obstacle** regime,
  where E1.2/E2.1 show the gradient degrades and becomes non-monotone, a zeroth-order / DPG / α-order hedge
  becomes competitive `[Suh2022]`.
- **H3.2** SHAC-style short-horizon + learned critic `[Xu2022shac]` is needed to stabilize family-C training
  through obstacle-rich rollouts.
- **H3.3** Family (A)/(B) gives the best robustness/constraint-satisfaction `[Romero2024acmpc][ZanonGros2021]`;
  diffmpc2's GPU throughput shifts the wall-clock axis toward MPC-in-the-loop methods.

### Experiments (E3.x) — not yet started
**E3.1** paradigm bake-off (APG / SHAC-style / DPG-SAC / pure model-free) on drone obstacles +
spacecraft/quad; **E3.2** α-order hedge `[Suh2022]`; **E3.3** MPC-as-Q prototype; **E3.4** throughput study.

---

## RQ4 — Real-time deploy-time auto-tuning via diffmpc

### Setup
Adapt MPC parameters online while the controller runs, using gradients from the differentiable solver.

### Established
- **Offline / between-trial gradient tuning works**, incl. on hardware `[Cheng2024difftune][Tao2024difftunempc]`
  (`[Tao2024difftunempc]` flags `∂u/∂θ=0` when inequalities activate — consistent with our obstacle findings).
- **Online closed-loop RL-tuning with stability/safety theory exists — in simulation**
  `[Gros2022learning][ZanonGros2021]`.
- **Online *model* adaptation on hardware works** (gradient instance `[Nagabandi2019grbal]`) — adapts the
  *model*, not MPC params.
- **BO tuning is mature but episode-level/offline** `[Loquercio2022autotune][Berkenkamp2016safeopt]`.
- **RTI substrate** `[Diehl2005rti][Verschueren2021acados]`; surveys put gradient online-tuning as
  "possible but not yet established" `[Mesbah2022fusion][Reiter2025survey]`.

### Partially / Open
- A differentiable-MPC policy **running onboard real-time** exists but learns offline `[Sun2025gate]`;
  online MAML *model* adaptation in NMPC is sim-only `[Mei2025]`.
- **The full combo {online + on-hardware + gradient-through-solver + per-loop update + safety} is
  undemonstrated.** Blockers: vanishing/non-smooth gradients at active constraints (RQ1!), model-mismatch
  bias `[Cheng2024difftune]`, per-step compute, unproven online safety. `[Dinev2022ddp]`: iLQR-level
  gradients diverge the outer loop (second-order terms required).

### Working hypotheses
- **H4.1** diffmpc2's GPU solve+backward latency attacks the per-step-compute blocker.
- **H4.2** Online updates are stable only with smoothing / trust-region limits (raw gradient non-smooth at
  active-set changes per RQ1, biased at finite tolerance per RQ2). Our E1.2/E2.1 finding that obstacle
  gradients are non-monotone in ADMM iters reinforces that **online updates must guard the solve quality**.

### Experiments (E4.x) — not yet started
**E4.1** simulated online-tuning loop on drone obstacles; **E4.2** smoothing/trust-region ablation;
**E4.3** real-time feasibility budget (solve+backward latency at the loosest training-viable tolerance).

---

## 2. Cross-cutting white space (project contributions, ranked)

1. **Controlled gradient-error characterization for inequality-constrained MPC** — *first results in hand*
   (E1.2): the box-vs-nonlinear distinction and the activity ladder. Extend with slack ablation and a
   genuinely-tight obstacle reference.
2. **ADMM-vs-IPM gradient-quality comparison for MPC**, with downstream-training impact (E2.4 — needs IPM
   baseline).
3. **Empirical "minimum solve for usable training gradient" law** — E2.1 gives the first cut (descent
   preserved to very loose tolerance; safe margin shrinks with activity).
4. **Head-to-head of RL paradigms on the *same* diffmpc-as-policy benchmark** (E3.1).
5. **Online gradient-based MPC tuning through the solver** (E4.x).

## 3. Shared experiment infrastructure (built 2026-06-16)

- **Ground-truth gradient oracle** — (a) **per-sample convergence-checked finite differences** (shrink eps
  over a decreasing sequence until a plateau; flag non-plateau samples as discontinuities — CLAUDE.md FD rule)
  + tight autodiff reference; `experiments/quadrotor/gradient_quality_sweep.py` (open-loop, drone) and
  `experiments/cartpole/benchmark_cartpole_coupling.py` (closed-loop coupling, cartpole/quadrotor). (b) an
  **independent acados exact-Hessian parametric adjoint** as a second solver reference
  (`experiments/cartpole/acados_cartpole_gradient.py`, run in the `turbompc-acados` Docker).
- **Gradient-quality metrics** — cosine, relative ℓ₂, sign agreement, descent test, bias/variance over
  batches `[Suh2022]`; achieved convergence + active-constraint counts recorded.
- **Knob surface** — `tol_convergence`, SQP/ADMM iteration caps, ADMM tolerances, `use_slack_variables`,
  control bounds, obstacles on/off, forward/backward backends.
- **Testbeds** — point-mass (analytic sanity) → cartpole / quadrotor swing-up·hover (smooth, no-inequality;
  per-sample coupling × tolerance, E1.6/E2.5) → drone obstacles (general inequalities, main, E1.2/E2.1) →
  spacecraft (smooth RL). Reuse `examples/reinforcement_learning/.../apg/`.

## 4. Open decisions / parking lot
- External **IPM differentiable-MPC baseline** for E2.4: acados `[Frey2025]` vs OptNet vs mpc.pytorch.
  *Update (2026-06-18):* an **acados exact-Hessian parametric adjoint** is now wired up and validated as a
  gradient-correctness *reference* (cartpole cross-check, `acados_comparison.md`) — but that is an
  exact-Hessian *reference*, not yet the **ADMM-vs-IPM gradient-quality head-to-head** E2.4 needs (no
  tolerance/active-set sweep against acados yet).
- A better-converging obstacle SQP (more iters / trust region / globalization) to get a genuinely tight
  obstacle reference and cleanly separate active-set effect from non-convergence (RQ1 caveat).
- Whether RQ4 hardware deployment is in-scope this cycle (currently: simulation first).
- Critic architecture for SHAC-style family-C training (E3.1).

## 5. Changelog
- **2026-06-15** — Note created. Framing locked (drone-obstacle testbed; RQ3 scope = MPC-as-Q + pure
  model-free + multiple RL methods for MPC-as-policy). Literature review completed (5 agents, verified);
  `REFERENCES.md` written; codebase grounding done (§1).
- **2026-06-16** — Experiment infrastructure built (`experiments/`, `gradient_quality_sweep.py`,
  `plot_gradient_quality.py`). First run E1.2+E2.1 (drone obstacles, 2×2 activity grid × tolerance/ADMM
  sweeps, GPU). Findings (`experiments/quadrotor/results/RESULTS.md`): **linear box constraints keep the implicit gradient
  exact even when many are active; nonlinear obstacle constraints degrade it; ADMM-iteration dependence is
  monotone for box but non-monotone for obstacles; the descent direction survives even very loose solves.**
  H1.1/H1.3 confirmed; H2.1/H2.3 supported (constraint-type dependent). Caveat: obstacle NLP doesn't
  converge tightly, entangling obstacle gradient error with non-convergence. Recreated NOTE/REFERENCES/
  CLAUDE after a workspace reset dropped the first-phase docs.
- **2026-06-18** — Closed-loop coupling × NLP-tolerance benchmark on *smooth* (no-inequality) systems
  (cartpole swing-up, quadrotor-about-hover; cuDSS FFI backends), with a **per-sample convergence-checked FD**
  methodology (each x0 = one sample; flag non-plateau samples as discontinuities — CLAUDE.md FD rule). Records:
  `experiments/cartpole/results/cartpole_sweep.md`, `…/quadrotor/results/quadrotor_sweep.md`,
  `…/cartpole/results/acados_comparison.md`; diagnosis in `CARTPOLE_COUPLING_HANDOFF.md`. Findings (E1.6/E2.5):
  **on smooth systems the AD gradient direction is correct (cos=1.0) and tolerance-insensitive down to set-tol
  1e-1, coupling-robust through K=20.** Found + fixed a **backward dynamics-Hessian magnitude bug** (hardcoded
  Euler `dt·λᵀ∇²f` for all integrators → ~16% low magnitude on cartpole; commit `cccba81` differentiates the
  configured RK4 map → rel-ℓ₂ ~0.16→~5e-5), **independently confirmed by an acados exact-Hessian adjoint**
  (relmag 0.837 pre-fix → 1.0000 reference). Diagnosed the lone K=20 cos=−0.56 sample as an **FD step-size
  artifact at a nonconvex local-min basin boundary** (argmin jump; measure-zero; AD = FD(eps→0), reliable a.e.).
  Caveat logged: E1.2's obstacle *magnitude* numbers predate the fix and only ever measured direction vs FD →
  re-run with per-sample adaptive FD on the fixed backward. Repo housekeeping (same day): solver consolidated to
  a single `diffmpc2/` on `release-cleanup` (carries `cccba81`); the research workspace was put under its own git.
- **2026-06-23** — Built the **κ-relaxed (central-path) backward / VJP** for the inequality-constrained NMPC
  layer (new `src/diffmpc_learning/` package, branch `central-path-admm`; forward + NLP-KKT convergence were
  built earlier this session). The reduced retraction-KKT folds inequalities into the Hessian via the smooth
  weight `W = y_g/(s + y_g/γ)` (C¹ analogue of the hard active-set mask; reuses the exact `λᵀ∇²f` Lagrangian
  Hessian). Verified at a converged NLP-KKT on cartpole: QP-VJP vs FD (`dL/d{q,D,E}`, rel-ℓ∞<1e-4);
  **interior** NLP (`umax=50`, bounds inactive) AD == unrelaxed TurboMPC (`cos=1.0`, `rel-ℓ₂≈2e-8`) == FD;
  **bounded** NLP (`umax=2`, active) AD == FD of the *same* relaxed solver (`cos>1−1e-5`, `rel-ℓ₂<2e-3`) at
  both κ regimes (annealed→1e-6, fixed 1e-4), with a per-eps sweep confirming FD(eps→0)=AD to 7 digits.
  Whole-branch adversarial review (re-derived `W` from the IFT, sign-flip/zeroing probes) found **no gradient
  bug**; closed its flagged test-coverage gaps (interior W path, `dL/dE`, nontrivial-D guard). Establishes
  backward *correctness*; the across-active-set *smoothness* advantage (H1.2) is the next measurement (E1.7).
  Tests: `tests/python/solvers/test_backward_central_path.py` (5/5; full suite 14/14).
- **2026-06-24** — **E1.7 / H1.2 measured (RQ1): the hard active-set backward produces closed-loop gradient
  outliers; the soft/slack (incl. log-barrier) formulation eliminates them.** First made the κ-relaxed ADMM
  fast: ported TurboMPC's accelerated ADMM (over-relax α=1.6, OSQP adaptive ρ + Schur rebuild, residual stop)
  into `solve_qp_central_path` → **iteration parity 1:1** with TurboMPC (was ~100×; `test_central_path_speed.py`;
  the earlier "B ~100× worse / fragile" sweep result was a solver-tuning + step-norm-criterion artifact, now
  removed). Then the **50-step closed-loop** per-sample gradient-accuracy sweep ("as in diffmpc", 1 SQP iter on
  random linear MPC; built a custom_vjp B-solve differentiable w.r.t. weights **and** rollout state, FD-verified):
  **A** (TurboMPC **hard box**, the diffmpc default) is an outlier on **42/56 (75%)** samples across 3 systems
  (median cos 0.20–0.88, negative cosines, rel-ℓ₂ up to ~10³); at the faithful H=40 it is **non-differentiable**
  on 11/12 (FD won't plateau). **B** (log-barrier) has **~0** outliers (cos=1.0, rel-ℓ₂~1e-6 on every sample
  incl. all of A's failures). **Mechanism (measured, not assumed):** a κ-sweep shows B is outlier-free across
  κ∈[1e-11,1e-3] (insensitive); a direct control with TurboMPC's **own slack box** (`use_slack=True`) is *also*
  ~0 outliers → **the soft/slack forward (C¹) is the driver, not the barrier κ** (the hard active-set box is the
  pathological one; κ adds only marginal smoothing). **Generalizes to nonlinear** (cartpole regulation, nu=1):
  gated by active-set engagement — mild box (rarely binds) clean for both; **tight box** (umax=0.5, control
  saturates) → A-hardbox 8/8 non-differentiable (cos −0.81…+0.88), A-slack cos=1.0. Settles H1.2 (smoothing
  fixes the across-active-set gradient pathology *in closed loop*) and sharpens it: the fix is the soft-box
  *forward*. Artifacts: `experiments/linear_system/{closed_loop_accuracy.py,cl_b.py,cl_b_kappa_sweep.py,
  results/closed_loop_outliers.md}`, `experiments/cartpole/{closed_loop_cartpole.py,results/closed_loop_cartpole.md}`.
  Quadrotor (nu=4 nonlinear) closed-loop: **tractable; NO dramatic pathology in the hover regime.** (Two
  corrected false starts: "intractable" = GPU contention; "cos=0.616 outlier" = FD below the ~8e-8-rel
  cuDSS noise floor — fixed with eps≈1e-3; see [[isolate-gpu-contention-before-boundary]],
  [[fd-noise-floor-is-system-size-dependent]].) Final (noise-floor-corrected FD, 3 boxes umax 1.2/1.05/1.0):
  the hard box never drops below cos 0.989 and is *cleaner* at the tightest box (cos≥0.9998); slack is
  always slightly cleaner (a *weak* consistent directional match to the mechanism), but nothing like the
  linear/cartpole negative-cosine collapse. **Hypothesis (not measured):** the dramatic pathology needs
  active-set *switching*; at near-hover the thrust is continuously saturated → stable active set → few
  switches → smooth, even though the box binds firmly. So the pathology is **not universal** — it depends
  on the regime, not just nu. Clean/dramatic form stands on linear (nu=4) + cartpole (nu=1 nonlinear);
  the quadrotor adds "scales to 13 states" + this regime-dependence refinement. See
  `experiments/quadrotor/results/closed_loop_quadrotor_NOTE.md`. Still open: a switching-heavy nonlinear
  regime (aggressive tracking, not hover) to test the dramatic form at nu=4; B on nonlinear (multi-SQP
  custom_vjp); RL-impact (RQ3).
- **2026-06-25** — **Single-common-GT redesign + TRI-benchmark reproduction with a validated FD.** (a)
  Rebuilt the 4-variant linear closed-loop sweep against **one** common ground truth — the FD of the *true
  hard-constrained* loss, same 64 samples for all variants (`closed_loop_gradient_accuracy.py`,
  `results/gradient_accuracy.md`, B=64 H=160). Fixes the earlier per-config-GT apples-to-oranges. Findings:
  hard box & log-barrier are **faithful** (cos 1.0) once solved tightly; the **Moreau slack** gradient is
  biased from the hard-constrained gradient by the **relaxation, not a bug** — error scales **exactly
  O(1/γ)** (`gamma_scaling.py`: rel-err 22→0.0024 for γ=1e4→1e8, 10×/decade) and **compounds with horizon**
  (cos 0.9998→0.205 over 5→50 steps), so γ=1e4 needs ~1e7 for a faithful 50-step gradient. (b) The hard-box
  `DIRECT_CUDSS_FFI` backward has **systematic-but-rare outliers** (~4–12% of samples, at near-boundary
  controls): the true sensitivity is well-defined (barrier κ→0 gives cos 1.0, stable; strict-comp/LICQ/SOSC
  all hold) and the **barrier computes it correctly**, so it is a **data-dependent DIRECT-backward failure**
  (solver-independent: cuDSS≡JAX-dense give the identical wrong value; stable across fwd tol) — a fixable-flaw
  vs inherent-active-set-fragility distinction left **unsettled** (review corrected two over-claims of mine:
  the multiplier magnitude does not enter `dx/dθ`, and an interior barrier has no active set to compare).
  (c) **Reproduced `run_sweeps_gradient_accuracy.sh`** (horizon 40, 100 seeds, fused/DIRECT) with a
  **convergence-validated FD** (`benchmark_repro.py`, `results/benchmark_repro.{md,png,npz}`). The
  benchmark's single `fd_eps=1e-5` is **essentially fine** (converged per-sample; only marginally into the
  batch-sum noise floor; validated `1e-4` gives the same answer). Key result: the benchmark's **batch-summed
  cosine** (`cos(Σ_i g_i)`, its headline metric) is **outlier-dominated → median ~0.07** at tight tol, while
  the **per-sample median is 1.000**; removing the ~4–8/64 outliers per seed sends the batch-sum cos to
  **1.000**. So `cos(Σg)` is governed by whichever sample has the largest wrong ‖g‖, not by typical accuracy
  — per CLAUDE.md (per-sample, never batch-summed). **Cross-check DONE: ran the actual benchmark — its own
  `cos_all` is median ~0.36 (mean 0.31, min −0.38), low/outlier-dominated, NOT ~1.0**; my reproduction
  matches it per-seed (median 0.32 on seeds 0–9; per-seed scatter is the horizon 40-vs-39 off-by-one).
  So the reference figure `grad_box_accuracy_scp1_fdref.png` (median 1.0) is a **per-sample** cosine, which
  `run_sweeps` never computes — the two figures are the same gradients under two metrics (per-sample 1.0 vs
  batch-summed 0.36), not a contradiction. (My earlier assumption that the benchmark outputs ~1.0 was wrong.)
- **2026-06-26** — **The "diffmpc2 hard-box DIRECT-backward outliers" are a diffmpc2 `release-cleanup`
  BUG, not fundamental; `external/turbompc` (GitHub main) is correct.** This resolves the benchmark
  discrepancy and *corrects the 2026-06-25 entry above*: the reference figure is external's **batch-summed
  `cos_all`**, which on external is median **1.0** (not a "per-sample" metric). **Measured** (built
  external's cuDSS FFI against the installed cuDSS 0.7.1 via a backport — external's `.cu` ship the
  cuDSS-0.8 API; rename macros + drop the extra `cudssMatrixCreateCsr` index-type arg): same
  config/seeds/cuDSS, external hard-box `cos_all` median **1.0** (9/10 seeds exactly 1.0) vs diffmpc2
  **0.36**. **Cause** (`turbompc/solvers/turbompc_solver.py`; `backward_kkt_jax.py` is byte-identical, so
  the DIRECT backward is fed wrong *inputs*): diffmpc2 is missing external's inequality-multiplier
  **sign-correction** (`y_ineq = −sign·y_g`, `sign = lower − upper`) and **lower/upper dual-sign
  disambiguation** → wrong-signed / mis-classified active multipliers at near-active constraints →
  wrong-direction gradients. The rollout warm-start (`timing.py` `stop_gradient`) was **refuted** as the
  cause (`warmstart_test.py`). The project's own log-barrier backward (`src/diffmpc_learning`) is
  **immune** — it uses one-sided rows (`to_one_sided`: `[G;−G]x≤[u;−l]`, so lower/upper are separate
  ≥0-multiplier rows) + a smooth complementarity weight `W = y_g/(s + y_g/γ)`, never a two-sided hard
  active-set. CLAUDE.md (top callout) + HANDOFF.md updated to use `external/turbompc` as canonical;
  memory `diffmpc2-hardbox-outliers-are-release-cleanup-specific`. **Re-validate prior gradient_accuracy.md
  results** (the "weakly-active DIRECT-backward outlier" findings are this diffmpc2 bug).
- **2026-06-26** — **4-variant benchmark on the correct solver** (`experiments/linear_system/
  four_variant_benchmark.{py,md,png,npz}`; external/turbompc, common hard-box GT, horizon 40, 10 seeds,
  per-sample **and** batch-summed). At tight tol over 640 samples: **the hard box and the pure log-barrier
  are FAITHFUL — cos 1.0, ZERO outliers** (the hard box's 0 outliers is the clincher that external is
  correct — diffmpc2 has ~4–8/seed); **both Moreau-slack variants** (turbompc-Moreau and barrier+Moreau)
  are **biased ~0.45**, identically and flat across tol — the Moreau-relaxation bias (not a solve issue,
  not formulation-specific; 116/640 even go cos<0). Per-sample ≈ batch-sum for every variant here
  (**systematic** bias), unlike diffmpc2's hard box (per-sample 1.0 but batch-sum ~0.1 — **outlier**-driven):
  the two metrics together separate systematic bias from outlier corruption. **Upshot: the log-barrier
  (no slack) gives both a smooth forward and a faithful gradient.**
