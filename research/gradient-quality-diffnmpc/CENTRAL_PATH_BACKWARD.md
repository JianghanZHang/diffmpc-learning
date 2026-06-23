# The κ-Relaxed (Central-Path) Backward Pass / VJP

How the gradient of the central-path NMPC solver w.r.t. the cost weights is computed, as
implemented in `src/diffmpc_learning/solvers/backward.py` (branch `central-path-admm`).
Companion to `SLACK_PENALTY_ADMM.md` (the forward retraction derivation).

---

## 1. What the backward computes

The differentiable parameter is the **cost-weight vector** `θ` (the Q/R diagonals — the
diffmpc-as-policy parameter). Given a scalar loss `L` on the converged trajectory
`(states*, controls*)`, the backward returns

    dL/dθ.

The forward is an SQP outer loop whose inner QP is solved by the **central-path ADMM**
(closed-form elastic log-barrier retraction; `central_path_admm.py`). The backward does **not**
differentiate through the iterations of either loop. It uses **implicit differentiation**: it
differentiates the *optimality conditions* (the KKT system) of the converged solution. This is
the same principle as TurboMPC's own backward (`diffmpc2/.../turbompc_solver.py`) and PrismQP's
`diff_qp` — the difference is purely in **how inequalities enter the KKT system**.

The key design choice: we differentiate the **same relaxed fixed point the forward converges
to**, so the gradient is the *exact* gradient of the relaxed solution map (it matches finite
differences of the relaxed solver), and is **smooth across active-set changes** by construction
(the relaxed map is C¹ for any `κ > 0`).

---

## 2. The relaxed fixed point (what the forward converges to)

Work in the **one-sided** form (`to_one_sided` stacks the box `l ≤ Gx ≤ u` into rows
`G1 x ≤ h`, `G1 = [G; −G]`, `h = [u; −l]`). Let `γ = slack_weight`, `κ = target_kappa`.
At convergence the ADMM produces a primal–dual point `(x, y_f, y_g)` satisfying, per one-sided
row `i`:

| condition | equation |
|---|---|
| stationarity | `P x + q + Cᵀ y_f + G1ᵀ y_g = 0` |
| equality (dynamics + initial) | `C x = c` |
| **relaxed complementarity** | `s_i · y_g,i = κ`,  with barrier slack `s_i = h_i − (G1 x)_i + y_g,i/γ` |

Here `P, q` are the QP cost blocks, `C` the equality Jacobian (`A0`, `A_minus`, `A_plus`),
`y_f = (y_f_0, y_f_dyn)` the equality multipliers, `y_g ≥ 0` the one-sided inequality
multipliers. The relaxed complementarity `s·y = κ` (instead of the hard `s·y = 0`) is what the
log-barrier retraction enforces — derived in `SLACK_PENALTY_ADMM.md`; the elastic slack gives
`ξ = y_g/γ`, so `s = h − Gx + ξ`.

As `κ → 0` this recovers the soft-box KKT (`s·y = 0`, `y_g = γ·ξ`); additionally `γ → ∞`
recovers the hard box.

---

## 3. Implicit differentiation → the backward KKT

Collect the three conditions as `F(u; θ) = 0` with `u = (x, y_f, y_g)`. The implicit function
theorem gives `du/dθ = −M⁻¹ ∂F/∂θ`, where `M = ∂F/∂u`. For a loss `L = L(x)`, the reverse-mode
(VJP) form solves the **adjoint** `Mᵀ λ = (dL/dx, 0, 0)` and returns `dL/dθ = −λᵀ ∂F/∂θ`.

`M` in block form (rows = stationarity / equality / relaxed-complementarity; columns = `x / y_f / y_g`):

```
        x                 y_f     y_g
stat [  P                 Cᵀ      G1ᵀ            ]
eq   [  C                 0       0              ]
ineq [ −diag(y_g) G1      0       diag(s + y_g/γ)]
```

(The ineq row is `∂/∂x[(h − G1 x + y_g/γ)·y_g] = −diag(y_g)·G1` and
`∂/∂y_g[…] = h − G1 x + 2 y_g/γ = s + y_g/γ`.)

**Eliminate the inequality block.** Solving `Mᵀ λ = (dL/dx, 0, 0)` and substituting the
`λ_g` equation back into the stationarity row collapses `M` to a **reduced symmetric system**
in `(λ_x, λ_f)` only:

```
[ P + G1ᵀ diag(W) G1   Cᵀ ] [ λ_x ]   [ dL/dx ]
[ C                    0  ] [ λ_f ] = [   0   ]
```

with the **smooth per-one-sided-row weight**

    W_i = y_g,i / (s_i + y_g,i/γ),     s_i = h_i − (G1 x)_i + y_g,i/γ.

`W` is the entire content of the κ-relaxation. It is the **C¹ analogue of TurboMPC's hard
active-set mask**:

| regime | `y_g,i` | `s_i` | `W_i` | meaning |
|---|---|---|---|---|
| inactive row | `→ 0` | `> 0` | `→ 0` | row drops out (no Hessian contribution) |
| soft-active (`κ→0`, finite γ) | `> 0` | `→ 0` | `→ γ` | quadratic-penalty stiffness |
| hard-active (`κ→0, γ→∞`) | `> 0` | `→ 0` | `→ ∞` | enforces `G1·λ_x = 0` (the hard constraint) |

`W ∈ [0, γ]` always, never negative, never singular: the denominator
`s + y_g/γ = h − G1 x + 2 y_g/γ ≥ s > 0`, and `y_g = κ/s > 0`. Because `G1` is block-diagonal
per stage, `G1ᵀ diag(W) G1` adds only to the **diagonal blocks of `P`**, so the system stays
block-tridiagonal.

This is exactly TurboMPC's slack-variable backward (`_build_backward_qp`, which uses
`D + γ·(hard mask)·G_activeᵀ G_active` and an empty inequality), with the hard `γ·mask`
replaced by the smooth `W`.

---

## 4. The Hessian `P` — exact Lagrangian Hessian for the NLP

For a **single QP**, `P` is just the QP cost Hessian (the block-tridiagonal `D, E`).

For the **NLP** (nonlinear dynamics), the implicit diff is taken on the *NLP* KKT, so the (1,1)
block must be the exact **Hessian of the Lagrangian**:

    P = D_cost + Σ_t y_f_dyn,t · ∇²f(x_t, u_t)        ("D + λᵀ∇²f")

The cost part `D_cost` comes from `solver._build_qp_data`; the dynamics-curvature term is
`solver.program.get_dynamics_lagrangian_hessian(states, controls, params, y_f_dyn)` — the same,
acados-cross-checked term fixed in `cccba81` (see `NOTE.md` E1.6; dropping it gives a ~16%
magnitude error). Constraints (dynamics, bounds) do **not** depend on the cost weights `θ`, so
their `θ`-derivatives vanish (see §5).

---

## 5. From `λ_x` to the weight gradient

With `λ_x` (the primal block of the adjoint), `dL/dθ = −λᵀ ∂F/∂θ`. Only the **stationarity**
residual depends on `θ` (the constraints don't), and there only through the cost gradient. At
the converged point the QP stationarity equals the NLP cost gradient, `P x* + q = ∇cost(x*)`,
so

    dL/dθ = − λ_x · ∂(∇cost(x*; θ))/∂θ
          = − [ Σ_t (∂(∇_x cost)/∂θ)·λ_states,t  +  Σ_t (∂(∇_u cost)/∂θ)·λ_controls,t ].

This is computed by a one-line JAX autodiff of `program.cost` w.r.t. `θ`, contracted with `λ_x`
(`central_path_nlp_grad.contracted`). It matches TurboMPC's `_assemble_parameter_gradient`
(the cost-only collapse of its general mixed-partial formula — the constraint terms
`g_θ, h_θ` are zero for cost weights).

---

## 6. Code map (`src/diffmpc_learning/solvers/backward.py`)

| function | role |
|---|---|
| `relaxed_complementarity_weight(qp1, x*, y_g, γ)` | `W = y_g/(s + y_g/γ)`, `s = h − G1x + y_g/γ` (§3) |
| `augment_D_with_relaxed_ineq(D, G1, W)` | `D[t] += G1[t]ᵀ diag(W[t]) G1[t]` (stage-blockwise fold) |
| `solve_reduced_relaxed_kkt(D_aug, E, eq, x̄)` | builds an empty-ineq, homogeneous backward `QPData` (cost `q = −x̄`) and calls TurboMPC's dense `solve_backward_kkt` → `λ_x` |
| `qp_central_path_cost_vjp(qp1, x*, duals, x̄, γ)` | QP-level VJP → `(dL/dD, dL/dE, dL/dq)` via `−λ_x · ∂(Px*+q)/∂(D,E,q)` (JAX `vjp`) |
| `make_qp_central_path_diff(...)` | `jax.custom_vjp` wrapper over the inner QP solve (composes under `jax.grad`/`jit`) |
| `central_path_nlp_solve(solver, pp, θ, …)` | runs the weighted SQP forward (also serves FD) |
| `central_path_nlp_grad(solver, pp, θ, loss_grad_fn, …)` | the NLP backward: forward → re-solve at the converged point for a consistent `(x*, duals)` → exact-Hessian + `W` fold → adjoint solve → `dL/dθ` |

**Two entry points, by design:**

- **Inner QP — `jax.custom_vjp`.** `solve_qp_central_path` is jittable (a `lax.while_loop`),
  so its VJP is registered as a proper `custom_vjp` and composes with `jax.grad`/`jit`.
- **NLP — a manual `grad` function.** The SQP *outer* loop is plain Python (data-dependent
  `break` on `float(conv)`), so it is **not** traceable; wrapping it in `custom_vjp` is not
  possible without making the whole loop jittable. Instead `central_path_nlp_grad` runs the
  forward eagerly, then builds and solves the backward KKT once at the converged point. (Making
  the SQP jittable + `custom_vjp` is future work.)

The dense backward solve (`solve_backward_kkt`, `jnp.linalg.solve`) is reused unchanged from
diffmpc2; only the **inequality treatment** differs (the `W` fold replaces the hard active-set
rows). The cartpole KKT is small (~hundreds of vars), so the dense solve is trivial; a
structured/cuDSS backward is a later optimization.

---

## 7. Why this is correct (and the limits)

- **Interior (no active inequality):** `y_g → 0 ⇒ W → 0`, the fold vanishes, and the backward
  is the pure equality-constrained KKT — **identical to TurboMPC's unrelaxed backward.**
- **Active inequality:** `W` is finite and smooth in `(x, y_g)`, so `dL/dθ` is **continuous
  across activation** — the strict-complementarity singularity of the hard backward is replaced
  by the bounded `W`.
- **`κ → 0` (and `γ → ∞`):** `W` on active rows `→ ∞`, recovering the hard-constraint backward
  (O(κ) bias vanishes).

The backward is the *exact* gradient of the κ-relaxed forward — so for any fixed `κ > 0`,
finite differences of that same forward converge to it as `eps → 0`.

---

## 8. Verification (cartpole swing-up, GPU/cuDSS, x64)

`tests/python/solvers/test_backward_central_path.py` (5 tests), at a **converged NLP-KKT**:

1. **QP-level VJP vs convergence-checked FD** (bounded one-sided QP, active rows → `W`
   exercised): rel-ℓ∞ `< 1e-4` on `dL/dq`, `dL/dD`, `dL/dE`.
2. **Interior NLP** (`umax=50`, bounds present but inactive → `W ≈ 9e-11`): AD == **unrelaxed
   TurboMPC** backward (`cos = 1.0`, `rel-ℓ₂ ≈ 2e-8`) == FD (`rel-ℓ₂ < 1e-3`). Pins the
   exact-Hessian term and **all multiplier signs** (a sign flip diverges from both).
3. **Bounded NLP** (`umax=2`, control bound active): AD == convergence-checked FD of the *same*
   relaxed solver (`cos > 1−1e-5`, `rel-ℓ₂ < 2e-3`) at **both** κ regimes (annealed→1e-6, fixed
   1e-4). A per-eps sweep of the active-bound `dL/dR` confirms **FD(eps→0) = AD to 7 digits**
   (`−1.696323`); the apparent gap at coarse eps is FD truncation, not a backward bias.

A whole-branch adversarial review independently re-derived `W` from the IFT and confirmed every
sign (sign-flip / zeroing probes are rejected by these FD tests). **No gradient bug found.**

**Solver tolerances behind the verification** (fixed-κ bounded config): inner relaxed-QP
`cp_tol = 1e-11` (achieved `prim_res ≈ 3e-11`), SQP `tol = 1e-5` (achieved dynamics-eq `≈ 4e-12`),
barrier `κ = 1e-4`, soft penalty `γ = 1e2`. (κ is the relaxation parameter, not a tolerance —
AD differentiates the κ-relaxed solution exactly; the solve accuracy is what makes FD clean.)

---

## 9. Scope / not-yet-measured

**Established:** the κ-relaxed backward is a *correct* gradient of the relaxed solver and
*reduces to the exact unrelaxed gradient in the interior*. **Not yet measured** (per CLAUDE.md
report-only-what-was-measured): that the relaxed gradient is **smoother / better-conditioned
across an active-set boundary** than the hard backward — that is the design rationale (H1.2; IP
smoothing `[Frey2025]`) and the next experiment (`NOTE.md` E1.7). Also out of scope so far:
obstacle (nonlinear, one-sided) constraints, a structured/cuDSS backward, and RL-training impact.

---

## Appendix A — Eliminating the inequality block (the Schur complement that gives `W`)

§3 states the reduced system; here is the elimination in full. It is a **block Schur
complement**: the inequality-dual block is removed because its diagonal is trivially invertible.

### A.1 The adjoint system

The VJP solves `Mᵀ λ = (dL/dx, 0, 0)`, with `M = ∂F/∂u`, `u = (x, y_f, y_g)`,
`λ = (λ_x, λ_f, λ_g)`, and `F = (F_stat, F_eq, F_ineq)`. Block form (rows = residuals,
columns = `x / y_f / y_g`):

```
       x                y_f    y_g
M =  [ P               Cᵀ     G1ᵀ            ]   ← ∂F_stat
     [ C               0      0              ]   ← ∂F_eq
     [ −diag(y_g)·G1   0      diag(s+y_g/γ)  ]   ← ∂F_ineq
```

The inequality row is `∂/∂(·)` of `F_ineq,i = (h_i − (G1x)_i + y_g,i/γ)·y_g,i − κ`:

- `∂/∂x = −y_g,i·(G1 row i)`  →  `−diag(y_g)·G1`
- `∂/∂y_g,i = (h_i − (G1x)_i + y_g,i/γ) + y_g,i·(1/γ) = s_i + y_g,i/γ`  →  `diag(s + y_g/γ)`

### A.2 The three transposed equations

`P` is symmetric and the diagonal blocks transpose to themselves, so `Mᵀ λ = (dL/dx, 0, 0)` is:

```
(1)  P λ_x + Cᵀ λ_f − G1ᵀ diag(y_g) λ_g = dL/dx     ← x-column of M
(2)  C λ_x                               = 0         ← y_f-column
(3)  G1 λ_x + diag(s + y_g/γ) λ_g        = 0         ← y_g-column
```

### A.3 Eliminate `λ_g`

Equation **(3)** has a **diagonal, strictly positive** coefficient on `λ_g`
(`s + y_g/γ = h − G1x + 2y_g/γ ≥ s > 0`), so it inverts element-by-element:

```
λ_g = −diag( 1/(s + y_g/γ) ) · G1 λ_x
```

Substitute into **(1)**:

```
P λ_x + Cᵀ λ_f − G1ᵀ diag(y_g)·( −diag(1/(s+y_g/γ)) G1 λ_x ) = dL/dx
P λ_x + Cᵀ λ_f + G1ᵀ diag( y_g/(s+y_g/γ) ) G1 λ_x           = dL/dx
( P + G1ᵀ diag(W) G1 ) λ_x + Cᵀ λ_f                         = dL/dx,   W = y_g/(s+y_g/γ)
```

With **(2)** unchanged, this is the reduced symmetric system of §3. The eliminated `λ_g`
contributes exactly the rank-structured term `G1ᵀ diag(W) G1` to the `(x,x)` block — **that term
is `W`**. (Formally: `W` is the Schur complement of the `(y_g,y_g)` block onto `(x, y_f)`.)

### A.4 Why this is well-posed — and why `κ` is load-bearing

The elimination only works because `diag(s + y_g/γ)` is invertible. In the **hard** limit
`κ → 0`, active rows have `s = κ/y_g → 0`, so that block becomes **singular** — the
strict-complementarity failure that makes the unrelaxed KKT ill-conditioned at an active-set
boundary. The barrier `κ > 0` holds `s` away from 0, keeping the Schur complement finite and
smooth in `(x, y_g)`. This is the precise mechanism by which the relaxation makes the backward
C¹ across activation.

### A.5 What the code does

The implementation never forms `M` or eliminates symbolically — A.1–A.3 are done once here, and
the code builds the **already-reduced** system directly: compute `W`
(`relaxed_complementarity_weight`), fold `G1ᵀ diag(W) G1` into the diagonal blocks of `D`
(`augment_D_with_relaxed_ineq`), then solve `[[P+GWG, Cᵀ],[C,0]]` via `solve_backward_kkt` with
an **empty** inequality. `λ_g` is never reconstructed: for cost-only weights it would multiply
`∂h/∂θ = 0` (§5), so only `λ_x` is needed.
