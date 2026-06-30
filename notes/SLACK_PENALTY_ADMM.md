# The slack penalty in TurboMPC's ADMM

**Date:** 2026-06-22 · part of the **"Gradient Quality of Differentiable NMPC"** project (`NOTE.md`).

What `slack_penalization_weight` (γ) does to the inequality constraints inside TurboMPC's ADMM QP
solve, and why it is there — from the ADMM-on-the-augmented-Lagrangian perspective. Code paths are
relative to the `diffmpc2/` checkout.

## 1. Where the inequality lives in ADMM

ADMM solves each (SQP) QP subproblem by operator splitting on its augmented Lagrangian. The
inequality `l ≤ Gx ≤ u` (box `C = [l,u]`) is handled by giving `Gx` its own copy — split

```
Gx = z_g ,     z_g ∈ C .
```

The (scaled-dual) augmented Lagrangian of the QP is

```
L_ρ(x, z_g; y_f, y_g) = ½ xᵀP x + qᵀx                         [cost]
                      + y_fᵀ(Cx − c) + (ρ_f/2)‖Cx − c‖²       [dynamics, equality split]
                      + χ_C(z_g) + y_gᵀ(Gx − z_g) + (ρ/2)‖Gx − z_g‖²   [inequality split]
```

and ADMM alternates:

- **x-update** — minimize over `x`: a smooth strongly-convex quadratic. This is the block-tridiagonal
  Schur solve (`compute_S_Phiinv`, `turbompc/solvers/admm/admm.py:71`; `Dtilde = D + σI + ρ_ineq·GᵀG
  + ρ_f·(dynamics AᵀA)`, `:93`), factorized by the fused-cuDSS backend. (`ρ_ineq·GᵀG` is the standard
  ADMM penalty from the `z_g = Gx` split; `ρ_f` is the dynamics-equality penalty.)
- **z_g-update** — minimize over `z_g`: `z_g⁺ = argmin χ_C(z_g) + (ρ/2)‖z_g − z̃‖² = proj_C(z̃)`, with
  `z̃ = Gx + y_g/ρ`. A **hard box projection** (`_project_box`, `admm.py:177`, used at `:907`).
- **dual update** — `y_g ← y_g + ρ(Gx − z_g)`.

The crucial point: **the inequality lives entirely in the `z_g`-update, and that update is the
proximal operator of the constraint's indicator function** `χ_C` (which is `0` inside `C`, `+∞`
outside). `χ_C` is the one **non-smooth** term in the whole solve — an infinitely steep wall whose
"derivative" is the normal cone (set-valued, discontinuous).

## 2. What the slack penalty γ does: Moreau-regularize the indicator

The slack penalty replaces the hard indicator `χ_C` with its **Moreau envelope** — a smooth quadratic
bowl:

```
χ_C(z_g)  →  g_γ(z_g) = (γ/2)·dist(z_g, C)² = min_ξ { (γ/2)‖ξ‖² : z_g − ξ ∈ C } ,
```

where `ξ` is the slack variable (`xi_g`) and `γ = slack_penalization_weight`. `g_γ` is `0` inside `C`
and grows quadratically outside — the sharp wall softened into a ramp; as `γ→∞`, `g_γ → χ_C`.

The `z_g`-update is now the prox of a **smooth** function — a **damped projection** (`admm.py:912–919`):

```
z_g⁺ = (1 − frac)·z̃ + frac·proj_C(z̃) ,   frac = γ/(γ+ρ) ,   ξ = ρ/(γ+ρ)·(proj_C(z̃) − z̃) .
```

*(Derivation: per coordinate, a clamped one solves `γ(z − c̄) + ρ(z − z̃) = 0 ⇒ z = (γc̄ + ρz̃)/(γ+ρ)`;
an interior one solves `ρ(z − z̃) = 0 ⇒ z = z̃`.)* Limits: **`γ→∞ ⇒ frac→1 ⇒` hard projection** (the
constraint is enforced exactly); **`γ→0 ⇒ frac→0 ⇒ z_g = z̃`** (constraint ignored). So `γ` is the
**constraint stiffness**.

This is **not** over-relaxation. Over-relaxation is the separate `α` blend that forms `z̃` itself
(`z̃ = α·Gx + (1−α)·z_g_prev + y_g/ρ`, `admm.py:893,:903`) — a problem-*preserving* convergence
accelerator. The `frac` blend changes the QP being solved (soft vs hard constraint → a different
fixed point).

## 3. Derivation: the objective view and the update view are the same

**Objective view — the slack penalty is the Moreau envelope of the indicator.** The Moreau envelope
(Moreau–Yosida regularization) of a function `f` with parameter `λ > 0` is

```
e_λ f(z) = min_u { f(u) + (1/2λ)·‖z − u‖² } .
```

Take `f = χ_C` (the box indicator: `0` on `C`, `+∞` off it). The inner `min` keeps only `u ∈ C`:

```
e_λ χ_C(z) = min_{u∈C} (1/2λ)·‖z − u‖² = (1/2λ)·dist(z,C)² ,    argmin = proj_C(z) .
```

Setting `λ = 1/γ` gives exactly the slack penalty `g_γ(z) = (γ/2)·dist(z,C)²`. Its gradient is
continuous — this is the smoothing:

```
∇g_γ(z) = γ·(z − proj_C(z)) ,
```

zero inside `C`, growing linearly with the violation outside: C¹ and Lipschitz (constant γ), but
**not C²** — the gradient's slope jumps at `∂C`, where `proj_C` switches active faces.

**Update view — the z-update is the prox of `g_γ`.** Collecting the `z_g`-dependent terms of the
augmented Lagrangian (`§1`), with the ADMM target `z̃ = Gx + y_g/ρ`, the `z_g`-update is

```
z_g⁺ = argmin_{z_g}  g_γ(z_g) + (ρ/2)·‖z_g − z̃‖²  =  prox_{(1/ρ)g_γ}(z̃) .
```

Substitute `g_γ(z_g) = min_{u∈C} (γ/2)·‖z_g − u‖²` and minimize jointly over `(z_g, u∈C)`:

```
z_g⁺ = argmin_{z_g,  u∈C}  (γ/2)·‖z_g − u‖² + (ρ/2)·‖z_g − z̃‖² .
```

Inner minimization over `z_g` (unconstrained quadratic) for fixed `u`:

```
γ(z_g − u) + ρ(z_g − z̃) = 0   ⟹   z_g(u) = (γ·u + ρ·z̃)/(γ + ρ) .
```

Back-substituting collapses the objective to a multiple of `‖z̃ − u‖²` (the `(γ+ρ)` denominators
combine: `γρ²/2 + ργ²/2 = (γρ/2)(γ+ρ)`):

```
(γ/2)‖z_g(u) − u‖² + (ρ/2)‖z_g(u) − z̃‖²  =  ( γρ / (2(γ+ρ)) )·‖z̃ − u‖² ,
```

so the outer minimization over `u∈C` is just a projection, `u⋆ = proj_C(z̃)`. Hence

```
z_g⁺ = (γ·proj_C(z̃) + ρ·z̃)/(γ+ρ) = (1 − frac)·z̃ + frac·proj_C(z̃) ,    frac = γ/(γ+ρ) ,
```

with slack `ξ = proj_C(z̃) − z_g⁺ = (ρ/(γ+ρ))·(proj_C(z̃) − z̃)` (the `xi_g` of `admm.py:916`).
Limits: `γ→∞ ⇒ frac→1 ⇒` hard projection; `γ→0 ⇒ frac→0 ⇒ z_g⁺ = z̃`.

**The equivalence.** The two views are a single identity — *the prox of a Moreau envelope is a
relaxed prox of the original function*:

```
prox_{(1/ρ)·e_{1/γ}χ_C}(z̃) = z̃ + frac·( prox_{χ_C}(z̃) − z̃ ) ,   frac = (1/ρ)/((1/γ)+(1/ρ)) = γ/(γ+ρ) ,
```

and since `prox_{χ_C} = proj_C`, this is exactly the damped projection above. So "smooth the indicator
in the objective" (`χ_C → g_γ`) and "damp the projection in the z-update" (`proj_C → (1−frac)·I +
frac·proj_C`) are the same operation: **smoothing a function damps its proximal map.**

## 4. Why we need it (from the ADMM solve)

1. **Subproblem feasibility / convergence — the main reason.** The hard split needs an `x` with
   `Gx = z_g`, `z_g ∈ C`, and `Cx = c`. For NMPC the **linearized** obstacle constraint at an SQP
   iterate can be *infeasible* (no such `x`). On an infeasible QP, hard ADMM **cannot converge**: the
   x-update produces some `Gx`, the z-update snaps to `proj_C(z̃) ≠ Gx`, the primal residual
   `‖Gx − z_g‖` is bounded below by the infeasibility distance and never drops under tolerance, and
   `y_g` accumulates the persistent residual and drifts → it hits `max_iter` at a meaningless point.
   With the slack, the feasible set of `z_g` is all of `ℝᵖ` (any value, merely priced), so a minimizer
   always exists and ADMM converges. This is the elastic/relaxation role (akin to SQP elastic mode).
2. **Bounded, continuous multiplier.** The inequality multiplier is `y_g = γ·ξ` — bounded by
   `γ·violation` and ramping up continuously from `0` as the constraint activates, rather than
   switching `0 → positive` discontinuously.
3. **Conditioning.** The damped projection (`frac < 1`) is a contraction; it keeps the z-update from
   rigidly snapping against a changing active set, stabilizing/accelerating the iteration when
   constraints are tight.

**Cost.** A finite `γ` satisfies the hard constraint only approximately — the returned point violates
it by `O(y_g/γ) = O(ξ)`. `γ` trades robustness/feasibility (small `γ`) against constraint accuracy
(large `γ`); `γ→∞` recovers the hard constraint.

**In one sentence:** ADMM splits the QP so the inequality becomes a prox; the slack penalty replaces
the *prox-of-the-indicator* (hard projection) with the *prox-of-its-Moreau-envelope* (damped
projection) — i.e. a **Moreau–Yosida regularization** of the constraint with parameter `1/γ` — there
to keep the QP subproblem solvable and the multiplier well-behaved.
