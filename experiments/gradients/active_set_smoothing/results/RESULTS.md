# Active-set smoothing — Example 1 (Diehl/[Frey2025] Fig 1) reproduction + harm quantification

`paper_example1.py` reproduces the paper's tutorial example and measures **how much the active-set
switch hurts the gradient**, hard box vs central-path/log-barrier smoothing. CPU-only (scalar), x64.

Problem: `min_x (x − θ²)² s.t. −1 ≤ x ≤ 1`. Analytic `x*(θ) = clip(θ², −1, 1)`; the bound switches
active at `|θ| = 1`, where strict complementarity fails and `dx*/dθ` **jumps 2 → 0**. Smoothing
(`μ·s = τ`, realized as a log-barrier solved by Newton; AD through the solve = the diffmpc gradient)
gives a C¹ map; `τ → 0` recovers the hard clip.

## Measured (figure: `paper_example1.png`)

| metric | hard | smoothed τ=1e-2 | τ=1e-3 | τ=1e-4 |
|---|---:|---:|---:|---:|
| **M1** derivative jump at `\|θ\|=1` (max consecutive Δ of `dx/dθ`) | **1.996** (≈ true 2→0) | 0.026 | 0.087 | 0.271 |
| **M3** max finite-step linear-model error `\|x*(θ+h)−x*−h·g\|`, h=0.1 | **0.196** (at θ≈1.0) | 0.057 | 0.116 | 0.158 |

- **M2 — convergence-checked central FD vs AD.** Hard: FD **converges at all 1601 θ** (`0` non-converged);
  at the switch `θ=1`, **AD = 0** (jumped to the inactive branch) but **FD = 0.999**, gap ≈ **1.0**.
  Smoothed (τ=1e-3): AD = FD = 0.994, gap = 0.

## Findings

1. **The hard gradient jumps by ≈2 at the switch; smoothing rounds it** (M1, panel 2 = paper Fig 1).
   Smaller τ ⇒ sharper corner (jump 0.026 → 0.27 as τ: 1e-2 → 1e-4) — the O(τ) bias-vs-sharpness knob.
2. **The hard gradient is the *worst* local model exactly at the kink** (M3 = 0.196 at θ≈1.0): a real step
   `h=0.1` across the switch is mispredicted because the gradient says slope ≈2 while `x*` clips to 1.
   Smoothing cuts this ~3.4× (τ=1e-2). This is the concrete "hurts optimization" cost of the jump.
3. **KEY NUANCE — a convex box kink does NOT make FD diverge.** `x*(θ)` is *continuous* (only its
   derivative kinks), so central FD **converges to the average slope** (≈1) at the switch — it does **not**
   blow up `~1/eps`. So a convergence-checked-FD benchmark **does not flag** this switch (it reports a
   plateau), and at random θ the AD–FD gap is ~0 (measure-zero kink). **This is exactly why the
   `linear_system` 4-variant benchmark saw `0/640` non-converged and "both faithful"** — a control box is
   the benign case. The harm (M1 jump, M3 model error) is real but invisible to a point-wise FD-vs-AD score.
4. **Genuine FD non-convergence (`~1/eps` blow-up) needs a *discontinuous* solution map, i.e.
   NONCONVEXITY** — where the global optimum switches basins (e.g. passing an obstacle left vs right),
   `x*` jumps in value, not just in slope. That is the regime the obstacle surrogate (Phase 2) targets.

**Takeaway:** Example 1 confirms the mechanism (jump → smoothed C¹) and quantifies the harm, but also
shows the convex box switch is the *mildest* form — a derivative kink that point-wise FD-vs-AD cannot
see. The case for smoothing is expected to strengthen with nonconvex, state-coupled inequalities → the
Phase-2 linear-dynamics + obstacle surrogate.

---

# Phase-2 stepping stone — convex coupled-linear "corridor" (`corridor_surrogate.py`)

2-D double integrator (linear), quadratic tracking to a goal placed beyond a 2-wall corner
(`px≤1, py≤1`). The differentiable/swept parameter is the **position-tracking weight `wq`** (the faithful
diffmpc-as-policy parameter); the loss is the downstream task cost `L(wq)=Σ‖p_t−goal‖²`. Linear walls ⇒
`∇²g=0` ⇒ the existing central-path backward is exact, so this validates the sweep/activation/κ-tradeoff
**harness** before implementing the obstacle's curvature term. CPU-only (JAX_DENSE Schur). Fine sweep:
`wq∈[0.18,0.42]`, 70 pts; figure `corridor_surrogate.png`, data `corridor_surrogate.npz`.

## Measured (numbers from the npz; coupled activation at `wq≈0.2565`)

- **Both walls activate together** at `wq≈0.2565` (wall slack `a^Tp−b → 0` simultaneously, panel 3) — the
  coupled near-active set box constraints can't produce.
- **The hard diffmpc gradient `dL/dwq` JUMPS ≈ 66** across the activation (−78.1 → −11.8 over one 0.0035
  grid step). The convergence-checked FD **reproduces the jump** (max|hard AD − FD| = 0.09, ≈0 on each
  side) ⇒ the discontinuity is a real property of `L(wq)`, not a backward/solver artifact.
- **`L(wq)` is continuous with a kink** ⇒ FD-of-loss converges at **0/70** points. So a convergence-checked
  FD **accuracy** benchmark does **not** flag this (hard AD = FD on both sides); the jump is visible **only
  in the sweep**. (Same lesson as Example 1 — now on a real coupled-constraint OCP and the *cost-weight*
  gradient, confirming the activation discontinuity survives differentiating a task loss w.r.t. weights.)
- **The barrier smooths the jump; κ is the bias-vs-sharpness knob** (deviation from the true hard=FD grad):

  | κ | deviation in the transition band | deviation far from switch (\|wq−act\|>0.1) |
  |---|---:|---:|
  | 1e-6 | 13.6 (rounds only the jump) | **0.000** (matches hard exactly) |
  | 1e-4 | 27.4 | 0.020 |
  | 1e-2 | 44.6 (heavy rounding) | 3.08 (global O(κ) bias) |

## Findings

1. **The Example-1 phenomenon reproduces on a real coupled-constraint OCP with the faithful diffmpc
   gradient:** the hard `dL/dwq` is *discontinuous* at the coupled activation; the central-path barrier
   replaces the jump with a continuous transition. Small κ matches hard away from the switch and rounds
   only the jump; large κ rounds more but biases the whole curve O(κ).
2. **The hard gradient is *accurate* (= FD) yet *discontinuous*.** The problem smoothing fixes is the
   discontinuity, which an accuracy (cos / FD-vs-AD) benchmark is structurally blind to — reinforcing that
   RQ1's payoff must be measured by a *sweep across a switch*, not point-wise fidelity.
3. **Still convex (linear walls):** the loss is continuous (kinked), so FD never diverges. A *discontinuous
   loss* (FD blow-up) needs nonconvexity → the obstacle (Phase 3).

## Not yet measured / next

- **Backward-KKT condition number vs `wq`** (hard active-set spike at the weakly-active corner vs the
  barrier's bounded `W`) — the conditioning half of the story; harness hook exists, metric not yet added.
- **tol-band:** width of the corrupted-gradient band around the switch vs solver tol, hard vs barrier.
- **Obstacle (Phase 3):** swap the linear walls for `1−‖p−c‖/r≤0`; requires implementing `μ∇²g` in the
  forward QP-Hessian and the backward `D_aug` (NSD curvature ⇒ likely LM regularization). Only then does
  the *loss* become discontinuous (basin/activation switches), the regime where FD itself blows up.
