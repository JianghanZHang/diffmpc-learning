# Quadrotor closed-loop (nu=4 nonlinear): tractable; no dramatic pathology; two methodology lessons

**Run 2026-06-24.** Closed-loop hard-box-vs-slack gradient study on the **nu=4 nonlinear quadrotor**
(13 states, RK4, hover regulation, control box). Driver: `closed_loop_quadrotor.py`.

## Two false starts, both corrected (methodology)

1. **"Intractable" was GPU contention.** A leftover-process artifact; clean runs compile+run fine
   (sqp=10 n=2 ≈120 s; AD ≈82 s; FD slow ~33–40 min). See [[isolate-gpu-contention-before-boundary]].
2. **A "cos=0.616 hard-box outlier" was FD below the cost noise floor.** The rollout cost is
   non-deterministic at **~8e-8 relative** (cuDSS over 50 chained 13-state solves, ~100× the cartpole's
   floor). The original FD eps (3e-5…1e-6) sat below it → spurious noise → false outlier (that sample
   reads cos=0.99975 at eps=1e-3). Fixed by using eps `(1e-3,3e-4,1e-4)`. See
   [[fd-noise-floor-is-system-size-dependent]].

## Final results (noise-floor-corrected FD, sqp=8, n=6, seed 0)

| box (× hover) | hardbox cos min | hardbox rel max | slack cos min | slack rel max |
|---|---:|---:|---:|---:|
| umax=1.2  (1.22×) | 0.976 | — | 0.964 | — |
| umax=1.05 (1.07×) | 0.989 | 1.6e-1 | 0.99999 | 4.7e-3 |
| umax=1.00 (1.02×) | 0.99981 | 4.0e-2 | 0.99994 | 1.1e-2 |

## Findings

- **No dramatic hard-box pathology at any tested box.** Unlike the linear nu=4 (cos down to −0.47) and
  cartpole tight box (cos −0.81), the quadrotor hard box never drops below **cos 0.989**, and at the
  *tightest* box (umax=1.00) it is actually *cleaner* (cos ≥ 0.9998, #cos<0.99 = 0).
- **A consistent but weak directional signal**: the slack backward is always slightly more
  FD-consistent than the hard box (cos min 0.99999/0.99994 vs 0.989/0.99981; rel ~10–30× smaller). The
  *direction* matches the linear/cartpole mechanism (slack ≥ hard box), but the *magnitude* is tiny.
- **Tightening the box does NOT worsen the hard box** (umax 1.05→1.00 made it cleaner). This is the key
  clue.

## Interpretation (HYPOTHESIS — not directly measured)

The dramatic hard-box pathology is plausibly driven by active-set **switching** (controls flipping
active/inactive as the cost weights vary), not by constraint binding per se. At near-hover with a tight
box the quadrotor's thrust is **continuously saturated → the active set is stable → few switches →
smooth gradient**, even though the box is firmly active. The linear-random nu=4 systems and the cartpole
tight-regulation have frequent switching → the dramatic pathology; the quadrotor *hover* does not.
**This is a conjecture**: the confirming measurement would instrument the rollout to count active-set
changes (controls entering/leaving the bound) across the FD weight perturbations and correlate that
count with the per-sample hard-box cos. Not run this session.

## Bottom line

The clean, dramatic confirmation of "hard-box closed-loop outliers, fixed by the soft/slack box" stands
on the **linear (nu=4)** and **cartpole (nu=1 nonlinear)** systems. The quadrotor adds: (a) the
differentiable-rollout approach **scales to 13 states** (tractable, just a slow FD); (b) the pathology
is **not universal** — the nu=4 nonlinear *hover* regime shows only a weak directional version, likely
because its tight-box active set is stable rather than switching. Data:
`data/closed_loop_quadrotor_*_umax{1p2,1p05_bigeps,1p0}.npz`. A switching-heavy nonlinear regime (e.g.
aggressive trajectory tracking, not hover) would be the right testbed to see the dramatic form at nu=4.
