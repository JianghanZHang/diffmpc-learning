"""Reproduce [Frey2025] (reference_papers/Diehl_DiffNMPC.pdf) Example 1 / Fig 1, and QUANTIFY how
much the active-set switch hurts the gradient (hard box vs central-path/log-barrier smoothing).

Paper Example 1:   min_x (x - theta^2)^2   s.t.  -1 <= x <= 1
  analytic solution map  x*(theta) = clip(theta^2, -1, 1)
  analytic derivative    dx*/dtheta = 2*theta  for |theta|<1 ;  0 for |theta|>1 ;  UNDEFINED at |theta|=1
The bound switches active at |theta|=1, where strict complementarity fails (multiplier crosses 0) and
the derivative JUMPS from 2 to 0. The paper smooths complementarity (mu_i s_i = tau) -> a C^1 map.

We reproduce that with the SAME central-path smoothing realized as a log-barrier:
  x*(theta; tau) = argmin_x (x - theta^2)^2 - tau*[log(1-x) + log(1+x)]   (interior-point central path)
solved by Newton; the diffmpc gradient dx*/dtheta is obtained by AD through the solve. As tau->0 the
smoothed map -> the hard clip, and its derivative -> the discontinuous one.

HARM METRICS (what "hurts" means, all measured, per CLAUDE.md FD rule):
  M1  derivative jump at the switch:  hard |2 - 0| = 2 ;  smoothed ~ 0 (max finite-difference of dx*/dtheta).
  M2  convergence-checked central FD vs AD near the switch (shrink eps; report plateau + AD-FD gap).
  M3  finite-step linear-model error  E(theta,h) = |x*(theta+h) - x*(theta) - h*g(theta)|  (g = AD grad):
      how badly the gradient mispredicts a real step across the kink (hard) vs the rounded one (smoothed).

CPU-only (scalar problem) -> no GPU/cuDSS needed.  Run:
    python experiments/gradients/active_set_smoothing/paper_example1.py
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # scalar toy: stay off the (contended) GPU
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from util.plot import DATA_DIR, save_fig  # noqa: E402

TAUS = [1e-2, 1e-3, 1e-4]                               # paper Fig 1 uses tau = 1e-2,1e-3,1e-4
NEWTON_ITERS = 60                                       # plenty for this 1-D barrier subproblem
FD_EPS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]     # decreasing -> plateau check (CLAUDE.md)


# ---------- hard problem (analytic) ----------
def x_hard(theta):
    return jnp.clip(theta ** 2, -1.0, 1.0)

def dx_hard_analytic(theta):
    # 2*theta inside the box, 0 outside; the value AT |theta|=1 is a jump (left 2, right 0) -> not defined
    return jnp.where(jnp.abs(theta) < 1.0, 2.0 * theta, 0.0)


# ---------- central-path / log-barrier smoothed solve ----------
def x_smooth(theta, tau):
    """argmin_x (x-theta^2)^2 - tau*[log(1-x)+log(1+x)] via damped Newton from x=0. Stays in (-1,1)."""
    tau = jnp.asarray(tau)
    def fp(x):   # f'(x) = 2(x-theta^2) + tau/(1-x) - tau/(1+x)
        return 2.0 * (x - theta ** 2) + tau / (1.0 - x) - tau / (1.0 + x)
    def fpp(x):  # f''(x) = 2 + tau/(1-x)^2 + tau/(1+x)^2  (>0 -> strictly convex on (-1,1))
        return 2.0 + tau / (1.0 - x) ** 2 + tau / (1.0 + x) ** 2
    def step(x, _):
        dx = fp(x) / fpp(x)
        # damp so a full Newton step never leaves (-1,1)
        x_new = x - jnp.clip(dx, -0.49 * (1.0 - x), 0.49 * (1.0 + x))
        return x_new, None
    x0 = jnp.asarray(0.0)
    x_star, _ = jax.lax.scan(step, x0, None, length=NEWTON_ITERS)
    return x_star


def make_maps(tau):
    f = lambda th: x_smooth(th, tau)
    g = jax.grad(f)                                     # AD through the Newton solve = the diffmpc gradient
    return jax.vmap(f), jax.vmap(g)


# ---------- convergence-checked central FD (CLAUDE.md) ----------
def fd_converged(fmap, theta_grid):
    """Per-theta central FD over decreasing eps; return (fd_value, converged_mask, eps_used)."""
    cols = {e: np.asarray((fmap(theta_grid + e) - fmap(theta_grid - e)) / (2 * e)) for e in FD_EPS}
    eps_sorted = sorted(FD_EPS, reverse=True)
    n = theta_grid.shape[0]
    fd = np.array(cols[eps_sorted[0]]); conv = np.zeros(n, bool); eps_used = np.full(n, eps_sorted[0])
    for i in range(n):
        seq = [cols[e][i] for e in eps_sorted]
        chosen, ok = seq[0], False
        for k in range(len(seq) - 1):
            if abs(seq[k] - seq[k + 1]) <= 1e-2 * abs(seq[k + 1]) + 1e-9:
                chosen, ok, eps_used[i] = seq[k], True, eps_sorted[k]; break
        fd[i] = chosen; conv[i] = ok
    return fd, conv, eps_used


def main():
    theta = jnp.linspace(-1.6, 1.6, 1601)
    th_np = np.asarray(theta)

    xh = np.asarray(x_hard(theta)); dxh = np.asarray(dx_hard_analytic(theta))
    smaps = {tau: make_maps(tau) for tau in TAUS}
    xs = {tau: np.asarray(fmap(theta)) for tau, (fmap, _) in smaps.items()}
    dxs = {tau: np.asarray(gmap(theta)) for tau, (_, gmap) in smaps.items()}

    # ----- M1: derivative jump at the switch (max |consecutive diff| of dx/dtheta near |theta|=1) -----
    near = np.abs(th_np - 1.0) < 0.05
    jump_hard = float(np.max(np.abs(np.diff(dxh[near]))))
    jump_smooth = {tau: float(np.max(np.abs(np.diff(dxs[tau][near])))) for tau in TAUS}

    # ----- M2: convergence-checked central FD vs AD for hard and smoothed -----
    fd_hard, conv_hard, _ = fd_converged(jax.vmap(x_hard), theta)
    fd_tau = TAUS[1]                                    # representative tau for the FD panel
    fd_sm, conv_sm, _ = fd_converged(smaps[fd_tau][0], theta)
    # AD-FD gap right at the switch
    i1 = int(np.argmin(np.abs(th_np - 1.0)))
    gap_hard = abs(dxh[i1] - fd_hard[i1]); gap_sm = abs(dxs[fd_tau][i1] - fd_sm[i1])

    # ----- M3: finite-step linear-model error across the kink, E(theta,h)=|x*(th+h)-x*(th)-h*g| -----
    h = 0.1
    xh_step = np.asarray(x_hard(theta + h))
    E_hard = np.abs(xh_step - xh - h * dxh)
    E_sm = {tau: np.abs(np.asarray(smaps[tau][0](theta + h)) - xs[tau] - h * dxs[tau]) for tau in TAUS}

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {1e-2: "tab:orange", 1e-3: "tab:green", 1e-4: "tab:red"}

    ax[0].plot(th_np, xh, "k-", lw=2, label="hard (analytic)")
    for tau in TAUS:
        ax[0].plot(th_np, xs[tau], color=colors[tau], lw=1.3, label=f"smoothed tau={tau:g}")
    ax[0].set_title("solution map  x*(theta)"); ax[0].set_xlabel("theta"); ax[0].legend(fontsize=8)

    ax[1].plot(th_np, dxh, "k-", lw=2, label="hard (jumps 2->0)")
    for tau in TAUS:
        ax[1].plot(th_np, dxs[tau], color=colors[tau], lw=1.3, label=f"AD, tau={tau:g}")
    for s in (-1.0, 1.0):
        ax[1].axvline(s, color="0.7", ls=":", lw=1)
    ax[1].set_title("derivative  dx*/dtheta  (the gradient)"); ax[1].set_xlabel("theta"); ax[1].legend(fontsize=8)

    ax[2].plot(th_np, E_hard, "k-", lw=2, label="hard")
    for tau in TAUS:
        ax[2].plot(th_np, E_sm[tau], color=colors[tau], lw=1.3, label=f"smoothed tau={tau:g}")
    ax[2].set_title(f"linear-model error  |x*(th+h)-x*-h*grad|,  h={h}"); ax[2].set_xlabel("theta")
    ax[2].legend(fontsize=8)
    fig.suptitle("Diehl Example 1: active-set switch at |theta|=1 — hard gradient jumps, central-path smooths it",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = save_fig(fig, "paper_example1.png"); plt.close(fig)

    # ---------- report (measured) ----------
    print("=== Diehl Example 1 reproduction — measured harm ===")
    print(f"M1 derivative jump near |theta|=1 (max |consecutive d/dtheta diff|):")
    print(f"     hard            = {jump_hard:.4f}   (true jump 2 -> 0)")
    for tau in TAUS:
        print(f"     smoothed tau={tau:g} = {jump_smooth[tau]:.4f}")
    print(f"M2 convergence-checked central FD vs AD:")
    print(f"     hard:     FD non-converged at {int((~conv_hard).sum())}/{conv_hard.size} theta; "
          f"at theta=1  AD={dxh[i1]:.3f}  FD={fd_hard[i1]:.3f}  gap={gap_hard:.3f}")
    print(f"     smoothed(tau={fd_tau:g}): FD non-converged at {int((~conv_sm).sum())}/{conv_sm.size}; "
          f"at theta=1  AD={dxs[fd_tau][i1]:.3f}  FD={fd_sm[i1]:.3f}  gap={gap_sm:.3f}")
    print(f"M3 max linear-model error E(theta,h={h}) over the grid:")
    print(f"     hard            = {E_hard.max():.4f}  at theta={th_np[E_hard.argmax()]:.3f}")
    for tau in TAUS:
        print(f"     smoothed tau={tau:g} = {E_sm[tau].max():.4f}  at theta={th_np[E_sm[tau].argmax()]:.3f}")
    print(f"saved {os.path.relpath(out_png, _HERE)}")

    np.savez(os.path.join(DATA_DIR, "paper_example1.npz"),
             theta=th_np, x_hard=xh, dx_hard=dxh,
             **{f"x_smooth_{t:g}": xs[t] for t in TAUS},
             **{f"dx_smooth_{t:g}": dxs[t] for t in TAUS},
             fd_hard=fd_hard, conv_hard=conv_hard, E_hard=E_hard,
             jump_hard=jump_hard)


if __name__ == "__main__":
    main()
