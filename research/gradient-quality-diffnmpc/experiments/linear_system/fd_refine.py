"""Do the GT-flagged samples just need a smaller eps, or are they genuinely non-smooth?
For A no-slack (fused), TIGHT forward (tol=1e-11 so the cost noise floor is low), compute the FD over a
fine decreasing eps grid and watch each sample's FD-vs-eps convergence. A sample whose consecutive-eps
relative change drops below 1% somewhere has a plateau (converges -> just needed smaller eps); one that
never plateaus (or only grows at small eps = noise floor) is genuinely at a discontinuity.

    python research/gradient-quality-diffnmpc/experiments/linear_system/fd_refine.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from closed_loop_4way_sweep import _problem, _flat, _unflat, NW, ForwardBackend, LANE_CAP  # noqa: E402
from fused_check import cost_fn_A  # noqa: E402

EPS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]
FWD_TOL = 1e-11
B = 64
ORIG_LO = 3e-6        # original 4-way grid bottomed out here


def main():
    dyn, pp, w, x0b, A_sd, B_m, b_v = _problem(0, B)
    cf = cost_fn_A(dyn, pp, w, x0b, FWD_TOL, slack=False, fwd=ForwardBackend.ADMM_FUSED_CUDSS)
    single = jax.jit(jax.vmap(lambda wf, st: cf(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w)); E = jnp.eye(NW)
    G = np.zeros((len(EPS), B, NW))
    for ei, eps in enumerate(EPS):
        delta = eps * jnp.asarray([1.0, -1.0])[None, :, None] * E[:, None, :]   # [NW,2,NW]
        Wp = (w_flat + delta).reshape(-1, NW); P = Wp.shape[0]
        cP = max(1, LANE_CAP // B); costs = np.zeros((P, B))
        for s in range(0, P, cP):
            Wc = Wp[s:s + cP]; c = Wc.shape[0]
            costs[s:s + c] = np.asarray(single(jnp.repeat(Wc, B, axis=0),
                                               jnp.tile(x0b, (c, 1)))).reshape(c, B)
        costs = costs.reshape(NW, 2, B)
        G[ei] = ((costs[:, 0, :] - costs[:, 1, :]) / (2 * eps)).T
    # consecutive-eps relative change per sample
    rc = np.zeros((B, len(EPS) - 1))
    for nn in range(B):
        for ei in range(len(EPS) - 1):
            rc[nn, ei] = np.linalg.norm(G[ei][nn] - G[ei + 1][nn]) / (np.linalg.norm(G[ei + 1][nn]) + 1e-30)
    orig_pairs = [i for i in range(len(EPS) - 1) if EPS[i] >= ORIG_LO and EPS[i + 1] >= ORIG_LO]
    all_pairs = list(range(len(EPS) - 1))

    def flagged(pairs):
        return np.array([not any(rc[nn, i] < 1e-2 for i in pairs) for nn in range(B)])
    of, ef = flagged(orig_pairs), flagged(all_pairs)
    print(f"A no-slack fused, fwd_tol={FWD_TOL:g}, B={B}")
    print(f"eps grid: {EPS}")
    print(f"flagged ORIG grid (>= {ORIG_LO:g}):  {int(of.sum())}/{B}")
    print(f"flagged EXTENDED grid (.. {EPS[-1]:g}): {int(ef.sum())}/{B}")
    resolved = of & ~ef
    print(f"resolved by smaller eps: {int(resolved.sum())}  | still flagged: {int(ef.sum())}")
    show = sorted(set(np.where(of)[0]) | set(np.where(ef)[0]))
    print(f"\n{'samp':>4} {'orig':>5} {'ext':>4} | consecutive rel-change per eps-pair "
          f"({'  '.join(f'{e:.0e}' for e in EPS[1:])})")
    for nn in show:
        traj = "  ".join(f"{rc[nn, i]:5.0e}" for i in range(len(EPS) - 1))
        bi = int(np.argmin(rc[nn]))
        print(f"{nn:>4} {'F' if of[nn] else '.':>5} {'F' if ef[nn] else '.':>4} | {traj}   "
              f"min={rc[nn, bi]:.0e}@eps{EPS[bi + 1]:.0e}")


if __name__ == "__main__":
    main()
