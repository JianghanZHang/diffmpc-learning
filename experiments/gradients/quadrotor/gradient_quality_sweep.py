"""Gradient-quality sweep on the drone obstacle-avoidance problem.

Experiment E1.2 + E2.1/E2.3 of the "Gradient Quality of Differentiable NMPC"
project (docs/notes/NOTE.md).

We measure how good the diffmpc2 SQP-ADMM *backward* gradient is, as a function of
  (a) how many inequality constraints are active  -- a 2x2 constraint grid, and
  (b) how loosely the NLP is solved              -- tolerance and ADMM-iteration sweeps,
by comparing the autodiff gradient against a tight-solve ground truth (both autodiff at
tol->0 and central finite differences).

Differentiable target (the diffmpc-as-policy gradient): the cost weights
`weights_penalization_reference_state_trajectory` (6-vector) and, since slack is always
on, `slack_penalization_weight` (scalar). Objective J(weights) = sum(states^2)+sum(controls^2)
for a single open-loop OCP solve. Seeds (initial states) are batched with vmap on GPU.

Run from the repo root:
    python examples/drone_obstacles/gradient_quality_sweep.py --smoke
    python examples/drone_obstacles/gradient_quality_sweep.py            # full sweep
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
jax_config.update("jax_threefry_partitionable", True)

import jax
import jax.numpy as jnp

# This project lives next to the diffmpc2 solver checkout
# (diffmpc-learning/research/.../experiments  and  diffmpc-learning/diffmpc2). Put the diffmpc2
# repo root ahead of site-packages so `import diffmpc` resolves to that checkout (a different
# diffmpc v1.0.0 may be pip-installed and would otherwise shadow it), and the drone-example dir
# for sibling imports (timing_drone, benchmark_drone_params).
_HERE = os.path.dirname(os.path.abspath(__file__))                     # .../experiments/quadrotor
_DIFFMPC2 = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "diffmpc2"))
_DRONE = os.path.join(_DIFFMPC2, "examples", "drone_obstacles")
for _p in (_DIFFMPC2, _DRONE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import benchmark_drone_params as P  # noqa: E402
from timing_drone import make_drone_config  # noqa: E402

from diffmpc.solvers.sqp_admm import (  # noqa: E402
    SQPADMMSolver,
    ForwardBackend,
    BackwardBackend,
)
from diffmpc.utils.load_params import load_solver_params  # noqa: E402

# Pure-JAX GPU backends for both reference and swept runs, so the gradient-error
# measurement is not confounded by FFI-vs-JAX numerical differences.
FWD = ForwardBackend.ADMM_JAX_LOOP_PCG
BWD = BackwardBackend.DIRECT_JAX_DENSE

# Order in which weight leaves are flattened into a single per-seed gradient vector.
WEIGHT_ORDER = ["weights_penalization_reference_state_trajectory", "slack_penalization_weight"]

# 2x2 constraint grid (slack always on). umax values calibrated so "tight" binds and
# "loose" does not -- verified by the per-cell activity print. The unconstrained-optimal
# controls have max|u|~0.82, p90~0.12, so umax=0.1 saturates the upper ~10-15% of controls
# (clear box activity) while umax=1e4 never binds.
CELLS: Dict[str, Dict[str, Any]] = {
    "C1_loose_off": dict(umax=1e4, obstacles=False),
    "C2_tight_off": dict(umax=0.1, obstacles=False),
    "C3_loose_on": dict(umax=1e4, obstacles=True),
    "C4_tight_on": dict(umax=0.1, obstacles=True),
}

# Ground-truth reference solve. The obstacle SQP needs ~30 iterations (no line search; line
# search empirically slows it here) to converge to ~5e-5, vs ~5e-4 at 18 iters -- a tight
# reference is needed to separate the active-set gradient effect from non-convergence. The
# sweeps measure deviation from this reference.
REF = dict(tol=1e-9, admm_max_iter=1500, scp_iter=30)

# Sweeps hold SQP/ADMM structure fixed except the swept knob, to bound per-solve cost.
SWEEP_SCP = 15          # SQP iteration cap for all swept points
SWEEP_A_ADMM = 800      # ADMM cap while sweeping tolerance (Sweep A)
SWEEP_B_TOL = 1e-10     # (un-reachable) tolerance while sweeping ADMM budget (Sweep B)

# Default sweeps.
TOL_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
ADMM_ITER_LIST = [5, 10, 25, 50, 100, 300, 800]


# ---------------------------------------------------------------------------
# Problem / solver construction
# ---------------------------------------------------------------------------

def solver_params(tol: float, admm_max_iter: int, scp_iter: int) -> Dict[str, Any]:
    """Solver params with overridden NLP tolerance / iteration budget."""
    sp = load_solver_params("sqp_admm.yaml")
    sp["num_scp_iteration_max"] = scp_iter
    sp["tol_convergence"] = tol
    sp["linesearch"] = False
    sp["warm_start_backward"] = True
    sp["admm"]["max_iter"] = admm_max_iter
    sp["admm"]["eps_abs"] = tol
    sp["admm"]["eps_rel"] = tol
    sp["admm"]["check_termination_every"] = 1
    sp["admm"]["pcg"]["max_iter"] = 500
    sp["admm"]["pcg"]["tol_epsilon"] = 1e-18
    return sp


def cell_problem(cell: str, horizon: int = None):
    """Return (program, problem_params, weights0, umax, obstacles_on) for a grid cell."""
    spec = CELLS[cell]
    obstacles_on = spec["obstacles"]
    if obstacles_on:
        obs_c = jnp.asarray(P.OBS_CENTERS)
        obs_r = jnp.asarray(P.OBS_RADII)
    else:
        obs_c = jnp.zeros((0, 2), dtype=jnp.float64)
        obs_r = jnp.zeros((0,), dtype=jnp.float64)

    cfg = make_drone_config(
        use_slack=True,
        slack_weight=10.0,
        obs_centers=obs_c,
        obs_radii=obs_r,
        horizon=horizon,
    )
    pp = dict(cfg.problem_params)
    umax = float(spec["umax"])
    pp["control_min_bounds"] = jnp.full(P.DRONE_NU, -umax, dtype=jnp.float64)
    pp["control_max_bounds"] = jnp.full(P.DRONE_NU, umax, dtype=jnp.float64)

    program = cfg.problem_class(dynamics=cfg.dynamics, params=pp)
    weights0 = {
        "weights_penalization_reference_state_trajectory": jnp.asarray(
            pp["weights_penalization_reference_state_trajectory"], dtype=jnp.float64
        ),
        "slack_penalization_weight": jnp.asarray(pp["slack_penalization_weight"], dtype=jnp.float64),
    }
    return program, pp, weights0, umax, obstacles_on


# ---------------------------------------------------------------------------
# Solve / objective
# ---------------------------------------------------------------------------

def make_solve_fns(solver: SQPADMMSolver, pp: Dict[str, Any]):
    """Build (cost, solve) closures for one solver, pure in (weights, x0)."""

    def _solve(weights, x0):
        pp_seed = {**pp, "initial_state": x0}
        guess = solver.initial_guess(pp_seed)
        return solver.solve(guess, pp_seed, weights)

    def cost(weights, x0):
        sol = _solve(weights, x0)
        return jnp.sum(sol.states ** 2) + jnp.sum(sol.controls ** 2)

    return cost, _solve


def make_grad_solve(solver: SQPADMMSolver, pp: Dict[str, Any]):
    """Return (grad_solve, cost_only) jitted+vmapped over seeds for one solver.

    grad_solve(weights, x0_batch) -> (grad_pytree, solution)  [one compile, reuses the
    forward solve for both the gradient and diagnostics]. cost_only(weights, x0_batch) ->
    (n_seeds,) scalar objective, used for finite differences.
    """
    cost, _solve = make_solve_fns(solver, pp)

    def f(weights, x0):
        sol = _solve(weights, x0)
        return jnp.sum(sol.states ** 2) + jnp.sum(sol.controls ** 2), sol

    def vg(weights, x0):
        (_, sol), g = jax.value_and_grad(f, has_aux=True)(weights, x0)
        return g, sol

    grad_solve = jax.jit(jax.vmap(vg, in_axes=(None, 0)))
    cost_only = jax.jit(jax.vmap(lambda w, x: cost(w, x), in_axes=(None, 0)))
    return grad_solve, cost_only


def fd_grad_batched(cost_batched, weights0, x0_batch, eps: float = 1e-5):
    """Per-seed central-difference gradient pytree (leading seed axis).

    `cost_batched(weights, x0_batch) -> (n_seeds,)` is the jitted batched objective, so each
    perturbation is a single batched solve. ~ (sum of weight sizes) * 2 batched solves total.
    """
    n_seeds = x0_batch.shape[0]
    out: Dict[str, np.ndarray] = {}
    for k, v in weights0.items():
        v = jnp.asarray(v)
        flat = v.ravel()
        g = np.zeros((n_seeds, flat.size), dtype=np.float64)
        for i in range(flat.size):
            wp = {**weights0, k: flat.at[i].add(eps).reshape(v.shape)}
            wm = {**weights0, k: flat.at[i].add(-eps).reshape(v.shape)}
            num = np.asarray(cost_batched(wp, x0_batch)) - np.asarray(cost_batched(wm, x0_batch))
            g[:, i] = num / (2.0 * eps)
        out[k] = g.reshape((n_seeds,) + v.shape)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def flatten_grad(g: Dict[str, Any], n_seeds: int) -> np.ndarray:
    """Flatten a per-seed gradient pytree into (n_seeds, D)."""
    leaves = [np.asarray(g[k], dtype=np.float64).reshape(n_seeds, -1) for k in WEIGHT_ORDER]
    return np.concatenate(leaves, axis=1)


def grad_metrics(g: np.ndarray, gref: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-seed gradient-quality metrics of g vs reference gref (both (n_seeds, D))."""
    dot = np.sum(g * gref, axis=1)
    ng = np.linalg.norm(g, axis=1)
    nr = np.linalg.norm(gref, axis=1)
    eps = 1e-300
    cos = dot / (ng * nr + eps)
    rel_l2 = np.linalg.norm(g - gref, axis=1) / (nr + eps)
    sign_agree = np.mean(np.sign(g) == np.sign(gref), axis=1)
    descent = (dot > 0).astype(np.float64)
    return dict(cos=cos, rel_l2=rel_l2, sign_agree=sign_agree, descent=descent, gnorm=ng)


def activity_counts(sol, umax: float, obstacles_on: bool, n_seeds: int):
    """Per-seed (#active total, #control-bound active, #obstacle active) from a batched
    solution. Computed directly from the trajectory (backend-independent)."""
    u = np.asarray(sol.controls)        # (n_seeds, N+1, nu)
    x = np.asarray(sol.states)          # (n_seeds, N+1, nx)
    box_active = np.sum(np.abs(u) >= (umax - 1e-4), axis=(1, 2)).astype(float)
    obs_active = np.zeros(n_seeds)
    if obstacles_on:
        pos = x[:, :, :2]               # (n_seeds, N+1, 2)
        for c, r in zip(P.OBS_CENTERS, P.OBS_RADII):
            d = np.linalg.norm(pos - c[None, None, :], axis=2)
            margin = 1.0 - d / (r + 0.001)
            obs_active += np.sum(margin >= -1e-3, axis=1)
    return box_active + obs_active, box_active, obs_active


def achieved_convergence(sol, n_seeds: int):
    """Per-seed (convergence_error, num_iter, last ADMM iters)."""
    conv = np.asarray(sol.convergence_error).reshape(n_seeds)
    num_iter = np.asarray(sol.num_iter).reshape(n_seeds)
    admm = np.asarray(sol.admm_iters)   # (n_seeds, max_iter)
    # last nonzero ADMM-iter count per seed (the iters used by the final SQP step)
    last_admm = np.zeros(n_seeds)
    for s in range(n_seeds):
        nz = admm[s][admm[s] > 0]
        last_admm[s] = nz[-1] if nz.size else 0
    eq_v = ineq_v = np.full(n_seeds, np.nan)
    if sol.solver_stats is not None:
        ev = np.asarray(sol.solver_stats.eq_constraints_violations)
        iv = np.asarray(sol.solver_stats.ineq_constraints_violations)
        ni = np.clip(num_iter.astype(int) - 1, 0, ev.shape[1] - 1)
        eq_v = ev[np.arange(n_seeds), ni]
        ineq_v = iv[np.arange(n_seeds), ni]
    return conv, num_iter, last_admm, eq_v, ineq_v


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "cell", "umax", "obstacles", "sweep", "sweep_var", "tol", "admm_max_iter", "scp_iter",
    "seed", "cos_AD", "rel_l2_AD", "sign_AD", "descent_AD",
    "cos_FD", "rel_l2_FD", "sign_FD", "descent_FD", "gnorm",
    "conv_error", "num_iter", "admm_iters_last", "eq_viol", "ineq_viol",
    "n_active", "n_active_box", "n_active_obs", "cos_AD_FD_tight", "fwd_ms", "bwd_ms",
]


def run_cell(cell: str, x0_batch, tol_list, admm_iter_list, writer, horizon=None, ref=REF):
    n_seeds = x0_batch.shape[0]
    program, pp, weights0, umax, obstacles_on = cell_problem(cell, horizon=horizon)

    # ---- Ground truth at the (tightest affordable) reference solve ----
    tight_solver = SQPADMMSolver(program, params=solver_params(**ref),
                                 forward_backend=FWD, backward_backend=BWD)
    ref_gs, ref_cost = make_grad_solve(tight_solver, pp)
    g_ad_pytree, sol_t = ref_gs(weights0, x0_batch)
    jax.block_until_ready((g_ad_pytree, sol_t))
    g_ad_tight = flatten_grad(g_ad_pytree, n_seeds)
    g_fd_tight = flatten_grad(fd_grad_batched(ref_cost, weights0, x0_batch), n_seeds)
    n_active, n_box, n_obs = activity_counts(sol_t, umax, obstacles_on, n_seeds)
    m_adfd = grad_metrics(g_ad_tight, g_fd_tight)          # RQ1: converged AD vs FD
    cos_ad_fd_tight = m_adfd["cos"]
    print(f"[{cell}] tight: mean #active={n_active.mean():.1f} "
          f"(box={n_box.mean():.1f}, obs={n_obs.mean():.1f})  "
          f"cos(AD,FD)={np.nanmean(cos_ad_fd_tight):.4f}  "
          f"||g_AD||={np.linalg.norm(g_ad_tight, axis=1).mean():.3e}")

    # ---- Sweep configs ----
    configs: List[Tuple[str, float, Dict[str, Any]]] = []
    for tol in tol_list:
        configs.append(("tol", tol, dict(tol=tol, admm_max_iter=SWEEP_A_ADMM, scp_iter=SWEEP_SCP)))
    for it in admm_iter_list:
        configs.append(("admm_iter", it, dict(tol=SWEEP_B_TOL, admm_max_iter=it, scp_iter=SWEEP_SCP)))

    for sweep, sweep_var, spc in configs:
        solver = SQPADMMSolver(program, params=solver_params(**spc),
                               forward_backend=FWD, backward_backend=BWD)
        gs, _ = make_grad_solve(solver, pp)

        out = gs(weights0, x0_batch)           # compile + warm-up
        jax.block_until_ready(out)
        t0 = time.monotonic()
        g_pytree, sol = gs(weights0, x0_batch)
        jax.block_until_ready((g_pytree, sol))
        bwd_ms = (time.monotonic() - t0) * 1000.0   # combined forward+backward
        fwd_ms = float("nan")
        g = flatten_grad(g_pytree, n_seeds)

        conv, num_iter, last_admm, eq_v, ineq_v = achieved_convergence(sol, n_seeds)
        m_ad = grad_metrics(g, g_ad_tight)
        m_fd = grad_metrics(g, g_fd_tight)

        for s in range(n_seeds):
            writer.writerow({
                "cell": cell, "umax": umax, "obstacles": int(obstacles_on),
                "sweep": sweep, "sweep_var": sweep_var,
                "tol": spc["tol"], "admm_max_iter": spc["admm_max_iter"], "scp_iter": spc["scp_iter"],
                "seed": s,
                "cos_AD": m_ad["cos"][s], "rel_l2_AD": m_ad["rel_l2"][s],
                "sign_AD": m_ad["sign_agree"][s], "descent_AD": m_ad["descent"][s],
                "cos_FD": m_fd["cos"][s], "rel_l2_FD": m_fd["rel_l2"][s],
                "sign_FD": m_fd["sign_agree"][s], "descent_FD": m_fd["descent"][s],
                "gnorm": m_ad["gnorm"][s],
                "conv_error": conv[s], "num_iter": int(num_iter[s]),
                "admm_iters_last": last_admm[s], "eq_viol": eq_v[s], "ineq_viol": ineq_v[s],
                "n_active": n_active[s], "n_active_box": n_box[s], "n_active_obs": n_obs[s],
                "cos_AD_FD_tight": cos_ad_fd_tight[s], "fwd_ms": fwd_ms, "bwd_ms": bwd_ms,
            })
        print(f"  [{cell}] {sweep}={sweep_var:<8g}  cos(AD,ref)={np.nanmean(m_ad['cos']):.4f}  "
              f"relL2={np.nanmean(m_ad['rel_l2']):.2e}  conv={np.nanmean(conv):.1e}  "
              f"SQP_it={num_iter.mean():.1f}  bwd={bwd_ms:.0f}ms")

        del solver, gs
        jax.clear_caches()
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=list(CELLS), choices=list(CELLS))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--tol-list", type=float, nargs="*", default=TOL_LIST)
    ap.add_argument("--admm-iter-list", type=int, nargs="*", default=ADMM_ITER_LIST)
    ap.add_argument("--horizon", type=int, default=None, help="OCP horizon (default: drone.yaml = 50).")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: 1 cell, 2 seeds, 2 tol + 2 admm-iter values.")
    args = ap.parse_args()

    assert jax.config.jax_enable_x64, "x64 must be enabled for trustworthy FD."
    dev = jax.devices()[0]
    print(f"JAX device: {dev} ({dev.platform})")
    if dev.platform != "gpu":
        print("WARNING: not running on GPU.")

    if args.smoke:
        cells = ["C1_loose_off"]
        n_seeds = 2
        tol_list = [1e-1, 1e-4]
        admm_iter_list = [10, 200]
    else:
        cells = args.cells
        n_seeds = args.seeds
        tol_list = args.tol_list
        admm_iter_list = args.admm_iter_list

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(out_dir, f"grad_quality_{'smoke_' if args.smoke else ''}{stamp}.csv")

    x0_batch = jnp.asarray(P.generate_drone_x0(n_seeds, args.base_seed), dtype=jnp.float64)
    print(f"cells={cells}  n_seeds={n_seeds}  tol_list={tol_list}  admm_iter_list={admm_iter_list}")
    print(f"writing -> {out_path}\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for cell in cells:
            run_cell(cell, x0_batch, tol_list, admm_iter_list, writer, horizon=args.horizon)
            f.flush()

    print(f"\nDone. CSV -> {out_path}")


if __name__ == "__main__":
    main()
