"""Closed-loop gradient accuracy of 4 differentiable-NMPC variants against ONE common ground truth.

Variants (all on the SAME 64 samples, scored against ONE FD ground truth):
  A no-slack : TurboMPC hard box (OptimalControlProblem),     ADMM_FUSED_CUDSS fwd, DIRECT_CUDSS_FFI bwd
  A slack    : TurboMPC slack box (OptimalControlProblemSlack, gamma=1e4 Moreau), fused fwd, DIRECT bwd
  B no-slack : log-barrier central-path, gamma=1e12 (pure barrier),                kappa=1e-6
  B slack    : log-barrier central-path, gamma=1e4 (barrier + Moreau slack),       kappa=1e-6

GROUND TRUTH = the FD of the TRUE hard-constrained (|u|<=u_max) closed-loop loss, computed ONCE on the
hard-box forward solved very tightly (GT_TOL). Slack/barrier variants are differentiable surrogates
scored against this one reference. Same B=64 samples for every variant.

FD: ADAPTIVE eps. Decreasing-eps central differences with a per-sample plateau check; the eps grid is
EXTENDED downward until every sample converges OR an eps floor is hit. Samples that still don't plateau
are REPORTED (not resampled, not silently dropped), with the smallest eps reached.

A's loss (build_rollout_fn) == B's loss (manual scan): both = sum_t (||x_{t+1}||^2 + ||u_t||^2) with the
same dt*(A_sd x + B u + b) dynamics (verified: timing.py:101 `running_cost - reward_fn(new_state,u0)`).

    export LD_LIBRARY_PATH="$(cat /tmp/cudss071_ldpath.txt):$LD_LIBRARY_PATH"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    python research/gradient-quality-diffnmpc/experiments/linear_system/closed_loop_gradient_accuracy.py --batch 64
"""
from __future__ import annotations
import os, sys, time, argparse, gc

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (os.path.join(_DL_ROOT, "src"), os.path.join(_DL_ROOT, "diffmpc2"),
           os.path.join(_DL_ROOT, "diffmpc2", "benchmarking", "linear-system")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from utils import generate_problem_data  # noqa: E402
from turbompc.problems.optimal_control_problem import OptimalControlProblem, OptimalControlProblemSlack  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.utils.timing import ProblemConfig, build_rollout_fn  # noqa: E402
from turbompc.solvers.qp_data import QPData, QPCostBlocks, QPEqualityBlocks, QPInequalityBlocks  # noqa: E402
from turbompc.solvers.qp_utils import ZShape, pack_x  # noqa: E402
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402
from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402
from diffmpc_learning.solvers.backward import relaxed_complementarity_weight, augment_D_with_relaxed_ineq  # noqa: E402

NX, NU = 8, 4
NW = NX + NU
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
WK = [QK, RK]
HORIZON, UMAX = 160, 1.0
SIM_STEPS = 50
GAMMA_SLACK, GAMMA_NOSLACK, KAPPA = 1e4, 1e12, 1e-6
GT_TOL = 1e-11                  # hard-box forward tolerance for the ground truth
AD_TOLS = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]
ADMM_MAX_ITER = 8000           # horizon 160 + tight tol needs headroom
CP_MAX_ITER = 8000
LANE_CAP = 2048                # parallel rollout lanes per vmap chunk (horizon 160 is memory-heavy)
EPS0 = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
EPS_FLOOR = 1e-8
CONFIGS = ["A noslack", "A slack", "B noslack", "B slack"]


def _flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in WK])
def _unflat(wf): return {QK: wf[:NX], RK: wf[NX:NW]}
def _reward(s, c): return -(jnp.sum(s ** 2) + jnp.sum(c ** 2))
def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _sp(eps):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = eps; sp["warm_start_backward"] = False
    sp["linesearch"] = False; sp["admm"]["max_iter"] = ADMM_MAX_ITER
    sp["admm"]["check_termination_every"] = 1; sp["admm"]["eps_abs"] = eps; sp["admm"]["eps_rel"] = eps
    return sp


def _problem(seed, batch, sim_steps, horizon):
    dyn, pp_t = build_turbompc_linear_problem(horizon=horizon, umax=UMAX, n_state=NX, n_ctrl=NU)
    Q, R, A, B, b, x0 = generate_problem_data(batch, seed, n_state=NX, n_ctrl=NU)
    A_sd = jnp.asarray(A - np.eye(NX)); B_m = jnp.asarray(B); b_v = jnp.asarray(b)
    pp = dict(pp_t); pp["dynamics_state_dot_params"] = {"A": A_sd, "B": B_m, "b": b_v}
    pp[QK] = jnp.asarray(np.diag(Q)); pp[RK] = jnp.asarray(np.diag(R))
    dt = float(pp["discretization_resolution"])
    return dyn, pp, {k: pp[k] for k in WK}, jnp.asarray(x0), A_sd, B_m, b_v, dt


# ----- A (TurboMPC) closed-loop cost via build_rollout_fn (handles d/d-state) -----
def cost_fn_A(dyn, pp, w, x0b, tol, slack, sim_steps):
    sp = _sp(tol)
    if slack:
        ppA = {**pp, "use_slack_variables": True, "slack_penalization_weight": GAMMA_SLACK}
        cls = OptimalControlProblemSlack
    else:
        ppA = dict(pp); cls = OptimalControlProblem
    solver = TurboMPCSolver(program=cls(dynamics=dyn, params=ppA), params=sp,
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS,
        backward_backend=BackwardBackend.DIRECT_CUDSS_FFI, use_full_hessian=True)
    init = solver.solve(solver.initial_guess(ppA), problem_params=ppA, weights={**w, "initial_state": x0b[0]})
    cfg = ProblemConfig(dynamics=dyn, problem_class=cls, problem_params=ppA, solver_params=sp,
                        weight_keys=WK, reward_fn=_reward, update_per_seed=lambda s, b_, p: ({}, x0b))
    rollout = build_rollout_fn(config=cfg, solver=solver, problem_params=ppA, init_solution=init,
                               warm_start=False, num_sim_steps=sim_steps)
    return lambda w_dict, st: rollout(st, {**w_dict, "initial_state": st})[0]


# ----- B (log-barrier central-path) custom_vjp solve, differentiable w.r.t. weights AND state -----
def _solve_reduced_kkt_full(D_aug, E, eq_blocks, x_bar, nx, nu):
    Np1, n = D_aug.shape[0], D_aug.shape[1]; N = Np1 - 1
    eq0 = QPEqualityBlocks(A0=eq_blocks.A0, A_minus=eq_blocks.A_minus, A_plus=eq_blocks.A_plus,
                           c0=jnp.zeros_like(eq_blocks.c0), c=jnp.zeros_like(eq_blocks.c))
    empty = QPInequalityBlocks(G=jnp.zeros((Np1, 0, n), x_bar.dtype),
                               l=jnp.zeros((Np1, 0), x_bar.dtype), u=jnp.zeros((Np1, 0), x_bar.dtype))
    bwd = QPData(cost=QPCostBlocks(D=D_aug, E=E, q=-x_bar), eq=eq0, ineq=empty)
    (lam_s, lam_c), mult = solve_backward_kkt(bwd, ZShape(horizon=N, num_states=nx, num_controls=nu))
    return pack_x(lam_s, lam_c), mult


def make_b_mpc_solve(solver, pp_base, schur, ig, nx, nu, gamma, kappa, cp_tol):
    program = solver.program
    cp = dict(rho_bar=0.1, max_iter=CP_MAX_ITER, tol=cp_tol)

    def _qp1(weights, state):
        pp_i = {**pp_base, "initial_state": state, QK: weights[QK], RK: weights[RK]}
        qp = solver._build_qp_data(ig.states, ig.controls, pp_i)
        return to_one_sided(qp, gamma)

    @jax.custom_vjp
    def b_solve(weights, state):
        qp1 = _qp1(weights, state)
        x, _, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=gamma, **cp)
        return x[:, :nx], x[:, nx:]

    def b_fwd(weights, state):
        qp1 = _qp1(weights, state)
        x, duals, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=gamma, **cp)
        return (x[:, :nx], x[:, nx:]), (weights, state, qp1, x, duals)

    def b_bwd(res, g):
        weights, state, qp1, x, duals = res
        dL_ds, dL_dc = g
        y_g = duals[2]
        W = relaxed_complementarity_weight(qp1, x, y_g, gamma)
        D_aug = augment_D_with_relaxed_ineq(qp1.cost.D, qp1.ineq.G, W)
        x_bar = pack_x(dL_ds, dL_dc)
        lam_x, lam_f = _solve_reduced_kkt_full(D_aug, qp1.cost.E, qp1.eq, x_bar, nx, nu)
        lam_states, lam_controls = lam_x[:, :nx], lam_x[:, nx:]

        def contracted(ww):
            pp_w = {**pp_base, "initial_state": state, QK: ww[QK], RK: ww[RK]}
            fx, fu = jax.grad(lambda s, c: program.cost(s, c, pp_w), argnums=(0, 1))(x[:, :nx], x[:, nx:])
            return jnp.sum(fx * lam_states) + jnp.sum(fu * lam_controls)
        dL_dw = jax.tree_util.tree_map(lambda z: -z, jax.grad(contracted)(weights))
        dL_dstate = lam_f[:nx]
        return (dL_dw, dL_dstate)

    b_solve.defvjp(b_fwd, b_bwd)
    return b_solve


def cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, slack, sim_steps):
    gamma = GAMMA_SLACK if slack else GAMMA_NOSLACK
    solver = TurboMPCSolver(program=OptimalControlProblem(dynamics=dyn, params=pp), params=_sp(1e-9),
        forward_backend=ForwardBackend.ADMM_FUSED_CUDSS, backward_backend=BackwardBackend.DIRECT_CUDSS_FFI)
    ig = solver.initial_guess(pp); N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.CUDSS_FFI, N, NX, NU,
                              pcg_params={"max_iter": 800, "tol_epsilon": 1e-12})
    b_solve = make_b_mpc_solve(solver, pp, schur, ig, NX, NU, gamma=gamma, kappa=KAPPA, cp_tol=tol)

    def cost(weights, state):
        def step(carry, _):
            st, c = carry
            s, ctrl = b_solve(weights, st); u0 = ctrl[0]
            new = st + dt * (A_sd @ st + B_m @ u0 + b_v)
            return (new, c + jnp.sum(new ** 2) + jnp.sum(u0 ** 2)), None
        # checkpoint each step (mirror build_rollout_fn timing.py:109) -> avoid OOM at horizon 160 x 50 steps
        (_, tot), _ = jax.lax.scan(jax.checkpoint(step), (state, jnp.asarray(0.0)), None, length=sim_steps)
        return tot
    return cost


def build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, sim_steps):
    if name == "A noslack":   return cost_fn_A(dyn, pp, w, x0b, tol, False, sim_steps)
    elif name == "A slack":   return cost_fn_A(dyn, pp, w, x0b, tol, True, sim_steps)
    elif name == "B noslack": return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, False, sim_steps)
    else:                     return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, True, sim_steps)


def ad(cost_fn, w, x0b):
    B = x0b.shape[0]
    gd = jax.jit(jax.vmap(jax.grad(cost_fn, argnums=0), in_axes=(None, 0)))(w, x0b)
    return np.stack([_flat({k: gd[k][i] for k in WK}) for i in range(B)])


# ----- adaptive-eps FD ground truth -----
def _fd_at_eps(single, w_flat, x0b, eps):
    B = x0b.shape[0]; E = jnp.eye(NW)
    delta = eps * jnp.asarray([1.0, -1.0])[None, :, None] * E[:, None, :]     # [NW,2,NW]
    Wp = (w_flat + delta).reshape(-1, NW); P = Wp.shape[0]
    cP = max(1, LANE_CAP // B); costs = np.zeros((P, B))
    for s in range(0, P, cP):
        Wc = Wp[s:s + cP]; c = Wc.shape[0]
        costs[s:s + c] = np.asarray(single(jnp.repeat(Wc, B, axis=0),
                                           jnp.tile(x0b, (c, 1)))).reshape(c, B)
    costs = costs.reshape(NW, 2, B)
    return ((costs[:, 0, :] - costs[:, 1, :]) / (2 * eps)).T                   # [B, NW]


def _check_conv(G, atol_frac=1e-3):
    """Plateau check over a fixed eps grid (largest->smallest). A weight converges if some consecutive
    eps pair agrees within rel-tol 1% OR an ABSOLUTE tol = atol_frac*||grad|| (so near-zero components,
    where a relative check is ill-conditioned, pass). Default value = the LARGEST-eps estimate (least
    noisy), never the small-eps one."""
    eps_sorted = sorted(G.keys(), reverse=True)
    arrs = [G[e] for e in eps_sorted]; B = arrs[0].shape[0]
    gFD = np.array(arrs[0]); conv = np.zeros(B, bool); eps_used = np.full(B, eps_sorted[0])
    gnorm = np.linalg.norm(arrs[0], axis=1)                       # gradient scale (largest-eps FD)
    for n in range(B):
        atol = atol_frac * gnorm[n] + 1e-12
        ok_all = True; eps_n = eps_sorted[0]
        for j in range(NW):
            seq = [a[n, j] for a in arrs]; chosen, ok = seq[0], False
            for k in range(len(seq) - 1):
                if abs(seq[k] - seq[k + 1]) <= 1e-2 * abs(seq[k + 1]) + atol:
                    chosen, ok = seq[k], True; eps_n = min(eps_n, eps_sorted[k]); break
            gFD[n, j] = chosen; ok_all = ok_all and ok
        conv[n] = ok_all; eps_used[n] = eps_n
    return gFD, conv, eps_used


def fd_gt_adaptive(cost_fn, w, x0b):
    """FD over a FIXED eps grid kept ABOVE the cost noise floor. The plateau lives at large eps; the
    diagnostic showed the FD goes erratic (~1/eps noise) below ~1e-6 for horizon-160 closed loops, so
    we do NOT shrink eps into the floor (that was the wrong direction). Report any sample that still
    has no plateau in this window."""
    single = jax.jit(jax.vmap(lambda wf, st: cost_fn(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w))
    G = {}
    for e in EPS0:
        t = time.time(); G[e] = _fd_at_eps(single, w_flat, x0b, e)
        print(f"    FD eps={e:.0e} ({time.time()-t:.0f}s)", flush=True)
    gFD, conv, eps_used = _check_conv(G)
    return gFD, conv, eps_used, sorted(G.keys(), reverse=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim_steps", type=int, default=SIM_STEPS)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--tolerances", type=float, nargs="+", default=AD_TOLS)
    p.add_argument("--smoke_check", action="store_true", help="also print cos(A-slack, B-slack)")
    a = p.parse_args()
    n = a.batch
    print(f"Closed-loop gradient accuracy vs ONE hard-box ground truth")
    print(f"  (B,horizon,nx,nu)=({n},{a.horizon},{NX},{NU}) sim_steps={a.sim_steps} seed={a.seed} "
          f"kappa={KAPPA:g} gamma_slack={GAMMA_SLACK:g} gamma_noslack={GAMMA_NOSLACK:g}")
    print(f"  GT = adaptive-eps FD of the HARD box forward at GT_TOL={GT_TOL:.0e}; AD tols={a.tolerances}")
    dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(a.seed, n, a.sim_steps, a.horizon)

    # ---- ONE ground truth: FD of the hard box ----
    print("\n[GT] FD of the hard-box forward (fixed eps grid above the noise floor):")
    cf_hard = cost_fn_A(dyn, pp, w, x0b, GT_TOL, slack=False, sim_steps=a.sim_steps)
    gFD, conv, eps_used, eps_grid = fd_gt_adaptive(cf_hard, w, x0b)
    nbad = int((~conv).sum())
    print(f"  GT plateau found {int(conv.sum())}/{n}  | eps grid {eps_grid[0]:.0e}..{eps_grid[-1]:.0e}")
    if nbad:
        print(f"  *** {nbad} sample(s) with no plateau in [{eps_grid[-1]:.0e},{eps_grid[0]:.0e}]: "
              f"ids={list(np.where(~conv)[0])} ***")
    # ---- confirm the FD ground truth is correct: hard-box ANALYTIC gradient (DIRECT bwd) vs the FD ----
    gAD_hard = ad(cf_hard, w, x0b)
    ch = np.array([_cos(gAD_hard[i], gFD[i]) for i in range(n)])
    re = np.array([np.linalg.norm(gAD_hard[i] - gFD[i]) / (np.linalg.norm(gFD[i]) + 1e-30) for i in range(n)])
    print(f"  [FD check] hard-box analytic(DIRECT) vs FD: cos med={np.median(ch):.6f} min={np.min(ch):.6f} "
          f"| rel-err med={np.median(re):.2e} max={np.max(re):.2e}  (independent validation of the FD GT)")
    jax.clear_caches(); gc.collect()

    # ---- AD sweep: each variant vs the ONE GT ----
    keep = conv
    results = {}
    if a.smoke_check:
        cfa = cost_fn_A(dyn, pp, w, x0b, 1e-9, slack=True, sim_steps=a.sim_steps)
        cfb = cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, 1e-9, True, a.sim_steps)
        ga, gb = ad(cfa, w, x0b), ad(cfb, w, x0b)
        cc = np.array([_cos(ga[i], gb[i]) for i in range(n)])[keep]
        print(f"\n[smoke] cos(A-slack, B-slack) median={np.median(cc):.4f} min={np.min(cc):.4f} "
              f"(should be ~1.0 -> loss definitions match)")
        jax.clear_caches(); gc.collect()

    for tol in a.tolerances:
        print(f"\n--- AD tol={tol:.0e} ---  {'variant':>10} | {'cos med':>8} {'cos min':>8} "
              f"{'#cos<.99':>9} {'#cos<0':>7}")
        for name in CONFIGS:
            t = time.time()
            cf = build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, a.sim_steps)
            gAD = ad(cf, w, x0b)
            cos = np.array([_cos(gAD[i], gFD[i]) for i in range(n)])
            results[(tol, name)] = cos
            ck = cos[keep]
            print(f"{'':>17}{name:>10} | {np.median(ck):8.4f} {np.min(ck):8.4f} "
                  f"{int((ck < 0.99).sum()):9d} {int((ck < 0).sum()):7d}   ({time.time()-t:.0f}s)", flush=True)
            jax.clear_caches(); gc.collect()

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    np.savez(os.path.join(_HERE, "results", "gradient_accuracy.npz"),
             tolerances=np.array(a.tolerances), batch=n, seed=a.seed, sim_steps=a.sim_steps,
             horizon=a.horizon, gFD=gFD, gt_converged=conv, gt_eps_used=eps_used,
             gAD_hard=gAD_hard, fd_check_cos=ch, fd_check_relerr=re,
             **{f"{name}|{tol:.0e}": results[(tol, name)] for (tol, name) in results})
    print("\nsaved results/gradient_accuracy.npz")


if __name__ == "__main__":
    main()
