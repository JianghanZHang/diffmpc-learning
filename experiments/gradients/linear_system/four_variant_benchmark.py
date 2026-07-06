"""4-variant closed-loop gradient accuracy on the CORRECT solver (external/turbompc), 10 seeds, reporting
BOTH per-sample and batch-summed cosine vs ONE common hard-box ground truth.

Variants (same shape as closed_loop_gradient_accuracy.py, but on external/turbompc):
  1 turbompc hard box   : OptimalControlProblem,        ADMM_FUSED_CUDSS fwd, DIRECT_CUDSS_FFI bwd
  2 turbompc moreau     : OptimalControlProblemSlack (gamma=1e4), fused fwd, DIRECT bwd
  3 log-barrier no-slack: central-path, gamma=1e12 (pure barrier),  kappa=1e-6
  4 log-barrier slack   : central-path, gamma=1e4 (barrier+moreau), kappa=1e-6

GROUND TRUTH = convergence-checked FD of the TRUE hard-constrained closed-loop loss (external's hard box at
GT_TOL), computed ONCE per seed; same samples for all 4 variants. Per-sample cos (10x64=640) AND
batch-summed cos_all (10 values, one per seed) for each (variant, AD-tol).

NO FILTERING: every sample is kept and scored. Samples whose hard-box FD GT does NOT converge to a
plateau sit at/near an active-set discontinuity where the hard-box gradient is ill-defined (CLAUDE.md FD
rule) -- they are NOT dropped; they are FLAGGED (count + indices) and isolated in a 'non-conv subset'
column so the reader sees exactly where the GT is unreliable rather than having it hidden.

NOTE: external/turbompc is the verified-correct solver (diffmpc2 release-cleanup has the multiplier-sign
bug). Run with the .venv-cudss python (matches external's FFI build):

    /home/jianghan/Workspace/diffmpc2/.venv-cudss/bin/python \
        experiments/linear_system/four_variant_benchmark.py --nseeds 10
"""
from __future__ import annotations
import os, sys, time, argparse, gc
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# external/turbompc is the CANONICAL solver (see CLAUDE.md callout). NOT diffmpc2.
for _p in (os.path.join(_DL_ROOT, "src"),
           os.path.join(_DL_ROOT, "external", "diffmpc2"),
           os.path.join(_DL_ROOT, "external", "diffmpc2", "benchmarking", "linear-system")):
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
HORIZON, UMAX = 40, 1.0                 # benchmark horizon
SIM_STEPS = 50
GAMMA_SLACK, GAMMA_NOSLACK, KAPPA = 1e4, 1e12, 1e-6
GT_TOL = 1e-11
AD_TOLS = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]
ADMM_MAX_ITER = 4000
CP_MAX_ITER = 4000
LANE_CAP = 4096
EPS0 = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
CONFIGS = ["1 turbompc-hardbox", "2 turbompc-moreau", "3 barrier-noslack", "4 barrier-slack"]


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


# ----- A (TurboMPC) closed-loop cost -----
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


# ----- B (central-path / log-barrier) -----
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
        (_, tot), _ = jax.lax.scan(jax.checkpoint(step), (state, jnp.asarray(0.0)), None, length=sim_steps)
        return tot
    return cost


def build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, sim_steps):
    if name == "1 turbompc-hardbox": return cost_fn_A(dyn, pp, w, x0b, tol, False, sim_steps)
    if name == "2 turbompc-moreau":  return cost_fn_A(dyn, pp, w, x0b, tol, True, sim_steps)
    if name == "3 barrier-noslack":  return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, False, sim_steps)
    return cost_fn_B(dyn, pp, w, x0b, A_sd, B_m, b_v, dt, tol, True, sim_steps)


def ad(cost_fn, w, x0b):
    B = x0b.shape[0]
    gd = jax.jit(jax.vmap(jax.grad(cost_fn, argnums=0), in_axes=(None, 0)))(w, x0b)
    return np.stack([_flat({k: gd[k][i] for k in WK}) for i in range(B)])


# ----- adaptive-eps FD ground truth -----
def _fd_at_eps(single, w_flat, x0b, eps):
    B = x0b.shape[0]; E = jnp.eye(NW)
    delta = eps * jnp.asarray([1.0, -1.0])[None, :, None] * E[:, None, :]
    Wp = (w_flat + delta).reshape(-1, NW); P = Wp.shape[0]
    cP = max(1, LANE_CAP // B); costs = np.zeros((P, B))
    for s in range(0, P, cP):
        Wc = Wp[s:s + cP]; c = Wc.shape[0]
        costs[s:s + c] = np.asarray(single(jnp.repeat(Wc, B, axis=0),
                                           jnp.tile(x0b, (c, 1)))).reshape(c, B)
    costs = costs.reshape(NW, 2, B)
    return ((costs[:, 0, :] - costs[:, 1, :]) / (2 * eps)).T


def _check_conv(G, atol_frac=1e-3):
    eps_sorted = sorted(G.keys(), reverse=True)
    arrs = [G[e] for e in eps_sorted]; B = arrs[0].shape[0]
    gFD = np.array(arrs[0]); conv = np.zeros(B, bool); eps_used = np.full(B, eps_sorted[0])
    gnorm = np.linalg.norm(arrs[0], axis=1)
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
    single = jax.jit(jax.vmap(lambda wf, st: cost_fn(_unflat(wf), st)))
    w_flat = jnp.asarray(_flat(w)); G = {}
    for e in EPS0:
        G[e] = _fd_at_eps(single, w_flat, x0b, e)
    gFD, conv, eps_used = _check_conv(G)
    return gFD, conv, eps_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nseeds", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--smoke", action="store_true", help="2 seeds + sanity check")
    a = p.parse_args()
    nseeds = 2 if a.smoke else a.nseeds
    n = a.batch; tols = AD_TOLS
    print(f"4-variant benchmark on external/turbompc | horizon={HORIZON} batch={n} seeds={nseeds} "
          f"sim_steps={SIM_STEPS} | GT=hard-box FD @ {GT_TOL:.0e}, AD tols={tols}")

    ps = {(name, t): [] for name in CONFIGS for t in tols}   # per-sample cos arrays (one per seed), ALL samples
    bs = {(name, t): [] for name in CONFIGS for t in tols}   # batch-summed cos_all (one scalar per seed), ALL samples
    conv_per_seed = []          # per-seed FD-convergence mask (n,), shared GT across variants
    gt_bad = 0
    for seed in range(nseeds):
        t0 = time.time()
        dyn, pp, w, x0b, A_sd, B_m, b_v, dt = _problem(seed, n, SIM_STEPS, HORIZON)
        cf_hard = cost_fn_A(dyn, pp, w, x0b, GT_TOL, slack=False, sim_steps=SIM_STEPS)
        gFD, conv, _ = fd_gt_adaptive(cf_hard, w, x0b)
        conv = np.asarray(conv); conv_per_seed.append(conv)
        nbad = int((~conv).sum()); gt_bad += nbad
        if nbad:
            print(f"  [seed {seed}] WARNING: {nbad}/{n} GT-FD samples did NOT converge to a plateau "
                  f"(at/near an active-set boundary; hard-box GT ill-defined). KEPT and scored anyway; "
                  f"flagged in the 'non-conv subset' column. idx={list(np.nonzero(~conv)[0])}", flush=True)
        if a.smoke:
            gAD0 = ad(cf_hard, w, x0b)
            cs = np.array([_cos(gAD0[i], gFD[i]) for i in range(n)])
            print(f"  [sanity seed {seed}] cos(hardbox AD@{GT_TOL:.0e}, hardbox GT) over ALL {n} samples "
                  f"med={np.median(cs):.4f} min={np.min(cs):.4f}  (should be ~1.0 on converged samples)", flush=True)
        for name in CONFIGS:
            for t in tols:
                cf = build_cost_fn(name, dyn, pp, w, x0b, A_sd, B_m, b_v, dt, t, SIM_STEPS)
                gAD = ad(cf, w, x0b)
                cos = np.array([_cos(gAD[i], gFD[i]) for i in range(n)])
                ps[(name, t)].append(cos)                            # ALL samples (no keep filter)
                bs[(name, t)].append(_cos(gAD.sum(0), gFD.sum(0)))   # batch-sum over ALL samples
                jax.clear_caches(); gc.collect()
        print(f"  seed {seed} done ({time.time()-t0:.0f}s); GT non-converged so far={gt_bad}", flush=True)

    # ---- report ----
    conv_all = np.concatenate(conv_per_seed) if conv_per_seed else np.zeros(0, bool)
    n_tot = conv_all.size; n_bad = int((~conv_all).sum())
    print(f"\n=== cos(variant AD, common hard-box GT) | {nseeds} seeds, {n_tot} samples "
          f"(ALL KEPT, none dropped); GT-FD non-converged={n_bad} ===")
    if n_bad:
        print(f"!! {n_bad}/{n_tot} samples sit at/near an active-set discontinuity where the hard-box FD GT is\n"
              f"   ill-defined (CLAUDE.md FD rule). They are KEPT in every 'per-sample' / 'batch-sum' number\n"
              f"   below; the 'non-conv subset' column isolates them (median over just those samples) so the\n"
              f"   contamination is visible, not hidden. A LOW headline + LOW non-conv-subset = the GT itself is\n"
              f"   suspect there, not necessarily the AD.")
    for t in tols:
        print(f"\n-- AD tol={t:.0e} --   {'variant':<20} | {'per-sample med(min)':>20} {'#<.99':>6} "
              f"{'#<0':>5} | {'batch-sum med(min)':>20} | {'non-conv subset med':>22}")
        for name in CONFIGS:
            psall = np.concatenate(ps[(name, t)]); bsall = np.array(bs[(name, t)])
            sub = psall[~conv_all]
            sub_str = f"{np.median(sub):8.3f} (n={sub.size})" if sub.size else f"{'--':>8} (n=0)"
            print(f"{'':>9}{name:<20} | {np.median(psall):8.3f} ({np.min(psall):6.3f})    "
                  f"{int((psall < 0.99).sum()):6d} {int((psall < 0).sum()):5d} | "
                  f"{np.median(bsall):8.3f} ({np.min(bsall):6.3f}) | {sub_str:>22}")

    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)
    save = {"configs": np.array(CONFIGS), "tols": np.array(tols), "nseeds": nseeds, "batch": n,
            "horizon": HORIZON, "gt_nonconverged": gt_bad, "gt_conv_mask": conv_all}
    for (name, t), v in ps.items():
        save[f"ps|{name}|{t:.0e}"] = np.concatenate(v)
    for (name, t), v in bs.items():
        save[f"bs|{name}|{t:.0e}"] = np.array(v)
    out = "four_variant_benchmark_smoke.npz" if a.smoke else "four_variant_benchmark.npz"
    np.savez(os.path.join(_HERE, "results", out), **save)   # smoke -> separate file, never clobbers the canonical npz
    print(f"\nsaved results/{out}")


if __name__ == "__main__":
    main()
