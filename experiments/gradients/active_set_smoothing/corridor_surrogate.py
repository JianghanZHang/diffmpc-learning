"""Convex coupled-linear 'corridor' surrogate — Phase-2 stepping stone (CPU, no inequality Hessian).

2-D double integrator, quadratic tracking to a goal placed just OUTSIDE a 2-wall corner. Sweeping the
position-tracking weight `wq` presses the path into the corner, where BOTH walls activate together
(coupled near-active set). We differentiate a downstream task loss L(wq) w.r.t. wq (the diffmpc-as-policy
gradient) and compare HARD (TurboMPC, DIRECT backward) vs BARRIER (central-path) across the activation.

Linear walls => grad^2 g = 0 => the existing central-path backward is exact => this validates the
sweep / multiplier-crossing / conditioning / tol-band HARNESS before the obstacle's curvature term.

CPU-only (JAX_DENSE Schur, JAX_LOOP forward) -> no cuDSS/GPU; avoids the signal_mpc contention.

    python experiments/gradients/active_set_smoothing/corridor_surrogate.py --check  # setup sanity (fwd only)
    python experiments/gradients/active_set_smoothing/corridor_surrogate.py          # full sweep
"""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # convex stepping stone: stay off the GPU
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_DL_ROOT, "src"),
           os.path.join(_DL_ROOT, "external", "diffmpc2"),
           os.path.join(_DL_ROOT, "external", "diffmpc2", "benchmarking", "linear-system"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_problem_setup import build_turbompc_linear_problem  # noqa: E402
from turbompc.solvers.turbompc_solver import TurboMPCSolver, ForwardBackend, BackwardBackend  # noqa: E402
from turbompc.utils.load_params import load_solver_params  # noqa: E402
from turbompc.solvers.linear_systems_solvers.backends import SchurSolverBackend  # noqa: E402
from turbompc.solvers.linear_systems_solvers.schur_solver import make_schur_solver  # noqa: E402
from turbompc.solvers.qp_data import QPData, QPCostBlocks, QPEqualityBlocks, QPInequalityBlocks  # noqa: E402
from turbompc.solvers.qp_utils import ZShape, pack_x  # noqa: E402
from turbompc.solvers.backward.backward_kkt_jax import solve_backward_kkt  # noqa: E402
from diffmpc_learning.solvers.central_path_admm import to_one_sided, solve_qp_central_path  # noqa: E402
from diffmpc_learning.solvers.backward import relaxed_complementarity_weight, augment_D_with_relaxed_ineq  # noqa: E402
from env.corridor_problem import OptimalControlProblemCorridor  # noqa: E402
from util.plot import DATA_DIR, save_fig  # noqa: E402

# ----- problem constants -----
NX, NU = 4, 2                                          # (px,py,vx,vy), (ax,ay)
HORIZON, DT = 25, 0.1
X0 = jnp.array([0.0, 0.0, 0.0, 0.0])
GOAL = jnp.array([1.5, 1.5, 0.0, 0.0])                 # beyond the corner (1,1)
CORRIDOR_A = jnp.array([[1.0, 0.0], [0.0, 1.0]])       # walls: px<=1, py<=1
CORRIDOR_B = jnp.array([1.0, 1.0])
QV = 0.0                                               # velocity-tracking weight (off; only track position)
R_VEC = jnp.full((NU,), 1.0)                           # control weight
UMAX = 1.0e4                                           # control box inert -> only the walls can bind
QK = "weights_penalization_reference_state_trajectory"
RK = "weights_penalization_control_squared"
GAMMA, KAPPA = 1.0e12, 1.0e-6                          # pure barrier (gamma->inf), small kappa
CP_MAX_ITER, ADMM_MAX_ITER = 4000, 4000


def make_Q(wq):
    """State-tracking weight vector: position weight = wq (the swept/differentiated scalar)."""
    return jnp.array([wq, wq, QV, QV])


def build_problem():
    # complete, correct base params from the benchmark builder; then override what we need
    dyn, base = build_turbompc_linear_problem(horizon=HORIZON, umax=UMAX, n_state=NX, n_ctrl=NU)
    # continuous double integrator: pdot=v, vdot=u  ->  xdot = A_c x + B_c u
    A_c = jnp.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    B_c = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    pp = dict(base)
    pp["discretization_resolution"] = DT
    pp["initial_state"] = X0
    pp["reference_state_trajectory"] = jnp.tile(GOAL, (HORIZON + 1, 1))
    pp[QK] = make_Q(1.0); pp[RK] = R_VEC
    pp["weights_penalization_final_state"] = jnp.zeros((NX,))
    pp["control_min_bounds"] = -UMAX * jnp.ones((NU,))
    pp["control_max_bounds"] = UMAX * jnp.ones((NU,))
    pp["dynamics_state_dot_params"] = {"A": A_c, "B": B_c, "b": jnp.zeros((NX,))}
    # corridor params (read by OptimalControlProblemCorridor.step_inequality_constraints)
    pp["corridor_A"] = CORRIDOR_A; pp["corridor_b"] = CORRIDOR_B
    # construction params: corridor_dimension + disable variable rescaling
    cons = dict(pp); cons["corridor_dimension"] = 2; cons["rescale_optimization_variables"] = False
    return dyn, pp, cons


def _sp(tol):
    sp = dict(load_solver_params("turbompc.yaml"))
    sp["num_sqp_iteration_max"] = 1; sp["tol_convergence"] = tol; sp["warm_start_backward"] = False
    sp["linesearch"] = False
    sp["admm"]["max_iter"] = ADMM_MAX_ITER; sp["admm"]["check_termination_every"] = 1
    sp["admm"]["eps_abs"] = tol; sp["admm"]["eps_rel"] = tol
    return sp


def make_hard_solver(dyn, cons, tol):
    program = OptimalControlProblemCorridor(dynamics=dyn, params=cons)
    return TurboMPCSolver(program=program, params=_sp(tol),
                          forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
                          backward_backend=BackwardBackend.DIRECT_JAX_DENSE, use_full_hessian=True)


def solve_hard(solver, pp, wq):
    weights = {QK: make_Q(wq), RK: R_VEC, "initial_state": X0}
    sol = solver.solve(solver.initial_guess(pp), problem_params=pp, weights=weights)
    return sol.states, sol.controls


def make_barrier_pieces(dyn, cons, pp):
    solver = make_hard_solver(dyn, cons, 1e-9)         # reused only to _build_qp_data / initial_guess
    ig = solver.initial_guess(pp); N = solver.program.horizon
    schur = make_schur_solver(SchurSolverBackend.JAX_DENSE, N, NX, NU,
                              pcg_params={"max_iter": 800, "tol_epsilon": 1e-12})
    return solver, ig, schur


def solve_barrier(solver, ig, schur, pp, wq, kappa, tol):
    pp_i = {**pp, "initial_state": X0, QK: make_Q(wq), RK: R_VEC}
    qp = solver._build_qp_data(ig.states, ig.controls, pp_i)
    qp1 = to_one_sided(qp, GAMMA)
    x, duals, info = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=GAMMA,
                                           rho_bar=0.1, max_iter=CP_MAX_ITER, tol=tol)
    return x[:, :NX], x[:, NX:], duals, info, qp1


def _wall_slack(states):
    """max over the trajectory of (a_i^T p_t - b_i); ~0 => wall i touched (active)."""
    P = np.asarray(states)[:, :2]
    g = P @ np.asarray(CORRIDOR_A).T - np.asarray(CORRIDOR_B)    # (T, n_walls)
    return g.max(axis=0)                                         # per-wall worst (closest to active)


def task_loss(states):
    """Downstream task loss = position tracking to the goal (the diffmpc-as-policy objective)."""
    return jnp.sum((states[:, :2] - GOAL[:2]) ** 2)


# ----- barrier custom_vjp (relaxed central-path backward), mirrors the 4-variant benchmark -----
def _solve_reduced_kkt_full(D_aug, E, eq_blocks, x_bar, nx, nu):
    Np1, n = D_aug.shape[0], D_aug.shape[1]; N = Np1 - 1
    eq0 = QPEqualityBlocks(A0=eq_blocks.A0, A_minus=eq_blocks.A_minus, A_plus=eq_blocks.A_plus,
                           c0=jnp.zeros_like(eq_blocks.c0), c=jnp.zeros_like(eq_blocks.c))
    empty = QPInequalityBlocks(G=jnp.zeros((Np1, 0, n), x_bar.dtype),
                               l=jnp.zeros((Np1, 0), x_bar.dtype), u=jnp.zeros((Np1, 0), x_bar.dtype))
    bwd = QPData(cost=QPCostBlocks(D=D_aug, E=E, q=-x_bar), eq=eq0, ineq=empty)
    (lam_s, lam_c), mult = solve_backward_kkt(bwd, ZShape(horizon=N, num_states=nx, num_controls=nu))
    return pack_x(lam_s, lam_c), mult


def make_b_solve(solver, pp_base, schur, ig, kappa, gamma, cp_tol):
    program = solver.program
    cp = dict(rho_bar=0.1, max_iter=CP_MAX_ITER, tol=cp_tol)

    def _qp1(weights, state):
        pp_i = {**pp_base, "initial_state": state, QK: weights[QK], RK: weights[RK]}
        qp = solver._build_qp_data(ig.states, ig.controls, pp_i)
        return to_one_sided(qp, gamma)

    @jax.custom_vjp
    def b_solve(weights, state):
        x, _, _ = solve_qp_central_path(_qp1(weights, state), schur,
                                        target_kappa=kappa, slack_weight=gamma, **cp)
        return x[:, :NX], x[:, NX:]

    def b_fwd(weights, state):
        qp1 = _qp1(weights, state)
        x, duals, _ = solve_qp_central_path(qp1, schur, target_kappa=kappa, slack_weight=gamma, **cp)
        return (x[:, :NX], x[:, NX:]), (weights, state, qp1, x, duals)

    def b_bwd(res, g):
        weights, state, qp1, x, duals = res
        dL_ds, dL_dc = g
        W = relaxed_complementarity_weight(qp1, x, duals[2], gamma)
        D_aug = augment_D_with_relaxed_ineq(qp1.cost.D, qp1.ineq.G, W)
        lam_x, lam_f = _solve_reduced_kkt_full(D_aug, qp1.cost.E, qp1.eq, pack_x(dL_ds, dL_dc), NX, NU)
        lam_states, lam_controls = lam_x[:, :NX], lam_x[:, NX:]

        def contracted(ww):
            pp_w = {**pp_base, "initial_state": state, QK: ww[QK], RK: ww[RK]}
            fx, fu = jax.grad(lambda s, c: program.cost(s, c, pp_w), argnums=(0, 1))(x[:, :NX], x[:, NX:])
            return jnp.sum(fx * lam_states) + jnp.sum(fu * lam_controls)
        dL_dw = jax.tree_util.tree_map(lambda z: -z, jax.grad(contracted)(weights))
        return (dL_dw, lam_f[:NX])

    b_solve.defvjp(b_fwd, b_bwd)
    return b_solve


# ----- convergence-checked central FD of a scalar cost(wq) (CLAUDE.md) -----
FD_EPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]


def fd_converged(cost, wq_grid):
    cols = {}
    for e in FD_EPS:
        cols[e] = np.array([(float(cost(w + e)) - float(cost(w - e))) / (2 * e) for w in wq_grid])
    eps_sorted = sorted(FD_EPS, reverse=True)
    n = len(wq_grid); fd = np.array(cols[eps_sorted[0]]); conv = np.zeros(n, bool)
    for i in range(n):
        seq = [cols[e][i] for e in eps_sorted]; chosen, ok = seq[0], False
        for k in range(len(seq) - 1):
            if abs(seq[k] - seq[k + 1]) <= 1e-2 * abs(seq[k + 1]) + 1e-9:
                chosen, ok = seq[k], True; break
        fd[i] = chosen; conv[i] = ok
    return fd, conv


def check():
    dyn, pp, cons = build_problem()
    hard = make_hard_solver(dyn, cons, 1e-9)
    bsolver, ig, schur = make_barrier_pieces(dyn, cons, pp)
    print(f"corridor surrogate | horizon={HORIZON} dt={DT} goal={np.asarray(GOAL)[:2]} "
          f"walls: px<=1, py<=1 | gamma={GAMMA:g} kappa={KAPPA:g}")
    print(f"{'wq':>8} | {'hard term p':>20} {'wall slack (px,py)':>22} | "
          f"{'barrier term p':>20} | {'||x_h-x_b||':>12}")
    for wq in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0]:
        xh, uh = solve_hard(hard, pp, wq)
        xb, ub, duals, info, _ = solve_barrier(bsolver, ig, schur, pp, wq, KAPPA, 1e-9)
        th = np.asarray(xh)[-1, :2]; tb = np.asarray(xb)[-1, :2]
        wall = _wall_slack(xh)
        diff = float(np.linalg.norm(np.asarray(xh) - np.asarray(xb)))
        print(f"{wq:8.3f} | ({th[0]:7.4f},{th[1]:7.4f}) {str(np.round(wall,4)):>22} | "
              f"({tb[0]:7.4f},{tb[1]:7.4f}) | {diff:12.2e}  "
              f"[cp iters={int(info['iters'])} prim={float(info['prim_res']):.1e}]")
    print("\n(expect: low wq -> p interior, wall slack < 0; high wq -> p->(1,1), wall slack ~0 BOTH walls;\n"
          " ||x_h - x_b|| small everywhere = hard and barrier(kappa->0) agree.)")


def run_sweep(npts, wq_lo, wq_hi, kappas):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    dyn, pp, cons = build_problem()
    hard = make_hard_solver(dyn, cons, 1e-9)
    bsolver, ig, schur = make_barrier_pieces(dyn, cons, pp)

    def cost_hard(wq):
        weights = {QK: make_Q(wq), RK: R_VEC, "initial_state": X0}
        sol = hard.solve(hard.initial_guess(pp), problem_params=pp, weights=weights)
        return task_loss(sol.states)
    ch = jax.jit(cost_hard); gh = jax.jit(jax.grad(cost_hard))

    barrier = {}
    for kap in kappas:
        b_solve = make_b_solve(bsolver, pp, schur, ig, kap, GAMMA, 1e-9)
        cb = (lambda bs: (lambda wq: task_loss(bs({QK: make_Q(wq), RK: R_VEC}, X0)[0])))(b_solve)
        barrier[kap] = (jax.jit(cb), jax.jit(jax.grad(cb)))

    wq = np.linspace(wq_lo, wq_hi, npts)
    L_h = np.array([float(ch(w)) for w in wq])
    d_h = np.array([float(gh(w)) for w in wq])
    fd, conv = fd_converged(lambda w: float(ch(w)), wq)
    wall = np.array([_wall_slack(_solve_states_hard(hard, pp, w)) for w in wq])  # (npts, 2)
    d_b = {kap: np.array([float(gb(w)) for w in wq]) for kap, (cb, gb) in barrier.items()}

    # activation wq: first grid point where either wall slack reaches ~0
    act_i = int(np.argmax(wall.max(axis=1) > -1e-3))
    wq_act = wq[act_i]

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {1e-2: "tab:orange", 1e-4: "tab:green", 1e-6: "tab:red"}
    ax[0].plot(wq, L_h, "k-", lw=2); ax[0].axvline(wq_act, color="0.6", ls=":")
    ax[0].set_title("task loss  L(wq)"); ax[0].set_xlabel("wq (position-tracking weight)")
    ax[1].plot(wq, d_h, "k-", lw=2, label="hard AD")
    ax[1].plot(wq, fd, "k--", lw=1, label="hard FD (conv-checked)")
    for kap in kappas:
        ax[1].plot(wq, d_b[kap], color=colors.get(kap, "tab:blue"), lw=1.3, label=f"barrier AD k={kap:g}")
    ax[1].axvline(wq_act, color="0.6", ls=":"); ax[1].legend(fontsize=8)
    ax[1].set_title("gradient  dL/dwq  (hard jumps at activation, barrier smooths)"); ax[1].set_xlabel("wq")
    ax[2].plot(wq, wall[:, 0], label="wall px<=1"); ax[2].plot(wq, wall[:, 1], "--", label="wall py<=1")
    ax[2].axhline(0, color="0.6", lw=0.8); ax[2].axvline(wq_act, color="0.6", ls=":")
    ax[2].set_title("wall slack  a^T p - b  (->0 = active; both => coupled)"); ax[2].set_xlabel("wq")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Corridor surrogate: coupled 2-wall activation at wq~{wq_act:.3f} "
                 f"(convex, linear walls) — hard vs central-path gradient", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "corridor_surrogate.png"); plt.close(fig)

    far = np.abs(wq - wq_act) > 0.1                              # away from the switch
    win = np.abs(wq - wq_act) < 0.02                             # transition band
    jump_h = float(np.max(np.abs(np.diff(d_h[win])))) if win.sum() > 1 else float("nan")
    print(f"=== corridor surrogate sweep | wq in [{wq_lo},{wq_hi}] x{npts} | activation wq~{wq_act:.4f} ===")
    print(f"FD non-converged: {int((~conv).sum())}/{npts}  (loss continuous -> FD converges; jump only in the gradient)")
    print(f"HARD gradient JUMP at activation (max consec |Δ dL/dwq| in band): {jump_h:.2f}")
    print(f"hard AD vs FD: max|diff|={np.max(np.abs(d_h - fd)):.4f}  -> hard grad is ACCURATE yet discontinuous")
    print(f"barrier deviation from hard (= smoothing): {'band (|wq-act|<.02)':>22} | {'far-field (>.1) = O(k) bias':>28}")
    for kap in kappas:
        banddev = float(np.max(np.abs((d_h - d_b[kap])[win]))) if win.sum() else float("nan")
        fardev = float(np.max(np.abs((d_h - d_b[kap])[far]))) if far.sum() else float("nan")
        print(f"   kappa={kap:g}: {banddev:22.3f} | {fardev:28.3f}")
    np.savez(os.path.join(DATA_DIR, "corridor_surrogate.npz"),
             wq=wq, L_hard=L_h, dL_hard=d_h, fd=fd, conv=conv, wall=wall, wq_act=wq_act,
             **{f"dL_barrier_{k:g}": d_b[k] for k in kappas})
    print(f"saved {os.path.join(DATA_DIR, 'corridor_surrogate.npz')}")


def _solve_states_hard(solver, pp, wq):
    weights = {QK: make_Q(wq), RK: R_VEC, "initial_state": X0}
    return solver.solve(solver.initial_guess(pp), problem_params=pp, weights=weights).states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--npts", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.8)
    a = ap.parse_args()
    if a.check:
        check(); return
    run_sweep(a.npts, a.lo, a.hi, kappas=[1e-2, 1e-4, 1e-6])


if __name__ == "__main__":
    main()
