"""acados QP-solver-level gradient accuracy vs the QP-KKT residual tolerance — the SAME check as
our ADMM solver (sweep_admm_tolerance.py), but on acados's HPIPM inner QP instead of TurboMPC's ADMM.

Same linear MPC (loaded from results/acados_linear_problem.npz). 1 SQP iter (convex ⇒ the NLP is one
QP). Differentiable params = cost weights (Q,R diag) via `p_global`. Loss L = 0.5·Σ(x²+u²) over the
trajectory ⇒ adjoint seed = (x*, u*). Sweep `qp_solver_tol_{stat,eq,ineq,comp}`; per-x0:
  - achieved QP-KKT residual = max(get_residuals())
  - acados exact-Hessian adjoint gradient dL/d(Q,R) = eval_adjoint_solution_sensitivity
vs a convergence-checked FD ground truth (tight qp_tol). Reports cos/rel vs the achieved residual.

RUN IN THE turbompc-acados DOCKER:
  docker run --rm -v $PWD:/work -w /work turbompc-acados python3 \
    research/gradient-quality-diffnmpc/experiments/linear_system/acados_qp_gradient.py
"""
import os, sys
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
D = np.load(os.path.join(RES, "acados_linear_problem.npz"))
A, B, bvec = D["A"], D["B"], D["b"]
Qd0, Rd0 = D["Q_diag"], D["R_diag"]
X0 = D["x0"]
NX, NU, N = int(D["nx"]), int(D["nu"]), int(D["horizon"])
UMAX = float(D["umax"])
P0 = np.concatenate([Qd0, Rd0])                       # p_global = [Q_diag, R_diag]

TOLS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9]
FD_EPS = [1e-3, 3e-4, 1e-4, 3e-5]                     # convergence-checked
GT_TOL = 1e-12                                        # tight QP solve for the FD ground truth


def build_solver(qp_tol):
    ocp = AcadosOcp()
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    ocp.model.x = x; ocp.model.u = u; ocp.model.name = "lin_qp"
    Qp = ca.SX.sym("Qp", NX); Rp = ca.SX.sym("Rp", NU)
    ocp.model.p_global = ca.vertcat(Qp, Rp)
    ocp.p_global_values = P0.copy()
    ocp.model.disc_dyn_expr = ca.DM(A) @ x + ca.DM(B) @ u + ca.DM(bvec.reshape(-1, 1))
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.cost.cost_type = "EXTERNAL"; ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(Qp * x ** 2) + ca.sum1(Rp * u ** 2)   # turbompc form (no 1/2)
    ocp.model.cost_expr_ext_cost_e = ca.sum1(Qp * x ** 2)
    ocp.constraints.lbu = -UMAX * np.ones(NU); ocp.constraints.ubu = UMAX * np.ones(NU)
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.x0 = X0[0].copy()
    ocp.solver_options.N_horizon = N; ocp.solver_options.tf = float(N)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"          # quadratic cost ⇒ exact = GN, and gives sens
    ocp.solver_options.nlp_solver_max_iter = 2           # convex ⇒ converges immediately
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.qp_solver_cond_ric_alg = 0
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_iter_max = 1000
    for c in ("stat", "eq", "ineq", "comp"):
        setattr(ocp.solver_options, f"qp_solver_tol_{c}", qp_tol)
        setattr(ocp.solver_options, f"nlp_solver_tol_{c}", min(qp_tol, 1e-9))   # outer SQP not the bottleneck
    bd = os.path.join("/tmp", f"acados_lin_{qp_tol:.0e}".replace("-", "n"))
    ocp.solver_options.build_dir = bd
    return AcadosOcpSolver(ocp, json_file=os.path.join(bd, "ocp.json"), verbose=False)


def solve_one(solver, x0, p):
    solver.set_p_global_and_precompute_dependencies(p)
    solver.set(0, "lbx", x0); solver.set(0, "ubx", x0)
    st = solver.solve()
    xs = np.array([solver.get(k, "x") for k in range(N + 1)])     # (N+1, nx)
    us = np.array([solver.get(k, "u") for k in range(N)])         # (N, nu)
    res = float(np.max(solver.get_residuals()))
    return xs, us, res, st


def loss(xs, us):
    return 0.5 * float(np.sum(xs ** 2) + np.sum(us ** 2))


def adjoint_grad(solver, x0, p):
    xs, us, res, st = solve_one(solver, x0, p)
    solver.setup_qp_matrices_and_factorize()
    seed_x = [(k, xs[k].reshape(NX, 1)) for k in range(N + 1)]    # dL/dx_k = x_k
    seed_u = [(k, us[k].reshape(NU, 1)) for k in range(N)]        # dL/du_k = u_k
    g = np.asarray(solver.eval_adjoint_solution_sensitivity(seed_x=seed_x, seed_u=seed_u)).ravel()
    return g, res, st                                            # g = dL/d[Q_diag,R_diag] (nx+nu,)


def fd_grad_converged(gt_solver, x0):
    """convergence-checked central-difference dL/d(Q,R) at the tight QP solve."""
    def L(p):
        xs, us, _, _ = solve_one(gt_solver, x0, p)
        return loss(xs, us)
    per = []
    for eps in FD_EPS:
        g = np.zeros(len(P0))
        for i in range(len(P0)):
            pp = P0.copy(); pp[i] += eps; pm = P0.copy(); pm[i] -= eps
            g[i] = (L(pp) - L(pm)) / (2 * eps)
        per.append(g)
    g = per[-1].copy(); ok = True
    for i in range(len(P0)):
        seq = [pe[i] for pe in per]; chosen, okk = seq[-1], False
        for a, bb in zip(seq[:-1], seq[1:]):
            if abs(a - bb) <= 1e-2 * abs(bb) + 1e-7:
                chosen, okk = a, True; break
        g[i] = chosen; ok = ok and okk
    return g, ok


def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    n = X0.shape[0]
    print(f"acados QP-level sweep: nx={NX} nu={NU} H={N} umax={UMAX} n_x0={n} | HPIPM, 1 SQP iter")
    # ground truth: convergence-checked FD at the tight QP solve
    gt_solver = build_solver(GT_TOL)
    gFD = np.zeros((n, len(P0))); flagged = np.zeros(n, bool)
    for j in range(n):
        gFD[j], okj = fd_grad_converged(gt_solver, X0[j]); flagged[j] = not okj
    print(f"FD ground truth: flagged {int(flagged.sum())}/{n}")

    print(f"\n{'qp_tol':>8} {'achieved_res':>13} {'cos med':>9} {'cos min':>9} {'rel med':>9}")
    rows = []
    for tol in TOLS:
        solver = build_solver(tol)
        cos, rel, ress = [], [], []
        for j in range(n):
            if flagged[j]:
                continue
            g, res, stt = adjoint_grad(solver, X0[j], P0)
            cos.append(_cos(g, gFD[j])); rel.append(_rel(g, gFD[j])); ress.append(res)
        cos, rel, ress = map(np.array, (cos, rel, ress))
        rows.append((tol, np.median(ress), np.median(cos), np.min(cos), np.median(rel)))
        print(f"{tol:8.0e} {np.median(ress):13.2e} {np.median(cos):9.4f} {np.min(cos):9.4f} {np.median(rel):9.2e}")
    np.savez(os.path.join(RES, "acados_qp_sweep.npz"),
             tols=np.array([r[0] for r in rows]), rows=np.array([r[1:] for r in rows]),
             gFD=gFD, flagged=flagged, n=n)
    print("\nsaved results/acados_qp_sweep.npz")


if __name__ == "__main__":
    main()
