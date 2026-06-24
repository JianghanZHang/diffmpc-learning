"""acados QP-solver-level gradient accuracy vs the ACHIEVED QP solve accuracy (HPIPM, hard box).

We showed HPIPM OVERSHOOTS the set qp_solver_tol (IPM, quadratic convergence), so the genuine
accuracy knob is the QP ITERATION COUNT (`qp_solver_iter_max`). Sweep k=1..K; per x0:
  - rel_sol_err = ||x_k - x*||_inf / ||x*||_inf   (x* = full-iter HPIPM solution)
  - exact-Hessian adjoint gradient dL/d(Q,R) at the k-iter solution (eval_adjoint_solution_sensitivity)
compared to a convergence-checked FD ground truth. Reports cos/rel vs rel_sol_err — the common axis
on which A (ADMM-slack) and B (log-barrier) are also plotted (run separately in the GPU env).

L = 0.5*sum(x^2+u^2) ⇒ adjoint seed = (x*, u*). Same linear MPC as acados_linear_problem.npz.

RUN IN THE turbompc-acados DOCKER (renderer bind-mounted):
  docker run --rm -v $PWD:/work -w /work -v /path/t_renderer:/opt/acados/bin/t_renderer \
    turbompc-acados python3 research/gradient-quality-diffnmpc/experiments/linear_system/acados_qp_gradient.py
"""
import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
D = np.load(os.path.join(RES, "acados_linear_problem.npz"))
A, B, bvec = D["A"], D["B"], D["b"]
Qd0, Rd0, X0 = D["Q_diag"], D["R_diag"], D["x0"]
NX, NU, N, UMAX = int(D["nx"]), int(D["nu"]), int(D["horizon"]), float(D["umax"])
P0 = np.concatenate([Qd0, Rd0])

KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]      # qp_solver_iter_max sweep (the IPM accuracy knob)
K_TIGHT = 30                               # "exact" reference solve
FD_EPS = [1e-3, 3e-4, 1e-4, 3e-5]


def build(itmax):
    ocp = AcadosOcp()
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    ocp.model.x = x; ocp.model.u = u; ocp.model.name = "lin_qp"
    Qp = ca.SX.sym("Qp", NX); Rp = ca.SX.sym("Rp", NU)
    ocp.model.p_global = ca.vertcat(Qp, Rp); ocp.p_global_values = P0.copy()
    ocp.model.disc_dyn_expr = ca.DM(A) @ x + ca.DM(B) @ u + ca.DM(bvec.reshape(-1, 1))
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.cost.cost_type = "EXTERNAL"; ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(Qp * x ** 2) + ca.sum1(Rp * u ** 2)
    ocp.model.cost_expr_ext_cost_e = ca.sum1(Qp * x ** 2)
    ocp.constraints.lbu = -UMAX * np.ones(NU); ocp.constraints.ubu = UMAX * np.ones(NU)
    ocp.constraints.idxbu = np.arange(NU); ocp.constraints.x0 = X0[0].copy()
    ocp.solver_options.N_horizon = N; ocp.solver_options.tf = float(N)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.nlp_solver_max_iter = 1            # 1 SQP iter; the inner QP runs <= itmax HPIPM iters
    ocp.solver_options.qp_solver_iter_max = itmax
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.qp_solver_cond_ric_alg = 0; ocp.solver_options.qp_solver_ric_alg = 0
    for c in ("stat", "eq", "ineq", "comp"):
        setattr(ocp.solver_options, f"qp_solver_tol_{c}", 1e-15)   # never stop early -> run exactly itmax
        setattr(ocp.solver_options, f"nlp_solver_tol_{c}", 1e-12)
    bd = os.path.join("/tmp", f"aqp_{itmax}"); ocp.solver_options.build_dir = bd
    return AcadosOcpSolver(ocp, json_file=os.path.join(bd, "ocp.json"), verbose=False)


def solve(solver, x0, p):
    solver.set_p_global_and_precompute_dependencies(p)
    solver.set(0, "lbx", x0); solver.set(0, "ubx", x0)
    solver.solve()
    xs = np.array([solver.get(k, "x") for k in range(N + 1)])
    us = np.array([solver.get(k, "u") for k in range(N)])
    return xs, us


def flat(xs, us):
    return np.concatenate([np.concatenate([xs[k], us[k]]) for k in range(N)] + [xs[N]])


def adjoint(solver, x0, p):
    xs, us = solve(solver, x0, p)
    solver.setup_qp_matrices_and_factorize()
    seed_x = [(k, xs[k].reshape(NX, 1)) for k in range(N + 1)]
    seed_u = [(k, us[k].reshape(NU, 1)) for k in range(N)]
    g = np.asarray(solver.eval_adjoint_solution_sensitivity(seed_x=seed_x, seed_u=seed_u)).ravel()
    return g, xs, us


def fd_gt(tight, x0):
    def L(p):
        xs, us = solve(tight, x0, p); return 0.5 * float(np.sum(xs ** 2) + np.sum(us ** 2))
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
            if abs(a - bb) <= 1e-2 * abs(bb) + 1e-7: chosen, okk = a, True; break
        g[i] = chosen; ok = ok and okk
    return g, ok


def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def _rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    n = X0.shape[0]
    print(f"acados QP-level: nx={NX} nu={NU} H={N} umax={UMAX} n_x0={n} | HPIPM, sweep qp_solver_iter_max")
    tight = build(K_TIGHT)
    xstar = [flat(*solve(tight, X0[j], P0)) for j in range(n)]
    gFD = np.zeros((n, len(P0))); flagged = np.zeros(n, bool)
    for j in range(n):
        gFD[j], okj = fd_gt(tight, X0[j]); flagged[j] = not okj
    keep = ~flagged
    print(f"FD ground truth: flagged {int(flagged.sum())}/{n}; comparing on {int(keep.sum())} samples")

    print(f"\n{'k':>3} {'rel_sol_err':>12} {'cos med':>9} {'cos min':>9} {'rel med':>9}")
    rows = []; gAD = np.full((n, len(P0)), np.nan)
    for k in KS:
        s = build(k)
        sol_err, cosv, relv = [], [], []
        for j in range(n):
            if flagged[j]:
                continue
            g, xs, us = adjoint(s, X0[j], P0)
            if k == KS[-1]:
                gAD[j] = g                                 # converged acados adjoint per sample
            xk = flat(xs, us)
            sol_err.append(np.max(np.abs(xk - xstar[j])) / (np.max(np.abs(xstar[j])) + 1e-30))
            cosv.append(_cos(g, gFD[j])); relv.append(_rel(g, gFD[j]))
        sol_err, cosv, relv = map(np.array, (sol_err, cosv, relv))
        rows.append((k, np.median(sol_err), np.median(cosv), np.min(cosv), np.median(relv)))
        print(f"{k:>3} {np.median(sol_err):12.2e} {np.median(cosv):9.4f} {np.min(cosv):9.4f} {np.median(relv):9.2e}")
    np.savez(os.path.join(RES, "acados_qp_itersweep.npz"),
             ks=np.array([r[0] for r in rows]), rows=np.array([r[1:] for r in rows]),
             gFD=gFD, gAD=gAD, flagged=flagged, n=n, x0=X0, Q_diag=Qd0, R_diag=Rd0)
    print("\nsaved results/acados_qp_itersweep.npz")


if __name__ == "__main__":
    main()
