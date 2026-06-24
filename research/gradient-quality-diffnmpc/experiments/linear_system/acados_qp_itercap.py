"""acados solve-accuracy via qp_solver_iter_max (the real IPM accuracy knob, vs qp_solver_tol which
overshoots). Two-solver pattern: a capped FORWARD solver (no sensitivity options, so the iter-cap is
respected) + a separate EXACT sensitivity solver that loads the capped iterate.

Fixed smoothing tau_min (so the CONVERGED gradient is well-defined, cos=1.0). Sweep qp_iter_max k:
per x0 -> rel_sol_err = ||x_k - x*|| / ||x*|| (x* = full-iter solution) and the adjoint dL/d(Q,R) at
the k-iter iterate, vs FD of the converged (full-iter) tau_min-smoothed forward. This is the acados
analogue of B's ADMM-eps sweep at fixed kappa. RUN IN turbompc-acados DOCKER.
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

TAU_MIN = 1e-4
KS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 1000]      # qp_solver_iter_max; 1000 = "converged"
FD_EPS = [1e-3, 3e-4, 1e-4, 3e-5]


def build(sens, itmax=1000):
    ocp = AcadosOcp()
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    ocp.model.x = x; ocp.model.u = u; ocp.model.name = "lic_" + ("s" if sens else "f")
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
    ocp.solver_options.nlp_solver_max_iter = 1 if not sens else 50
    ocp.solver_options.qp_solver_iter_max = itmax                  # BUILD-TIME (not runtime-settable)
    for c in ("stat", "eq", "ineq", "comp"):
        setattr(ocp.solver_options, f"qp_solver_tol_{c}", 1e-15)   # never stop on tol -> iter-cap binds
    if sens:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.qp_solver_cond_ric_alg = 0; ocp.solver_options.qp_solver_ric_alg = 0
    bd = os.path.join("/tmp", f"aic_{'s' if sens else 'f'}_{itmax}"); ocp.solver_options.build_dir = bd
    return AcadosOcpSolver(ocp, json_file=os.path.join(bd, "ocp.json"), verbose=False)


def traj(s, x0, p):
    s.set_p_global_and_precompute_dependencies(p)
    s.set(0, "lbx", x0); s.set(0, "ubx", x0); s.solve()
    xs = np.array([s.get(k, "x") for k in range(N + 1)]); us = np.array([s.get(k, "u") for k in range(N)])
    return xs, us


def flat(xs, us): return np.concatenate([np.concatenate([xs[k], us[k]]) for k in range(N)] + [xs[N]])
def loss(xs, us): return 0.5 * float(np.sum(xs ** 2) + np.sum(us ** 2))


def adjoint(fwd, sens, x0, p):
    xs, us = traj(fwd, x0, p)
    sens.set_p_global_and_precompute_dependencies(p)
    sens.set_iterate(fwd.get_flat_iterate()); sens.setup_qp_matrices_and_factorize()
    g = np.asarray(sens.eval_adjoint_solution_sensitivity(
        seed_x=[(k, xs[k].reshape(NX, 1)) for k in range(N + 1)],
        seed_u=[(k, us[k].reshape(NU, 1)) for k in range(N)])).ravel()
    return g, flat(xs, us)


def fd_gt(fwd, x0):
    def L(p): xs, us = traj(fwd, x0, p); return loss(xs, us)
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
    print(f"acados qp_solver_iter_max sweep at tau_min={TAU_MIN:g}: nx={NX} nu={NU} H={N} n_x0={n}")
    sens = build(True, 1000); sens.options_set("tau_min", TAU_MIN)
    fwd_t = build(False, 1000); fwd_t.options_set("tau_min", TAU_MIN)
    xstar = [flat(*traj(fwd_t, X0[j], P0)) for j in range(n)]
    gFD = np.zeros((n, len(P0))); flagged = np.zeros(n, bool)
    for j in range(n):
        gFD[j], okj = fd_gt(fwd_t, X0[j]); flagged[j] = not okj
    print(f"FD GT (converged, tau_min): flagged {int(flagged.sum())}/{n}\n")
    print(f"{'qp_iter_max':>11} {'rel_sol_err':>12} {'cos med':>9} {'cos min':>9} {'rel med':>9}")
    for k in KS:
        fwd = build(False, int(k)); fwd.options_set("tau_min", TAU_MIN)   # iter-cap is build-time
        se, cv, rv = [], [], []
        for j in range(n):
            if flagged[j]:
                continue
            g, xk = adjoint(fwd, sens, X0[j], P0)
            se.append(np.max(np.abs(xk - xstar[j])) / (np.max(np.abs(xstar[j])) + 1e-30))
            cv.append(_cos(g, gFD[j])); rv.append(_rel(g, gFD[j]))
        se, cv, rv = map(np.array, (se, cv, rv))
        print(f"{int(k):>11} {np.median(se):12.2e} {np.median(cv):9.4f} {np.min(cv):9.4f} {np.median(rv):9.2e}")


if __name__ == "__main__":
    main()
