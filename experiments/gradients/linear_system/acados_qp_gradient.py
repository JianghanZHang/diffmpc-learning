"""acados SMOOTHED gradient vs the barrier parameter tau_min — the acados analogue of B's log-barrier
kappa sweep (Frey/Diehl 2025, "Differentiable NMPC", Eq.10 / Thm.3 / Fig.1).

CORRECTION: acados DOES smooth the gradient — by keeping the interior-point barrier at tau_min>0
(complementarity mu_i h_i = tau_min instead of 0), the solution map is continuously differentiable
(Thm.3) and the adjoint gives the correct sensitivity of the SMOOTHED map even when strict
complementarity fails (Remark 2). tau_min=0 is the exact/nonsmooth case (ill-defined at degenerate
active sets). This is the SAME mechanism as B's log-barrier kappa.

Recipe (per acados smooth_policy_gradients.py): two solvers; options_set('tau_min', tau) on both;
forward solve -> set_iterate -> setup_qp_matrices_and_factorize -> eval_adjoint_solution_sensitivity.
Loss L = 0.5*sum(x^2+u^2) => adjoint seed = (x*, u*). Per tau_min, FD of the SAME tau_min-smoothed
forward is the (reliable, smooth) ground truth. Same linear MPC as acados_linear_problem.npz.

RUN IN turbompc-acados DOCKER (renderer bind-mounted).
"""
import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "data")
D = np.load(os.path.join(RES, "acados_linear_problem.npz"))
A, B, bvec = D["A"], D["B"], D["b"]
Qd0, Rd0, X0 = D["Q_diag"], D["R_diag"], D["x0"]
NX, NU, N, UMAX = int(D["nx"]), int(D["nu"]), int(D["horizon"]), float(D["umax"])
P0 = np.concatenate([Qd0, Rd0])

TAUS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 0.0]    # barrier smoothing; 0 = exact/nonsmooth
FD_EPS = [1e-3, 3e-4, 1e-4, 3e-5]


def build(sens):
    ocp = AcadosOcp()
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    ocp.model.x = x; ocp.model.u = u; ocp.model.name = "lin_sm" + ("_s" if sens else "_f")
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
    ocp.solver_options.hessian_approx = "EXACT"          # exact Hessian (Remark 3: required for sens)
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_iter_max = 1000
    if sens:
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.qp_solver_cond_ric_alg = 0; ocp.solver_options.qp_solver_ric_alg = 0
    bd = os.path.join("/tmp", "asm_s" if sens else "asm_f"); ocp.solver_options.build_dir = bd
    return AcadosOcpSolver(ocp, json_file=os.path.join(bd, "ocp.json"), verbose=False)


def traj(s, x0, p):
    s.set_p_global_and_precompute_dependencies(p)
    s.set(0, "lbx", x0); s.set(0, "ubx", x0); s.solve()
    xs = np.array([s.get(k, "x") for k in range(N + 1)]); us = np.array([s.get(k, "u") for k in range(N)])
    return xs, us


def loss(xs, us): return 0.5 * float(np.sum(xs ** 2) + np.sum(us ** 2))


def adjoint(fwd, sens, x0, p):
    xs, us = traj(fwd, x0, p)
    sens.set_p_global_and_precompute_dependencies(p)
    sens.set_iterate(fwd.get_flat_iterate()); sens.setup_qp_matrices_and_factorize()
    seed_x = [(k, xs[k].reshape(NX, 1)) for k in range(N + 1)]
    seed_u = [(k, us[k].reshape(NU, 1)) for k in range(N)]
    g = np.asarray(sens.eval_adjoint_solution_sensitivity(seed_x=seed_x, seed_u=seed_u)).ravel()
    return g


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
    print(f"acados SMOOTHED gradient vs tau_min: nx={NX} nu={NU} H={N} umax={UMAX} n_x0={n}")
    fwd, sens = build(False), build(True)
    print(f"\n{'tau_min':>9} {'FD-flag':>8} {'cos med':>9} {'cos min':>9} {'rel med':>9}")
    rows = []
    for tau in TAUS:
        for s in (fwd, sens):
            s.options_set("tau_min", tau)
        gFD = np.zeros((n, len(P0))); flagged = np.zeros(n, bool)
        for j in range(n):
            gFD[j], okj = fd_gt(fwd, X0[j]); flagged[j] = not okj
        cosv, relv = [], []
        for j in range(n):
            if flagged[j]:
                continue
            g = adjoint(fwd, sens, X0[j], P0)
            cosv.append(_cos(g, gFD[j])); relv.append(_rel(g, gFD[j]))
        cosv, relv = map(np.array, (cosv, relv))
        cm = np.median(cosv) if cosv.size else float("nan")
        rows.append((tau, int(flagged.sum()), cm, np.min(cosv) if cosv.size else float("nan"),
                     np.median(relv) if relv.size else float("nan")))
        print(f"{tau:9.0e} {int(flagged.sum()):8d} {cm:9.4f} "
              f"{(np.min(cosv) if cosv.size else float('nan')):9.4f} "
              f"{(np.median(relv) if relv.size else float('nan')):9.2e}")
    np.savez(os.path.join(RES, "acados_tau_sweep.npz"), taus=np.array([r[0] for r in rows]),
             rows=np.array([r[1:] for r in rows]), n=n)
    print("\nsaved results/data/acados_tau_sweep.npz")


if __name__ == "__main__":
    main()
