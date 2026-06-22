"""acados replica of the turbompc cartpole OCP, for an INDEPENDENT backward-gradient comparison.

Canonical acados differentiable-MPC pattern (per differentiable_nmpc/solution_sensitivity_example.py):
  - FORWARD solver: hessian_approx=GAUSS_NEWTON (PSD QP -> robust; solves the swing-up cold).
  - SENSITIVITY solver: a SEPARATE solver with hessian_approx=EXACT + with_solution_sens_wrt_params;
    load the forward solution iterate, setup_qp_matrices_and_factorize, eval_adjoint_solution_sensitivity.
  -> GN for the forward, EXACT Hessian for the backward gradient.

OCP replicates turbompc EXACTLY: cartpole RK4 DISCRETE (dt=0.04, N=25); cost J = sum_k [Q_i x_i^2 + R u^2],
Q=[1,1,10,1], R=[0.1], ref=0, NO 1/2. K=1 closed-loop eval cost C(x0)=||x1||^2+||u0||^2, x1=RK4(x0,u0):
  dC/d(Q,R) = adjoint sensitivity with seed on u0:  s = 2*u0 + (dRK4/du0)^T (2*x1).

Runs in the acados Docker container.  Modes: --check_dynamics | --gradient
"""
import argparse
import os
import re

import numpy as np
import casadi as ca

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M_C, M_P, L, G = 1.0, 0.1, 0.5, 9.81
NX, NU, N_HOR, DT = 4, 1, 25, 0.04
Q_NOM = np.array([1.0, 1.0, 10.0, 1.0])
R_NOM = np.array([0.1])
P_GLOBAL = np.concatenate([Q_NOM, R_NOM])   # [Q(4), R(1)]


def cartpole_state_dot(x, u):
    x_dot, th, th_dot = x[1], x[2], x[3]
    F = u[0]
    M = M_C + M_P
    pml = M_P * L
    s, c = ca.sin(th), ca.cos(th)
    temp = (F + pml * th_dot ** 2 * s) / M
    th_acc = (G * s - c * temp) / (L * (4.0 / 3.0 - (M_P * c ** 2) / M))
    x_acc = temp - pml * th_acc * c / M
    return ca.vertcat(x_dot, x_acc, th_dot, th_acc)


def rk4_step(x, u, dt=DT):
    k1 = cartpole_state_dot(x, u)
    k2 = cartpole_state_dot(x + 0.5 * dt * k1, u)
    k3 = cartpole_state_dot(x + 0.5 * dt * k2, u)
    k4 = cartpole_state_dot(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_fn():
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    return ca.Function("rk4", [x, u], [rk4_step(x, u)])


def _ju_fn():
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    return ca.Function("ju", [x, u], [ca.jacobian(rk4_step(x, u), u)])  # (NX, NU)


def check_dynamics():
    d = np.load(os.path.join(RES, "rk4_ref.npz"))
    XU, ref = d["XU"], d["ref"]
    f = _rk4_fn()
    got = np.array([np.asarray(f(xu[:NX], xu[NX:])).ravel() for xu in XU])
    err = np.abs(got - ref).max()
    print(f"[check_dynamics] max|acados_RK4 - turbompc_RK4| over {len(XU)} pts = {err:.3e}  "
          f"{'PASS' if err < 1e-10 else 'FAIL'}")


def build_ocp(hessian="GAUSS_NEWTON", globalization="MERIT_BACKTRACKING", umax=1e7):
    import tempfile
    from acados_template import AcadosModel, AcadosOcp
    m = AcadosModel()
    m.name = "cp_" + re.sub(r"[^a-z0-9]", "", hessian.lower())[:10]   # UNIQUE per hessian (no symbol clash)
    x = ca.SX.sym("x", NX); u = ca.SX.sym("u", NU)
    Q = ca.SX.sym("Q", NX); R = ca.SX.sym("R", NU)
    m.x = x; m.u = u; m.p_global = ca.vertcat(Q, R)
    m.disc_dyn_expr = rk4_step(x, u)
    ocp = AcadosOcp(); ocp.model = m
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.cost.cost_type = "EXTERNAL"; ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = ca.sum1(Q * x ** 2) + ca.sum1(R * u ** 2)   # NO 1/2
    ocp.model.cost_expr_ext_cost_e = ca.sum1(Q * x ** 2)
    if hessian == "GAUSS_NEWTON":
        # acados forms NO GN Hessian for a generic EXTERNAL cost -> give it explicitly.
        # GN Hessian of J = sum Q_i x_i^2 + R u^2  is  diag(2Q, 2R) (PSD); dynamics curvature dropped.
        ocp.model.cost_expr_ext_cost_custom_hess = ca.diag(ca.vertcat(2 * Q, 2 * R))
        ocp.model.cost_expr_ext_cost_custom_hess_e = ca.diag(2 * Q)
    ocp.p_global_values = P_GLOBAL.copy()
    ocp.solver_options.N_horizon = N_HOR; ocp.solver_options.tf = float(N_HOR)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = globalization
    ocp.solver_options.hessian_approx = hessian
    if hessian == "GAUSS_NEWTON":
        # turbompc's forward is GN + ADMM proximal rho*I; mirror the proximal with Levenberg-Marquardt
        # (adds mu*I to the Hessian; biases the STEPS, not the converged KKT point).
        ocp.solver_options.levenberg_marquardt = 1e-2
    if hessian == "EXACT":                       # sensitivity solver
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.tol = 1e-9
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.constraints.lbu = np.array([-umax]); ocp.constraints.ubu = np.array([umax])
    ocp.constraints.idxbu = np.arange(NU)
    ocp.constraints.x0 = np.zeros(NX)
    d = tempfile.mkdtemp(prefix="acados_cp_")
    ocp.code_export_directory = os.path.join(d, "c_generated_code")
    ocp.solver_options.build_dir = d
    return ocp


def make_solver(hessian, globalization="MERIT_BACKTRACKING"):
    from acados_template import AcadosOcpSolver
    ocp = build_ocp(hessian=hessian, globalization=globalization)
    return AcadosOcpSolver(ocp, json_file=os.path.join(ocp.solver_options.build_dir, "ocp.json"))


def _cold_solve(fwd, x0j):
    """Cold start from a feasible-ish guess (states held at x0, u=0). NO external warm-start."""
    for k in range(N_HOR + 1):
        fwd.set(k, "x", x0j)
    for k in range(N_HOR):
        fwd.set(k, "u", np.zeros(NU))
    fwd.set(0, "lbx", x0j); fwd.set(0, "ubx", x0j)
    fwd.solve()
    return fwd.get(0, "u"), int(fwd.status), float(np.max(fwd.get_stats("residuals")))


def check_forward(n_check=16):
    """Evidence that acados's INDEPENDENT GN forward converges to turbompc's solution EXACTLY:
    compare the full converged trajectory (states + controls), not just u0."""
    x0 = np.load(os.path.join(RES, "x0_cartpole_128.npy"))[:n_check]
    ts = np.load(os.path.join(RES, "turbompc_solution.npz"))
    st_tm, ct_tm = ts["states"][:n_check], ts["controls"][:n_check]   # (n,26,4),(n,26,1)
    fwd = make_solver("GAUSS_NEWTON")
    fwd.set_p_global_and_precompute_dependencies(P_GLOBAL)
    print(f"acados GN forward (cold, NO warm-start) vs turbompc full trajectory, {n_check} samples:")
    dxs, dus, rrs = [], [], []
    for j in range(n_check):
        u0, st, rr = _cold_solve(fwd, x0[j])
        xs = np.array([fwd.get(k, "x") for k in range(N_HOR + 1)])    # (26,4)
        us = np.array([fwd.get(k, "u") for k in range(N_HOR)])        # (25,1)
        dx = float(np.abs(xs - st_tm[j]).max())
        du = float(np.abs(us - ct_tm[j, :N_HOR, :]).max())
        dxs.append(dx); dus.append(du); rrs.append(rr)
        print(f"  s{j:2d}: acados KKTres={rr:.1e}  max|x-x_tm|={dx:.2e}  max|u-u_tm|={du:.2e}")
    print(f"OVERALL: max|x diff|={max(dxs):.2e}  max|u diff|={max(dus):.2e}  max KKTres={max(rrs):.1e}")


def gradient():
    x0 = np.load(os.path.join(RES, "x0_cartpole_128.npy"))
    n = x0.shape[0]
    ts = np.load(os.path.join(RES, "turbompc_solution.npz"))
    st_tm, ct_tm = ts["states"], ts["controls"]              # full turbompc traj (128,26,4),(128,26,1)
    fwd = make_solver("GAUSS_NEWTON")            # robust forward
    sens = make_solver("EXACT")                  # exact-Hessian backward
    fwd.set_p_global_and_precompute_dependencies(P_GLOBAL)
    sens.set_p_global_and_precompute_dependencies(P_GLOBAL)
    f_rk4, f_ju = _rk4_fn(), _ju_fn()
    gQ = np.zeros((n, NX)); gR = np.zeros((n, NU))
    u0_all = np.zeros((n, NU)); status = np.zeros(n, int); res = np.zeros(n)
    dxmax = np.zeros(n); dumax = np.zeros(n)
    for j in range(n):
        u0, st, rr = _cold_solve(fwd, x0[j]); status[j] = st; u0_all[j] = u0; res[j] = rr
        xs = np.array([fwd.get(k, "x") for k in range(N_HOR + 1)])    # acados full traj
        us = np.array([fwd.get(k, "u") for k in range(N_HOR)])
        dxmax[j] = np.abs(xs - st_tm[j]).max(); dumax[j] = np.abs(us - ct_tm[j, :N_HOR]).max()
        x1 = np.asarray(f_rk4(x0[j], u0)).ravel()
        Ju = np.asarray(f_ju(x0[j], u0))
        seed = (2.0 * u0 + Ju.T @ (2.0 * x1)).reshape(NU, 1)        # dC/du0
        sens.set(0, "lbx", x0[j]); sens.set(0, "ubx", x0[j])
        sens.load_iterate_from_flat_obj(fwd.store_iterate_to_flat_obj())
        sens.setup_qp_matrices_and_factorize()
        sa = np.asarray(sens.eval_adjoint_solution_sensitivity(
            seed_x=None, seed_u=[(0, seed)])).ravel()              # d C / d p_global (5,)
        gQ[j] = sa[:NX]; gR[j] = sa[NX:NX + NU]
    # SAME-SOLUTION confirmation (the backward comparison is only valid where the FULL forward
    # solution matches): require the entire state+control trajectory to agree, not just u0.
    matched = (dxmax < 1e-6) & (dumax < 1e-6)
    print(f"[gradient] acados GN+LM forward reached turbompc's SAME solution (full traj: "
          f"max|x|<1e-6 AND max|u|<1e-6) for {int(matched.sum())}/{n}  "
          f"(rest = different local min). max acados KKT res={res.max():.1e}")
    print(f"  grad finite={np.isfinite(gQ).all() and np.isfinite(gR).all()}  "
          f"sample0 grad_Q={np.round(gQ[0],4)} grad_R={np.round(gR[0],4)}")
    np.savez(os.path.join(RES, "acados_grad.npz"), grad_Q=gQ, grad_R=gR, u0=u0_all,
             status=status, res=res, dxmax=dxmax, dumax=dumax, matched=matched)
    print(f"  saved acados_grad.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check_dynamics", action="store_true")
    p.add_argument("--check_forward", action="store_true")
    p.add_argument("--gradient", action="store_true")
    a = p.parse_args()
    if a.check_dynamics: check_dynamics()
    if a.check_forward: check_forward()
    if a.gradient: gradient()
