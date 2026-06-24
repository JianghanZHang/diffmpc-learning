import os,sys
for p in ("src","diffmpc2","research/gradient-quality-diffnmpc/experiments/linear_system"): sys.path.insert(0,os.path.abspath(p))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import cl_b
NX,NU=cl_b.NX,cl_b.NU; QK,RK=cl_b.QK,cl_b.RK
dyn,pp,solver,ig,schur,x0,A_sd,B_m,b_v = cl_b._setup(0)     # linear MPC, 8 x0, H=20, umax=1
w={QK:jnp.asarray(pp[QK]),RK:jnp.asarray(pp[RK])}
def loss(sc): s,c=sc; return 0.5*jnp.sum(s**2)+0.5*jnp.sum(c**2)
def flat(d): return np.concatenate([np.asarray(d[k]).reshape(-1) for k in (QK,RK)])
def cosm(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
FD=[1e-3,3e-4,1e-4,3e-5]
def run(SW,kappa):
    cl_b.GAMMA=SW                                          # monkeypatch slack weight (1e12 = no slack)
    b=cl_b.make_b_mpc_solve(solver,pp,schur,ig,NX,NU,kappa=kappa,cp_tol=1e-9)
    gfn=jax.jit(jax.vmap(jax.grad(lambda ww,st: loss(b(ww,st))), in_axes=(None,0)))
    cv =jax.jit(jax.vmap(lambda ww,st: loss(b(ww,st)), in_axes=(None,0)))
    gB=gfn(w,x0); gB=np.stack([flat({QK:gB[QK][i],RK:gB[RK][i]}) for i in range(x0.shape[0])])
    per=[]
    for eps in FD:
        cols=[]
        for k in (QK,RK):
            base=np.asarray(w[k],float)
            for i in range(base.size):
                ap=base.copy();ap[i]+=eps; am=base.copy();am[i]-=eps
                cols.append((np.asarray(cv({**w,k:jnp.asarray(ap)},x0))-np.asarray(cv({**w,k:jnp.asarray(am)},x0)))/(2*eps))
        per.append(np.stack(cols,1))
    g=np.array(per[-1]); fl=np.zeros(x0.shape[0],bool)
    for n in range(x0.shape[0]):
        ok=True
        for j in range(per[0].shape[1]):
            seq=[pe[n,j] for pe in per]; ch,o=seq[-1],False
            for a,bb in zip(seq[:-1],seq[1:]):
                if abs(a-bb)<=1e-2*abs(bb)+1e-7: ch,o=a,True;break
            g[n,j]=ch; ok=ok and o
        fl[n]=not ok
    keep=~fl; cs=np.array([cosm(gB[n],g[n]) for n in range(x0.shape[0])])
    return (np.median(cs[keep]) if keep.any() else np.nan), int(fl.sum())
print(f"{'kappa':>7} | {'B slack g=1e4':>14} {'flag':>5} | {'B NO-slack g=1e12':>18} {'flag':>5}")
for kp in [1e-2,1e-3,1e-4,1e-6,1e-9]:
    c1,f1=run(1e4,kp); c2,f2=run(1e12,kp)
    print(f"{kp:7.0e} | {c1:14.4f} {f1:5d} | {c2:18.4f} {f2:5d}")
