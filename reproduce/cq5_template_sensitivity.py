"""CQ5: Q_LP projector diagnostics + sensitivity of C_LP to the control suite.
Baseline + 10 leave-one-template-out + 5 random 10-dim subspaces. Python only."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, sys
from scipy import sparse
from scipy.optimize import linprog
from multiprocessing import Pool
from pathlib import Path as _P
D=str(_P(__file__).resolve().parents[1] / "generators")
SC=str(_P(__file__).resolve().parents[1] / "data" / "certificate")
sys.path.insert(0,D)
import compute_T170_1_pipeline as M
Aeq=sparse.load_npz(SC+"/A_eq.npz").tocsr(); Aineq=sparse.load_npz(SC+"/A_ineq.npz").tocsr()
b_ub=np.load(SC+"/vectors_B1p5.npz")["b_ub"]
N=Aeq.shape[1]; B=1.5; beq=np.zeros(Aeq.shape[0]); NORM=5.826049575311591/15.897143289973446
Phi,cells,_,_=M.build_basis_matrix(32,N_dir=12)
K_at,kind=M.kernel_value_at_cells("delta_H",cells); k_a=M.kernel_to_basis(K_at,kind,Phi,cells)
Mmat=M.matched_control_matrix(32,Phi,cells)
Md=Mmat.toarray() if sparse.issparse(Mmat) else np.asarray(Mmat)   # 10 x 2880 templates
print(f"control matrix shape (templates x dim) = {Md.shape}")
Qf,_=np.linalg.qr(Md.T); P=Qf@Qf.T
print(f"projector: ||P^2-P||_F={np.linalg.norm(P@P-P):.3e}  ||P-P^T||_F={np.linalg.norm(P-P.T):.3e}  rank={np.linalg.matrix_rank(Qf)}")
def ellfrom(Qsub):
    kr=k_a-Qsub@(Qsub.T@k_a); return kr/(np.max(np.abs(kr))+1e-300)
# build objective list
objs=[("baseline",ellfrom(Qf))]
for i in range(10):
    keep=[r for r in range(10) if r!=i]; Qi,_=np.linalg.qr(Md[keep].T); objs.append((f"drop_tmpl_{i}",ellfrom(Qi)))
for s in range(5):
    rng=np.random.default_rng(1000+s); Gr=rng.standard_normal((N,10)); Qr,_=np.linalg.qr(Gr); objs.append((f"rand_seed_{1000+s}",ellfrom(Qr)))
def solve(item):
    name,ell=item
    r=linprog(c=-ell,A_eq=Aeq,b_eq=beq,A_ub=Aineq,b_ub=b_ub,bounds=[(-B,B)]*N,method="highs-ipm",options={"time_limit":150,"presolve":True})
    return (name, NORM*(-r.fun) if r.success else float("nan"), int(getattr(r,"status",-1)))
with Pool(8) as p:
    res=p.map(solve,objs)
print("\n case                 C_LP        delta_vs_baseline")
base=[c for n,c,s in res if n=="baseline"][0]
for n,c,s in res:
    print(f"  {n:18s}  {c:.6f}   {c-base:+.6f}")
vals=[c for n,c,s in res if n!="baseline"]
print(f"\nbaseline C_LP = {base:.6f} (declared 5.826)")
print(f"leave-one-out spread: [{min(vals[:10]):.4f}, {max(vals[:10]):.4f}]  max|Delta|={max(abs(x-base) for x in vals[:10]):.4f}")
print(f"random-subspace spread: [{min(vals[10:]):.4f}, {max(vals[10:]):.4f}]")
print("CQ5: sensitivity measurement (no pass/fail). Small spread => template choice not load-bearing.")
