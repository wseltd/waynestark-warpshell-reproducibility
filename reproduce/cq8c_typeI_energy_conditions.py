"""CQ8(c): exact pointwise type-I (Hawking-Ellis) WEC/DEC test on the frozen LP optimum tau*.
Assemble T_munu per cell from tau*, form T^mu_nu, eigen-decompose, classify, test WEC/DEC."""
import numpy as np, sys
from scipy import sparse
from pathlib import Path as _P
D=str(_P(__file__).resolve().parents[1] / "generators")
SC=str(_P(__file__).resolve().parents[1] / "data" / "certificate")
sys.path.insert(0,D)
import compute_T170_1_pipeline as M
tau=np.load(SC+"/vectors_B1p5.npz")["tau_star"]
Phi,cells,_,_=M.build_basis_matrix(32,N_dir=12)
Nc,Nm=Phi.shape; NS=10
comp=np.array([Phi@tau[q*Nm:(q+1)*Nm] for q in range(NS)])   # (10, Ncells): rho,Sx,Sy,Sz,Pxx,Pyy,Pzz,Pxy,Pxz,Pyz
g=np.diag([-1.0,1,1,1])
scale=np.max(np.abs(comp))+1e-300
tol=1e-8*scale
nWEC=nDEC=nTypeI=0; viol_examples=[]
for c in range(Nc):
    rho,Sx,Sy,Sz,Pxx,Pyy,Pzz,Pxy,Pxz,Pyz=comp[:,c]
    Tdn=np.array([[rho,Sx,Sy,Sz],[Sx,Pxx,Pxy,Pxz],[Sy,Pxy,Pyy,Pyz],[Sz,Pxz,Pyz,Pzz]])
    Tmix=g@Tdn   # T^mu_nu
    w,V=np.linalg.eig(Tmix)
    if np.max(np.abs(w.imag))>1e-6*scale:   # complex -> not type-I
        continue
    w=w.real; V=V.real
    gn=np.array([V[:,k]@g@V[:,k] for k in range(4)])
    tl=np.where(gn<0)[0]
    if len(tl)!=1:   # not exactly one timelike eigenvector
        continue
    nTypeI+=1
    lam_t=w[tl[0]]; ps=np.delete(w,tl[0])
    rho_rf=-lam_t   # rest-frame energy density
    wec=(rho_rf>=-tol) and all(rho_rf+p>=-tol for p in ps)
    dec=(rho_rf>=-tol) and all(abs(p)<=rho_rf+tol for p in ps)
    nWEC+=wec; nDEC+=dec
    if not wec and len(viol_examples)<3: viol_examples.append((c,rho_rf,ps))
print(f"cells={Nc}  type-I (real eig, one timelike)={nTypeI}  ({100*nTypeI/Nc:.1f}%)")
print(f"WEC-satisfied cells: {nWEC}/{Nc} ({100*nWEC/Nc:.1f}%)")
print(f"DEC-satisfied cells: {nDEC}/{Nc} ({100*nDEC/Nc:.1f}%)")
print(f"(tol={tol:.2e}, scale max|T|={scale:.3e})")
if viol_examples:
    print("sample WEC violations (cell, rho_rf, principal pressures):")
    for c,r,ps in viol_examples: print(f"  cell {c}: rho={r:+.3e} p={np.round(ps,4)}")
print("\nCQ8(c) INTERPRETATION: this is the exact covariant test the surrogate rows do NOT enforce.")
print("  high WEC/DEC fraction => surrogate optimum largely respects pointwise conditions (title defensible);")
print("  significant violations => 'positive-energy' names the BM background class, not the surrogate optimum.")
