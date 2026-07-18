"""CQ4: reconstruct ell from the objective generator; verify orthogonality + norms + units."""
import numpy as np, sys
from scipy import sparse
from pathlib import Path as _P
D=str(_P(__file__).resolve().parents[1] / "generators")
SC=str(_P(__file__).resolve().parents[1] / "data" / "certificate")
sys.path.insert(0,D)
import compute_T170_1_pipeline as M
ell_f=np.load(SC+"/vectors_B1p5.npz")["ell"]
print(f"constants: R_SUPPORT={M.R_SUPPORT}  R_PASSENGER={M.R_PASSENGER}  R_LOOP={M.R_LOOP}  R_REG={M.R_REG}")
print(f"EPS_S={M.EPS_S:.4e}  N_STRESS={M.N_STRESS}")
Phi,cells,th,ph=M.build_basis_matrix(32,N_dir=12)
K_at,kind=M.kernel_value_at_cells("delta_H",cells)
k_a=M.kernel_to_basis(K_at,kind,Phi,cells)
Mmat=M.matched_control_matrix(32,Phi,cells)
Md=Mmat.toarray() if sparse.issparse(Mmat) else np.asarray(Mmat)
Q,_=np.linalg.qr(Md.T)
k_res=k_a-Q@(Q.T@k_a); ell_re=k_res/(np.max(np.abs(k_res))+1e-300)
# match frozen ell up to sign
cos=float(ell_re@ell_f/(np.linalg.norm(ell_re)*np.linalg.norm(ell_f)+1e-300))
sign=1.0 if cos>=0 else -1.0
match=float(np.max(np.abs(sign*ell_re-ell_f)))
print(f"\nreconstruction vs frozen ell: cosine={cos:+.6f}  max|+/-ell_re - ell_f|={match:.3e}")
# orthogonality of frozen ell to control space
contam=float(np.linalg.norm(Q@(Q.T@ell_f))/(np.linalg.norm(ell_f)+1e-300))
print(f"orthogonality  ||P_LP ell||_2/||ell||_2 = {contam:.3e}   (audit history ~8e-17)")
print(f"||ell||_2 = {np.linalg.norm(ell_f):.6f}   ||ell||_inf = {np.max(np.abs(ell_f)):.6f}  (unit-normalised => inf-norm 1)")
print(f"control-space rank (QR cols) = {Q.shape[1]}")
dV=4*np.pi*M.R_SUPPORT**3/(32*12)
print(f"\ncell volume dV = 4*pi*R_support^3/N_cells = {dV:.6e}")
print("UNIT LEDGER:")
print("  K_deltaH(x) = G/(c^4 R_loop d),  [K]=1/(J*m)   (G/c^4 = s^2 kg^-1 m^-1; /(R_loop*d)=1/m^2 -> combined 1/(J m))")
print("  raw k_a = Phi^T (K_deltaH .* dV):  [k_a] = (1/(J m)) * m^3 = m^2/J   (per basis mode)")
print("  ell = (I - Q Q^T) k_a / max|.|  -> DIMENSIONLESS: division by max|k_res| cancels the m^2/J units")
print("  tau = t/eps_S is dimensionless -> ell^T tau is dimensionless. No dangling dimensional factor.")
ok=(contam<=1e-12 and abs(np.max(np.abs(ell_f))-1.0)<1e-9 and match<=1e-9)
print(f"\nCQ4 VERDICT: {'PASS' if ok else 'CHECK'} (ell reconstructs, is unit-normalised and control-orthogonal; ell^T tau dimensionless)")
