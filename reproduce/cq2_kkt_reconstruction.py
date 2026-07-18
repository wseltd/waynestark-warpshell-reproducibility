"""CQ2c: verify the bounded-variable KKT certificate with the physically correct
multiplier signs. Stored mu,alpha are <=0, beta >=0; KKT multipliers (all >=0) are
mu_K=-mu, alpha_K=-alpha, beta_K=+beta. Stationarity: ell = Aeq^T lam + Aineq^T mu_K + alpha_K - beta_K."""
import numpy as np
from scipy import sparse
from pathlib import Path as _P
SC = str(_P(__file__).resolve().parents[1] / "data" / "certificate")
Aeq = sparse.load_npz(SC+"/A_eq.npz").tocsr(); Aineq = sparse.load_npz(SC+"/A_ineq.npz").tocsr()
v = np.load(SC+"/vectors_B1p5.npz")
ell=v["ell"]; b=v["b_ub"]; tau=v["tau_star"]; lam=v["lambda_eq"]
muK=-v["mu_ineq"]; alK=-v["alpha_upper"]; beK=v["beta_lower"]; B=1.5
# lam~0; try both signs, pick the one closing stationarity
best=None
for sL in (1,-1):
    r=float(np.max(np.abs(ell-(sL*(Aeq.T@lam)+Aineq.T@muK+alK-beK))))
    if best is None or r<best[0]: best=(r,sL)
r_stat,sL=best; laK=sL*lam
obj=float(ell@tau)
dual=float(b@muK + B*np.sum(alK+beK))
gap=dual-obj
slack=b-(Aineq@tau)
comp_mu=float(np.max(np.abs(muK*slack)))
comp_al=float(np.max(np.abs(alK*(tau-B))))
comp_be=float(np.max(np.abs(beK*(tau+B))))
print(f"KKT multiplier nonnegativity: min(mu_K)={muK.min():.3e}  min(alpha_K)={alK.min():.3e}  min(beta_K)={beK.min():.3e}")
print(f"stationarity r_stat        = {r_stat:.3e}   (manuscript ~1.19e-14)")
print(f"primal  ell^T tau*         = {obj:.15f}")
print(f"dual    b^T mu + B*(a+b)    = {dual:.15f}")
print(f"duality gap                = {gap:.3e}   (manuscript ~2.06e-13)")
print(f"complementarity: mu={comp_mu:.3e}  alpha={comp_al:.3e}  beta={comp_be:.3e}")
print(f"feas: eq={float(np.max(np.abs(Aeq@tau))):.3e}  ineq={float(max(0,np.max(Aineq@tau-b))):.3e}  box={float(max(0,np.max(np.abs(tau))-B)):.3e}")
print(f"active bounds: {int(np.sum(np.abs(tau)>=B-1e-7))}")
NORM=5.826049575311591/15.897143289973446
print(f"C_LP = {NORM*obj:.15f}")
PASS=(r_stat<=1e-10 and comp_mu<=1e-10 and comp_al<=1e-10 and comp_be<=1e-10 and abs(gap)<=1e-9
      and muK.min()>=-1e-12 and alK.min()>=-1e-12 and beK.min()>=-1e-12)
print(f"\nCQ2 VERDICT: {'PASS (class-A certificate independently reconstructed)' if PASS else 'FAIL/CHECK'}")
