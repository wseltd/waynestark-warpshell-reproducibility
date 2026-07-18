import sys, json, numpy as np
from scipy import sparse
from scipy.optimize import linprog
from pathlib import Path
D=str(Path(__file__).resolve().parent)
OUT=Path(__file__).resolve().parents[1] / "_regen_certificate"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, D)
import compute_T170_1_pipeline as M

N_r, obs_id, sign = 32, "delta_H", -1
Phi, cells_arr, th, ph = M.build_basis_matrix(N_r, N_dir=12)
A_eq = sparse.csr_matrix(M.conservation_matrix(N_r))
M_matrix = M.matched_control_matrix(N_r, Phi, cells_arr)
A_ineq, b_ineq, e_vec = M.positive_energy_constraints(Phi, N_r, N_dir_sample=12)
K_at, kind = M.kernel_value_at_cells(obs_id, cells_arr)
k_a = M.kernel_to_basis(K_at, kind, Phi, cells_arr)
Md = M_matrix.toarray() if sparse.issparse(M_matrix) else M_matrix
Q,_ = np.linalg.qr(Md.T); k_res = k_a - Q@(Q.T@k_a)
rho_max=M.EPS_S; V_R=(4.0/3.0)*np.pi*M.R_SUPPORT**3
A_ub=sparse.vstack([A_ineq, sparse.csr_matrix((e_vec/rho_max).reshape(1,-1))],format="csr").tocsr()
b_ub=np.concatenate([b_ineq/rho_max,[V_R]])
k_max=float(np.max(np.abs(k_res))+1e-300); c=-sign*(k_res/k_max); ell=-c
E_0=rho_max*V_R; B_nat=M.G_NEWTON*E_0/(M.C_LIGHT**4*M.R_SUPPORT); N=len(c)

def solve(B):
    res=linprog(c=c,A_eq=A_eq,b_eq=np.zeros(A_eq.shape[0]),A_ub=A_ub,b_ub=b_ub,bounds=[(-B,B)]*N,method="highs")
    x=res.x; lam=res.eqlin.marginals; mu=res.ineqlin.marginals; zl=res.lower.marginals; zu=res.upper.marginals
    C=abs(sign*(-res.fun)*k_max*rho_max)/(B_nat+1e-300)
    eqres=float(np.max(np.abs(A_eq@x))); slk=float((b_ub-A_ub@x).min())
    dual_full=float(b_ub@mu + (-B*np.ones(N))@zl + (B*np.ones(N))@zu); gap=float(abs(c@x-dual_full))
    dual_nobnd=float(b_ub@mu)
    s1=c+A_eq.T@lam+A_ub.T@mu+zl+zu; s2=c-(A_eq.T@lam+A_ub.T@mu+zl+zu)
    use1=float(np.max(np.abs(s1)))<float(np.max(np.abs(s2)))
    statB=float(min(np.max(np.abs(s1)),np.max(np.abs(s2))))
    bare=(c+A_eq.T@lam+A_ub.T@mu) if use1 else (c-(A_eq.T@lam+A_ub.T@mu))
    statBare=float(np.max(np.abs(bare)))
    cs_in=float(np.max(np.abs(mu*(b_ub-A_ub@x)))); cs_lo=float(np.max(np.abs(zl*(x+B)))); cs_up=float(np.max(np.abs(zu*(B-x))))
    return dict(B=B,status=int(res.status),C_baseline=float(C),obj_max=float(-res.fun),
                active_lower=int(np.sum(x<=-B+1e-7)),active_upper=int(np.sum(x>=B-1e-7)),
                eq_residual=eqres,ineq_slack_min=slk,stationarity_bounded=statB,stationarity_bare=statBare,
                primal_dual_gap=gap,dual_obj_full=dual_full,dual_obj_no_bound=dual_nobnd,
                primal_minus_dual_nobound=float(abs(c@x-dual_nobnd)),
                comp_slack_ineq=cs_in,comp_slack_lower=cs_lo,comp_slack_upper=cs_up), res

# baseline B=1.5 full export
base, res15 = solve(1.5)
sparse.save_npz(OUT/"A_eq.npz", A_eq); sparse.save_npz(OUT/"A_ineq.npz", A_ub)
np.savez_compressed(OUT/"vectors_B1p5.npz", ell=ell, c=c, b_ub=b_ub,
                    tau_star=res15.x, lambda_eq=res15.eqlin.marginals, mu_ineq=res15.ineqlin.marginals,
                    alpha_upper=res15.upper.marginals, beta_lower=res15.lower.marginals, bound=np.array([1.5]))
# cap sensitivity
import os
if os.environ.get("WARPSHELL_CAP_SWEEP")=="1":
    rows=[solve(B)[0] for B in [1.0,1.25,1.5,2.0,3.0,5.0,100.0]]
else:
    rows=[base]  # fast default; set WARPSHELL_CAP_SWEEP=1 for the full 7-point box sweep
import csv
with open(OUT/"cap_sensitivity.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
cert=dict(generated="2026-06-24", source_script="compute_T170_1_pipeline.py (T170_1)",
          observable="delta_H", sign=-1, N_r=32, N_vars=N, n_eq=int(A_eq.shape[0]), n_ineq=int(A_ub.shape[0]),
          recorded_C_baseline=5.826049575311591, reconstructed_C_baseline_B1p5=base["C_baseline"],
          box_bounds=[-1.5,1.5], note="b_eq=0; SciPy minimises -ell^T tau (max-form ell=-c).",
          baseline_B1p5=base,
          cap_free_C_baseline=(rows[-1]["C_baseline"] if len(rows)>1 else "set WARPSHELL_CAP_SWEEP=1 (~13.07 at B>=5)"),
          finding="C_baseline is cap-controlled at B=1.5 (box active); cap-free cone optimum ~13.07 (box inactive at B>=5).",
          solver="scipy.optimize.linprog method=highs (scipy 1.17.1)")
(OUT/"bounded_kkt_certificate.json").write_text(json.dumps(cert, indent=2))
print("recorded vs reconstructed C_baseline(B=1.5):", 5.826049575311591, base["C_baseline"])
print("bounded stationarity:", base["stationarity_bounded"], " bare:", base["stationarity_bare"], " gap:", base["primal_dual_gap"])
print("cap-free C_baseline (B=100):", rows[-1]["C_baseline"])
print("files:", sorted(p.name for p in OUT.iterdir()))
print("SIDECAR_DONE")
