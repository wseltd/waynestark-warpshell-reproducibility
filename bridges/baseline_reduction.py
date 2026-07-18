"""V14 baseline constrained reduction (finite, scoped).
160 deterministic feasible perturbed-objective snapshots -> rank check -> POD + greedy
adaptive basis -> reduced constrained LPs with full bounded-KKT diagnostics.
tau_0 is EXCLUDED from training (held-out evaluation only). Read-only on frozen matrices.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, json, time, csv
from scipy import sparse
from scipy.optimize import linprog
from multiprocessing import Pool

BASE = "/home/onur/projects/warp-drive/paper/arxiv_warpshell_constitutive_closure/article_extraction_v10_author_workflow"
SC = BASE + "/final_deliverables_optionA_freeze_20260624/bounded_kkt_certificate"
OUT = BASE + "/bridge_validation_v14"
C_FULL = 5.826049575311591; NORM = C_FULL / 15.897143289973446
Aeq = sparse.load_npz(SC + "/A_eq.npz").tocsr(); Aineq = sparse.load_npz(SC + "/A_ineq.npz").tocsr()
_v = np.load(SC + "/vectors_B1p5.npz"); ell = _v["ell"]; b_ub = _v["b_ub"]; tau0 = _v["tau_star"]
N = Aeq.shape[1]; B = 1.5; beq = np.zeros(Aeq.shape[0])
N_SNAP = 160; SEEDS = list(range(N_SNAP))

def solve_perturbed(seed):
    # identical snapshot definition to bridge_core (0.5*g perturbed objective on the
    # feasible optimal face); only the solver differs -- interior-point is far more
    # predictable than simplex on this degenerate 2880-var/23809-row polytope.
    g = np.sin(0.7 * seed + np.arange(N) * (0.31 + 0.013 * seed)); g /= (np.linalg.norm(g) + 1e-300)
    t0 = time.time()
    r = linprog(c=-(ell + 0.5 * g), A_eq=Aeq, b_eq=beq, A_ub=Aineq, b_ub=b_ub,
                bounds=[(-B, B)] * N, method="highs-ipm",
                options={"time_limit": 150, "presolve": True})
    dt = time.time() - t0
    return (seed, r.x.copy() if r.success else None, float(dt), int(getattr(r, "status", -1)))

def reduced_LP_kkt(Vr, Bx):
    r = Vr.shape[1]
    A_ub = sparse.vstack([sparse.csr_matrix(Aineq @ Vr), sparse.csr_matrix(Vr), sparse.csr_matrix(-Vr)], format="csr")
    b = np.concatenate([b_ub, Bx * np.ones(N), Bx * np.ones(N)])
    cz = -(Vr.T @ ell)
    t0 = time.time(); res = linprog(c=cz, A_ub=A_ub, b_ub=b, bounds=[(None, None)] * r, method="highs"); dt = time.time() - t0
    if not res.success:
        return dict(r=r, success=False, status=res.status)
    z = res.x; tau = Vr @ z; obj = -res.fun; Cred = NORM * obj
    mu = -res.ineqlin.marginals                      # >=0 marginals for A_ub z <= b (min form)
    stat = float(np.max(np.abs(cz + A_ub.T @ (res.ineqlin.marginals))))   # c + A^T y = 0 stationarity
    slack = b - A_ub @ z
    comp = float(np.max(np.abs(res.ineqlin.marginals * slack)))
    dual_obj = float(-b @ (-res.ineqlin.marginals)) if False else float(b @ res.ineqlin.marginals)
    gap = float(abs(cz @ z - (b @ res.ineqlin.marginals)))
    return dict(r=r, success=True, Cred=Cred, relerr=abs(Cred - C_FULL) / C_FULL,
                ineq_viol=float(max(0.0, np.max(Aineq @ tau - b_ub))), box_viol=float(max(0.0, np.max(np.abs(tau) - Bx))),
                eq_res=float(np.max(np.abs(Aeq @ tau))), stat=stat, comp=comp, gap=gap,
                dual_sign_viol=int(np.sum(res.ineqlin.marginals > 1e-9)),  # marginals should be <=0 for <= in min-form
                active_ineq=int(np.sum(slack[:Aineq.shape[0]] < 1e-7)),
                active_box=int(np.sum(np.abs(tau) >= Bx - 1e-7)), dt=dt, orth=float(np.linalg.norm(Vr.T @ Vr - np.eye(r))))

if __name__ == "__main__":
    t0 = time.time()
    snaps = []; done = 0; skipped = 0; dts = []
    with Pool(24) as p:
        for seed, x, dt, status in p.imap_unordered(solve_perturbed, SEEDS):
            done += 1; dts.append(dt)
            if x is not None:
                snaps.append(x)
            else:
                skipped += 1
            if done % 8 == 0 or done == N_SNAP:
                print(f"[snap] {done}/{N_SNAP} solved ({len(snaps)} feasible, {skipped} skipped) "
                      f"last_dt={dt:.1f}s median_dt={float(np.median(dts)):.1f}s elapsed={time.time()-t0:.0f}s",
                      flush=True)
    S = np.array(snaps).T
    np.savez_compressed(OUT + "/bases/snapshots.npz", S=S, seeds=np.array(SEEDS))
    print(f"[snap] DONE {S.shape[1]} feasible snapshots in {time.time()-t0:.0f}s "
          f"({skipped} skipped)  A_eq@S max={float(np.max(np.abs(Aeq@S))):.1e}", flush=True)
    U, sig, _ = np.linalg.svd(S, full_matrices=False)
    tol = 1e-10 * sig[0]; nrank = int(np.sum(sig > tol))
    print(f"[rank] numerical_rank={nrank} (tol={tol:.2e})  sig[0]={sig[0]:.3e} sig[-1]={sig[-1]:.3e}", flush=True)
    with open(OUT + "/BASELINE_BASIS_SPECTRUM.csv", "w", newline="") as f:  # baseline-specific name (bridge_core owns BASIS_SPECTRUM.csv)
        w = csv.writer(f); w.writerow(["mode", "singular_value", "cum_energy"]); tot = np.sum(sig**2)
        for i, s in enumerate(sig): w.writerow([i+1, s, float(np.sum(sig[:i+1]**2)/tot)])
    # train/val split (exclude tau0 always)
    ntr = int(0.8 * S.shape[1]); Str = S[:, :ntr]; Sval = S[:, ntr:]
    Utr, sigtr, _ = np.linalg.svd(Str, full_matrices=False)
    # POD reduced LPs
    rows = []
    for r in [16, 32, 64, 128]:
        if r > nrank:
            print(f"[POD r={r}] SKIPPED: numerical_rank {nrank} < {r}", flush=True); continue
        res = reduced_LP_kkt(U[:, :r], B); res["basis"] = "POD"; rows.append(res)
        print(f"[POD r={r:3d}] C_red={res.get('Cred',float('nan')):.6f} relerr={res.get('relerr',float('nan')):.4%} "
              f"gap={res.get('gap',0):.1e} stat={res.get('stat',0):.1e} ineq_v={res.get('ineq_viol',0):.1e} t={res.get('dt',0):.1f}s", flush=True)
    # greedy adaptive basis from TRAINING snapshots only
    V = Utr[:, :16].copy()
    used = set()
    while V.shape[1] < min(128, nrank):
        resid_err = []
        for j in range(Str.shape[1]):
            if j in used: resid_err.append(-1); continue
            x = Str[:, j]; rj = x - V @ (V.T @ x); resid_err.append(np.linalg.norm(rj) / (np.linalg.norm(x)+1e-300))
        j = int(np.argmax(resid_err))
        if resid_err[j] <= 1e-12: break
        x = Str[:, j]; rj = x - V @ (V.T @ x); rj = rj - V @ (V.T @ rj)  # reorthogonalise
        V = np.hstack([V, (rj/np.linalg.norm(rj)).reshape(-1,1)]); used.add(j)
    greedy_full = V
    for r in [32, 64, 128]:
        if r > greedy_full.shape[1]:
            print(f"[greedy r={r}] SKIPPED: greedy dim {greedy_full.shape[1]} < {r}", flush=True); continue
        res = reduced_LP_kkt(greedy_full[:, :r], B); res["basis"] = "greedy"; rows.append(res)
        print(f"[greedy r={r:3d}] C_red={res.get('Cred',float('nan')):.6f} relerr={res.get('relerr',float('nan')):.4%} "
              f"gap={res.get('gap',0):.1e} ineq_v={res.get('ineq_viol',0):.1e} t={res.get('dt',0):.1f}s", flush=True)
    # held-out + tau0 reconstruction diagnostics
    def recon_err(Vb, X):
        R = X - Vb @ (Vb.T @ X); return float(np.median(np.linalg.norm(R,axis=0)/(np.linalg.norm(X,axis=0)+1e-300))), float(np.max(np.linalg.norm(R,axis=0)/(np.linalg.norm(X,axis=0)+1e-300)))
    pod64 = U[:, :64]
    ho_med, ho_max = recon_err(pod64, Sval)
    tau0_err = float(np.linalg.norm(tau0 - pod64 @ (pod64.T @ tau0))/np.linalg.norm(tau0))
    # principal angles POD vs greedy (r=64)
    from numpy.linalg import svd as _svd
    _, s_pa, _ = _svd(U[:, :64].T @ greedy_full[:, :min(64, greedy_full.shape[1])])
    pa = np.degrees(np.arccos(np.clip(s_pa, -1, 1)))
    print(f"[diag] rank={nrank} POD64 held-out recon med={ho_med:.3e} max={ho_max:.3e}  tau0 recon err(POD64)={tau0_err:.3e}", flush=True)
    print(f"[diag] principal angles POD64 vs greedy (deg): min={pa.min():.2f} max={pa.max():.2f}", flush=True)
    with open(OUT + "/BASELINE_ADAPTIVE_BASIS_RESULTS.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["basis","r","C_LP_red","rel_error","eq_res","ineq_viol","box_viol","gap","stat","comp","active_ineq","active_box","solve_time_s","orth"])
        for x in rows:
            w.writerow([x["basis"],x["r"],x.get("Cred"),x.get("relerr"),x.get("eq_res"),x.get("ineq_viol"),x.get("box_viol"),x.get("gap"),x.get("stat"),x.get("comp"),x.get("active_ineq"),x.get("active_box"),x.get("dt"),x.get("orth")])
    json.dump(dict(n_snapshots=int(S.shape[1]), numerical_rank=nrank, train=ntr, val=int(Sval.shape[1]),
                   tau0_recon_err_POD64=tau0_err, heldout_recon_med=ho_med, heldout_recon_max=ho_max,
                   principal_angle_min_deg=float(pa.min()), principal_angle_max_deg=float(pa.max())),
              open(OUT + "/baseline_reduction_summary.json", "w"), indent=2)
    print("DONE", flush=True)
