#!/usr/bin/env python3
"""PATH B: regenerate the LP matrices from source and compare to the frozen artefacts.

Runs generators/regenerate_certificate.py (which rebuilds A_eq, A_ineq, ell and re-solves
the LP from compute_T170_1_pipeline.py), then compares the regenerated arrays to the frozen
ones by DATA (not file hash -- .npz zips embed timestamps). Deterministic construction
(A_eq, A_ineq, ell) must match to machine zero; the LP solution matches to solver tolerance.
"""
import subprocess, sys
from pathlib import Path
import numpy as np
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "data" / "certificate"
REGEN = REPO / "_regen_certificate"
C_LP_EXPECTED = 5.826049575311591

def arr_maxdiff(a, b):
    a = a.tocsr(); b = b.tocsr()
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs((a - b).data))) if (a - b).nnz else 0.0

def main():
    print("=" * 70)
    print("WarpShell certificate reproduction  (PATH B: regenerate from source)")
    print("=" * 70)
    print("\n[1] running generators/regenerate_certificate.py ...")
    r = subprocess.run([sys.executable, str(REPO / "generators" / "regenerate_certificate.py")],
                       capture_output=True, text=True)
    print("\n".join(r.stdout.splitlines()[-6:]))
    if r.returncode != 0:
        print("[FAIL] regeneration errored:\n", r.stderr[-1500:]); return 1

    print("\n[2] compare regenerated vs frozen (array data)")
    ok = True
    for name in ["A_eq.npz", "A_ineq.npz"]:
        d = arr_maxdiff(sparse.load_npz(REGEN / name), sparse.load_npz(CERT / name))
        s = "PASS" if d == 0.0 else ("PASS~" if d <= 1e-12 else "FAIL")
        print(f"    {name:14s} max|regen-frozen| = {d:.2e}  [{s}]")
        if d > 1e-12: ok = False
    fz, rg = np.load(CERT / "vectors_B1p5.npz"), np.load(REGEN / "vectors_B1p5.npz")
    for k in ["ell", "b_ub", "tau_star"]:
        if k in fz and k in rg:
            d = float(np.max(np.abs(fz[k] - rg[k])))
            tol = 0.0 if k in ("ell", "b_ub") else 1e-9   # ell/b_ub deterministic; tau_star from solver
            s = "PASS" if d <= max(tol, 1e-15) else "FAIL"
            print(f"    vectors[{k}]    max|regen-frozen| = {d:.2e}  [{s}]")
            if d > max(tol, 1e-12): ok = False

    import json
    cert = json.loads((REGEN / "bounded_kkt_certificate.json").read_text())
    c_regen = cert.get("reconstructed_C_baseline_B1p5", cert.get("recorded_C_baseline"))
    dC = abs(float(c_regen) - C_LP_EXPECTED)
    print(f"\n[3] regenerated C_LP = {c_regen}   |delta| = {dC:.2e}")
    if dC > 1e-9: ok = False

    print("\n" + "=" * 70)
    print(f"PATH B OVERALL: {'PASS -- matrices + C_LP regenerated from source match the frozen artefacts' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
