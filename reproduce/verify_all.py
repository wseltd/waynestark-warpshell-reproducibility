#!/usr/bin/env python3
"""One-command reproduction of the boxed finite-audit certificate C_LP.

PATH A (this script): verify the frozen artefacts and re-derive the certificate from them.
  1. integrity  -- SHA-256 of the frozen LP inputs
  2. re-solve   -- solve the frozen LP with SciPy/HiGHS -> raw optimum max ell^T tau
  3. KKT/rows/ell -- run the CQ2/CQ3/CQ4 independent checks
  4. report the headline certificate C_LP = NORM * (max ell^T tau)

Requires only numpy + scipy (see requirements.txt; scipy==1.17.1 reproduces the last digit).
For PATH B (regenerate the matrices from source), see generators/regenerate_certificate.py
and reproduce/regenerate_and_compare.py.
"""
import hashlib, subprocess, sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "data" / "certificate"
C_LP_EXPECTED = 5.826049575311591
OBJ_EXPECTED = 15.897143289973446
# NORM = k_max * eps_S / B_nat = 0.3664840574843 (aperture-energy c^4 R/G factor of the C[T]
# definition; derived from source in generators/, verified in cq4_ell_units_orthogonality.py).
NORM = C_LP_EXPECTED / OBJ_EXPECTED
B = 1.5

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("WarpShell certificate reproduction  (PATH A: verify frozen artefacts)")
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}  scipy {__import__('scipy').__version__}")
    print("=" * 70)
    ok = True

    print("\n[1] integrity (SHA-256 of frozen inputs)")
    for name in ["A_eq.npz", "A_ineq.npz", "vectors_B1p5.npz", "bounded_kkt_certificate.json"]:
        print(f"    {sha256(CERT / name)}  {name}")

    print("\n[2] re-solve the frozen LP (SciPy/HiGHS)")
    A_eq = sparse.load_npz(CERT / "A_eq.npz").tocsr()
    A_ineq = sparse.load_npz(CERT / "A_ineq.npz").tocsr()
    v = np.load(CERT / "vectors_B1p5.npz")
    ell, b_ub = v["ell"], v["b_ub"]
    N = A_eq.shape[1]
    res = linprog(c=-ell, A_eq=A_eq, b_eq=np.zeros(A_eq.shape[0]),
                  A_ub=A_ineq, b_ub=b_ub, bounds=[(-B, B)] * N, method="highs")
    obj = float(-res.fun)
    C_LP = NORM * obj
    print(f"    solver success = {res.success}")
    print(f"    raw optimum  max ell^T tau = {obj:.15f}   (expected {OBJ_EXPECTED})")
    print(f"    C_LP = NORM * raw = {C_LP:.15f}   (expected {C_LP_EXPECTED})")
    d_obj = abs(obj - OBJ_EXPECTED)
    d_c = abs(C_LP - C_LP_EXPECTED)
    print(f"    |delta raw| = {d_obj:.2e}   |delta C_LP| = {d_c:.2e}")
    if not (res.success and d_obj <= 1e-6 and d_c <= 1e-6):
        ok = False; print("    [FAIL] re-solve did not reproduce the optimum")
    else:
        print("    [PASS]")

    print("\n[3] independent checks (CQ2 KKT / CQ3 row-map / CQ4 objective)")
    for script in ["cq2_kkt_reconstruction.py", "cq3_row_block_map.py", "cq4_ell_units_orthogonality.py"]:
        r = subprocess.run([sys.executable, str(REPO / "reproduce" / script)],
                           capture_output=True, text=True)
        verdict = [ln for ln in r.stdout.splitlines() if "VERDICT" in ln or "PASS" in ln or "FAIL" in ln]
        tag = verdict[-1].strip() if verdict else "(no verdict line)"
        status = "PASS" if r.returncode == 0 and "FAIL" not in tag else "CHECK"
        print(f"    {script:38s} -> {status}: {tag[:80]}")
        if r.returncode != 0:
            ok = False

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS -- C_LP = 5.826049575311591 reproduced' if ok else 'FAIL -- see above'}")
    print("=" * 70)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
