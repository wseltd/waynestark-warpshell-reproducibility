#!/usr/bin/env python3
"""Export the repo's frozen certificate matrices to certificate.mat for the MATLAB checks.
Reads data/certificate/ (the same frozen inputs verify_all.py uses) so the MATLAB cross-solve
runs on the repo's data, not a private copy."""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.io import savemat

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "data" / "certificate"
C_FULL = 5.826049575311591
NORM = C_FULL / 15.897143289973446

v = np.load(CERT / "vectors_B1p5.npz")
payload = dict(
    Aeq=sparse.load_npz(CERT / "A_eq.npz").tocsc(),
    Aineq=sparse.load_npz(CERT / "A_ineq.npz").tocsc(),
    ell=v["ell"].reshape(-1, 1), b_ub=v["b_ub"].reshape(-1, 1),
    tau_star=v["tau_star"].reshape(-1, 1), B=1.5, NORM=NORM, C_FULL=C_FULL,
)
out = REPO / "matlab" / "certificate.mat"
savemat(out, payload, do_compression=True)
print(f"wrote {out}  (Aeq {payload['Aeq'].shape}, Aineq {payload['Aineq'].shape})")
