# WarpShell — reproducibility package

Reproduces the numerical results of *A Stress-Energy-First Finite Audit Framework for
Positive-Energy Warp-Shell Geometries: Boxed LP Certificates, Reduced-Operator Sensitivity
and Source-Realisation Gates* (v18). Everything here runs from source; the load-bearing
inputs are frozen and hash-anchored.

**What the paper is (and is not).** It defines a continuum response coefficient `C[T]` (never
evaluated), then studies two *mathematically distinct* finite objects: (1) a boxed 2880-variable
LP with a bounded-variable KKT certificate, `C_LP = 5.826049575311591`; and (2) a separate
32-dimensional reduced spectral model giving calibrated indices `I_S`. Under an illustrative
scale mapping it reconstructs one conditional source scale and shows no ordinary compact-object
EOS, material, operational regime or standard realises it. It is **not** a warp-drive
construction, a universal no-go, or a GR-derived bound.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy 2.4.4, scipy 1.17.1, pandas 3.0.2

# 1) headline certificate -- verify the frozen artefacts + re-derive C_LP (seconds)
python reproduce/verify_all.py

# 2) regenerate the LP matrices FROM SOURCE and check they match the frozen ones (~4 min)
python reproduce/regenerate_and_compare.py

# 3) reduced-operator ladder (13.41, 26.12; clean 10.44/21.18) from source (several min)
python reduced_model/verify_reduced.py

# 4) matter-realisation gate (dependency-free) -- 29.4 M_sun vs compact-object ceilings
python matter_gate/verify_matter_gate.py
```

Every script prints an explicit `PASS`/`FAIL` and the reproduced numbers.

---

## What reproduces, and to what tolerance

| Result | How | Expected | Tolerance |
|---|---|---|---|
| **Certificate `C_LP`** | `verify_all.py` (Path A, frozen) | `5.826049575311591` | bit-exact (Δ ~1e-15) |
| **`C_LP` from source** | `regenerate_and_compare.py` (Path B) | matrices Δ = 0; `C_LP` Δ = 0 | bit-exact with pinned scipy |
| KKT certificate (stationarity, gap, complementarity) | CQ2 in `verify_all.py` | r_stat 1.19e-14, gap 2.06e-13 | ≤ 1e-10 |
| Row map (23809 = 384 cells × blocks) | CQ3 | max row-diff 0 | 0 |
| Objective ℓ (def, units, orthogonality) | CQ4 | ‖P_LPℓ‖/‖ℓ‖ ≈ 8e-17 | ≤ 1e-12 |
| **Reduced ladder** | `verify_reduced.py` | joint 13.41, sequential 26.12 | ≤ 0.01 |
| **Matter gate** | `verify_matter_gate.py` | 29.4 M☉ ≫ 3.2 M☉ (causal) | analytic |

`C_LP = NORM · max ℓᵀτ`, with `max ℓᵀτ = 15.897143289973446` and `NORM = k_max·ε_S/B_nat =
0.3664841` (the `c⁴R/G` aperture-energy factor; ε_S cancels, `NORM = k_max c⁴ R_SUPPORT/(G V_R)`).

---

## Layout

```
data/certificate/     frozen LP inputs: A_eq (128×2880), A_ineq (23809×2880), vectors, bounded_kkt json
generators/           compute_T170_1_pipeline.py (builds the matrices) + regenerate_certificate.py
reproduce/            verify_all.py, regenerate_and_compare.py, cq2/cq3/cq4/cq5/cq8c (independent checks)
reduced_model/        compute_T216 (joint) / T217 (sequential) / T215 (KILL) + verify_reduced.py
matter_gate/          verify_matter_gate.py (dependency-free) + compute_T224_audit.py + frozen T224 *.csv
bridges/              baseline constrained reduction (B-level) + preserved bridge_core result + PROVENANCE_NOTE
audits/               mechanism-mapping D audit (MECHANISM_MAPPING_DERIVATION.md) + v15_audit.py
paper/                main_v18 .tex, references.bib, built PDF
MANIFEST.sha256       SHA-256 of the frozen inputs and the manuscript
```

---

## Environment (why the pin matters)

Python 3.13, **scipy 1.17.1** (the HiGHS build fixes the last digit of `C_LP`, and
`scipy.special.sph_harm_y` is scipy≥1.15), numpy 2.4.4, pandas 3.0.2. With a different HiGHS,
Path A/B still reproduce `C_LP` to ~1e-9 but not necessarily bit-for-bit. Verify you have the
exact frozen inputs first: `sha256sum -c MANIFEST.sha256`.

## Honest scope / caveats (these are results too)

- The finite LP's energy-condition rows are **sampled stress-cone surrogates**, not exact
  covariant NEC/WEC/DEC. `cq8c` shows the LP optimum satisfies the exact pointwise conditions at
  only a minority of cells — so "positive-energy" names the Bobrick–Martire *background class*,
  not the optimum. This is by design and disclosed.
- The attempted `C_LP ↔ r_S` mechanism bridge is **mapping-not-established** (see `audits/`); the
  reduced indices are sensitivity quantities, never claimed equal to `C_LP`.
- The baseline constrained reduction is a **B-level** feasible lower-bound surrogate (~3.6–5%), not
  a high-accuracy replacement (`bridges/`).
- **MATLAB / WarpFactory cross-checks** (verifyTensor, cross-solve, grid convergence) are *not*
  in this repo — they need a MATLAB licence + WarpFactory (commit `03b10cb0`) and are run
  separately; the pure-Python paths above stand alone.
- `matter_gate/compute_T224_audit.py`'s γ=2 polytrope-TOV leg uses the optional `CompactObject-TOV`
  package (it returned NaN in the frozen run and is not decisive); the decisive causal-bound leg is
  reproduced dependency-free by `verify_matter_gate.py`.

## Citation & licence

See `paper/main_v18_final_external_review.pdf`. Licensed MIT (`LICENSE`).
