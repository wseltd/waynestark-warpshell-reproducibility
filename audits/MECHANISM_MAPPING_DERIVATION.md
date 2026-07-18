# Mechanism mapping derivation — result: D (MAPPING NOT ESTABLISHED)

This documents why the predeclared `C_LP`-to-`r_S` finite-amplitude mechanism bridge
**cannot be derived** from the frozen artefacts. The favourable surrogate number does not
override this: a numerically favourable result does not repair a missing operator mapping.

## 1. How the full LP objective `ell_0` is constructed

Traced from `compute_T170_1_pipeline.py`:
```
K_at, kind = kernel_value_at_cells("delta_H", cells)   # the delta_H OBSERVABLE kernel at cells
k_a        = kernel_to_basis(K_at, kind, Phi, cells)   # = Phi^T ( K_deltaH . dV )   (observable in basis)
Q_LP       = qr( matched_control_matrix.T )            # control-space ONB (10 physical templates)
k_res      = k_a - Q_LP (Q_LP^T k_a)                   # matched-control-orthogonal residual
ell_0      = k_res / max|k_res|                        # the LP objective (up to sign)
```
`ell_0` is built from the **delta_H observable** (a fixed Sagnac/holonomy kernel), matched-control
-orthogonalised. **The construction contains no source-to-response operator** — the "kernel" here
is the observable, not a response operator.

## 2. The reduced mechanism has no full-order object to act on

The reduced spectral model uses a **surrogate** Gaussian response operator
`K_0 = build_K(32, sigma=0.10)` (32x32), and the finite-amplitude mechanism is an **additive
kernel perturbation**
```
K_S = K_0 + outer(b, b) / N_r ,   b = 0.4*0.005 / ((z-0.5)^2 + 0.005^2)   (N_r = 32)
```
`K_0` is a 32-dimensional surrogate that **does not appear anywhere in the full 2880-dimensional
delta_H LP construction**. There is no frozen `K_full` corresponding to `K_0`. Therefore the additive
perturbation `K_S - K_0 = outer(b,b)/N_r` **has no defined object to act on in the full LP**. The
mechanism cannot be mapped to the full problem from the existing operator/observable definitions.

## 3. The surrogate lift used in the run was not a derivation

The v15 run computed `ell_S = M_rad @ ell_0` with `M_rad = I + outer(b,b)/N_r` applied to the radial
index of the **already-projected** objective. This is a **surrogate objective transformation**, not a
derivation of the mechanism:
- If a response operator existed with `ell_0 = K_0^T q`, the additive perturbation would give
  `ell_S = K_S^T q = ell_0 + b (b^T q)/N_r` — which uses `b^T q`, **not** `b^T ell_0`. Since `q != ell_0`,
  `M_rad @ ell_0 != K_S^T q`. The two are not equal.
- Applying `M_rad` **after** `Q_LP` (to `ell_0`, which is already matched-control-orthogonal) reintroduces
  control-subspace components, because `M_rad` and `Q_LP` do not commute.

Measured (audit `v15_audit.py`):
- `orthogonality_ratio_0`  (frozen `ell_0`)     = 8.0e-17   (correctly orthogonal)
- `orthogonality_ratio_S`  (surrogate `ell_S`)  = 7.8e-2    (**control components reintroduced**)
- `||ell_S - ell_0|| / ||ell_0||`               = 0.437
- `cosine_similarity(ell_0, ell_S)`             = 0.926

Reapplying `Q_LP` last would fix **only** the orthogonality defect; it would **not** supply the missing
common source-to-response operator. The mapping remains underivable.

## 4. Lorentzian normalisation (internally consistent; App C text is wrong)

- code / frozen `r_S` / surrogate lift all use `b = w*gamma / (...)`  ->  `||b_code|| = 10.00`.
- Appendix C text prints `w*gamma^2 / (...)`  ->  `||b_manuscript|| = 0.050`.
- ratio = **200x**. The `gamma^2` numerator is a **manuscript error**; it was not used in any computation,
  so the bridge numbers are internally consistent. App C must be corrected (v15 pass, item B).

## 5. Classification

**D: MAPPING NOT ESTABLISHED.** The additive reduced-kernel mechanism cannot be defensibly mapped from
the frozen full-order operator/observable definitions, because no common source-to-response operator links
the reduced Gaussian `K_0` to the full delta_H LP objective. The numerical surrogate result is preserved in
`FAILED_SURROGATE_LIFT_RESULTS.csv` as evidence that numerical agreement is insufficient without an operator
derivation. It is **not** a predictive bridge and is **not** used in the manuscript main results.

Deriving a common full-order source-to-response operator is a defined **grant work package** for the
source-realisation workbench.
