# Provenance note — bridge_validation_v14 (two distinct studies share this directory)

This directory holds **two separate, independently valid** reduced-order studies. They must
not be confused.

## 1. bridge_core (the preserved B-level POD bridge)

- Script: `code_python/bridge_core.py`
- Certified result: `BRIDGE_RESULTS.csv`, `bridge_summary.json`
  (71 snapshots incl. `tau_star`; POD r=16/32/64 -> C_red = 5.5455 / 5.5688 / 5.6408;
  rel. error 4.82% / 4.42% / 3.18%; classification **B**).
- Backup copy of the certified result: `_bridge_core_preserved/`.
- This result is **unchanged** and is the preserved provenance referred to in the v15 task.

## 2. baseline_reduction (the v15 tau_0-excluded honest reduction)

- Script: `code_python/baseline_reduction.py` (+ `fix_kkt_diagnostics.py`, `export_for_matlab.py`)
- Result: `BASELINE_ADAPTIVE_BASIS_RESULTS.csv`, `baseline_reduction_summary.json`,
  `BASELINE_BASIS_SPECTRUM.csv`, `bases/snapshots.npz` (160 snapshots, tau_0 **excluded**).
- Independent MATLAB check: `MATLAB_INDEPENDENT_CHECK.md`, `code_matlab/`.
- POD r=16/32/64/128 -> C_red = 5.532 / 5.541 / 5.553 / 5.577 (5.04% -> 4.28%);
  greedy r=64 -> 5.617 (3.58%); held-out tau_0 recon error 65%.

## Collision that was corrected

The v15 `baseline_reduction.py` originally wrote `BASIS_SPECTRUM.csv` and thereby
**overwrote bridge_core's spectrum intermediate** (71-snapshot) with the 160-snapshot
baseline spectrum. This was corrected:

- The baseline spectrum was re-attributed to `BASELINE_BASIS_SPECTRUM.csv`, and
  `baseline_reduction.py` now writes that name.
- bridge_core's **certified result files were never touched** and are additionally backed
  up in `_bridge_core_preserved/`.
- bridge_core's `BASIS_SPECTRUM.csv` is a **regenerable intermediate**: re-running
  `bridge_core.py` (deterministic; seeds `range(70)` + `tau_star`) reproduces it and the
  certified `BRIDGE_RESULTS.csv`. It was not regenerated here to avoid a long simplex
  re-solve; the certified B-level numbers do not depend on it.
