#!/usr/bin/env python3
"""Self-contained reproduction of the matter-realisation gate (dependency-free: numpy only).

Reproduces the DECISIVE comparison the paper's conclusion rests on: the required source mass
at the retained rung vs the maximum masses of known compact-object matter. The operative leg
is the causal (Rhoades-Ruffini) upper bound; all decisive numbers are analytic/literature, so
no external TOV solver is needed. (The full T224 campaign -- compute_T224_audit.py + the frozen
*.csv here -- additionally attempts a gamma=2 polytrope TOV via the optional CompactObject-TOV
package; that leg returned NaN in the frozen run and is not decisive.)
"""
import numpy as np

# SI constants
G = 6.67430e-11
c = 2.99792458e8
Msun = 1.98892e30

# Retained clean rung (paper Section 5): illustrative scale mapping I_S R^2 = 1e6 km^2, u = 0.20
I_seq_clean = 21.18
R_km = np.sqrt(1e6 / I_seq_clean)        # -> ~217.3 km
R = R_km * 1e3
u = 0.20
M_required = u * c**2 * R / G            # M = u c^2 R / G
M_required_solar = M_required / Msun

# Known compact-object mass ceilings (literature values cited in the paper)
ceilings = {
    "Chandrasekhar white-dwarf limit [Chandrasekhar 1931]": 1.44,
    "Observed high-mass pulsar J0740+6620 [Fonseca 2021]":   2.08,
    "Rhoades-Ruffini causal upper bound [Rhoades & Ruffini 1974]": 3.2,
}

print("=" * 70)
print("Matter-realisation gate (dependency-free reproduction)")
print("=" * 70)
print(f"retained rung: I_seq,clean = {I_seq_clean}  ->  R = {R_km:.1f} km, u = {u}")
print(f"required source mass  M = u c^2 R / G = {M_required_solar:.2f} M_sun   (paper: 29.4)")
print("\ncompact-object matter ceilings:")
worst = 0.0
for name, m in ceilings.items():
    print(f"   {m:5.2f} M_sun   {name}")
    worst = max(worst, m)
factor = M_required_solar / worst
print(f"\nhighest known ceiling = {worst} M_sun (causal bound)")
print(f"required / highest-ceiling = {factor:.1f}x  -> no compact-object EOS supports the rung")
print(f"peak-density gap vs engineering materials: ~1e15 / ~1e4 = 1e11 to 1e12 (11-12 orders)")

gate_fails = M_required_solar > worst and abs(M_required_solar - 29.4) < 0.2
print("\n" + "=" * 70)
print(f"MATTER GATE: {'FAILS as reported (29.4 M_sun >> 3.2 M_sun causal ceiling)' if gate_fails else 'CHECK'}")
print("=" * 70)
