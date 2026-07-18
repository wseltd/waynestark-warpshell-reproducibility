#!/usr/bin/env python3
"""Re-run the reduced-operator ladder from source and check the reported indices.

Runs the frozen T216 (joint) and T217 (sequential) compositions and confirms the
provenance values I_joint,prov = 13.41 and I_seq,prov = 26.12 (via A_HT = 6.39). The
clean carried-forward values are these with the topological factor removed: 10.44 and
21.18. Self-contained (numpy/scipy/pandas). Takes a few minutes (real spectral sweeps).
"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = [("compute_T216_stacking_no_kill.py", 13.41, "I_joint,prov"),
          ("compute_T217_engineered_orthogonality.py", 26.12, "I_seq,prov")]

def main():
    print("=" * 70)
    print("Reduced-operator ladder reproduction (T216 joint / T217 sequential)")
    print("=" * 70)
    ok = True
    for script, expected, label in CHECKS:
        print(f"\n[run] {script}  (expect {label} = {expected}) ...")
        r = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True)
        nums = [float(x) for x in re.findall(r"\b(\d{2}\.\d{2,})\b", r.stdout)]
        hit = any(abs(n - expected) < 0.01 for n in nums)
        got = next((n for n in nums if abs(n - expected) < 0.01), None)
        print(f"      exit={r.returncode}  reproduced {label} = {got}  [{'PASS' if (r.returncode==0 and hit) else 'FAIL'}]")
        if not (r.returncode == 0 and hit):
            ok = False
    print("\n" + "=" * 70)
    print(f"REDUCED LADDER: {'PASS -- 13.41 (joint) and 26.12 (sequential) reproduced;' if ok else 'FAIL'}")
    print("  clean carried-forward values = these minus the KILL-tainted topological factor: 10.44, 21.18")
    print("=" * 70)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
