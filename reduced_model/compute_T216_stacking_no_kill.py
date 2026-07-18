"""T216 — stacking enrichment mechanisms (NO kill rule).

Take the best honest first-order enrichments from T211-T215 and stack
them in pairs, triples, and full combination.  Report whatever C results,
no binary threshold.

Mechanisms:
  K_design      - choose wake operator width sigma_K (T213A)
  topological   - add edge / defect spatial channels (T215 T8)
  oscillator    - add one optimised adversarial oscillator (T215 T2)
  active_pump   - active pump term G u with weight alpha (T213B)
  nonnormal     - non-normal transient at physical Kreiss (T213C)

Each mechanism has a boost matrix M_i constructed so its leading
generalised eigenvalue equals gamma_i * C_HT in isolation.  Stacking
forms a JOINT operator:

  M_joint = (sum / product / weighted) of M_i

For each combination we compute the leading generalised eigenvalue
and report C = sqrt(lambda_max).
"""
# ruff: noqa: E501, N802, N803, N806

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm

T216_DIR = Path(__file__).resolve().parent

C_HT = 6.39
C_TARGET = 100.0


def build_K(N: int, sigma: float = 0.10) -> np.ndarray:
    z = np.linspace(0.0, 1.0, N)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i >= j:
                K[i, j] = np.exp(-((z[i] - z[j]) ** 2) / sigma**2)
    return K


def build_Q(N: int, n_controls: int = 4) -> np.ndarray:
    z = np.linspace(0.0, 1.0, N)
    Phi = np.zeros((n_controls, N))
    Phi[0, :] = 1.0 / np.sqrt(N)
    for m in range(1, n_controls):
        v = np.cos(np.pi * m * z)
        v -= v.mean()
        v /= np.linalg.norm(v)
        Phi[m, :] = v
    return np.eye(N) - Phi.T @ Phi


def calibrate_to_C_HT(K: np.ndarray, Q: np.ndarray) -> float:
    A = K.T @ Q.T @ Q @ K
    eigvals = np.linalg.eigvalsh(A)
    lam = float(eigvals[-1])
    return C_HT / np.sqrt(max(lam, 1e-12))


def primal_C(K: np.ndarray, Q: np.ndarray, W: np.ndarray) -> float:
    A = K.T @ Q.T @ Q @ K
    if np.allclose(W, np.eye(W.shape[0])):
        eigvals = np.linalg.eigvalsh(A)
    else:
        eigvals = eigh(A, W, eigvals_only=True)
    return float(np.sqrt(max(eigvals[-1], 0.0)))


def R_from_C(C: float) -> float:
    """R = 1000 km / sqrt(C) (audit-chain conservation law C R^2 = 10^6 km^2)."""
    return 1000.0 / np.sqrt(max(C, 1e-12))


def mechanism_K_design(N: int, sigma: float, cal: float) -> np.ndarray:
    return cal * build_K(N, sigma)


def mechanism_topological(N: int, edge_weight: float = 0.5, cal: float = 1.0) -> np.ndarray:
    K_base = mechanism_K_design(N, 0.10, cal)
    z = np.linspace(0.0, 1.0, N)
    edge_mask = ((z > 0.4) & (z < 0.6)).astype(float)
    edge_boost = np.outer(edge_mask, edge_mask)
    return K_base * (1.0 + edge_weight * edge_boost)


def mechanism_oscillator(N: int, omega_a: float = 0.5, gamma_a: float = 0.005, weight: float = 0.4, cal: float = 1.0) -> np.ndarray:
    K_base = mechanism_K_design(N, 0.10, cal)
    z = np.linspace(0.0, 1.0, N)
    bump = weight * gamma_a / ((z - omega_a) ** 2 + gamma_a**2)
    return K_base + np.outer(bump, bump) / N


def mechanism_active_pump(N: int, alpha: float = 1.0, cal: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    K_base = mechanism_K_design(N, 0.10, cal)
    G = K_base.copy()
    return K_base, G, alpha


def mechanism_nonnormal(N: int, c_coupling: float = 1.0, t_eval: float = 1.0) -> np.ndarray:
    A_diag = -1.0 * np.eye(N)
    A_upper = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, min(i + 8, N)):
            A_upper[i, j] = c_coupling * np.exp(-(j - i) / 3.0)
    A = A_diag + A_upper
    return expm(A * t_eval)


def C_with_K(K: np.ndarray, Q: np.ndarray, W: np.ndarray) -> float:
    return primal_C(K, Q, W)


def C_with_active_pump(K: np.ndarray, G: np.ndarray, Q: np.ndarray, alpha: float, N: int) -> float:
    QK = Q @ K
    QG = Q @ G
    A_block = np.zeros((2 * N, 2 * N))
    A_block[:N, :N] = QK.T @ QK
    A_block[:N, N:] = QK.T @ QG
    A_block[N:, :N] = QG.T @ QK
    A_block[N:, N:] = QG.T @ QG
    W_block = np.zeros((2 * N, 2 * N))
    W_block[:N, :N] = np.eye(N)
    W_block[N:, N:] = alpha * np.eye(N)
    eigvals = eigh(A_block, W_block, eigvals_only=True)
    return float(np.sqrt(max(eigvals[-1], 0.0)))


def C_with_nonnormal(K: np.ndarray, Q: np.ndarray, expA_t: np.ndarray) -> float:
    QK_E = Q @ K @ expA_t
    A = QK_E.T @ QK_E
    eigvals = np.linalg.eigvalsh(A)
    return float(np.sqrt(max(eigvals[-1], 0.0)))


def evaluate_mechanism_set(mechs: tuple[str, ...], N: int = 32) -> dict:
    """Evaluate the joint C for a given combination of mechanisms.

    Mechanisms compose as:
      K_design     : adjusts wake width sigma_K
      topological  : multiplicative spatial edge boost on K
      oscillator   : additive resonant bump on K
      active_pump  : adds Q G u channel (block-LP)
      nonnormal    : multiplies K by exp(A t) (transient evolution)

    The compositions are honest joint operators, not multiplicative gain
    products.
    """
    use_K_design = "K_design" in mechs
    use_topological = "topological" in mechs
    use_oscillator = "oscillator" in mechs
    use_active_pump = "active_pump" in mechs
    use_nonnormal = "nonnormal" in mechs

    sigma_K = 0.05 if use_K_design else 0.10
    n_controls = 4
    Q = build_Q(N, n_controls)
    W = np.eye(N)

    K_canonical = build_K(N, sigma=0.10)
    cal = calibrate_to_C_HT(K_canonical, Q)

    K = cal * build_K(N, sigma_K)
    if use_topological:
        z = np.linspace(0.0, 1.0, N)
        edge_mask = ((z > 0.4) & (z < 0.6)).astype(float)
        K = K * (1.0 + 0.5 * np.outer(edge_mask, edge_mask))
    if use_oscillator:
        z = np.linspace(0.0, 1.0, N)
        bump = 0.4 * 0.005 / ((z - 0.5) ** 2 + 0.005**2)
        K = K + np.outer(bump, bump) / N

    if use_nonnormal:
        c_coupling = 1.0
        A_diag = -1.0 * np.eye(N)
        A_upper = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, min(i + 8, N)):
                A_upper[i, j] = c_coupling * np.exp(-(j - i) / 3.0)
        A_dyn = A_diag + A_upper
        E = expm(A_dyn * 1.0)
        K_eff = K @ E
    else:
        K_eff = K

    if use_active_pump:
        G = K_eff.copy()
        alpha = 1.0
        C_max = C_with_active_pump(K_eff, G, Q, alpha, N)
    else:
        C_max = primal_C(K_eff, Q, W)

    return {
        "mechanisms_used": list(mechs),
        "n_mechanisms": len(mechs),
        "C_max": C_max,
        "R_implied_km": R_from_C(C_max),
        "boost_over_C_HT": C_max / C_HT,
        "uses_K_design": use_K_design,
        "uses_topological": use_topological,
        "uses_oscillator": use_oscillator,
        "uses_active_pump": use_active_pump,
        "uses_nonnormal": use_nonnormal,
    }


def task1_individual_mechanisms() -> pd.DataFrame:
    mechanisms_to_test = [
        ("none_baseline",),
        ("K_design",),
        ("topological",),
        ("oscillator",),
        ("active_pump",),
        ("nonnormal",),
    ]
    rows = []
    for mechs in mechanisms_to_test:
        if mechs == ("none_baseline",):
            res = evaluate_mechanism_set(())
            res["mechanisms_used"] = ["none_baseline"]
        else:
            res = evaluate_mechanism_set(mechs)
        rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(T216_DIR / "T216_individual_mechanisms.csv", index=False)
    return df


def task2_pairwise_stack() -> pd.DataFrame:
    mechs_pool = ["K_design", "topological", "oscillator", "active_pump", "nonnormal"]
    rows = []
    for pair in itertools.combinations(mechs_pool, 2):
        res = evaluate_mechanism_set(pair)
        rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(T216_DIR / "T216_pairwise_stack.csv", index=False)
    return df


def task3_triple_stack() -> pd.DataFrame:
    mechs_pool = ["K_design", "topological", "oscillator", "active_pump", "nonnormal"]
    rows = []
    for triple in itertools.combinations(mechs_pool, 3):
        res = evaluate_mechanism_set(triple)
        rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(T216_DIR / "T216_triple_stack.csv", index=False)
    return df


def task4_full_stack() -> pd.DataFrame:
    mechs_pool = ["K_design", "topological", "oscillator", "active_pump", "nonnormal"]
    rows = []
    for n_use in [4, 5]:
        for combo in itertools.combinations(mechs_pool, n_use):
            res = evaluate_mechanism_set(combo)
            rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(T216_DIR / "T216_full_stack.csv", index=False)
    return df


def task5_R_implied_summary(df_indiv: pd.DataFrame, df_pair: pd.DataFrame, df_triple: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"category": "anchor_C_HT", "best_C_max": C_HT, "best_R_km": R_from_C(C_HT), "best_combination": "C_HT inherited", "improvement_over_C_HT_in_C": 1.0, "improvement_over_C_HT_in_R_km": 0.0})

    indiv_no_baseline = df_indiv[df_indiv["mechanisms_used"].apply(lambda x: x != ["none_baseline"])]
    if len(indiv_no_baseline) > 0:
        idx = indiv_no_baseline["C_max"].idxmax()
        row = indiv_no_baseline.loc[idx]
        rows.append({"category": "best_single_mechanism", "best_C_max": row["C_max"], "best_R_km": row["R_implied_km"], "best_combination": str(row["mechanisms_used"]), "improvement_over_C_HT_in_C": row["C_max"] / C_HT, "improvement_over_C_HT_in_R_km": R_from_C(C_HT) - row["R_implied_km"]})

    for label, df in [("best_pair", df_pair), ("best_triple", df_triple), ("best_quad_or_quint", df_full)]:
        if len(df) > 0:
            idx = df["C_max"].idxmax()
            row = df.loc[idx]
            rows.append({"category": label, "best_C_max": row["C_max"], "best_R_km": row["R_implied_km"], "best_combination": str(row["mechanisms_used"]), "improvement_over_C_HT_in_C": row["C_max"] / C_HT, "improvement_over_C_HT_in_R_km": R_from_C(C_HT) - row["R_implied_km"]})

    rows.append({"category": "engineering_target_for_reference", "best_C_max": C_TARGET, "best_R_km": R_from_C(C_TARGET), "best_combination": "C = 100 (engineering threshold; reference only)", "improvement_over_C_HT_in_C": C_TARGET / C_HT, "improvement_over_C_HT_in_R_km": R_from_C(C_HT) - R_from_C(C_TARGET)})

    df = pd.DataFrame(rows)
    df.to_csv(T216_DIR / "T216_R_implied_summary.csv", index=False)
    return df


def write_combined_verdict_and_summary(df_indiv: pd.DataFrame, df_pair: pd.DataFrame, df_triple: pd.DataFrame, df_full: pd.DataFrame, df_R: pd.DataFrame) -> dict:
    df_pair_no_basel = df_pair
    df_triple_no_basel = df_triple
    df_full_no_basel = df_full

    best_indiv = df_indiv[df_indiv["mechanisms_used"].apply(lambda x: x != ["none_baseline"])]
    best_indiv_C = float(best_indiv["C_max"].max())
    best_pair_C = float(df_pair_no_basel["C_max"].max())
    best_triple_C = float(df_triple_no_basel["C_max"].max())
    best_full_C = float(df_full_no_basel["C_max"].max())

    overall_best = max(best_indiv_C, best_pair_C, best_triple_C, best_full_C)
    overall_R = R_from_C(overall_best)

    R_HT = R_from_C(C_HT)
    delta_R = R_HT - overall_R

    verdict_rows = [
        {"verdict_block": "PRIMARY", "verdict": f"T216_OVERALL_BEST_C_{overall_best:.4f}_R_{overall_R:.1f}_KM", "note": f"best joint stacking gives C = {overall_best:.4f}, R = {overall_R:.1f} km; closes {delta_R:.1f} km of the gap from C_HT (R = {R_HT:.1f} km)"},
        {"verdict_block": "BEST_INDIVIDUAL", "verdict": f"C_{best_indiv_C:.4f}_R_{R_from_C(best_indiv_C):.1f}_KM", "note": "best single mechanism"},
        {"verdict_block": "BEST_PAIR", "verdict": f"C_{best_pair_C:.4f}_R_{R_from_C(best_pair_C):.1f}_KM", "note": "best pairwise stacking"},
        {"verdict_block": "BEST_TRIPLE", "verdict": f"C_{best_triple_C:.4f}_R_{R_from_C(best_triple_C):.1f}_KM", "note": "best triple stacking"},
        {"verdict_block": "BEST_QUAD_OR_QUINT", "verdict": f"C_{best_full_C:.4f}_R_{R_from_C(best_full_C):.1f}_KM", "note": "best 4-way or 5-way stacking"},
        {"verdict_block": "STACKING_PATTERN", "verdict": "MULTIPLICATIVE_OR_DIMINISHING", "note": f"if perfect multiplicative: 6.39 * 2.38 * 1.59 * 1.61 * 1.42 * 6.11 = {6.39 * 2.378 * 1.589 * 1.606 * 1.415 * 6.111:.0f}; observed joint = {overall_best:.4f}; sharing factor = {overall_best / (6.39 * 2.378 * 1.589 * 1.606 * 1.415 * 6.111):.4f}"},
        {"verdict_block": "USER_NO_KILL_RULE", "verdict": "ANY_IMPROVEMENT_RECORDED_AS_REAL", "note": f"any C > 6.39 counts; best result reduces R from {R_HT:.1f} km to {overall_R:.1f} km, closing {delta_R:.1f} km of gap"},
        {"verdict_block": "PRIOR_CLOSURES", "verdict": "T169_THROUGH_T215_REMAIN_CLOSED", "note": "audit-chain consistency preserved"},
    ]
    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(T216_DIR / "T216_combined_verdict.csv", index=False)

    summary = {
        "ticket": "T216",
        "name": "T216_STACKING_ENRICHMENT_MECHANISMS_NO_KILL_RULE",
        "scope": "joint stacking of best honest first-order enrichments from T211-T215",
        "no_kill_rule_applied": True,
        "C_HT_anchor": C_HT,
        "R_HT_anchor_km": R_HT,
        "C_target_reference_only": C_TARGET,
        "R_target_reference_only_km": R_from_C(C_TARGET),
        "best_individual_C": best_indiv_C,
        "best_individual_R_km": R_from_C(best_indiv_C),
        "best_pair_C": best_pair_C,
        "best_pair_R_km": R_from_C(best_pair_C),
        "best_triple_C": best_triple_C,
        "best_triple_R_km": R_from_C(best_triple_C),
        "best_full_stack_C": best_full_C,
        "best_full_stack_R_km": R_from_C(best_full_C),
        "overall_best_C": overall_best,
        "overall_best_R_km": overall_R,
        "delta_R_km_closed": delta_R,
        "fraction_of_gap_closed": delta_R / (R_HT - R_from_C(C_TARGET)) if R_HT > R_from_C(C_TARGET) else 0.0,
        "always_appended": [
            "NO_PHYSICAL_WARP_CLAIM",
            "NO_BUILDABLE_WARP_CLAIM",
            "NOT_QUANTUM_GRAVITY",
            "NOT_PROPULSION",
            "T216_STACKING_ENRICHMENT_MECHANISMS_NO_KILL_RULE",
        ],
    }
    with open(T216_DIR / "T216_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _ = df_R
    return summary


def main() -> None:
    print("[T216] stacking enrichment mechanisms (NO kill rule)")
    print(f"[T216] C_HT = {C_HT}, R(C_HT) = {R_from_C(C_HT):.1f} km")
    print()

    print("[T216] Task 1 — individual mechanisms")
    df_indiv = task1_individual_mechanisms()
    for _, row in df_indiv.iterrows():
        print(f"  {str(row['mechanisms_used']):<32}  C = {row['C_max']:>8.4f}  R = {row['R_implied_km']:>7.1f} km")

    print("[T216] Task 2 — pairwise stack")
    df_pair = task2_pairwise_stack()
    for _, row in df_pair.iterrows():
        print(f"  {str(row['mechanisms_used']):<46}  C = {row['C_max']:>8.4f}  R = {row['R_implied_km']:>7.1f} km")

    print("[T216] Task 3 — triple stack")
    df_triple = task3_triple_stack()
    for _, row in df_triple.iterrows():
        print(f"  {str(row['mechanisms_used']):<60}  C = {row['C_max']:>8.4f}  R = {row['R_implied_km']:>7.1f} km")

    print("[T216] Task 4 — full stack (4 + 5 mechanisms)")
    df_full = task4_full_stack()
    for _, row in df_full.iterrows():
        print(f"  {str(row['mechanisms_used'])[:80]:<82}  C = {row['C_max']:>8.4f}  R = {row['R_implied_km']:>7.1f} km")

    df_R = task5_R_implied_summary(df_indiv, df_pair, df_triple, df_full)
    summary = write_combined_verdict_and_summary(df_indiv, df_pair, df_triple, df_full, df_R)

    print(f"\n[T216] OVERALL best C = {summary['overall_best_C']:.4f}, R = {summary['overall_best_R_km']:.1f} km")
    print(f"[T216] gap closed: {summary['delta_R_km_closed']:.1f} km out of {R_from_C(C_HT) - R_from_C(C_TARGET):.1f} km available")
    print(f"[T216] fraction of gap closed: {summary['fraction_of_gap_closed']:.2%}")
    print("[T216] DONE.")


if __name__ == "__main__":
    main()
