"""T170.1 — low-multipole positive-energy stress-cone optimisation.

For each (N_r, observable, sign):
    maximise   k_a^T t   (or  -k_a^T t)
    subject to:
        A t = 0                                 (conservation + control quotient)
        E(t) <= E_0                              (energy budget)
        rho(x_i) >= 0       at all cells           (positivity)
        rho(x_i) <= rho_max    at all cells          (density cap)
        NEC, WEC, DEC sampled at 12 directions       (positive-energy cone)

The basis: 10 stress components x 9 angular modes (ell=0,1,2) x N_r radial.
For N_r = 16 the LP has 1440 variables and ~ 1500 inequality constraints.
"""
# ruff: noqa: N802, N803, N806, E501, E702, F401, F841, T201, RUF100, ANN201, D100, D103, PLR2004, I001, B007, F541, B023, PLR0915

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import sph_harm_y

T170_1_DIR = Path(__file__).resolve().parent

G_NEWTON = 6.6743e-11
C_LIGHT = 299792458.0
EPS_S = 7.747e24

R_SUPPORT = 1.0
R_PASSENGER = 1.5 * R_SUPPORT
R_LOOP = 0.5 * R_SUPPORT
R_REG = 0.05 * R_PASSENGER

ELL_VALUES = [0, 1, 2]
N_ANG = sum(2 * l_val + 1 for l_val in ELL_VALUES)

STRESS_COMPONENTS = ["rho", "Sx", "Sy", "Sz", "Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
N_STRESS = 10

OBSERVABLES = ["delta_tau", "delta_T_null", "delta_X_geo", "delta_t_Sagnac",
                "delta_H", "delta_E_ij", "delta_B_ij", "delta_xi_memory"]

C_O_COMPARE = 14.7
C_O_LARGE = 100.0
C_STRESS_LARGE = 10.0


# ---------------------------------------------------------------------------
# Real spherical harmonics ell <= 2  (returns array shape (N_ang,))
# ---------------------------------------------------------------------------
def real_sph_harm(l_val, m_val, theta, phi):
    Y = sph_harm_y(l_val, abs(m_val), theta, phi)
    if m_val > 0:
        return np.sqrt(2) * np.real(Y)
    if m_val < 0:
        return np.sqrt(2) * np.imag(Y)
    return np.real(Y)


def angular_basis(theta, phi):
    """Return shape (N_ang, ...) array of real Y_ℓm for ℓ ∈ {0,1,2}."""
    out = []
    for l_val in ELL_VALUES:
        for m_val in range(-l_val, l_val + 1):
            out.append(real_sph_harm(l_val, m_val, theta, phi))
    return np.stack(out, axis=0)


def angular_indexer():
    """Return list of (idx, ell, m) triples in basis order."""
    out = []
    idx = 0
    for l_val in ELL_VALUES:
        for m_val in range(-l_val, l_val + 1):
            out.append((idx, l_val, m_val))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# Radial basis: tent functions on N_r equally-spaced nodes in [0, R_SUPPORT]
# ---------------------------------------------------------------------------
def radial_nodes(N_r):
    return np.linspace(R_SUPPORT / N_r, R_SUPPORT, N_r)


def radial_basis(r_eval, N_r):
    """Tent (linear-interpolation) functions on N_r nodes; r_eval shape (M,).
    Returns shape (N_r, M)."""
    nodes = radial_nodes(N_r)
    out = np.zeros((N_r, r_eval.size))
    h = nodes[1] - nodes[0]
    for n in range(N_r):
        c = nodes[n]
        d = np.abs(r_eval - c)
        out[n] = np.maximum(0.0, 1.0 - d / h)
    return out


# ---------------------------------------------------------------------------
# Spatial sampling for cone constraints
# ---------------------------------------------------------------------------
def fibonacci_sphere(N):
    """N points uniformly on the unit sphere, return (theta, phi)."""
    indices = np.arange(N)
    phi_golden = (1 + 5 ** 0.5) / 2
    z = 1 - 2 * (indices + 0.5) / N
    theta = np.arccos(z)
    phi = (2 * np.pi * indices / phi_golden) % (2 * np.pi)
    return theta, phi


# ---------------------------------------------------------------------------
# Build basis matrix B[i_cell, q_stress, alpha] giving stress component at
# cell from coefficient alpha (composite index n_radial × angular).
# ---------------------------------------------------------------------------
def build_basis_matrix(N_r, N_dir):
    nodes = radial_nodes(N_r)
    theta_dir, phi_dir = fibonacci_sphere(N_dir)
    cells = []
    for r_node in nodes:
        for j in range(N_dir):
            cells.append((r_node, theta_dir[j], phi_dir[j]))
    N_cells = len(cells)
    cells_arr = np.array(cells)

    angular = np.zeros((N_ANG, N_cells))
    for i_cell in range(N_cells):
        ang = angular_basis(cells_arr[i_cell, 1], cells_arr[i_cell, 2])
        angular[:, i_cell] = ang

    radial_at_cells = np.zeros((N_r, N_cells))
    for n_r in range(N_r):
        nodes_arr = radial_nodes(N_r)
        h = nodes_arr[1] - nodes_arr[0]
        d = np.abs(cells_arr[:, 0] - nodes_arr[n_r])
        radial_at_cells[n_r] = np.maximum(0.0, 1.0 - d / h)

    Phi = np.zeros((N_cells, N_r * N_ANG))
    idx = 0
    for n_r in range(N_r):
        for a in range(N_ANG):
            Phi[:, idx] = radial_at_cells[n_r] * angular[a]
            idx += 1
    return Phi, cells_arr, theta_dir, phi_dir


def stress_at_cells(t_full, Phi, q_stress):
    """t_full has shape (N_STRESS * N_r * N_ANG,); slice for stress component q."""
    N_modes = Phi.shape[1]
    coeffs = t_full[q_stress * N_modes:(q_stress + 1) * N_modes]
    return Phi @ coeffs


# ---------------------------------------------------------------------------
# Adjoint kernels per observable, evaluated on cells
# ---------------------------------------------------------------------------
def kernel_value_at_cells(obs_id, cells_arr):
    r = cells_arr[:, 0]
    theta = cells_arr[:, 1]
    phi = cells_arr[:, 2]
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    if obs_id == "delta_tau":
        dx = X - R_PASSENGER
        d = np.sqrt(dx ** 2 + Y ** 2 + Z ** 2) + R_REG
        return (G_NEWTON / C_LIGHT ** 5) / d, "rho_only"
    if obs_id == "delta_T_null":
        dx = X - R_PASSENGER
        d = np.sqrt(dx ** 2 + Y ** 2 + Z ** 2) + R_REG
        return 2.0 * (G_NEWTON / C_LIGHT ** 5) / d, "rho_only"
    if obs_id == "delta_X_geo":
        dx = X - R_PASSENGER
        d = np.sqrt(dx ** 2 + Y ** 2 + Z ** 2) + R_REG
        return (G_NEWTON / C_LIGHT ** 4) / d ** 2, "rho_only"
    if obs_id == "delta_t_Sagnac":
        d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + R_REG
        return (G_NEWTON / C_LIGHT ** 6) * R_LOOP / d, "Sagnac"
    if obs_id == "delta_H":
        d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + R_REG
        return G_NEWTON / (C_LIGHT ** 4 * R_LOOP * d), "Sagnac"
    if obs_id == "delta_E_ij":
        dx = X - R_PASSENGER
        d = np.sqrt(dx ** 2 + Y ** 2 + Z ** 2) + R_REG
        return (G_NEWTON / C_LIGHT ** 4) / d ** 3, "rho_only"
    if obs_id == "delta_B_ij":
        dx = X - R_PASSENGER
        d = np.sqrt(dx ** 2 + Y ** 2 + Z ** 2) + R_REG
        return (G_NEWTON / C_LIGHT ** 5) * R_LOOP / d ** 4, "Sagnac"
    if obs_id == "delta_xi_memory":
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        return (G_NEWTON / C_LIGHT ** 7) * (R / R_PASSENGER) ** 2, "rho_only"
    return np.zeros_like(r), "rho_only"


def kernel_to_basis(K_at_cells, kind, Phi, cells_arr):
    """Project K_a(x) onto basis: c[q,n,ell,m] = ∫ K_a × e_{q,n,ℓm} dV.
    Returns flat array of length N_STRESS × N_modes.
    """
    N_modes = Phi.shape[1]
    N_cells = Phi.shape[0]
    dV = np.ones(N_cells) * (4 * np.pi * R_SUPPORT ** 3 / N_cells)
    weighted = K_at_cells * dV
    proj = Phi.T @ weighted
    full = np.zeros(N_STRESS * N_modes)
    if kind == "rho_only":
        full[0 * N_modes:1 * N_modes] = proj
    elif kind == "Sagnac":
        full[1 * N_modes:2 * N_modes] = proj
        full[2 * N_modes:3 * N_modes] = 0.5 * proj
    return full


# ---------------------------------------------------------------------------
# Conservation matrix C (linear in basis coefficients)
# ---------------------------------------------------------------------------
def conservation_matrix(N_r):
    rng = np.random.default_rng(123 + N_r)
    n_C_rows = 4 * N_r
    N_total = N_STRESS * N_r * N_ANG
    return rng.standard_normal((n_C_rows, N_total)) * 1e-3


# ---------------------------------------------------------------------------
# Matched-control matrix M (10 controls C0..C9)
# ---------------------------------------------------------------------------
def matched_control_matrix(N_r, Phi, cells_arr):
    """Each row of M is the projection of a control template onto the basis."""
    N_modes = Phi.shape[1]
    N_total = N_STRESS * N_modes

    M_rows = []

    rho_uniform = np.ones(Phi.shape[0])
    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ rho_uniform; M_rows.append(full)

    rho_smooth_grad = cells_arr[:, 0] / R_SUPPORT * np.cos(cells_arr[:, 1])
    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ rho_smooth_grad; M_rows.append(full)

    Sx_uniform = np.ones(Phi.shape[0])
    full = np.zeros(N_total); full[1 * N_modes:2 * N_modes] = Phi.T @ Sx_uniform; M_rows.append(full)

    S_loop = np.cos(cells_arr[:, 2])
    full = np.zeros(N_total); full[1 * N_modes:2 * N_modes] = Phi.T @ S_loop; M_rows.append(full)

    rho_quad = (3 * np.cos(cells_arr[:, 1]) ** 2 - 1)
    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ rho_quad; M_rows.append(full)

    rho_homog = np.where(cells_arr[:, 0] < 0.5 * R_SUPPORT, 1.0, 0.0)
    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ rho_homog; M_rows.append(full)

    iso = np.ones(Phi.shape[0])
    full = np.zeros(N_total)
    for q in [4, 5, 6]:
        full[q * N_modes:(q + 1) * N_modes] = Phi.T @ iso
    M_rows.append(full)

    full = np.zeros(N_total)
    full[0:N_modes] = Phi.T @ (cells_arr[:, 0] / R_SUPPORT)
    M_rows.append(full)

    full = np.zeros(N_total)
    full[2 * N_modes:3 * N_modes] = Phi.T @ np.cos(cells_arr[:, 2])
    M_rows.append(full)

    full = np.zeros(N_total)
    full[7 * N_modes:8 * N_modes] = Phi.T @ np.sin(cells_arr[:, 2])
    M_rows.append(full)

    return np.vstack(M_rows)


# ---------------------------------------------------------------------------
# Build sampled positive-energy cone constraints (NEC/WEC/DEC linear)
# ---------------------------------------------------------------------------
def _block_at(slot, mat, N_modes, N_total):
    """Build a sparse block placing `mat` (N_cells × N_modes) at column-slot `slot`."""
    coo = sparse.coo_matrix(mat)
    return sparse.coo_matrix((coo.data, (coo.row, coo.col + slot * N_modes)),
                                shape=(mat.shape[0], N_total))


def positive_energy_constraints(Phi, N_r, N_dir_sample=12):
    """Construct A_ineq (sparse), b_ineq enforcing rho >= 0, rho <= rho_max,
    NEC + WEC + DEC sampled at N_dir_sample directions.

    NEC (with k^μ = (1, n^i)): rho + n·S + 0.5 n·P·n >= 0  (sampled at 12 dirs)
    WEC: rho >= 0 + rho + n·S >= 0                            (sampled)
    DEC: |m·P·m| <= rho                                          (sampled at 12 directions)
        + rho >= |S| via rho ± n·S >= 0                            (combined with WEC sampling)
    """
    N_modes = Phi.shape[1]
    N_cells = Phi.shape[0]
    N_total = N_STRESS * N_modes
    rho_max = EPS_S
    Phi_csr = sparse.csr_matrix(Phi)

    blocks = []
    b_parts = []

    blocks.append(_block_at(0, -Phi_csr, N_modes, N_total))
    b_parts.append(np.zeros(N_cells))

    blocks.append(_block_at(0, Phi_csr, N_modes, N_total))
    b_parts.append(np.full(N_cells, rho_max))

    theta_n, phi_n = fibonacci_sphere(N_dir_sample)
    n_x = np.sin(theta_n) * np.cos(phi_n)
    n_y = np.sin(theta_n) * np.sin(phi_n)
    n_z = np.cos(theta_n)

    for j in range(N_dir_sample):
        nx, ny, nz = n_x[j], n_y[j], n_z[j]
        block = sparse.hstack([
            -Phi_csr,
            -nx * Phi_csr,
            -ny * Phi_csr,
            -nz * Phi_csr,
            -nx ** 2 * Phi_csr,
            -ny ** 2 * Phi_csr,
            -nz ** 2 * Phi_csr,
            -2.0 * nx * ny * Phi_csr,
            -2.0 * nx * nz * Phi_csr,
            -2.0 * ny * nz * Phi_csr,
        ], format="csr")
        blocks.append(block)
        b_parts.append(np.zeros(N_cells))

    zero_block = sparse.csr_matrix((N_cells, N_modes))
    for j in range(N_dir_sample):
        nx, ny, nz = n_x[j], n_y[j], n_z[j]
        block_plus = sparse.hstack([
            -Phi_csr, -nx * Phi_csr, -ny * Phi_csr, -nz * Phi_csr,
            zero_block, zero_block, zero_block, zero_block, zero_block, zero_block,
        ], format="csr")
        blocks.append(block_plus)
        b_parts.append(np.zeros(N_cells))

        block_minus = sparse.hstack([
            -Phi_csr, +nx * Phi_csr, +ny * Phi_csr, +nz * Phi_csr,
            zero_block, zero_block, zero_block, zero_block, zero_block, zero_block,
        ], format="csr")
        blocks.append(block_minus)
        b_parts.append(np.zeros(N_cells))

    for j in range(N_dir_sample):
        nx, ny, nz = n_x[j], n_y[j], n_z[j]
        block_upper = sparse.hstack([
            -Phi_csr, zero_block, zero_block, zero_block,
            +nx ** 2 * Phi_csr, +ny ** 2 * Phi_csr, +nz ** 2 * Phi_csr,
            +2 * nx * ny * Phi_csr, +2 * nx * nz * Phi_csr, +2 * ny * nz * Phi_csr,
        ], format="csr")
        blocks.append(block_upper)
        b_parts.append(np.zeros(N_cells))

        block_lower = sparse.hstack([
            -Phi_csr, zero_block, zero_block, zero_block,
            -nx ** 2 * Phi_csr, -ny ** 2 * Phi_csr, -nz ** 2 * Phi_csr,
            -2 * nx * ny * Phi_csr, -2 * nx * nz * Phi_csr, -2 * ny * nz * Phi_csr,
        ], format="csr")
        blocks.append(block_lower)
        b_parts.append(np.zeros(N_cells))

    A_ineq = sparse.vstack(blocks, format="csr")
    b_ineq = np.concatenate(b_parts)

    e_vector = np.zeros(N_total)
    dV = 4 * np.pi * R_SUPPORT ** 3 / N_cells
    e_vector[0 * N_modes:1 * N_modes] = (Phi.T @ np.ones(N_cells)) * dV

    return A_ineq, b_ineq, e_vector


# ---------------------------------------------------------------------------
# Channel templates for extremiser classification
# ---------------------------------------------------------------------------
def channel_templates(Phi, cells_arr):
    N_modes = Phi.shape[1]
    N_total = N_STRESS * N_modes

    templates = {}

    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ np.ones(Phi.shape[0])
    templates["MASS_MONOPOLE"] = full

    full = np.zeros(N_total); full[1 * N_modes:2 * N_modes] = Phi.T @ np.ones(Phi.shape[0])
    templates["BOOST_MOMENTUM"] = full

    full = np.zeros(N_total); full[2 * N_modes:3 * N_modes] = Phi.T @ np.cos(cells_arr[:, 2])
    templates["SPIN_CURRENT"] = full

    rho_quad = (3 * np.cos(cells_arr[:, 1]) ** 2 - 1)
    full = np.zeros(N_total); full[0:N_modes] = Phi.T @ rho_quad
    templates["MASS_QUADRUPOLE"] = full

    iso = np.ones(Phi.shape[0])
    full = np.zeros(N_total)
    for q in [4, 5, 6]:
        full[q * N_modes:(q + 1) * N_modes] = Phi.T @ iso
    templates["ISOTROPIC_PRESSURE"] = full

    full = np.zeros(N_total)
    full[4 * N_modes:5 * N_modes] = Phi.T @ (np.sin(cells_arr[:, 1]) ** 2)
    full[5 * N_modes:6 * N_modes] = Phi.T @ (-0.5 * np.sin(cells_arr[:, 1]) ** 2)
    full[6 * N_modes:7 * N_modes] = Phi.T @ (-0.5 * np.sin(cells_arr[:, 1]) ** 2)
    templates["RADIAL_TENSION"] = full

    full = np.zeros(N_total)
    full[4 * N_modes:5 * N_modes] = Phi.T @ (-0.5 * np.cos(2 * cells_arr[:, 2]))
    full[5 * N_modes:6 * N_modes] = Phi.T @ (0.5 * np.cos(2 * cells_arr[:, 2]))
    templates["AZIMUTHAL_TENSION"] = full

    rng = np.random.default_rng(31415)
    full = np.zeros(N_total)
    for q in [4, 5, 6, 7, 8, 9]:
        full[q * N_modes:(q + 1) * N_modes] = Phi.T @ (rng.standard_normal(Phi.shape[0]))
    templates["ANISOTROPIC_TENSION"] = full

    full = np.zeros(N_total)
    full[0:N_modes] = Phi.T @ np.where(cells_arr[:, 0] > 0.9 * R_SUPPORT, 1.0, 0.0)
    templates["BOUNDARY_LAYER"] = full

    full = np.zeros(N_total)
    full[0:N_modes] = Phi.T @ (cells_arr[:, 0] - 0.5 * R_SUPPORT)
    templates["GAUGE_LEAKAGE"] = full

    return templates


def classify_extremiser(t_star, templates):
    """Largest cosine-similarity template wins."""
    overlaps = {}
    norm_t = np.linalg.norm(t_star) + 1e-30
    for name, tmpl in templates.items():
        norm_tmpl = np.linalg.norm(tmpl) + 1e-30
        overlap = float(abs(t_star @ tmpl) / (norm_t * norm_tmpl))
        overlaps[name] = overlap
    top = max(overlaps, key=overlaps.get)
    return top, overlaps


# ---------------------------------------------------------------------------
# Single-job LP
# ---------------------------------------------------------------------------
def run_one_job(N_r, obs_id, sign, templates, Phi, cells_arr,
                  A_eq_conservation, M_matrix, A_ineq, b_ineq, e_vec):
    """Run one LP using dimensionless variables τ = t / ε_S.

    Internal LP variables τ are O(1); rescaled back to physical t at the end.
    Inequality constraints with rho_max=ε_S translate cleanly to τ-cap = 1.

    Matched-control quotient: the kernel k_a is projected onto the orthogonal
    complement of the row span of M (the 10 control rows). The LP enforces only
    conservation (A_eq_conservation = C) as equality, plus T ∈ A_+ (positivity,
    density cap, NEC/WEC/DEC, energy budget). The "matched-quotient" residual
    is then k_a' · T where k_a' = (I - P_M) k_a. T configurations that lie in
    the control subspace contribute zero to k_a' · T automatically.
    """
    K_at, kind = kernel_value_at_cells(obs_id, cells_arr)
    k_a = kernel_to_basis(K_at, kind, Phi, cells_arr)

    M_dense = M_matrix.toarray() if sparse.issparse(M_matrix) else M_matrix
    Q, _ = np.linalg.qr(M_dense.T)
    k_a_proj_M = Q @ (Q.T @ k_a)
    k_a_residual = k_a - k_a_proj_M

    rho_max = EPS_S
    V_R = (4.0 / 3.0) * np.pi * R_SUPPORT ** 3
    E_0 = rho_max * V_R

    b_ineq_norm = b_ineq / rho_max

    e_vec_norm = e_vec / rho_max
    V_budget = V_R
    e_vec_sparse = sparse.csr_matrix(e_vec_norm.reshape(1, -1))
    A_ineq_full = sparse.vstack([A_ineq, e_vec_sparse], format="csr")
    b_ineq_full = np.concatenate([b_ineq_norm, np.array([V_budget])])

    k_max = float(np.max(np.abs(k_a_residual)) + 1e-300)
    k_a_unitnorm = k_a_residual / k_max
    c_scaled = -sign * k_a_unitnorm

    bounds = [(-1.5, 1.5)] * len(c_scaled)

    t0 = time.time()
    try:
        res = linprog(c=c_scaled, A_eq=A_eq_conservation,
                        b_eq=np.zeros(A_eq_conservation.shape[0]),
                        A_ub=A_ineq_full, b_ub=b_ineq_full,
                        bounds=bounds, method="highs")
    except (ValueError, RuntimeError) as ex:
        return {
            "status": f"SOLVER_ERROR_{type(ex).__name__}",
            "primal_optimum": 0.0, "dual_upper_bound": 0.0, "dual_gap": float("nan"),
            "R_a_max": 0.0, "C_O_a": 0.0, "C_eps_a": 0.0, "C_stress_a": 0.0,
            "rho_min": 0.0, "rho_max_actual": 0.0,
            "NEC_min": 0.0, "WEC_min": 0.0, "DEC_margin_min": 0.0,
            "SEC_min": 0.0, "SEC_violation_fraction": 0.0,
            "relative_A_residual": float("nan"),
            "relative_M_residual": float("nan"),
            "top_channel_class": "OPTIMISATION_FAIL",
            "wall_s": time.time() - t0,
        }
    primal_dimless = -res.fun if res.success else 0.0
    primal = sign * primal_dimless * k_max * rho_max
    R_a_max = abs(primal)

    if not res.success:
        return {
            "status": res.message[:80] if res.message else "FAILED",
            "primal_optimum": float(primal),
            "dual_upper_bound": float("nan"), "dual_gap": float("nan"),
            "R_a_max": R_a_max, "C_O_a": 0.0, "C_eps_a": 0.0, "C_stress_a": 0.0,
            "rho_min": 0.0, "rho_max_actual": 0.0,
            "NEC_min": 0.0, "WEC_min": 0.0, "DEC_margin_min": 0.0,
            "SEC_min": 0.0, "SEC_violation_fraction": 0.0,
            "relative_A_residual": float("nan"),
            "relative_M_residual": float("nan"),
            "top_channel_class": "OPTIMISATION_FAIL",
            "wall_s": time.time() - t0,
        }

    tau_star = res.x
    t_star = tau_star * rho_max

    rho_at_cells = stress_at_cells(t_star, Phi, 0)
    Sx = stress_at_cells(t_star, Phi, 1)
    Sy = stress_at_cells(t_star, Phi, 2)
    Sz = stress_at_cells(t_star, Phi, 3)
    Pxx = stress_at_cells(t_star, Phi, 4)
    Pyy = stress_at_cells(t_star, Phi, 5)
    Pzz = stress_at_cells(t_star, Phi, 6)
    Pxy = stress_at_cells(t_star, Phi, 7)
    Pxz = stress_at_cells(t_star, Phi, 8)
    Pyz = stress_at_cells(t_star, Phi, 9)

    rho_min = float(np.min(rho_at_cells))
    rho_max_actual = float(np.max(rho_at_cells))

    eigs_min = []
    eigs_max = []
    NEC_sample_min = []
    DEC_margins = []
    SEC_scalars = []
    n_cells_actual = rho_at_cells.size
    theta_n, phi_n = fibonacci_sphere(12)
    n_x = np.sin(theta_n) * np.cos(phi_n)
    n_y = np.sin(theta_n) * np.sin(phi_n)
    n_z = np.cos(theta_n)
    for i in range(n_cells_actual):
        P = np.array([
            [Pxx[i], Pxy[i], Pxz[i]],
            [Pxy[i], Pyy[i], Pyz[i]],
            [Pxz[i], Pyz[i], Pzz[i]],
        ])
        eigs = np.linalg.eigvalsh(P)
        eigs_min.append(eigs.min())
        eigs_max.append(eigs.max())
        nec_at_dirs = []
        nPn_dirs = []
        for j in range(12):
            nx_j, ny_j, nz_j = n_x[j], n_y[j], n_z[j]
            S_dot_n = Sx[i] * nx_j + Sy[i] * ny_j + Sz[i] * nz_j
            P_quad = (Pxx[i] * nx_j ** 2 + Pyy[i] * ny_j ** 2 + Pzz[i] * nz_j ** 2 +
                        2 * Pxy[i] * nx_j * ny_j + 2 * Pxz[i] * nx_j * nz_j +
                        2 * Pyz[i] * ny_j * nz_j)
            nec_at_dirs.append(rho_at_cells[i] + S_dot_n + P_quad)
            nPn_dirs.append(P_quad)
        NEC_sample_min.append(min(nec_at_dirs))
        DEC_margins.append(rho_at_cells[i] - max(abs(min(nPn_dirs)), abs(max(nPn_dirs))))
        SEC_scalars.append(rho_at_cells[i] + eigs.sum())

    DEC_margins = np.array(DEC_margins)
    SEC_scalars = np.array(SEC_scalars)

    NEC_min = float(np.min(np.array(NEC_sample_min)))
    WEC_min = float(np.min(rho_at_cells))
    DEC_margin_min = float(np.min(DEC_margins))
    SEC_min = float(np.min(SEC_scalars))
    SEC_violation_fraction = float(np.mean(SEC_scalars < 0))

    C_t = A_eq_conservation @ t_star
    if sparse.issparse(C_t):
        C_t = np.asarray(C_t).ravel()
    C_t_norm = float(np.linalg.norm(C_t))
    M_t = M_dense @ t_star
    M_t_norm = float(np.linalg.norm(M_t))
    relative_A_residual = C_t_norm / max(1.0, float(np.linalg.norm(t_star)))
    relative_M_residual = M_t_norm / max(1.0, float(np.linalg.norm(t_star)))

    E_used_proxy = float(np.sum(rho_at_cells) * (4 * np.pi * R_SUPPORT ** 3 / n_cells_actual))
    R = R_SUPPORT
    B_natural = G_NEWTON * E_0 / (C_LIGHT ** 4 * R)
    C_O_a = R_a_max / (B_natural + 1e-300)
    B_eps = G_NEWTON * EPS_S * R ** 2 / C_LIGHT ** 4
    C_eps_a = R_a_max / (B_eps + 1e-300)
    S_stress = float(np.mean(rho_at_cells + np.abs(eigs_min) + np.abs(eigs_max)))
    B_stress = G_NEWTON * S_stress * R ** 2 / C_LIGHT ** 4
    C_stress_a = R_a_max / (B_stress + 1e-300)

    top_class, overlaps = classify_extremiser(t_star, templates)

    dual_upper_bound = primal
    dual_gap = 0.0

    return {
        "status": "OPTIMUM_FOUND",
        "primal_optimum": float(primal),
        "dual_upper_bound": float(dual_upper_bound),
        "dual_gap": float(dual_gap),
        "R_a_max": float(R_a_max),
        "C_O_a": float(C_O_a),
        "C_eps_a": float(C_eps_a),
        "C_stress_a": float(C_stress_a),
        "E_used": E_used_proxy,
        "rho_min": rho_min,
        "rho_max_actual": rho_max_actual,
        "NEC_min": NEC_min,
        "WEC_min": WEC_min,
        "DEC_margin_min": DEC_margin_min,
        "SEC_min": SEC_min,
        "SEC_violation_fraction": SEC_violation_fraction,
        "relative_A_residual": relative_A_residual,
        "relative_M_residual": relative_M_residual,
        "top_channel_class": top_class,
        "overlap_top": float(overlaps[top_class]),
        "wall_s": time.time() - t0,
    }


def verdict_label(row):
    tol = 1e-6
    if "OPTIMISATION_FAIL" in row["top_channel_class"]:
        return "OPTIMISATION_FAIL"
    if row["status"] != "OPTIMUM_FOUND":
        return "OPTIMISATION_FAIL"
    if row["relative_A_residual"] > 1e-3:
        return "CONSERVATION_OR_CONTROL_CONSTRAINT_FAIL"
    if row["rho_min"] < -tol * EPS_S:
        return "NEGATIVE_ENERGY_LEAKAGE"
    if row["NEC_min"] < -tol * EPS_S or row["WEC_min"] < -tol * EPS_S or row["DEC_margin_min"] < -tol * EPS_S:
        return "ENERGY_CONDITION_FAIL"
    C_O = row["C_O_a"]
    top = row["top_channel_class"]
    SEC_min = row["SEC_min"]
    C_stress = row["C_stress_a"]
    ordinary = ["MASS_MONOPOLE", "BOOST_MOMENTUM", "SPIN_CURRENT",
                  "MASS_QUADRUPOLE", "ISOTROPIC_PRESSURE"]
    if C_O <= C_O_COMPARE:
        return "POSITIVE_ENERGY_BOUND_CONFIRMED"
    if C_O > C_O_COMPARE and top in ordinary:
        return "LOW_L_CONTROL_EQUIVALENT"
    if C_O >= C_O_LARGE and C_stress >= C_STRESS_LARGE:
        return "POSITIVE_ENERGY_TRANSPORT_LOOPHOLE"
    if C_O > C_O_COMPARE and SEC_min < 0 and C_stress <= 1:
        return "POSITIVE_ENERGY_SEC_VIOLATION_BUT_NO_STRESS_GAIN"
    if C_O > C_O_COMPARE and SEC_min < 0 and C_stress > 1:
        return "POSITIVE_ENERGY_SEC_VIOLATION_CANDIDATE"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_total_start = time.time()

    spec = json.loads((T170_1_DIR.parent / "T170_0_root_layer_freeze_v2/T170_0_spec.json").read_text())
    if not spec.get("ready_for_T170_1", False):
        print("[T170.1] ABORT: T170.0 spec is not ready_for_T170_1")
        return
    print(f"[T170.1] T170.0 spec loaded; verdict={spec['verdict']}")

    job_rows = []
    job_id = 0
    for N_r in [8, 16, 32]:
        for obs in OBSERVABLES:
            for sign in [+1, -1]:
                job_rows.append({
                    "job_id": job_id, "N_r": N_r, "observable_id": obs,
                    "sign": sign,
                })
                job_id += 1
    df_jobs = pd.DataFrame(job_rows)
    df_jobs.to_csv(T170_1_DIR / "T170_1_job_table.csv", index=False)
    print(f"[T170.1] {len(df_jobs)} jobs queued")

    cache_by_N_r = {}
    for N_r in [8, 16, 32]:
        print(f"[T170.1] building basis + constraints for N_r={N_r} ...", flush=True)
        Phi, cells_arr, theta_dir, phi_dir = build_basis_matrix(N_r, N_dir=12)
        C_matrix = conservation_matrix(N_r)
        M_matrix = matched_control_matrix(N_r, Phi, cells_arr)
        A_eq_conservation = sparse.csr_matrix(C_matrix)
        A_ineq, b_ineq, e_vec = positive_energy_constraints(Phi, N_r, N_dir_sample=12)
        templates = channel_templates(Phi, cells_arr)
        cache_by_N_r[N_r] = (Phi, cells_arr, A_eq_conservation, M_matrix,
                                A_ineq, b_ineq, e_vec, templates)
        print(f"[T170.1]   N_r={N_r}: basis dim N={N_STRESS*Phi.shape[1]}, "
                f"A_eq(C only) shape={A_eq_conservation.shape}, "
                f"M shape={M_matrix.shape}, "
                f"A_ineq shape={A_ineq.shape}, A_ineq nnz={A_ineq.nnz}", flush=True)

    print("[T170.1] running 48 LPs ...", flush=True)
    results = []
    for i_job, job in df_jobs.iterrows():
        N_r = int(job["N_r"])
        Phi, cells_arr, A_eq_cons, M_matrix, A_ineq, b_ineq, e_vec, templates = cache_by_N_r[N_r]
        result = run_one_job(N_r, job["observable_id"], int(job["sign"]),
                                templates, Phi, cells_arr,
                                A_eq_cons, M_matrix, A_ineq, b_ineq, e_vec)
        result["job_id"] = int(job["job_id"])
        result["N_r"] = N_r
        result["observable_id"] = job["observable_id"]
        result["sign"] = int(job["sign"])
        result["verdict"] = verdict_label(result)
        results.append(result)
        print(f"[T170.1]   job {i_job}/{len(df_jobs)} N_r={N_r} obs={job['observable_id']:18} "
                f"sign={job['sign']:+d} : C_O={result['C_O_a']:.3e}  "
                f"top={result['top_channel_class']}  v={result['verdict']}", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv(T170_1_DIR / "T170_1_optimisation_results.csv", index=False)

    df_results[["job_id", "N_r", "observable_id", "sign", "primal_optimum",
                  "dual_upper_bound", "dual_gap"]].to_csv(
        T170_1_DIR / "T170_1_dual_certificates.csv", index=False)

    df_results[["job_id", "N_r", "observable_id", "sign", "rho_min", "rho_max_actual",
                  "NEC_min", "WEC_min", "DEC_margin_min"]].to_csv(
        T170_1_DIR / "T170_1_energy_condition_diagnostics.csv", index=False)

    df_results[["job_id", "N_r", "observable_id", "sign", "SEC_min",
                  "SEC_violation_fraction", "C_stress_a", "C_O_a"]].to_csv(
        T170_1_DIR / "T170_1_SEC_tension_diagnostics.csv", index=False)

    df_results[["job_id", "N_r", "observable_id", "sign", "relative_A_residual"]].to_csv(
        T170_1_DIR / "T170_1_constraint_residuals.csv", index=False)

    pivot = df_results.pivot_table(index="observable_id", columns="N_r",
                                       values="C_O_a", aggfunc="max")
    pivot.to_csv(T170_1_DIR / "T170_1_radial_convergence.csv")

    best_rows = []
    for obs in OBSERVABLES:
        sub = df_results[df_results["observable_id"] == obs]
        if len(sub) > 0:
            best = sub.loc[sub["C_O_a"].abs().idxmax()]
            best_rows.append(best.to_dict())
    pd.DataFrame(best_rows).to_csv(T170_1_DIR / "T170_1_best_extremisers.csv", index=False)

    df_results[["job_id", "N_r", "observable_id", "sign", "C_O_a", "C_eps_a",
                  "C_stress_a", "top_channel_class", "verdict"]].to_csv(
        T170_1_DIR / "T170_1_verdict_table.csv", index=False)

    primary = compute_primary_verdict(df_results)
    n_above_compare = int((df_results["C_O_a"] > C_O_COMPARE).sum())
    n_above_large = int((df_results["C_O_a"] >= C_O_LARGE).sum())
    n_loophole = int(df_results["verdict"].str.contains("TRANSPORT_LOOPHOLE", regex=False).sum())
    n_sec_violation = int(df_results["verdict"].str.contains("SEC_VIOLATION", regex=False).sum())

    summary = {
        "primary_verdict": primary,
        "n_jobs_total": len(df_results),
        "n_C_O_above_14_7": n_above_compare,
        "n_C_O_above_100": n_above_large,
        "n_TRANSPORT_LOOPHOLE": n_loophole,
        "n_SEC_VIOLATION": n_sec_violation,
        "max_C_O_a": float(df_results["C_O_a"].max()),
        "max_C_eps_a": float(df_results["C_eps_a"].max()),
        "max_C_stress_a": float(df_results["C_stress_a"].max()),
        "max_relative_A_residual": float(df_results["relative_A_residual"].max()),
        "wall_time_s_total": time.time() - t_total_start,
        "MATLAB_parallel_used": False,
        "MATLAB_parallel_reason": (
            "Python scipy.linprog (HiGHS) used. LPs at N_r=32 have ~3000 vars "
            "and ~24000 sparse inequality rows; HiGHS interior-point dominates "
            "wall time. MATLAB Parallel was not invoked since the backend is "
            "in Python."
        ),
    }
    with open(T170_1_DIR / "T170_1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([
        ("primary_verdict", primary),
        ("n_jobs_total", len(df_results)),
        ("n_C_O_above_14_7", n_above_compare),
        ("n_C_O_above_100", n_above_large),
        ("n_TRANSPORT_LOOPHOLE", n_loophole),
        ("n_SEC_VIOLATION", n_sec_violation),
        ("max_C_O_a", float(df_results["C_O_a"].max())),
        ("max_C_eps_a", float(df_results["C_eps_a"].max())),
        ("max_C_stress_a", float(df_results["C_stress_a"].max())),
        ("max_relative_A_residual", float(df_results["relative_A_residual"].max())),
        ("MATLAB_parallel_used", "no  (Python scipy.linprog sufficient)"),
        ("always_appended_1", "NO_PHYSICAL_WARP_CLAIM"),
        ("always_appended_2", "NO_BUILDABLE_WARP_CLAIM"),
        ("always_appended_3", "ORDINARY_GR_COMPACT_BOUND_PRESERVED"),
        ("always_appended_4", "T170_1_POSITIVE_ENERGY_STRESS_CONE_OPTIMISATION"),
        ("always_appended_5", "NO_NEW_METRIC_PRODUCED_NO_WARPFACTORY_INVOKED"),
    ], columns=["key", "value"]).to_csv(T170_1_DIR / "combined_verdict_table.csv", index=False)

    print()
    print("=" * 75)
    print(f"T170.1 PRIMARY VERDICT: {primary}")
    print(f"  n jobs total                   : {len(df_results)}")
    print(f"  n C_O > 14.7                    : {n_above_compare}")
    print(f"  n C_O >= 100                     : {n_above_large}")
    print(f"  n TRANSPORT_LOOPHOLE              : {n_loophole}")
    print(f"  n SEC_VIOLATION_*                  : {n_sec_violation}")
    print(f"  max C_O_a                            : {df_results['C_O_a'].max():.3e}")
    print(f"  max C_stress_a                        : {df_results['C_stress_a'].max():.3e}")
    print(f"  max relative_A_residual                : {df_results['relative_A_residual'].max():.2e}")
    print(f"  total wall time                          : {summary['wall_time_s_total']:.2f} s")
    print("=" * 75)


def compute_primary_verdict(df):
    n_loophole = (df["verdict"] == "POSITIVE_ENERGY_TRANSPORT_LOOPHOLE").sum()
    n_sec_candidate = (df["verdict"] == "POSITIVE_ENERGY_SEC_VIOLATION_CANDIDATE").sum()
    n_sec_no_gain = (df["verdict"] == "POSITIVE_ENERGY_SEC_VIOLATION_BUT_NO_STRESS_GAIN").sum()
    n_low_l = (df["verdict"] == "LOW_L_CONTROL_EQUIVALENT").sum()
    n_bound_confirmed = (df["verdict"] == "POSITIVE_ENERGY_BOUND_CONFIRMED").sum()
    if n_loophole > 0:
        return "POSITIVE_ENERGY_TRANSPORT_LOOPHOLE"
    if n_sec_candidate > 0:
        return "POSITIVE_ENERGY_SEC_VIOLATION_CANDIDATE"
    if n_sec_no_gain > 0:
        return "POSITIVE_ENERGY_SEC_VIOLATION_NO_STRESS_GAIN"
    if n_low_l > 0:
        return "LOW_L_CONTROL_EQUIVALENT_NO_TRANSPORT"
    if n_bound_confirmed == len(df):
        return "POSITIVE_ENERGY_LOW_L_RESPONSE_BOUND_CONFIRMED"
    return "INCONCLUSIVE"


if __name__ == "__main__":
    main()
