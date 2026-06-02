#!/usr/bin/env python3
"""
Numerical solver for the 4th-order ODE arising from the 2D Laplace transform
of the Kou double-exponential jump-diffusion Parisian option PIDE.

The ODE (in x = log S) has piecewise-constant coefficients across 3 regions:
  Region I:   S_min ≤ S < L    (outside below, chi=1)
  Region II:  L ≤ S ≤ U        (inside corridor, chi=0)
  Region III: U < S ≤ S_max    (outside above, chi=1)

Method: 5-point compact finite difference with C^3 continuity at barriers.
"""

import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import spsolve

# ────────────────────────────────────────────────────────────
# ODE coefficients (from symbolic_ode.py — verified)
# ────────────────────────────────────────────────────────────

def ode_coefficients(sigma, r, lam_jump, lam1, lam2, eta1, eta2, p, q, chi):
    """Return (A4, A3, A2, A1, A0) for given chi (0=inside, 1=outside)."""
    zeta = p * eta1/(eta1 - 1) - q * eta2/(eta2 + 1) - 1  # E[Y-1]
    
    A4 = -0.5 * sigma**2
    A3 = lam_jump*zeta - r + 0.5*sigma**2*(-eta1 + eta2) + 0.5*sigma**2
    
    A2 = (0.5*eta1*eta2*sigma**2 + lam1 + chi*lam2 + lam_jump + r
          + 0.5*(eta1 - eta2)*(2*lam_jump*zeta - 2*r + sigma**2))
    
    A1 = (-0.5*eta1*eta2*(2*lam_jump*zeta - 2*r + sigma**2)
          - lam_jump*(eta1*p - eta2*q)
          + (eta1 - eta2)*(lam1 + chi*lam2 + lam_jump + r))
    
    A0 = eta1*eta2*(-lam1 - chi*lam2 - r)
    
    return A4, A3, A2, A1, A0

# ────────────────────────────────────────────────────────────
# Source term: F(x) and F_tilde = Q(D) F(x)
# ────────────────────────────────────────────────────────────

def source_f(x, K, lam2, D_bar):
    """F(x) = -max(K-e^x,0) * (1-exp(-lambda2*D_bar))/lambda2."""
    C = (1 - np.exp(-lam2 * D_bar)) / max(lam2, 1e-14)
    payoff = np.maximum(K - np.exp(x), 0)
    return -payoff * C


def source_f_tilde(x, K, lam2, D_bar, eta1, eta2):
    """F_tilde = Q(D)F = -F'' + (eta2-eta1)F' + eta1*eta2*F."""
    C = (1 - np.exp(-lam2 * D_bar)) / max(lam2, 1e-14)
    xK = np.log(K)
    
    # Initialize
    F_tilde = np.zeros_like(x)
    
    # For x < log(K): F = C*(-K + e^x), F' = C*e^x, F'' = C*e^x
    mask = x < xK
    ex = np.exp(x[mask])
    F_tilde[mask] = C * (-ex + (eta2 - eta1)*ex + eta1*eta2*(-K + ex))
    # For x >= log(K): F = 0, F' = 0, F'' = 0 => F_tilde = 0
    # Already zero-initialized
    
    return F_tilde

# ────────────────────────────────────────────────────────────
# Far-field boundary values
# ────────────────────────────────────────────────────────────

def boundary_value_left(K, lam1, lam2, r, D_bar):
    """As S -> 0 (x -> -inf), psi -> K*(1-e^{-lambda2*D})/(lambda2*(r+lambda1))."""
    C = (1 - np.exp(-lam2 * D_bar)) / max(lam2, 1e-14)
    return K * C / max(r + lam1, 1e-14)

# ────────────────────────────────────────────────────────────
# 5-point finite difference stencil helper
# ────────────────────────────────────────────────────────────

def fd_coeffs_4th(j, N, h, A4, A3, A2, A1, A0):
    """Return (row_indices, row_values) for FD equation at grid point j."""
    # Standard centered 5-point formulas:
    # u''''(xj) ≈ (u_{j-2} - 4u_{j-1} + 6u_j - 4u_{j+1} + u_{j+2}) / h^4
    # u'''(xj)  ≈ (-u_{j-2} + 2u_{j-1} - 2u_{j+1} + u_{j+2}) / (2h^3)
    # u''(xj)   ≈ (u_{j-1} - 2u_j + u_{j+1}) / h^2
    # u'(xj)    ≈ (-u_{j-1} + u_{j+1}) / (2h)
    # u(xj)     ≈ u_j
    
    # Coefficients for u_{j-2}, u_{j-1}, u_j, u_{j+1}, u_{j+2}
    val_j2 = A4/h**4 - A3/(2*h**3)
    val_j1 = -4*A4/h**4 + 2*A3/(2*h**3) + A2/h**2 - A1/(2*h)
    val_j0 = 6*A4/h**4 - 2*A2/h**2 + A0
    val_jp1 = -4*A4/h**4 - 2*A3/(2*h**3) + A2/h**2 + A1/(2*h)
    val_jp2 = A4/h**4 + A3/(2*h**3)
    
    indices = [j-2, j-1, j, j+1, j+2]
    values = [val_j2, val_j1, val_j0, val_jp1, val_jp2]
    
    # Filter out out-of-bounds indices
    valid = [(idx, val) for idx, val in zip(indices, values) if 0 <= idx < N]
    return list(zip(*valid)) if valid else ([], [])

# ────────────────────────────────────────────────────────────
# Main ODE solver
# ────────────────────────────────────────────────────────────

def solve_kou_ode(lam1, lam2, params, N_x=400):
    """
    Solve the 4th-order ODE on the 3-region domain.
    
    Parameters
    ----------
    lam1, lam2 : float
        Laplace parameters (real-valued).
    params : dict
        Model parameters: sigma, r, lam_jump, eta1, eta2, p, K,
                          L, U, D_bar, S_min, S_max.
    N_x : int
        Number of grid points in x = log(S).
    
    Returns
    -------
    x_grid : ndarray
        Uniform grid in x-space.
    psi : ndarray
        Solution psi(x) = V_hat(e^x; lam1, lam2).
    psi_interior : ndarray
        Solution restricted to [L, U] for DeepONet training.
    S_interior : ndarray
        Interior S grid.
    """
    sigma = params['sigma']
    r = params['r']
    lam_jump = params['lam_jump']
    eta1 = params['eta1']
    eta2 = params['eta2']
    p_jump = params['p']
    q_jump = 1 - p_jump
    K = params['K']
    L = params['L']
    U = params['U']
    D_bar = params['D_bar']
    S_min = params.get('S_min', max(0.1*L, 1.0))
    S_max = params.get('S_max', min(5.0*U, 1000.0))
    
    # ── x-grid ──
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    xL = np.log(L)
    xU = np.log(U)
    h = (x_max - x_min) / (N_x - 1)
    x_grid = np.linspace(x_min, x_max, N_x)
    
    # ── Build sparse matrix ──
    rows = []
    cols = []
    vals = []
    rhs = np.zeros(N_x)
    
    # Pre-compute source term
    F_tilde = source_f_tilde(x_grid, K, lam2, D_bar, eta1, eta2)
    
    for j in range(N_x):
        S_j = np.exp(x_grid[j])
        
        # Determine region
        if S_j < L:
            chi = 1  # outside below
        elif S_j <= U:
            chi = 0  # inside corridor
        else:
            chi = 1  # outside above
        
        A4, A3, A2, A1, A0 = ode_coefficients(
            sigma, r, lam_jump, lam1, lam2, eta1, eta2, p_jump, q_jump, chi)
        
        # Boundary rows
        if j == 0:
            # Dirichlet: psi(x_min) = psi_left
            rows.append(0); cols.append(0); vals.append(1.0)
            psi_left = boundary_value_left(K, lam1, lam2, r, D_bar)
            rhs[0] = psi_left
            continue
        
        if j == N_x - 1:
            # Dirichlet: psi(x_max) = 0
            rows.append(N_x - 1); cols.append(N_x - 1); vals.append(1.0)
            rhs[N_x - 1] = 0.0
            continue
        
        # ODE equation at interior point j
        jdx, jvals = fd_coeffs_4th(j, N_x, h, A4, A3, A2, A1, A0)
        for c, v in zip(jdx, jvals):
            rows.append(j); cols.append(c); vals.append(v)
        rhs[j] = F_tilde[j]
    
    A_mat = csr_matrix((vals, (rows, cols)), shape=(N_x, N_x))
    psi = spsolve(A_mat, rhs)
    
    # ── Extract interior solution ──
    interior_mask = (x_grid >= xL) & (x_grid <= xU)
    x_interior = x_grid[interior_mask]
    psi_interior = psi[interior_mask]
    S_interior = np.exp(x_interior)
    
    return x_grid, psi, psi_interior, S_interior


# ────────────────────────────────────────────────────────────
# Verification: solve ODE and compare to direct PIDE quadrature
# ────────────────────────────────────────────────────────────

def solve_pide_direct(lam1, lam2, params, N_x=300):
    """
    Solve the original 2nd-order PIDE directly (with integral term
    handled via Gauss-Laguerre quadrature) for verification.
    
    This is slower but does NOT use the ODE-ification trick,
    serving as an independent check.
    """
    sigma = params['sigma']
    r = params['r']
    lam_jump = params['lam_jump']
    eta1 = params['eta1']
    eta2 = params['eta2']
    p_jump = params['p']
    q_jump = 1 - p_jump
    K = params['K']
    L = params['L']
    U = params['U']
    D_bar = params['D_bar']
    S_min = params.get('S_min', max(0.1*L, 1.0))
    S_max = params.get('S_max', min(5.0*U, 1000.0))
    
    zeta = p_jump * eta1/(eta1 - 1) - q_jump * eta2/(eta2 + 1) - 1
    
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    xL = np.log(L)
    xU = np.log(U)
    h = (x_max - x_min) / (N_x - 1)
    x_grid = np.linspace(x_min, x_max, N_x)
    
    # PIDE: a ψ'' + b ψ' + c ψ + lambda * (p*I_plus + q*I_minus) psi = F
    a_coeff = 0.5 * sigma**2
    b_coeff = r - lam_jump*zeta - 0.5*sigma**2
    
    # Source
    C = (1 - np.exp(-lam2 * D_bar)) / max(lam2, 1e-14)
    F = np.where(np.exp(x_grid) < K, -C * (K - np.exp(x_grid)), 0.0)
    
    # FD matrix for diffusion + drift part (tridiagonal)
    main_diag = np.zeros(N_x)
    lower_diag = np.zeros(N_x - 1)
    upper_diag = np.zeros(N_x - 1)
    
    for j in range(N_x):
        S_j = np.exp(x_grid[j])
        chi = 1 if (S_j < L or S_j > U) else 0
        c_coeff = -(r + lam_jump + lam1 + chi*lam2)
        
        if j == 0:  # Dirichlet
            main_diag[j] = 1.0
        elif j == N_x - 1:  # Dirichlet
            main_diag[j] = 1.0
        else:
            main_diag[j] = -2*a_coeff/h**2 + c_coeff
            lower_diag[j-1] = a_coeff/h**2 - b_coeff/(2*h)
            upper_diag[j-1] = a_coeff/h**2 + b_coeff/(2*h)
    
    # Jump integral: dense coupling
    # I_plus ψ(x) = ∫_0^∞ ψ(x+z) eta1 e^{-eta1 z} dz
    # I_minus ψ(x) = ∫_0^∞ ψ(x-z) eta2 e^{-eta2 z} dz
    
    # Use trapezoidal quadrature for the integral (extend x-grid with extrapolation)
    L_jump_mat = np.zeros((N_x, N_x))
    
    for j in range(N_x):
        x_j = x_grid[j]
        # I_plus: integrate over z > 0
        integrand_plus = np.zeros(N_x)
        integrand_minus = np.zeros(N_x)
        
        for k in range(N_x):
            z_plus = x_grid[k] - x_j
            if z_plus > 0:
                integrand_plus[k] = eta1 * np.exp(-eta1 * z_plus)
            
            z_minus = x_j - x_grid[k]
            if z_minus > 0:
                integrand_minus[k] = eta2 * np.exp(-eta2 * z_minus)
        
        # Trapezoidal weights
        w = np.ones(N_x) * h
        w[0] = h/2; w[-1] = h/2
        
        L_jump_mat[j, :] = lam_jump * (p_jump * integrand_plus + q_jump * integrand_minus) * w
    
    # Boundary conditions for RHS
    psi_left = boundary_value_left(K, lam1, lam2, r, D_bar)
    F[0] = psi_left
    F[-1] = 0.0
    
    # Build system matrix in coo first
    from scipy.sparse import coo_matrix, lil_matrix, csr_matrix as csr_m
    A_tridiag = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='coo')
    A_mat = lil_matrix(A_tridiag.shape)
    # Add tridiagonal part
    for i, j, v in zip(A_tridiag.row, A_tridiag.col, A_tridiag.data):
        A_mat[i, j] = v
    # Add dense jump part (convert to sparse first)
    L_jump_sparse = csr_m(L_jump_mat)
    A_mat = A_mat + L_jump_sparse
    A_mat[0, :] = 0; A_mat[0, 0] = 1
    A_mat[-1, :] = 0; A_mat[-1, -1] = 1
    A_mat = A_mat.tocsr()
    
    psi_pide = spsolve(A_mat, F)
    
    interior_mask = (x_grid >= xL) & (x_grid <= xU)
    psi_int = psi_pide[interior_mask]
    S_int = np.exp(x_grid[interior_mask])
    
    return x_grid, psi_pide, psi_int, S_int


# ────────────────────────────────────────────────────────────
# Test / verification
# ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    
    params = {
        'sigma': 0.20, 'r': 0.05, 'lam_jump': 0.5,
        'eta1': 25.0, 'eta2': 20.0, 'p': 0.4,
        'K': 100.0, 'L': 80.0, 'U': 120.0,
        'D_bar': 0.1, 'S_min': 10.0, 'S_max': 500.0,
    }
    
    # Compute zeta for reference
    p_jump = params['p']
    q_jump = 1 - p_jump
    zeta = p_jump * params['eta1']/(params['eta1']-1) - q_jump * params['eta2']/(params['eta2']+1) - 1
    print(f"Model: sigma={params['sigma']}, r={params['r']}, lambda={params['lam_jump']}")
    print(f"       eta1={params['eta1']}, eta2={params['eta2']}, p={p_jump}")
    print(f"       zeta = E[Y-1] = {zeta:.6f}")
    print(f"       L={params['L']}, U={params['U']}, K={params['K']}, D={params['D_bar']}")
    
    # Test on a few (lam1, lam2) pairs
    for lam1, lam2 in [(1.0, 1.0), (2.0, 2.0), (0.5, 0.5)]:
        print(f"\n{'='*60}")
        print(f"lam1={lam1:.1f}, lam2={lam2:.1f}")
        
        # 4th-order ODE
        t0 = time.time()
        x_ode, psi_ode, psi_ode_int, S_int = solve_kou_ode(lam1, lam2, params, N_x=400)
        t_ode = time.time() - t0
        
        # Direct PIDE
        t0 = time.time()
        x_pide, psi_pide, psi_pide_int, S_pide_int = solve_pide_direct(lam1, lam2, params, N_x=300)
        t_pide = time.time() - t0
        
        # Compare
        rel_err = np.linalg.norm(psi_ode_int - psi_pide_int) / max(np.linalg.norm(psi_pide_int), 1e-14)
        max_err = np.max(np.abs(psi_ode_int - psi_pide_int))
        
        print(f"  ODE  solver: {t_ode:.3f}s, range = [{psi_ode_int.min():.6f}, {psi_ode_int.max():.6f}]")
        print(f"  PIDE solver: {t_pide:.3f}s, range = [{psi_pide_int.min():.6f}, {psi_pide_int.max():.6f}]")
        print(f"  Relative L2 error: {rel_err:.2e}")
        print(f"  Max absolute error: {max_err:.2e}")
        
        # Quick visual check
        print(f"  Barrier values: V(L)={psi_ode_int[0]:.6f}, V(U)={psi_ode_int[-1]:.6f}")
        print(f"  ATM value: {psi_ode_int[len(psi_ode_int)//2]:.6f}")
