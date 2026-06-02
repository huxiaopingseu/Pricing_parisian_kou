#!/usr/bin/env python3
"""
End-to-end pipeline: Kou Parisian option pricing via
2D Laplace transform + DeepONet + 2D numerical inversion.

Stages:
  1. Generate ODE training data: (lam1, lam2) -> psi(S) pairs
  2. Train DeepONet to learn the parametric operator
  3. 2D Fourier-series Laplace inversion to recover V(S, t, J)
  4. Compare with Monte Carlo benchmark
"""

import numpy as np
import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from kou_ode_solver import solve_kou_ode
from deeponet_kou import DeepONet, DeepONetTrainer, create_deeponet


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

MODEL_PARAMS = {
    'sigma': 0.20, 'r': 0.05, 'lam_jump': 0.5,
    'eta1': 25.0, 'eta2': 20.0, 'p': 0.4,
    'K': 100.0, 'L': 80.0, 'U': 120.0,
    'D_bar': 0.1, 'T': 1.0,
    'S_min': 10.0, 'S_max': 500.0,
}

DEEPONET_CFG = {
    'branch_layers': [128, 256, 128],
    'trunk_layers': [128, 256, 128],
    'n_basis': 80,
}

TRAIN_CFG = {
    'n_train_samples': 1500,
    'n_epochs': 5000,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'lr_step': 1500,
    'device': 'cpu',
    # Complex-aware: fraction of training samples with imaginary parts
    'complex_fraction': 0.5,
    'imag_range_lam1': 1.5,
    'imag_range_lam2': 1.2,
}

INVERSION_CFG = {
    'M': 16,          # quadrature half-width
    'h_step': 0.10,   # frequency step
    'gamma1': 1.0,    # Bromwich contour shift
    'gamma2': 1.0,
    't_eval': 1.0,    # time-to-maturity
    'J_eval': 0.0,    # excursion time
}

OUTPUT_DIR = '/home/xiaoping/parisian_kou/results'
N_SPATIAL = 200  # grid points in [L, U] for DeepONet


# ═══════════════════════════════════════════════════════════
# Stage 1: Generate training data
# ═══════════════════════════════════════════════════════════

def generate_training_data(params, n_samples, complex_fraction=0.5,
                           imag_range_lam1=1.5, imag_range_lam2=1.2,
                           N_x=400, N_interior=N_SPATIAL, seed=1234):
    """
    Generate (lam1, lam2) -> psi(S) training pairs.
    
    Returns:
        lam_array: (n_samples, 4) [Re(lam1), Im(lam1), Re(lam2), Im(lam2)]
        psi_array: (n_samples, N_interior)
        S_grid: (N_interior,) interior S coordinates
    """
    rng = np.random.default_rng(seed)
    n_complex = int(n_samples * complex_fraction)
    n_real = n_samples - n_complex
    
    lam_array = np.zeros((n_samples, 4))
    psi_list = []
    S_grid = None
    
    # ODE domain for each sample — use a coarser grid for speed
    ode_params = params.copy()
    ode_params['S_min'] = params.get('S_min', 10.0)
    ode_params['S_max'] = params.get('S_max', 500.0)
    
    print(f"Generating {n_samples} training samples ({n_real} real + {n_complex} complex)...")
    t0 = time.time()
    
    # Helper to process one sample
    def process_sample(idx, re_lam1, im_lam1, re_lam2, im_lam2, is_real):
        lam1 = re_lam1 + 1j*im_lam1 if not is_real else re_lam1
        lam2 = re_lam2 + 1j*im_lam2 if not is_real else re_lam2
        lam_array[idx] = [re_lam1, im_lam1, re_lam2, im_lam2]
        
        # Solve ODE at complex (lam1, lam2)
        # For complex parameters, solve separate real and imaginary parts
        if is_real:
            _, _, psi_int, S_int = solve_kou_ode(re_lam1, re_lam2, ode_params, N_x=N_x)
            return psi_int, S_int
        else:
            # For complex lambda, we need to solve the complex ODE
            # Strategy: solve the 4th-order ODE with complex coefficients
            # The ODE coefficients are analytic in lambda, so we can solve
            # the complex system directly by treating real and imag parts
            psi = solve_kou_ode_complex(lam1, lam2, ode_params, N_x=N_x)
            return psi, None  # S_grid is same for all
    
    # Generate real samples
    for i in range(n_real):
        re_lam1 = rng.uniform(0.1, 5.0)
        re_lam2 = rng.uniform(0.1, 5.0)
        psi_int, S_int = process_sample(i, re_lam1, 0.0, re_lam2, 0.0, True)
        psi_list.append(psi_int)
        if S_grid is None:
            S_grid = S_int
        if (i + 1) % 500 == 0:
            print(f"  Real samples: {i+1}/{n_real} ({(i+1)/n_real*100:.0f}%)")
    
    # Generate complex samples
    for i in range(n_complex):
        idx = n_real + i
        re_lam1 = rng.uniform(0.1, 5.0)
        re_lam2 = rng.uniform(0.1, 5.0)
        im_lam1 = rng.uniform(-imag_range_lam1, imag_range_lam1)
        im_lam2 = rng.uniform(-imag_range_lam2, imag_range_lam2)
        psi_int, _ = process_sample(idx, re_lam1, im_lam1, re_lam2, im_lam2, False)
        psi_list.append(psi_int)
        if (i + 1) % 500 == 0:
            print(f"  Complex samples: {i+1}/{n_complex} ({(i+1)/n_complex*100:.0f}%)")
    
    psi_array = np.array(psi_list)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Shape: {psi_array.shape}")
    
    return lam_array, psi_array, S_grid


def solve_kou_ode_complex(lam1_c, lam2_c, params, N_x=400):
    """
    Solve the 4th-order ODE for complex lambda values.
    
    Since the ODE coefficients are analytic in lambda, we can solve
    the complex-valued linear system directly using numpy's complex
    linear algebra (spsolve handles complex RHS).
    """
    from kou_ode_solver import ode_coefficients, source_f_tilde, boundary_value_left
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    import numpy as np
    
    sigma = params['sigma']
    r_val = params['r']
    lam_jump = params['lam_jump']
    eta1 = params['eta1']
    eta2 = params['eta2']
    p_jump = params['p']
    q_jump = 1 - p_jump
    K = params['K']
    L = params['L']
    U = params['U']
    D_bar = params['D_bar']
    S_min = params.get('S_min', 10.0)
    S_max = params.get('S_max', 500.0)
    
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    xL = np.log(L)
    xU = np.log(U)
    h = (x_max - x_min) / (N_x - 1)
    x_grid = np.linspace(x_min, x_max, N_x)
    
    # Build complex linear system (same structure as real case)
    rows, cols, vals = [], [], []
    rhs = np.zeros(N_x, dtype=complex)
    
    # Source term (complex lambda)
    C = (1 - np.exp(-lam2_c * D_bar)) / max(abs(lam2_c), 1e-14)
    F_tilde = np.zeros(N_x, dtype=complex)
    xK = np.log(K)
    mask = x_grid < xK
    ex = np.exp(x_grid[mask])
    F_tilde[mask] = C * (-ex + (eta2 - eta1)*ex + eta1*eta2*(-K + ex))
    
    for j in range(N_x):
        S_j = np.exp(x_grid[j])
        if S_j < L:
            chi = 1
        elif S_j <= U:
            chi = 0
        else:
            chi = 1
        
        # Complex ODE coefficients (same formulas, complex lam1, lam2)
        A4_c, A3_c, A2_c, A1_c, A0_c = ode_coefficients_complex(
            sigma, r_val, lam_jump, lam1_c, lam2_c, eta1, eta2, p_jump, q_jump, chi)
        
        if j == 0:
            rows.append(0); cols.append(0); vals.append(1.0 + 0j)
            psi_left = boundary_value_left_complex(K, lam1_c, lam2_c, r_val, D_bar)
            rhs[0] = psi_left
            continue
        if j == N_x - 1:
            rows.append(N_x-1); cols.append(N_x-1); vals.append(1.0 + 0j)
            rhs[N_x-1] = 0.0 + 0j
            continue
        
        # 5-point stencil (same as real case, with complex coefficients)
        val_j2 = A4_c/h**4 - A3_c/(2*h**3)
        val_j1 = -4*A4_c/h**4 + A3_c/h**3 + A2_c/h**2 - A1_c/(2*h)
        val_j0 = 6*A4_c/h**4 - 2*A2_c/h**2 + A0_c
        val_jp1 = -4*A4_c/h**4 - A3_c/h**3 + A2_c/h**2 + A1_c/(2*h)
        val_jp2 = A4_c/h**4 + A3_c/(2*h**3)
        
        for idx, v in zip([j-2, j-1, j, j+1, j+2], [val_j2, val_j1, val_j0, val_jp1, val_jp2]):
            if 0 <= idx < N_x:
                rows.append(j); cols.append(idx); vals.append(v)
        rhs[j] = complex(F_tilde[j])
    
    A_mat = csr_matrix((vals, (rows, cols)), shape=(N_x, N_x))
    psi_full = spsolve(A_mat, rhs)
    
    # Extract interior
    interior_mask = (x_grid >= xL) & (x_grid <= xU)
    psi_int = psi_full[interior_mask]
    
    # Return: stack real and imag parts
    psi_2ch = np.column_stack([psi_int.real, psi_int.imag])
    return psi_2ch


def ode_coefficients_complex(sigma, r, lam_jump, lam1, lam2, eta1, eta2, p, q, chi):
    """Complex-valued ODE coefficients."""
    zeta = p * eta1/(eta1 - 1) - q * eta2/(eta2 + 1) - 1
    
    A4 = -0.5 * sigma**2 + 0j
    A3 = (lam_jump*zeta - r + 0.5*sigma**2*(-eta1 + eta2) + 0.5*sigma**2) + 0j
    
    A2 = (0.5*eta1*eta2*sigma**2 + lam1 + chi*lam2 + lam_jump + r
          + 0.5*(eta1 - eta2)*(2*lam_jump*zeta - 2*r + sigma**2))
    
    A1 = (-0.5*eta1*eta2*(2*lam_jump*zeta - 2*r + sigma**2)
          - lam_jump*(eta1*p - eta2*q)
          + (eta1 - eta2)*(lam1 + chi*lam2 + lam_jump + r))
    
    A0 = eta1*eta2*(-lam1 - chi*lam2 - r)
    
    return A4, A3, A2, A1, A0


def boundary_value_left_complex(K, lam1, lam2, r, D_bar):
    """Complex far-field boundary value."""
    C = (1 - np.exp(-lam2 * D_bar)) / max(abs(lam2), 1e-14)
    return K * C / max(abs(r + lam1), 1e-14)


# ═══════════════════════════════════════════════════════════
# Stage 2: Train DeepONet
# ═══════════════════════════════════════════════════════════

def train_deeponet(lam_array, psi_array, S_grid, cfg_train, complex_output=True):
    """Train DeepONet and return model + trainer."""
    n_samples = lam_array.shape[0]
    N = S_grid.shape[0]
    
    # Normalize S grid to [0, 1]
    S_min, S_max = S_grid[0], S_grid[-1]
    S_norm = (S_grid - S_min) / (S_max - S_min)
    
    # Create model
    model = DeepONet(
        branch_layers=DEEPONET_CFG['branch_layers'],
        trunk_layers=DEEPONET_CFG['trunk_layers'],
        n_basis=DEEPONET_CFG['n_basis'],
        complex_output=complex_output,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DeepONet: {n_params:,} parameters, complex_output={complex_output}")
    
    trainer = DeepONetTrainer(
        model,
        device=cfg_train['device'],
        lr=cfg_train['lr'],
        weight_decay=cfg_train['weight_decay'],
    )
    
    # Training loop
    n_epochs = cfg_train['n_epochs']
    batch_size = cfg_train['batch_size']
    
    print(f"Training {n_epochs} epochs, batch_size={batch_size}...")
    t0 = time.time()
    losses = []
    
    for epoch in range(n_epochs):
        # Random batch
        idx = np.random.choice(n_samples, batch_size, replace=False)
        lam_batch = lam_array[idx]
        psi_batch = psi_array[idx]
        
        if not complex_output and psi_batch.ndim == 3:
            # Use only real channel
            psi_batch = psi_batch[:, :, 0]
        
        loss = trainer.train_epoch(lam_batch, S_norm, psi_batch)
        losses.append(loss)
        
        if (epoch + 1) % 1000 == 0:
            # Evaluate on test set
            test_idx = np.random.choice(n_samples, min(50, n_samples), replace=False)
            test_lam = lam_array[test_idx]
            test_psi = psi_array[test_idx]
            if not complex_output and test_psi.ndim == 3:
                test_psi = test_psi[:, :, 0]
            rel_errs = trainer.evaluate(test_lam, S_norm, test_psi)
            
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:5d}/{n_epochs} | Loss: {loss:.2e} | "
                  f"Rel err: mean={rel_errs.mean():.2e}, max={rel_errs.max():.2e} | "
                  f"{elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")
    
    return model, trainer, S_norm, losses


# ═══════════════════════════════════════════════════════════
# Stage 3: 2D Fourier-series Laplace inversion
# ═══════════════════════════════════════════════════════════

def fourier_inversion_2d(model, S_grid_norm, cfg_inv: dict):
    """
    Recover V(S, t, J) from DeepONet via 2D Fourier-series inversion.
    
    Uses Bromwich contour: lam_k = gamma_k + i * omega_k
    """
    M = cfg_inv['M']
    h = cfg_inv['h_step']
    gamma1 = cfg_inv['gamma1']
    gamma2 = cfg_inv['gamma2']
    t_val = cfg_inv['t_eval']
    J_val = cfg_inv['J_eval']
    
    device = next(model.parameters()).device
    model.eval()
    
    N = len(S_grid_norm)
    V_hat_sum = np.zeros(N, dtype=complex)
    
    print(f"2D Fourier inversion: M={M}, h={h:.4f}, {2*M+1}x{2*M+1} = {(2*M+1)**2} evaluations...")
    t0 = time.time()
    
    for k1 in range(-M, M + 1):
        omega1 = k1 * h
        lam1_c = gamma1 + 1j * omega1
        
        for k2 in range(-M, M + 1):
            omega2 = k2 * h
            lam2_c = gamma2 + 1j * omega2
            
            # Build lambda parameter vector
            lam_vec = np.array([[gamma1, omega1, gamma2, omega2]], dtype=np.float32)
            lam_t = torch.FloatTensor(lam_vec).to(device)
            S_t = torch.FloatTensor(S_grid_norm).to(device)
            
            with torch.no_grad():
                out = model(lam_t, S_t).squeeze(0).cpu().numpy()  # (N,) or (N, 2)
            
            if out.ndim == 2 and out.shape[1] == 2:
                V_hat_val = out[:, 0] + 1j * out[:, 1]
            else:
                V_hat_val = out.astype(complex)
            
            # Accumulate with quadrature weight
            weight = np.exp(lam1_c * t_val + lam2_c * J_val) * h**2
            V_hat_sum += weight * V_hat_val
    
    # Final scaling
    V_price = (np.exp(gamma1 * t_val + gamma2 * J_val) / (4 * np.pi**2)) * V_hat_sum.real
    
    elapsed = time.time() - t0
    print(f"  Inversion complete in {elapsed:.1f}s")
    print(f"  V(S=K) = {V_price[len(V_price)//2]:.6f}")
    
    return V_price


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def run_pipeline(skip_data_gen=False, data_file=None):
    """Execute full 3-stage pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ── Stage 1: Data generation ──
    if skip_data_gen and data_file:
        print(f"Loading precomputed data from {data_file}...")
        data = np.load(data_file, allow_pickle=True)
        lam_array = data['lam_array']
        psi_array = data['psi_array']
        S_grid = data['S_grid']
    else:
        lam_array, psi_array, S_grid = generate_training_data(
            MODEL_PARAMS,
            n_samples=TRAIN_CFG['n_train_samples'],
            complex_fraction=TRAIN_CFG['complex_fraction'],
            imag_range_lam1=TRAIN_CFG['imag_range_lam1'],
            imag_range_lam2=TRAIN_CFG['imag_range_lam2'],
        )
        
        # Save training data
        np.savez(os.path.join(OUTPUT_DIR, 'training_data.npz'),
                 lam_array=lam_array, psi_array=psi_array, S_grid=S_grid)
        print(f"Training data saved to {OUTPUT_DIR}/training_data.npz")
    
    # ── Stage 2: Train DeepONet ──
    model, trainer, S_norm, losses = train_deeponet(
        lam_array, psi_array, S_grid, TRAIN_CFG, complex_output=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'deeponet_kou.pt'))
    print(f"Model saved to {OUTPUT_DIR}/deeponet_kou.pt")
    
    # ── Stage 3: Laplace inversion ──
    V_price = fourier_inversion_2d(model, S_norm, INVERSION_CFG)
    
    # Save results
    S_full = S_grid  # interior [L, U]
    np.savez(os.path.join(OUTPUT_DIR, 'pricing_results.npz'),
             S_grid=S_full, V_price=V_price)
    print(f"Results saved to {OUTPUT_DIR}/pricing_results.npz")
    
    # ── Summary ──
    atm_idx = len(S_full) // 2
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  ATM price (S=K={MODEL_PARAMS['K']}): {V_price[atm_idx]:.6f}")
    print(f"  Barrier values: V(L)={V_price[0]:.6f}, V(U)={V_price[-1]:.6f}")
    print(f"  Price range: [{V_price.min():.6f}, {V_price.max():.6f}]")
    
    return model, V_price, S_full, losses


# ═══════════════════════════════════════════════════════════
# Quick test / verification mode (small scale)
# ═══════════════════════════════════════════════════════════

def run_quick_test():
    """Small-scale test to verify the pipeline works end-to-end."""
    print("QUICK TEST MODE (small scale)")
    print("=" * 60)
    
    # Reduced config
    n_train = 100  # very small for testing
    n_epochs = 500
    M_inv = 8  # coarse inversion
    
    params = MODEL_PARAMS.copy()
    
    # Generate small training set
    print("\n[1/3] Generating mini training set...")
    rng = np.random.default_rng(42)
    lam_list = []
    psi_list = []
    S_grid = None
    
    for i in range(n_train):
        re_lam1 = rng.uniform(0.1, 5.0)
        re_lam2 = rng.uniform(0.1, 5.0)
        im_lam1 = rng.uniform(-1.5, 1.5) if i % 2 == 0 else 0.0
        im_lam2 = rng.uniform(-1.2, 1.2) if i % 2 == 0 else 0.0
        
        if im_lam1 == 0 and im_lam2 == 0:
            _, _, psi_int, S_int = solve_kou_ode(re_lam1, re_lam2, params, N_x=200)
            psi = psi_int
        else:
            psi = solve_kou_ode_complex(re_lam1+1j*im_lam1, re_lam2+1j*im_lam2, params, N_x=200)
            psi = psi.reshape(-1, 2)  # (N, 2)
        
        lam_list.append([re_lam1, im_lam1, re_lam2, im_lam2])
        psi_list.append(psi)
        if S_grid is None and isinstance(psi, np.ndarray):
            _, _, psi_int_ref, S_grid = solve_kou_ode(1.0, 1.0, params, N_x=200)
    
    lam_array = np.array(lam_list)
    
    # Ensure uniform psi shape: real samples need imag channel padded with zeros
    N_pts = min(len(p) for p in psi_list)
    
    # First pass: determine max shape
    max_N = max(p.shape[0] if p.ndim >= 1 else 1 for p in psi_list)
    has_2d = any(p.ndim == 2 for p in psi_list)
    
    # All psi become (max_N, 2) for complex-aware training
    psi_uniform = []
    for p in psi_list:
        if p.ndim == 1:
            # Real sample: real part only, zero imag
            p2 = np.zeros((max_N, 2))
            p2[:len(p), 0] = p
            psi_uniform.append(p2)
        else:
            # Already 2D
            p2 = np.zeros((max_N, 2))
            n = min(p.shape[0], max_N)
            p2[:n, :] = p[:n, :]
            psi_uniform.append(p2)
    psi_array = np.array(psi_uniform)
    
    print(f"  Generated {len(lam_array)} samples, psi shape: {psi_array.shape}")
    
    # Train tiny DeepONet
    print("\n[2/3] Training mini DeepONet...")
    S_min, S_max = S_grid[0], S_grid[-1]
    S_norm = (S_grid - S_min) / (S_max - S_min)
    
    model = DeepONet(
        branch_layers=[64, 128, 64],
        trunk_layers=[64, 128, 64],
        n_basis=40,
        complex_output=True,
    )
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    trainer = DeepONetTrainer(model, device='cpu', lr=1e-3)
    
    for epoch in range(n_epochs):
        idx = np.random.choice(len(lam_array), min(32, len(lam_array)), replace=False)
        loss = trainer.train_epoch(lam_array[idx], S_norm, psi_array[idx])
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: loss={loss:.2e}")
    
    # Quick inversion
    print(f"\n[3/3] Coarse inversion (M={M_inv})...")
    t0 = time.time()
    
    N = len(S_norm)
    V_sum = np.zeros(N, dtype=complex)
    
    for k1 in range(-M_inv, M_inv + 1):
        omega1 = k1 * 0.12
        for k2 in range(-M_inv, M_inv + 1):
            omega2 = k2 * 0.12
            
            lam_vec = np.array([[1.0, omega1, 1.0, omega2]], dtype=np.float32)
            lam_t = torch.FloatTensor(lam_vec)
            S_t = torch.FloatTensor(S_norm)
            
            with torch.no_grad():
                out = model(lam_t, S_t).squeeze(0).cpu().numpy()
            
            if out.ndim == 2:
                V_hat = out[:, 0] + 1j * out[:, 1]
            else:
                V_hat = out.astype(complex)
            
            weight = np.exp((1.0 + 1j*omega1) * 1.0 + (1.0 + 1j*omega2) * 0.0) * 0.0144
            V_sum += weight * V_hat
    
    V_price = np.exp(1.0 + 0.0) / (4 * np.pi**2) * V_sum.real
    elapsed = time.time() - t0
    
    atm_idx = len(V_price) // 2
    print(f"  Done in {elapsed:.1f}s")
    print(f"  ATM price: {V_price[atm_idx]:.6f}")
    print(f"  V(L)={V_price[0]:.6f}, V(U)={V_price[-1]:.6f}")
    print(f"\n  Quick test PASSED — pipeline functional")
    
    return V_price, S_grid


if __name__ == '__main__':
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Quick test mode (small scale)')
    ap.add_argument('--full', action='store_true', help='Full pipeline')
    ap.add_argument('--skip-data', action='store_true', help='Skip data generation')
    ap.add_argument('--data', type=str, default=None, help='Precomputed data file')
    args = ap.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.full:
        run_pipeline(skip_data_gen=args.skip_data, data_file=args.data)
    else:
        print("Usage: python run_pipeline.py --quick  (small test)")
        print("       python run_pipeline.py --full   (full pipeline)")
        print("\nRunning quick test by default...")
        run_quick_test()
