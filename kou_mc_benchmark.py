#!/usr/bin/env python3
"""
Monte Carlo benchmark for double-barrier Parisian options under
Kou's double-exponential jump-diffusion model.

The Parisian knock-out condition: option is worthless if cumulative
excursion time outside [L, U] exceeds D_bar before maturity T.
"""

import numpy as np
from numpy.random import default_rng


def simulate_kou_parisian(params, n_paths=50000, n_steps=500, seed=42):
    """
    Simulate double-barrier Parisian knock-out put under Kou model.
    
    Parameters
    ----------
    params : dict
        sigma, r, lam_jump, eta1, eta2, p, K, L, U, D_bar, T
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Time discretization steps.
    seed : int
        Random seed.
    
    Returns
    -------
    price : float
        Estimated Parisian put price.
    se : float
        Standard error.
    survival_rate : float
        Fraction of paths not knocked out.
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
    T = params.get('T', 1.0)
    
    rng = default_rng(seed)
    dt = T / n_steps
    
    # Expected relative jump size
    zeta = p_jump * eta1/(eta1 - 1) - q_jump * eta2/(eta2 + 1) - 1
    
    # Drift of GBM part
    mu = r - lam_jump*zeta - 0.5*sigma**2
    
    # Initialize
    S = np.full(n_paths, params.get('S0', K))
    excursion = np.zeros(n_paths)
    knocked_out = np.zeros(n_paths, dtype=bool)
    
    for step in range(n_steps):
        if knocked_out.all():
            break
        
        # Active paths
        active = ~knocked_out
        
        # GBM diffusion
        Z = rng.normal(0, 1, size=n_paths)
        dW = np.sqrt(dt) * Z
        
        # Jump component
        # Number of jumps: Poisson(lam_jump * dt)
        n_jumps = rng.poisson(lam_jump * dt, size=n_paths)
        
        # Process each path
        for i in range(n_paths):
            if knocked_out[i]:
                continue
            
            # Diffusion
            S[i] *= np.exp(mu * dt + sigma * dW[i])
            
            # Jumps
            for _ in range(n_jumps[i]):
                u = rng.uniform()
                if u < p_jump:
                    # Upward jump: Y ~ eta1 * exp(-eta1*(y-1)), y > 1
                    jump_size = 1 + rng.exponential(1/eta1)
                else:
                    # Downward jump: Y = e^{-J} where J ~ exponential(eta2)
                    jump_size = np.exp(-rng.exponential(1/eta2))
                S[i] *= jump_size
            
            # Excursion clock
            if S[i] < L or S[i] > U:
                excursion[i] += dt
            
            # Knock-out check
            if excursion[i] >= D_bar:
                knocked_out[i] = True
    
    # Payoff
    payoff = np.where(knocked_out, 0.0, np.maximum(K - S, 0.0))
    discounted = np.exp(-r * T) * payoff
    
    price = np.mean(discounted)
    se = np.std(discounted) / np.sqrt(n_paths)
    survival_rate = 1 - np.mean(knocked_out)
    
    return price, se, survival_rate


def mc_price_grid(params, S_grid, n_paths=50000, n_steps=500, seed=42):
    """Price Parisian put for a grid of initial asset prices."""
    # Run one simulation; use path information to estimate at multiple S0
    # For simplicity, run separate simulations per S0 (small S_grid)
    prices = np.zeros(len(S_grid))
    ses = np.zeros(len(S_grid))
    
    for i, S0 in enumerate(S_grid):
        p = params.copy()
        p['S0'] = S0
        price, se, surv = simulate_kou_parisian(p, n_paths=n_paths, n_steps=n_steps, seed=seed+i)
        prices[i] = price
        ses[i] = se
    
    return prices, ses


if __name__ == '__main__':
    import time
    
    params = {
        'sigma': 0.20, 'r': 0.05, 'lam_jump': 0.5,
        'eta1': 25.0, 'eta2': 20.0, 'p': 0.4,
        'K': 100.0, 'L': 80.0, 'U': 120.0,
        'D_bar': 0.1, 'T': 1.0, 'S0': 100.0,
    }
    
    print("Kou Model Double-Barrier Parisian Put — Monte Carlo")
    print("=" * 60)
    
    zeta = params['p']*params['eta1']/(params['eta1']-1) - (1-params['p'])*params['eta2']/(params['eta2']+1) - 1
    print(f"sigma={params['sigma']}, r={params['r']}, lambda={params['lam_jump']}")
    print(f"eta1={params['eta1']}, eta2={params['eta2']}, p={params['p']}, zeta={zeta:.4f}")
    print(f"K={params['K']}, L={params['L']}, U={params['U']}, D={params['D_bar']}, T={params['T']}")
    
    t0 = time.time()
    price, se, surv = simulate_kou_parisian(params, n_paths=50000, n_steps=500)
    elapsed = time.time() - t0
    
    print(f"\nParisian Put ATM:")
    print(f"  Price = {price:.4f} +/- {se:.4f} (95% CI: [{price-1.96*se:.4f}, {price+1.96*se:.4f}])")
    print(f"  Survival rate = {surv:.1%}")
    print(f"  Wall time = {elapsed:.2f}s")
    
    # European put for comparison (no knock-out)
    print("\nBenchmarks:")
    
    # Kou European put (via characteristic function / Fourier)
    # For now, approximate with BS
    from scipy.stats import norm as scipy_norm
    d1 = (np.log(params['S0']/params['K']) + (params['r'] + 0.5*params['sigma']**2)*params['T']) / (params['sigma']*np.sqrt(params['T']))
    d2 = d1 - params['sigma']*np.sqrt(params['T'])
    bs_put = params['K']*np.exp(-params['r']*params['T'])*scipy_norm.cdf(-d2) - params['S0']*scipy_norm.cdf(-d1)
    print(f"  Black-Scholes European put (no jumps): {bs_put:.4f}")
    print(f"  (Note: Kou European put would be different due to jumps)")
