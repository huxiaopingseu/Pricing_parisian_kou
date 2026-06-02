#!/usr/bin/env python3
"""
Run Monte Carlo benchmark for Kou Parisian options and compare
with Laplace-DeepONet pricing results.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from kou_mc_benchmark import simulate_kou_parisian, mc_price_grid


PARAMS = {
    'sigma': 0.20, 'r': 0.05, 'lam_jump': 0.5,
    'eta1': 25.0, 'eta2': 20.0, 'p': 0.4,
    'K': 100.0, 'L': 80.0, 'U': 120.0,
    'D_bar': 0.1, 'T': 1.0,
}


def run_mc_benchmark(params, n_paths=50000, n_steps=500):
    """Run MC and print results."""
    print("=" * 60)
    print("Kou Model Parisian Put — Monte Carlo Benchmark")
    print("=" * 60)
    
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # Compute zeta
    p_j = params['p']; q_j = 1 - p_j
    zeta = p_j*params['eta1']/(params['eta1']-1) - q_j*params['eta2']/(params['eta2']+1) - 1
    print(f"  zeta = E[Y-1] = {zeta:.6f}")
    
    print(f"\nRunning {n_paths:,} paths x {n_steps} steps...")
    t0 = time.time()
    
    price, se, surv = simulate_kou_parisian(
        params, n_paths=n_paths, n_steps=n_steps, seed=42
    )
    
    elapsed = time.time() - t0
    ci_low = price - 1.96*se
    ci_high = price + 1.96*se
    
    print(f"\nResults:")
    print(f"  Parisian put price (ATM): {price:.6f}")
    print(f"  Standard error:           {se:.6f}")
    print(f"  95% CI:                   [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  Survival rate:            {surv:.2%}")
    print(f"  Wall time:                {elapsed:.1f}s")
    
    # European BS put for comparison
    from scipy.stats import norm
    S0 = params.get('S0', params['K'])
    d1 = (np.log(S0/params['K']) + (params['r']+0.5*params['sigma']**2)*params['T'])/(params['sigma']*np.sqrt(params['T']))
    d2 = d1 - params['sigma']*np.sqrt(params['T'])
    bs_put = params['K']*np.exp(-params['r']*params['T'])*norm.cdf(-d2) - S0*norm.cdf(-d1)
    print(f"\n  Black-Scholes European (no jumps, no barrier): {bs_put:.6f}")
    
    return price, se, surv


def run_mc_grid(params, n_S=11):
    """Price across S grid [70, 130]."""
    S_grid = np.linspace(70, 130, n_S)
    print(f"\n{'='*60}")
    print(f"MC price grid: {n_S} points, S in [{S_grid[0]}, {S_grid[-1]}]")
    print(f"{'='*60}")
    
    prices, ses = mc_price_grid(params, S_grid, n_paths=50000, n_steps=500, seed=42)
    
    for S0, p, s in zip(S_grid, prices, ses):
        print(f"  S={S0:6.1f}:  {p:.6f} +/- {s:.6f}")
    
    return S_grid, prices, ses


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', action='store_true', help='Price across S grid')
    ap.add_argument('--paths', type=int, default=50000, help='Number of MC paths')
    ap.add_argument('--steps', type=int, default=500, help='Time steps')
    args = ap.parse_args()
    
    price, se, surv = run_mc_benchmark(PARAMS, n_paths=args.paths, n_steps=args.steps)
    
    if args.grid:
        run_mc_grid(PARAMS, n_S=11)
