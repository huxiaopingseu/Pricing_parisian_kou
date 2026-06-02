# Kou Double-Exponential Jump-Diffusion Parisian Option Pricing via 2D Laplace + DeepONet

Extension of the DeepONet-Laplace pricing framework to the Kou (2002) model with hyper-exponential jumps.

## Pipeline

1. **Symbolic Derivation** → PIDE → 4th-order ODE via hyper-exponential ODE-ification
2. **ODE Solver** → 3-region domain with C^3 continuity at barriers
3. **DeepONet** → learns operator (lam1, lam2) → V_hat(S), complex-aware training
4. **2D Fourier Inversion** → recovers option price V(S, t, J)
5. **MC Benchmark** → Kou jump simulation + Parisian clock

## Files

| File | Purpose |
|------|---------|
| `symbolic_ode.py` | Sympy derivation: PIDE coefficients → 4th-order ODE |
| `kou_ode_solver.py` | Numerical 4th-order ODE solver + direct PIDE verification |
| `deeponet_kou.py` | DeepONet architecture (real + complex-aware) |
| `run_pipeline.py` | Full 3-stage pipeline (data → train → invert) |
| `kou_mc_benchmark.py` | Kou model Monte Carlo with Parisian clock |
| `run_mc.py` | MC benchmark runner + comparison |

## Quick Start

```bash
pip install -r requirements.txt

# Verify: ODE solver correctness
python kou_ode_solver.py

# Quick test: mini pipeline
python run_pipeline.py --quick

# Full pipeline (1500 samples, 5000 epochs)
python run_pipeline.py --full

# Monte Carlo benchmark
python run_mc.py
```

## Key mathematical identity

The hyper-exponential jump distribution allows ODE-ification:
```
I[f](x) = ∫ f(x+z) η e^{-ηz} dz  →  I = η (D + η)^{-1}
```
Applying (D+η_1)(η_2-D) to the entire PIDE eliminates the integral term,
converting the non-local PIDE into a local 4th-order ODE with piecewise-constant coefficients.

## Parameters

See `run_pipeline.py` MODEL_PARAMS for default configuration.
