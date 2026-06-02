#!/usr/bin/env python3
"""
Deep Operator Network (DeepONet) for learning the solution operator
(lam1, lam2) -> V_hat(S) of the Kou parametric ODE.

Architecture: branch net encodes lambda parameters, trunk net encodes S.
Supports complex-aware training with dual output channels.
"""

import numpy as np
import torch
import torch.nn as nn


class DeepONet(nn.Module):
    """Deep Operator Network for parametric ODE solution operator."""
    
    def __init__(self, branch_layers, trunk_layers, n_basis=80,
                 complex_output=False):
        """
        Parameters
        ----------
        branch_layers : list
            Hidden layer sizes for branch net (input: 4 for lam params).
        trunk_layers : list
            Hidden layer sizes for trunk net (input: 1 for S coord).
        n_basis : int
            Number of basis functions p.
        complex_output : bool
            If True, output 2 channels (real, imag) for complex lambda.
        """
        super().__init__()
        self.complex_output = complex_output
        n_output = n_basis * (2 if complex_output else 1)
        
        # Branch network: lambda params -> latent code
        branch_dims = [4] + list(branch_layers) + [n_output]
        branch = []
        for i in range(len(branch_dims) - 1):
            branch.append(nn.Linear(branch_dims[i], branch_dims[i+1]))
            if i < len(branch_dims) - 2:
                branch.append(nn.Tanh())
        self.branch = nn.Sequential(*branch)
        
        # Trunk network: S coordinate -> basis functions
        trunk_dims = [1] + list(trunk_layers) + [n_basis]
        trunk = []
        for i in range(len(trunk_dims) - 1):
            trunk.append(nn.Linear(trunk_dims[i], trunk_dims[i+1]))
            if i < len(trunk_dims) - 2:
                trunk.append(nn.Tanh())
        self.trunk = nn.Sequential(*trunk)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
        if complex_output:
            self.bias_imag = nn.Parameter(torch.zeros(1))
    
    def forward(self, lam_params, S_coords):
        """
        lam_params: (batch, 4) — [Re(lam1), Im(lam1), Re(lam2), Im(lam2)]
        S_coords:   (n_spatial,) — spatial grid
        
        Returns: (batch, n_spatial) or (batch, n_spatial, 2) if complex_output
        """
        branch_out = self.branch(lam_params)    # (batch, n_basis * channels)
        trunk_out = self.trunk(S_coords.unsqueeze(-1))  # (n_spatial, n_basis)
        
        if self.complex_output:
            n_basis = trunk_out.shape[-1]
            branch_real = branch_out[:, :n_basis]   # (batch, n_basis)
            branch_imag = branch_out[:, n_basis:]   # (batch, n_basis)
            
            real_part = branch_real @ trunk_out.T + self.bias       # (batch, n_spatial)
            imag_part = branch_imag @ trunk_out.T + self.bias_imag  # (batch, n_spatial)
            return torch.stack([real_part, imag_part], dim=-1)     # (batch, n_spatial, 2)
        else:
            output = branch_out @ trunk_out.T + self.bias  # (batch, n_spatial)
            return output
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class DeepONetTrainer:
    """Training harness for DeepONet on Kou parametric ODE data."""
    
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=1e-6):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1500, gamma=0.5
        )
        self.mse = nn.MSELoss()
    
    def train_epoch(self, lam_batch, S_grid, psi_batch):
        """
        lam_batch: (B, 4) lambda parameters
        S_grid:    (N,)  spatial grid [L, U] — normalized to [0,1]
        psi_batch: (B, N) or (B, N, 2) ground truth
        
        Returns: loss value.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        lam_t = torch.FloatTensor(lam_batch).to(self.device)
        S_t = torch.FloatTensor(S_grid).to(self.device)
        psi_t = torch.FloatTensor(psi_batch).to(self.device)
        
        pred = self.model(lam_t, S_t)
        loss = self.mse(pred, psi_t)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, lam_batch, S_grid, psi_batch):
        """Compute relative L2 error."""
        self.model.eval()
        lam_t = torch.FloatTensor(lam_batch).to(self.device)
        S_t = torch.FloatTensor(S_grid).to(self.device)
        psi_t = torch.FloatTensor(psi_batch).to(self.device)
        
        pred = self.model(lam_t, S_t)
        
        if self.model.complex_output:
            # Complex norm
            err = torch.norm(pred - psi_t, dim=(1, 2))
            norm = torch.norm(psi_t, dim=(1, 2))
        else:
            err = torch.norm(pred - psi_t, dim=1)
            norm = torch.norm(psi_t, dim=1)
        
        rel_err = (err / (norm + 1e-10)).cpu().numpy()
        return rel_err
    
    def predict(self, lam_params, S_grid):
        """Single forward pass."""
        self.model.eval()
        with torch.no_grad():
            lam_t = torch.FloatTensor(lam_params).to(self.device).unsqueeze(0)
            S_t = torch.FloatTensor(S_grid).to(self.device)
            out = self.model(lam_t, S_t)
        return out.squeeze(0).cpu().numpy()


def create_deeponet(complex_output=False):
    """Factory: create standard DeepONet architecture for Kou ODE."""
    return DeepONet(
        branch_layers=[128, 256, 128],
        trunk_layers=[128, 256, 128],
        n_basis=80,
        complex_output=complex_output
    )


if __name__ == '__main__':
    # Quick smoke test
    print("DeepONet architecture test...")
    
    model_real = create_deeponet(complex_output=False)
    print(f"  Real-only model: {model_real.count_params():,} parameters")
    
    model_complex = create_deeponet(complex_output=True)
    print(f"  Complex model:  {model_complex.count_params():,} parameters")
    
    # Test forward pass
    B, N = 8, 200
    lam = torch.randn(B, 4)
    S = torch.linspace(0, 1, N)
    
    out_real = model_real(lam, S)
    print(f"  Real output shape: {out_real.shape}")  # (8, 200)
    
    out_complex = model_complex(lam, S)
    print(f"  Complex output shape: {out_complex.shape}")  # (8, 200, 2)
    
    print("  OK — architecture functional")
