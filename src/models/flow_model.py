import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .components import ResidualMLP, ConditionalAffineCouplingLayer

# ==========================================
# 3. THE WRAPPER: THE NORMALIZING FLOW
# ==========================================
class PipelineConditionalFlow(nn.Module):
    """
    The orchestrator. Stacks multiple coupling layers, handles the array
    swapping between layers, and calculates the final Log-Likelihood Loss.
    """
    def __init__(self, dim_theta, dim_condition, num_layers=6, hidden_dim=128):
        super().__init__()
        
        self.dim_theta = dim_theta
        
        # Define the pre-chosen Blueprint (Standard Normal Bell Curve)
        # We do this here so it lives on the correct hardware device (CPU/GPU)
        self.register_buffer('blueprint_loc', torch.zeros(dim_theta))
        self.register_buffer('blueprint_cov', torch.eye(dim_theta))
        
        # Stack multiple separate layers
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingLayer(dim_theta, dim_condition, hidden_dim)
            for _ in range(num_layers)
        ])

    def get_blueprint(self):
        return MultivariateNormal(self.blueprint_loc, self.blueprint_cov)

    def forward(self, theta, condition):
        """
        TRAINING PASS: Takes uncertain future (theta) and pushes it to Z.
        Returns the Z coordinates and the Total Accumulated Volume Penalty.
        """
        total_log_det = 0
        z = theta
        
        for i, layer in enumerate(self.layers):
            # Pass through the Coupling Layer
            z, log_det = layer(z, condition)
            total_log_det += log_det
            
            # [SLIDE 38, STEP 3: THE SWAP]
            # Swap Half A and Half B by rolling the tensor elements
            # so the next layer warps the OTHER half.
            z = torch.roll(z, shifts=self.dim_theta // 2, dims=-1)
            
        return z, total_log_det

    def compute_loss(self, theta, condition):
        """
        [SLIDE 38, STEP 5: FINAL OUTPUT & LOSS]
        """
        # 1. Run the entire forward relay race
        z_final, total_volume_penalty = self.forward(theta, condition)
        
        # 2. Evaluate Blueprint Score (Is Z a Bell Curve?)
        blueprint = self.get_blueprint()
        blueprint_score = blueprint.log_prob(z_final)
        
        # 3. Total Loss Objective (Minimize to zero)
        loss = -1 * (blueprint_score + total_volume_penalty)
        
        # Return the mean loss across the batch
        return loss.mean()

    def sample(self, num_samples, condition):
        """
        INFERENCE / MPC PASS: Run instantly on edge hardware to simulate the future.
        """
        batch_size = condition.shape[0]
        
        # Grab random clay from the Blueprint Bell Curve
        blueprint = self.get_blueprint()
        z = blueprint.sample((batch_size,))
        
        # Run the relay race BACKWARD through the layers
        for layer in reversed(self.layers):
            # Undo the swap first!
            z = torch.roll(z, shifts=-(self.dim_theta // 2), dims=-1)
            # Run the inverse math
            z = layer.inverse(z, condition)
            
        # Z has now morphed back into simulated real-world future (Theta)
        return z