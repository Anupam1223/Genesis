import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# IMPORTANT: We are now importing the Neural Spline layer instead of Affine
from .components import ResidualMLP, NeuralSplineCouplingLayer

# ==========================================
# 3. THE WRAPPER: THE NORMALIZING FLOW
# ==========================================
class PipelineConditionalFlow(nn.Module):
    """
    The orchestrator. Stacks multiple Neural Spline coupling layers, handles 
    the array swapping between layers, and calculates the final Log-Likelihood Loss.
    """
    def __init__(self, dim_theta, dim_condition, num_layers=6, hidden_dim=128, num_bins=8, bound=5.0,
                 mlp_layers=4, dropout_rate=0.1):
        super().__init__()
        
        self.dim_theta = dim_theta
        self.bound = bound
        
        # Define the pre-chosen Blueprint (Standard Normal Bell Curve: N(0, 1))
        # We do this here so it lives on the correct hardware device (CPU/GPU/MPS)
        self.register_buffer('blueprint_loc', torch.zeros(dim_theta))
        self.register_buffer('blueprint_cov', torch.eye(dim_theta))
        
        # Stack multiple Neural Spline Coupling Layers
        # We pass down the specific Spline hyperparameters (num_bins, bound)
        self.layers = nn.ModuleList([
            NeuralSplineCouplingLayer(
                dim_theta=dim_theta, 
                dim_condition=dim_condition, 
                hidden_dim=hidden_dim,
                num_bins=num_bins,
                bound=bound,
                mlp_layers=mlp_layers,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ])

    def get_blueprint(self):
        """Returns the continuous Bell Curve distribution for grading."""
        return MultivariateNormal(self.blueprint_loc, self.blueprint_cov)

    def forward(self, theta, condition):
        """
        TRAINING PASS: Takes uncertain future (theta) and pushes it to Z.
        Returns the Z coordinates and the Total Accumulated Volume Penalty.
        """
        total_log_det = 0
        z = theta
        
        for i, layer in enumerate(self.layers):
            # Pass through the Spline Coupling Layer
            z, log_det = layer(z, condition)
            total_log_det += log_det
            
            # [SLIDE 2: THE SWAP]
            # Match the diagram perfectly: Flip the tensor array entirely 
            # so the variables in Half B become the new Half A!
            # .contiguous() is REQUIRED on MPS: torch.flip() returns a non-contiguous
            # tensor, which causes torch.searchsorted to produce NaN in the spline math.
            z = torch.flip(z, dims=[-1]).contiguous()
            
        return z, total_log_det

    def compute_loss(self, theta, condition):
        """
        [SLIDE 16: FINAL OUTPUT & 3-STEP LOSS]
        """
        # 1. Run the entire forward relay race through the Splines
        z_final, total_volume_penalty = self.forward(theta, condition)
        
        # 2. Evaluate Blueprint Score (Is Z_final landing in the fat part of the Bell Curve?)
        blueprint = self.get_blueprint()
        blueprint_score = blueprint.log_prob(z_final)
        
        # 3. NLL objective — minimise negative log-likelihood
        loss = -1 * (blueprint_score + total_volume_penalty)
        
        return loss.mean()

    def sample(self, num_samples, condition):
        """
        INFERENCE / GENERATION PASS: Run instantly on edge hardware to simulate the future.
        Uses the exact Algebraic Quadratic Formula trick to instantly reverse the flow.
        """
        batch_size = condition.shape[0]
        
        # Grab random noise (Y) from the Blueprint Bell Curve
        blueprint = self.get_blueprint()
        z = blueprint.sample((batch_size,))
        
        # Run the relay race BACKWARD through the layers
        for layer in reversed(self.layers):
            # Undo the array flip first!
            # .contiguous() required: flip returns non-contiguous tensor on MPS
            z = torch.flip(z, dims=[-1]).contiguous()
            
            # Run the inverse Spline algebra
            z = layer.inverse(z, condition)
            
        # Z has now morphed back into simulated real-world SCADA futures (Theta)
        return z