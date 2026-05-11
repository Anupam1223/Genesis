import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE BRAIN: PHYSICS-INFORMED RESIDUAL MLP
# ==========================================
class ResidualMLP(nn.Module):
    """
    The 'Brain' of the Coupling Layer.
    Uses GELU for smooth physics gradients and Residual skip connections.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout_rate=0.1):
        super().__init__()
        
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        
        # Build residual blocks with Dropout
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # SLIDE 8: The Output Layer
        # Instead of 's' and 't', this will now output (3K - 1) raw numbers per dimension!
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize to zero so the flow starts as an Identity function
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        out = self.dropout(self.activation(self.initial_layer(x)))
        for block in self.blocks:
            out = self.activation(block(out) + out)
        return self.final_layer(out)


# ==========================================
# 2. THE MATH ENGINE: RATIONAL-QUADRATIC SPLINE
# ==========================================
def rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, 
                              unnormalized_derivatives, inverse=False, 
                              bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
    """
    SLIDE 13 & 14: The Continuous Spline Math Engine.
    Transforms inputs -> outputs using the exact algebraic cheat codes.
    """
    num_bins = unnormalized_widths.shape[-1]
    
    # ----------------------------------------------------
    # STEP 1: APPLY CONSTRAINTS (Slide 9)
    # Softmax forces widths and heights to sum perfectly to 2 * bound (e.g., 10.0)
    # Softplus forces slopes (D) to be strictly positive!
    # ----------------------------------------------------
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    
    # Pad derivatives with 1.0 at the absolute outer boundaries to connect smoothly to linear tails
    pad = torch.ones_like(derivatives[..., :1])
    derivatives = torch.cat([pad, derivatives, pad], dim=-1)

    # ----------------------------------------------------
    # STEP 2: BUILD THE SCAFFOLDING / KNOTS (Slide 10)
    # ----------------------------------------------------
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (bound * 2.0) * cumwidths / cumwidths[..., -1:] - bound # Map from 0 to [-bound, bound]
    
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (bound * 2.0) * cumheights / cumheights[..., -1:] - bound
    
    cumwidths[..., 0], cumwidths[..., -1] = -bound, bound
    cumheights[..., 0], cumheights[..., -1] = -bound, bound

    # ----------------------------------------------------
    # STEP 3: FIND THE BIN & LOCAL COORDS (Slide 11 & 12)
    # ----------------------------------------------------
    if not inverse:
        # Forward pass: Search the X-axis (widths)
        bin_idx = torch.searchsorted(cumwidths, inputs.unsqueeze(-1)).squeeze(-1) - 1
    else:
        # Inverse pass: Search the Y-axis (heights)
        bin_idx = torch.searchsorted(cumheights, inputs.unsqueeze(-1)).squeeze(-1) - 1

    # Edge cases (catch values exactly on boundaries)
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    # Gather the specific W, H, D for the bin our point landed in
    input_shape = inputs.shape
    bin_idx_expanded = bin_idx.unsqueeze(-1)
    
    W_k = torch.gather(widths, -1, bin_idx_expanded).squeeze(-1) * (bound * 2.0)
    H_k = torch.gather(heights, -1, bin_idx_expanded).squeeze(-1) * (bound * 2.0)
    D_0 = torch.gather(derivatives, -1, bin_idx_expanded).squeeze(-1)
    D_1 = torch.gather(derivatives, -1, bin_idx_expanded + 1).squeeze(-1)
    
    start_x = torch.gather(cumwidths, -1, bin_idx_expanded).squeeze(-1)
    start_y = torch.gather(cumheights, -1, bin_idx_expanded).squeeze(-1)

    # Helper Variable: Straight line slope (S)
    S = H_k / W_k

    # ----------------------------------------------------
    # STEP 4: THE ALGEBRA & CALCULUS (Slide 14 & 15)
    # ----------------------------------------------------
    if not inverse:
        # Local coordinate (Progress Bar)
        xi = (inputs - start_x) / W_k

        # Calculate the Polynomial Math (c_1 to c_6 inside the equation)
        numerator = H_k * (S * xi**2 + D_0 * xi * (1 - xi))
        denominator = S + (D_1 + D_0 - 2 * S) * xi * (1 - xi)
        
        # Final Output Y
        outputs = start_y + numerator / denominator
        
        # Calculus: Quotient Rule exact derivative for Volume Penalty
        derivative_numerator = S**2 * (D_1 * xi**2 + 2 * S * xi * (1 - xi) + D_0 * (1 - xi)**2)
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        
        return outputs, logabsdet
    
    else:
        # The Inverse Algebra Trick: Calculate a, b, c for Quadratic Equation
        y_shifted = inputs - start_y
        
        a = H_k * (S - D_0) + y_shifted * (D_1 + D_0 - 2 * S)
        b = H_k * D_0 - y_shifted * (D_1 + D_0 - 2 * S)
        c = -S * y_shifted
        
        # High-school Quadratic Formula: (-b ± sqrt(b² - 4ac)) / 2a
        discriminant = b.pow(2) - 4 * a * c
        
        # Numerical stability check for discriminant
        discriminant = torch.clamp(discriminant, min=0.0)
        
        # Solve for xi
        xi = (2 * c) / (-b - torch.sqrt(discriminant)) 
        
        outputs = start_x + xi * W_k
        
        # Derivative (Calculated same as forward, but inverted)
        derivative_numerator = S**2 * (D_1 * xi**2 + 2 * S * xi * (1 - xi) + D_0 * (1 - xi)**2)
        denominator = S + (D_1 + D_0 - 2 * S) * xi * (1 - xi)
        logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))
        
        return outputs, logabsdet


# ==========================================
# 3. THE MUSCLE: NEURAL SPLINE COUPLING
# ==========================================
class NeuralSplineCouplingLayer(nn.Module):
    """
    The orchestrator for a single Spline Layer.
    Splits the array, runs the MLP, and calls the Math Engine.
    """
    def __init__(self, dim_theta, dim_condition, hidden_dim=128, num_bins=8, bound=5.0):
        super().__init__()
        
        self.half_dim = dim_theta // 2
        self.num_bins = num_bins
        self.bound = bound
        
        brain_input_dim = self.half_dim + dim_condition
        
        # For each dimension in Half B, we need (3 * K - 1) parameters!
        # K widths, K heights, K-1 slopes
        self.params_per_dim = (3 * num_bins) - 1
        brain_output_dim = (dim_theta - self.half_dim) * self.params_per_dim
        
        self.brain = ResidualMLP(brain_input_dim, hidden_dim, brain_output_dim)

    def forward(self, theta, condition):
        # 1. THE SPLIT
        theta_1, theta_2 = theta[:, :self.half_dim], theta[:, self.half_dim:]
        
        # 2. THE BRAIN
        brain_input = torch.cat([theta_1, condition], dim=-1)
        raw_params = self.brain(brain_input)
        
        # Reshape so we have exactly (Batch, Dims, 3K-1)
        raw_params = raw_params.reshape(-1, theta_2.shape[-1], self.params_per_dim)
        
        # 3. SLICE THE PARAMS
        unnormalized_widths = raw_params[..., :self.num_bins]
        unnormalized_heights = raw_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = raw_params[..., 2*self.num_bins:]
        
        # 4. IDENTITY TAILS (Check Bounding Box)
        # If the input is outside [-bound, bound], it stays exactly the same (linear)
        inside_mask = (theta_2 > -self.bound) & (theta_2 < self.bound)
        
        outputs = torch.clone(theta_2)
        logabsdet = torch.zeros_like(theta_2)
        
        # Only transform points inside the box!
        if inside_mask.any():
            spline_out, spline_logdet = rational_quadratic_spline(
                inputs=theta_2[inside_mask],
                unnormalized_widths=unnormalized_widths[inside_mask, :],
                unnormalized_heights=unnormalized_heights[inside_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_mask, :],
                inverse=False,
                bound=self.bound
            )
            outputs[inside_mask] = spline_out
            logabsdet[inside_mask] = spline_logdet
        
        # 5. FINAL ASSEMBLY
        y_final = torch.cat([theta_1, outputs], dim=-1)
        
        # Sum the Log Jacobian Determinants for the Volume Penalty!
        total_log_det = logabsdet.sum(dim=-1)
        
        return y_final, total_log_det

    def inverse(self, z, condition):
        # INVERSE GENERATION PASS
        z_1, z_2 = z[:, :self.half_dim], z[:, self.half_dim:]
        
        brain_input = torch.cat([z_1, condition], dim=-1)
        raw_params = self.brain(brain_input)
        raw_params = raw_params.reshape(-1, z_2.shape[-1], self.params_per_dim)
        
        unnormalized_widths = raw_params[..., :self.num_bins]
        unnormalized_heights = raw_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = raw_params[..., 2*self.num_bins:]
        
        inside_mask = (z_2 > -self.bound) & (z_2 < self.bound)
        
        outputs = torch.clone(z_2)
        
        if inside_mask.any():
            spline_out, _ = rational_quadratic_spline(
                inputs=z_2[inside_mask],
                unnormalized_widths=unnormalized_widths[inside_mask, :],
                unnormalized_heights=unnormalized_heights[inside_mask, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_mask, :],
                inverse=True,
                bound=self.bound
            )
            outputs[inside_mask] = spline_out
            
        return torch.cat([z_1, outputs], dim=-1)