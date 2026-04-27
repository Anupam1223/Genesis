import torch
import torch.nn as nn

# ==========================================
# 1. THE BRAIN: PHYSICS-INFORMED RESIDUAL MLP
# ==========================================
class ResidualMLP(nn.Module):
    """
    The 'Brain' of the Coupling Layer.
    Uses GELU for smooth physics gradients and Residual skip connections.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        
        # Build residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.activation = nn.GELU()
        
        # [SLIDE 38, STEP 2: ZERO-INIT]
        # The final layer that outputs s and t. 
        # We initialize weights and biases to EXACTLY zero.
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        out = self.activation(self.initial_layer(x))
        
        # Pass through residual blocks (output = F(x) + x)
        for block in self.blocks:
            out = self.activation(block(out) + out)
            
        # Output random garbage initially? NO! Because of zero-init,
        # this safely outputs exactly zeros for s and t at the start.
        return self.final_layer(out)


# ==========================================
# 2. THE MUSCLE: CONDITIONAL AFFINE COUPLING
# ==========================================
class ConditionalAffineCouplingLayer(nn.Module):
    """
    A single gear in the machine. Splits the uncertainty, routes Half A and 
    the Condition to the Brain, and applies the Affine Math to Half B.
    """
    def __init__(self, dim_theta, dim_condition, hidden_dim=128):
        super().__init__()
        
        self.half_dim = dim_theta // 2
        
        # Input to Brain = Half A + Condition (x) + Controls (u)
        brain_input_dim = self.half_dim + dim_condition
        # Output of Brain = s and t vectors for Half B
        brain_output_dim = (dim_theta - self.half_dim) * 2
        
        self.brain = ResidualMLP(brain_input_dim, hidden_dim, brain_output_dim)

    def forward(self, theta, condition):
        # [SLIDE 38, STEP 1: THE SPLIT]
        # theta_1 is Half A, theta_2 is Half B
        theta_1, theta_2 = theta[:, :self.half_dim], theta[:, self.half_dim:]
        
        # [SLIDE 38, STEP 2: THE BRAIN]
        # Concatenate Half A with the Condition (x + u)
        brain_input = torch.cat([theta_1, condition], dim=-1)
        
        # Brain outputs the parameters
        st_params = self.brain(brain_input)
        s, t = st_params.chunk(2, dim=-1)
        
        # Physics Stability Constraint: Prevent extreme stretching
        s = torch.tanh(s) 
        
        # [SLIDE 38, STEP 2: THE MUSCLE (AFFINE MATH)]
        # Note: Condition is NOT warped here. Only theta_2!
        y_1 = theta_1
        y_2 = (theta_2 * torch.exp(s)) + t
        
        # Recombine the array for the next layer
        y_final = torch.cat([y_1, y_2], dim=-1)
        
        # [SLIDE 38, STEP 5: VOLUME PENALTY]
        # The Jacobian Determinant literally simplifies to sum(s)
        log_det = s.sum(dim=-1)
        
        return y_final, log_det

    def inverse(self, z, condition):
        # EDGE HARDWARE SAMPLING (Generation)
        # Reverses the math to generate simulated pipelines states!
        z_1, z_2 = z[:, :self.half_dim], z[:, self.half_dim:]
        
        # The Brain does the exact same forward pass because Z_1 == Theta_1
        brain_input = torch.cat([z_1, condition], dim=-1)
        st_params = self.brain(brain_input)
        s, t = st_params.chunk(2, dim=-1)
        s = torch.tanh(s)
        
        # Inverse Affine Math
        theta_1 = z_1
        theta_2 = (z_2 - t) * torch.exp(-s)
        
        return torch.cat([theta_1, theta_2], dim=-1)
