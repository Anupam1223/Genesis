import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the root directory to path so python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow

def main():
    # --- CONFIGURATION ---
    DATA_PATH = "data/raw/DataAllParts.xlsx"
    CHECKPOINT_PATH = "outputs/checkpoints/model_best.pt"
    SCALER_PATH = "outputs/checkpoints/scaler.pt"
    IMAGE_DIR = "outputs/images"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Starting Evaluation on {device.upper()}...")

    # 1. LOAD TEST DATA
    # Industry standard: Evaluate on completely unseen holdout TEST data
    print("Loading Test Dataset...")
    val_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='test', log_to_wandb=False)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # 2. LOAD SCALER (To un-scale predictions back to real PSI/Temp)
    scaler_stats = torch.load(SCALER_PATH, weights_only=False)
    theta_mean = scaler_stats['theta_mean']
    theta_std = scaler_stats['theta_std']

    # 3. INITIALIZE MODEL
    sample = val_dataset[0]
    dim_theta = sample['theta'].shape[0]
    dim_condition = sample['condition'].shape[0]
    
    model = PipelineConditionalFlow(dim_theta=dim_theta, dim_condition=dim_condition, num_layers=6)
    
    # 4. LOAD THE BEST WEIGHTS
    print(f"Loading Best Model Weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # Lock the model

    # 5. CONDITIONAL PROBABILITY DENSITY EVALUATION
    print("Evaluating specific Conditional Distributions...")
    
    # Select a random specific timestep from the validation set
    sample_idx = 150 # You can change this to look at different moments in time
    sample_data = val_dataset[sample_idx]
    
    single_condition = sample_data['condition'].to(device)
    true_theta_scaled = sample_data['theta'].cpu().numpy()
    
    # To see the PDF, we simulate what the Flow thinks COULD happen exactly at this moment
    NUM_SAMPLES = 5000
    
    # Repeat the identical condition 5000 times
    condition_batch = single_condition.unsqueeze(0).repeat(NUM_SAMPLES, 1)
    
    with torch.no_grad():
        # Generate 5000 different possible futures for this single context!
        pred_theta_batch = model.sample(num_samples=NUM_SAMPLES, condition=condition_batch)
        pred_theta_scaled = pred_theta_batch.cpu().numpy()

    # 6. UN-SCALE THE DATA (Convert Z-Scores back to real physical units)
    print("Un-scaling predictions back to real physical units...")
    
    # Ensure scaler stats are numpy arrays (they might be tensors if saved via torch.save)
    tm = theta_mean.cpu().numpy() if torch.is_tensor(theta_mean) else theta_mean
    ts = theta_std.cpu().numpy() if torch.is_tensor(theta_std) else theta_std

    true_theta_physical = (true_theta_scaled * ts) + tm
    pred_theta_physical = (pred_theta_scaled * ts) + tm

    # 7. VISUALIZE PROBABILITY DENSITIES
    print("Generating Probability Density Plots for single timestep...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Let's plot ALL 8 variables dynamically to see the full architecture working!
    features_to_plot = [
        (0, 'Seal Gas Filter DP', '#3b82f6'),
        (1, 'Lube Oil Level', '#ef4444'),
        (2, 'Thermal Cycle Efficiency', '#10b981'),
        (3, 'Isentropic Efficiency', '#f59e0b'),
        (4, 'Discharge Pressure (PSI)', '#8b5cf6'),
        (5, 'Discharge Temp', '#ec4899'),
        (6, 'Exhaust Temp Spread 1', '#14b8a6'),
        (7, 'Turbine Heat Rate', '#f43f5e')
    ]

    # Create a 4x2 grid to show all 8 theta variables
    fig, axes = plt.subplots(4, 2, figsize=(18, 16))
    fig.suptitle(f"Predicted Future Probability Density (Timestep {sample_idx})", fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, (idx, name, color) in zip(axes, features_to_plot):
        pred_vals = pred_theta_physical[:, idx]
        true_val = true_theta_physical[idx]
        
        # Plot the predicted probability distribution of futures
        ax.hist(pred_vals, bins=80, density=True, alpha=0.6, color=color, label='Predicted PDF (Flow)')
        
        # Plot the single actual reality that happened
        ax.axvline(true_val, color='black', linestyle='dashed', linewidth=2.5, label='Actual Realized Value')
        
        ax.set_title(f"Target: {name}", fontweight='bold')
        ax.set_ylabel("Probability Density")
        ax.set_xlabel(name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(IMAGE_DIR, "conditional_density_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Evaluation Complete! Conditional Density plots saved to: {plot_path}")

if __name__ == "__main__":
    main()