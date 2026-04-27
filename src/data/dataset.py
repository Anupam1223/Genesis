import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb

class SCADAPipelineDataset(Dataset):
    """
    In-Memory PyTorch Dataset for Emerson SCADA Pipeline Data.
    Optimized for high-RAM Apple Silicon (M4 Max 128GB).
    """
    def __init__(self, data_path: str, is_training: bool = True, log_to_wandb: bool = False, output_dir: str = "outputs/images"):
        super().__init__()
        self.is_training = is_training
        self.log_to_wandb = log_to_wandb
        self.output_dir = output_dir
        
        # 1. LOAD DATA TO RAM
        # With 128GB of Unified Memory, we load the entire SCADA dataset into Pandas instantly.
        print(f"Loading data from {data_path} into M4 Max Unified Memory...")
        raw_df = pd.read_csv(data_path) # Switch to pd.read_parquet() for large prod files
        
        # 2. SEPARATE THE 3 PARTITIONS
        # We explicitly separate the columns based on our Conditional Flow architecture.
        # (Replace these mock column names with your actual Emerson SCADA columns)
        
        # x: Measured Now (100% Certain Context)
        self.x_cols = ['inlet_pressure', 'ambient_temp', 'gas_composition_C1']
        
        # u: Controls (Operator Dials)
        self.u_cols = ['compressor_speed_rpm', 'recycle_valve_open_pct']
        
        # theta: Uncertain Future (The variables we want to simulate/warp)
        self.theta_cols = ['future_peak_demand', 'degradation_factor', 'trip_category']
        
        # 3. PREPROCESSING & TRACKING
        # We process the data and physically save the visualizations of the changes.
        self._process_and_track(raw_df)

    def _process_and_track(self, df: pd.DataFrame):
        """
        Handles scaling/standardization and triggers the visual tracker.
        """
        # A. Store RAW arrays for visualization before we alter them
        raw_theta = df[self.theta_cols].values
        raw_x = df[self.x_cols].values
        raw_u = df[self.u_cols].values

        # B. Apply Standard Scaling (Z-Score Normalization)
        # Note: In production, load a saved StandardScaler fit on the training set
        # to ensure the validation/test data is scaled identically.
        self.x_data = (raw_x - np.mean(raw_x, axis=0)) / (np.std(raw_x, axis=0) + 1e-8)
        self.u_data = (raw_u - np.mean(raw_u, axis=0)) / (np.std(raw_u, axis=0) + 1e-8)
        self.theta_data = (raw_theta - np.mean(raw_theta, axis=0)) / (np.std(raw_theta, axis=0) + 1e-8)

        # C. Convert to PyTorch Tensors (Stored in CPU RAM)
        # float32 is optimal for Apple Metal Performance Shaders (MPS) acceleration
        self.x_tensor = torch.tensor(self.x_data, dtype=torch.float32)
        self.u_tensor = torch.tensor(self.u_data, dtype=torch.float32)
        self.theta_tensor = torch.tensor(self.theta_data, dtype=torch.float32)
        
        # D. Combine conditions into a single tensor for the "Brain" (MLP)
        self.condition_tensor = torch.cat([self.x_tensor, self.u_tensor], dim=-1)

        # E. VISUAL TRACKING
        # Save a picture of the transformation to local disk (and wandb if enabled)
        if self.is_training:
            self._log_data_transformations(raw_theta, self.theta_data)

    def _log_data_transformations(self, raw_data, scaled_data):
        """
        Generates plots comparing raw vs. processed data.
        Saves them locally to outputs/images and optionally uploads to W&B.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Raw Data (e.g., erratic pipeline pressure)
        axes[0].plot(raw_data[:200, 0], color='#ef4444', linewidth=1.5)
        axes[0].set_title("Raw SCADA Data (Before)", fontweight="bold")
        axes[0].set_ylabel("Original Units")
        axes[0].set_xlabel("Timesteps")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Scaled Data (Prepared for the Normalizing Flow)
        axes[1].plot(scaled_data[:200, 0], color='#3b82f6', linewidth=1.5)
        axes[1].set_title("Standardized Data (After)", fontweight="bold")
        axes[1].set_ylabel("Z-Score (Mean=0, Std=1)")
        axes[1].set_xlabel("Timesteps")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # 1. Save Locally
        os.makedirs(self.output_dir, exist_ok=True)
        local_path = os.path.join(self.output_dir, "data_transformation_step.png")
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        print(f"📊 Local transformation image saved to: {local_path}")
        
        # 2. Upload to Cloud (W&B)
        if self.log_to_wandb:
            wandb.log({"Data_Transform_Visual": wandb.Image(fig)})
            print("☁️ Data transformation visualizations synced to Weights & Biases!")
            
        plt.close(fig)

    def __len__(self):
        return len(self.theta_tensor)

    def __getitem__(self, idx):
        """
        Data is entirely pre-cached in RAM. Fetching takes ~0ms.
        """
        return {
            "theta": self.theta_tensor[idx],
            "condition": self.condition_tensor[idx]
        }

# ==========================================
# USAGE EXAMPLE (Run this script directly to test)
# ==========================================
if __name__ == "__main__":
    print("Generating dummy SCADA dataset...")
    # Create a dummy CSV with sinusoidal waves and noise to mimic SCADA
    timesteps = np.linspace(0, 100, 2000)
    pressure_wave = np.sin(timesteps) * 50 + 500 + np.random.randn(2000) * 5
    
    dummy_data = pd.DataFrame({
        'inlet_pressure': pressure_wave,
        'ambient_temp': np.random.randn(2000) * 10 + 70,
        'gas_composition_C1': np.random.rand(2000),
        'compressor_speed_rpm': np.random.randn(2000) * 100 + 3000,
        'recycle_valve_open_pct': np.random.rand(2000) * 100,
        'future_peak_demand': pressure_wave * 1.5 + np.random.randn(2000) * 20,
        'degradation_factor': np.linspace(0, 1, 2000),
        'trip_category': np.random.randint(0, 3, 2000)
    })
    
    os.makedirs("data/raw", exist_ok=True)
    dummy_path = "data/raw/dummy_scada.csv"
    dummy_data.to_csv(dummy_path, index=False)
    
    # Initialize the dataset (This will trigger the visual save!)
    print("Testing SCADAPipelineDataset initialization...")
    dataset = SCADAPipelineDataset(data_path=dummy_path, is_training=True, log_to_wandb=False)
    
    # Fetch a sample
    sample = dataset[0]
    print("\n--- Test Sample Fetch ---")
    print(f"Theta shape: {sample['theta'].shape} (Future Variables)")
    print(f"Condition shape: {sample['condition'].shape} (Measured x + Controls u)")
    print("\nCheck the 'outputs/images' folder to see your data transformation graph!")