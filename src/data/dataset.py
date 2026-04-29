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
    def __init__(self, data_path: str, split: str = 'train', log_to_wandb: bool = False, output_dir: str = "outputs"):
        super().__init__()
        self.split = split
        self.log_to_wandb = log_to_wandb
        self.image_dir = os.path.join(output_dir, "images")
        self.scaler_path = os.path.join(output_dir, "checkpoints", "scaler.pt")
        
        # 1. LOAD DATA TO RAM
        print(f"Loading actual data from {data_path} into M4 Max Unified Memory (Split: {split.upper()})...")
        raw_df = pd.read_csv(data_path) 
        
        # Clean column names (strip hidden spaces, replace internal spaces with underscores)
        raw_df.columns = raw_df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        
        # 2. CHRONOLOGICAL SPLIT (80% Train / 20% Val)
        split_idx = int(len(raw_df) * 0.8)
        if self.split == 'train':
            df = raw_df.iloc[:split_idx].copy()
        elif self.split == 'val':
            df = raw_df.iloc[split_idx:].copy()
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        print(f"Loaded {len(df)} rows for {self.split.upper()}.")
        
        # 3. SEPARATE THE 3 PARTITIONS
        # x: Measured Now (100% Certain Context)
        self.x_cols = [
            'COMP_Suction_Pressure', 
            'COMP_Suction_Drum_Temperature', 
            'KPI_Fuel_Gas_Lower_Heating_Value'
        ]
        
        # u: Controls (Operator Dials)
        self.u_cols = [
            'Turbine_SHAFT_SPEED', 
            'UK_14PDCV-504_H-SEL', 
            'SEAL_GAS_SUP_DE'
        ]
        
        # theta: Uncertain Future (The variables we want to simulate/warp)
        self.theta_cols = [
            'SEAL_GAS_FLTR_DP', 
            'LUBE_OIL_LVL_XMTR_HI/LO_TNK', 
            'KPI_Turbine_Overall_Thermal_Cycle_Efficiency', 
            'KPI_Gas_COMP_Isentropic_Efficiency', 
            'COMP_Discharge_Pressure', 
            'COMP_Discharge_Temp', 
            'Exhaust_Temp_Spread_1',
            'KPI_Turbine_Heat_Rate'
        ]
        
        # 4. PREPROCESSING & TRACKING
        self._process_and_track(df)

    def _process_and_track(self, df: pd.DataFrame):
        """
        Handles scaling/standardization and triggers the visual tracker.
        """
        # A. Store RAW arrays
        raw_theta = df[self.theta_cols].values
        raw_x = df[self.x_cols].values
        raw_u = df[self.u_cols].values

        # B. Apply Standard Scaling (Z-Score Normalization) WITHOUT DATA LEAKAGE
        if self.split == 'train':
            # Calculate the math ONLY on training data
            self.scaler_stats = {
                'x_mean': np.mean(raw_x, axis=0), 'x_std': np.std(raw_x, axis=0) + 1e-8,
                'u_mean': np.mean(raw_u, axis=0), 'u_std': np.std(raw_u, axis=0) + 1e-8,
                'theta_mean': np.mean(raw_theta, axis=0), 'theta_std': np.std(raw_theta, axis=0) + 1e-8
            }
            # Save the math to disk so Edge devices and Validation loops can use it!
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            torch.save(self.scaler_stats, self.scaler_path)
            print(f"💾 Scaler state saved to {self.scaler_path}")
        else:
            # If we are Validation (or Inference), load the math calculated by the training phase!
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Run training first!")
            self.scaler_stats = torch.load(self.scaler_path)
            print("🔄 Loaded existing scaler state.")

        # Actually apply the scaling math
        self.x_data = (raw_x - self.scaler_stats['x_mean']) / self.scaler_stats['x_std']
        self.u_data = (raw_u - self.scaler_stats['u_mean']) / self.scaler_stats['u_std']
        self.theta_data = (raw_theta - self.scaler_stats['theta_mean']) / self.scaler_stats['theta_std']

        # C. Convert to PyTorch Tensors (Stored in CPU RAM)
        self.x_tensor = torch.tensor(self.x_data, dtype=torch.float32)
        self.u_tensor = torch.tensor(self.u_data, dtype=torch.float32)
        self.theta_tensor = torch.tensor(self.theta_data, dtype=torch.float32)
        
        # D. Combine conditions into a single tensor for the "Brain" (MLP)
        self.condition_tensor = torch.cat([self.x_tensor, self.u_tensor], dim=-1)

        # E. VISUAL TRACKING (Only generate plots during training phase to save time)
        if self.split == 'train':
            self._log_data_transformations(raw_theta, self.theta_data)

    def _log_data_transformations(self, raw_data, scaled_data):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(raw_data[:200, 0], color='#ef4444', linewidth=1.5)
        axes[0].set_title("Raw SCADA Data (Before)", fontweight="bold")
        axes[0].set_ylabel("Original Units")
        axes[0].set_xlabel("Timesteps")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        axes[1].plot(scaled_data[:200, 0], color='#3b82f6', linewidth=1.5)
        axes[1].set_title("Standardized Data (After)", fontweight="bold")
        axes[1].set_ylabel("Z-Score (Mean=0, Std=1)")
        axes[1].set_xlabel("Timesteps")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        os.makedirs(self.image_dir, exist_ok=True)
        local_path = os.path.join(self.image_dir, "data_transformation_step.png")
        plt.savefig(local_path, dpi=300, bbox_inches='tight')
        
        if self.log_to_wandb:
            wandb.log({"Data_Transform_Visual": wandb.Image(fig)})
            
        plt.close(fig)

    def __len__(self):
        return len(self.theta_tensor)

    def __getitem__(self, idx):
        return {
            "theta": self.theta_tensor[idx],
            "condition": self.condition_tensor[idx]
        }