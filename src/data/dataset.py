import os
import re
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
        
        # 1. DEFINE COLUMNS TO EXTRACT
        self.x_cols = [
            'COMP_Suction_Pressure', 
            'COMP_Suction_Drum_Temperature', 
            'KPI_Fuel_Gas_Lower_Heating_Value'
        ]
        
        self.u_cols = [
            'Turbine_SHAFT_SPEED', 
            'UK_14PDCV-504_H-SEL', 
            'SEAL_GAS_SUP_DE'
        ]
        
        # theta consists entirely of the pre-computed scenario projection components
        self.theta_cols = [f'PCA_Coefficient_{i+1}' for i in range(12)]
        
        self.all_required_cols = self.x_cols + self.u_cols + self.theta_cols
        
        # 2. FAST LOAD PREPROCESSED PARQUET
        # Construct the target file path based on the split
        base_dir = os.path.dirname(data_path)
        split_path = os.path.join(base_dir, f"{self.split}.parquet")
        
        print(f"⚡ FAST LOAD: Reading {len(self.all_required_cols)} columns from {split_path}...")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_path} not found. Make sure to run src/data/preprocessing.py first.")
        
        # The parquet is already pre-windowed, pre-scaled, pre-computed, AND pre-split!
        df = pd.read_parquet(split_path, columns=self.all_required_cols)
            
        print(f"Loaded {len(df)} rows for {self.split.upper()}.")
        
        # 4. PREPROCESSING & TRACKING
        self._process_and_track(df)

    def _process_and_track(self, df: pd.DataFrame):
        """
        Loads the preprocessed DataFrame into PyTorch tensors.
        Leaves tracking functionality untouched since it's already stripped out of this layer.
        """
        # Data is already float32 ready, scaled, and clean from the preprocessing script
        raw_theta = df[self.theta_cols].values
        raw_x = df[self.x_cols].values
        raw_u = df[self.u_cols].values

        # Convert to PyTorch Tensors
        self.x_tensor = torch.tensor(raw_x, dtype=torch.float32)
        self.u_tensor = torch.tensor(raw_u, dtype=torch.float32)
        self.theta_tensor = torch.tensor(raw_theta, dtype=torch.float32)
        
        # Combine conditions into a single tensor for the "Brain" (MLP)
        self.condition_tensor = torch.cat([self.x_tensor, self.u_tensor], dim=-1)

    def __len__(self):
        return len(self.theta_tensor)

    def __getitem__(self, idx):
        return {
            "theta": self.theta_tensor[idx],
            "condition": self.condition_tensor[idx]
        }