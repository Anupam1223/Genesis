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
        self.image_dir = os.path.join(output_dir, "images")
        self.scaler_path = os.path.join(output_dir, "checkpoints", "scaler.pt")
        
        # 1. DEFINE REQUIRED COLUMNS FIRST
        # We define these before loading so we can selectively read ONLY these from the Excel file
        
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
        
        self.all_required_cols = self.x_cols + self.u_cols + self.theta_cols
        
        # 2. LOAD DATA TO RAM (WITH PARQUET CACHING & COLUMN PRUNING)
        cache_path = data_path.replace('.xlsx', '.parquet').replace('.xls', '.parquet')
        
        if os.path.exists(cache_path):
            print(f"⚡ FAST LOAD: Reading ONLY {len(self.all_required_cols)} columns from cached Parquet (Split: {split.upper()})...")
            # Parquet reads incredibly fast when only requesting specific columns
            raw_df = pd.read_parquet(cache_path, columns=self.all_required_cols)
        else:
            print(f"🐌 SLOW LOAD: Reading ONLY {len(self.all_required_cols)} columns from Excel. This saves massive RAM...")
            
            # Helper function: Excel headers might be messy. We clean them on the fly 
            # to check if they belong in our required list before wasting RAM on them.
            def is_required_col(col_name):
                clean_name = re.sub(r'\s+', '_', str(col_name).strip())
                return clean_name in self.all_required_cols
                
            raw_df = pd.read_excel(data_path, usecols=is_required_col) 
            
            # Clean column names in the loaded dataframe to perfectly match our lists
            raw_df.columns = raw_df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
            
            # Deduplicate column names (Parquet strictly requires unique column names)
            cols = pd.Series(raw_df.columns)
            for dup in cols[cols.duplicated()].unique(): 
                mask = cols == dup
                cols[mask] = [f"{dup}_{i}" if i > 0 else dup for i in range(mask.sum())]
            raw_df.columns = cols
            
            # ---> FIX: Force numeric types BEFORE saving to Parquet! <---
            # This turns strings like "Bad Input" or "Offline" into blank NaNs so Parquet doesn't crash
            print("🧹 Sanitizing textual artifacts (like 'Bad Input') into NaNs before caching...")
            for col in self.all_required_cols:
                if col in raw_df.columns:
                    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            
            # Save the clean, slimmed-down dataframe to a fast binary format for next time
            print(f"💾 Caching slimmed data to {cache_path} for instant loading next time...")
            raw_df.to_parquet(cache_path, index=False)
            
        # 3. CHRONOLOGICAL SPLIT (80% Train / 20% Val)
        split_idx = int(len(raw_df) * 0.8)
        if self.split == 'train':
            df = raw_df.iloc[:split_idx].copy()
        elif self.split == 'val':
            df = raw_df.iloc[split_idx:].copy()
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
        print(f"Loaded {len(df)} rows for {self.split.upper()}.")
        
        # 4. PREPROCESSING & TRACKING
        self._process_and_track(df)

    def _process_and_track(self, df: pd.DataFrame):
        """
        Handles scaling/standardization and triggers the visual tracker.
        """
        # A. Store RAW arrays (Forcing numeric types to prevent np.std crashes)
        # 1. apply(pd.to_numeric) turns text like "Offline" into NaN (Already done, but safe to repeat)
        # 2. ffill() copies the last known good sensor value forward
        # 3. bfill() catches any NaNs at the very beginning of the dataset
        # 4. fillna(0.0) is the absolute final fallback
        raw_theta = df[self.theta_cols].apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0.0).values
        raw_x = df[self.x_cols].apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0.0).values
        raw_u = df[self.u_cols].apply(pd.to_numeric, errors='coerce').ffill().bfill().fillna(0.0).values

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
            # ---> FIX: Add weights_only=False to bypass PyTorch 2.6 security checks <---
            self.scaler_stats = torch.load(self.scaler_path, weights_only=False)
            print(f"🔄 Loaded existing scaler state from {self.scaler_path}")

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