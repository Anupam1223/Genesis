import os
import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib
import time

def main():
    print("🚀 Starting SCADA Trajectory Preprocessing...")
    start_time = time.time()

    # --- CONFIGURATION ---
    INPUT_PATH = "data/raw/DataAllParts.parquet"  # Assuming you have the fast parquet version
    OUTPUT_PATH = "data/processed/DataAllParts_PCA.parquet"
    CHECKPOINT_DIR = "outputs/checkpoints"
    
    # 14,400 seconds = 4 hours
    WINDOW_SIZE = 14400 
    # Take 1 point every 60 seconds (Smooths the curve, saves massive RAM/Compute)
    DOWNSAMPLE_RATE = 60 
    N_COMPONENTS = 4

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. DEFINE COLUMNS
    x_cols = [
        'COMP_Suction_Pressure', 
        'COMP_Suction_Drum_Temperature', 
        'KPI_Fuel_Gas_Lower_Heating_Value'
    ]
    u_cols = [
        'Turbine_SHAFT_SPEED', 
        'UK_14PDCV-504_H-SEL', 
        'SEAL_GAS_SUP_DE'
    ]
    theta_cols = [
        'SEAL_GAS_FLTR_DP', 
        'LUBE_OIL_LVL_XMTR_HI/LO_TNK', 
        'KPI_Turbine_Overall_Thermal_Cycle_Efficiency', 
        'KPI_Gas_COMP_Isentropic_Efficiency', 
        'COMP_Discharge_Pressure', 
        'COMP_Discharge_Temp', 
        'Exhaust_Temp_Spread_1',
        'KPI_Turbine_Heat_Rate'
    ]

    # 2. LOAD DATA
    print(f"📦 Loading raw data from {INPUT_PATH}...")
    # If the parquet doesn't exist yet, fall back to excel
    if not os.path.exists(INPUT_PATH):
        print("Parquet not found. Loading Excel (this will be slow)...")
        df = pd.read_excel(INPUT_PATH.replace('.parquet', '.xlsx'))
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    else:
        df = pd.read_parquet(INPUT_PATH)

    # Ensure everything is numeric and fill NaNs
    print("🧹 Cleaning data...")
    for col in x_cols + u_cols + theta_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.ffill().bfill().fillna(0.0)

    # 3. PREPARE THE THETA (TARGET) TRAJECTORIES
    print("⚖️ Scaling Theta variables for PCA...")
    theta_array = df[theta_cols].values
    
    # PCA is extremely sensitive to scale. We must standard-scale the targets 
    # before PCA so Discharge Pressure (1000s) doesn't dominate Efficiency (0.8s).
    theta_scaler = StandardScaler()
    theta_scaled = theta_scaler.fit_transform(theta_array)

    print(f"🪟 Creating {WINDOW_SIZE}-step sliding windows (Virtually)...")
    # MAGIC TRICK 1: sliding_window_view creates rolling windows using ZERO extra memory
    # Shape becomes: (Number_of_Windows, 8_Features, Window_Size)
    windows = np.lib.stride_tricks.sliding_window_view(theta_scaled, window_shape=WINDOW_SIZE, axis=0)
    
    # Swap axes so time is in the middle: (Number_of_Windows, Window_Size, 8_Features)
    windows = np.swapaxes(windows, 1, 2)
    
    # MAGIC TRICK 2: Downsample the trajectory
    print(f"📉 Downsampling trajectories by factor of {DOWNSAMPLE_RATE}...")
    windows_downsampled = windows[:, ::DOWNSAMPLE_RATE, :]
    
    # Flatten the curves into a single long array for PCA
    # Shape: (Number_of_Windows, (14400/60) * 8) -> (M, 1920)
    pca_input = windows_downsampled.reshape(windows.shape[0], -1)

    # 4. FUNCTIONAL PCA COMPRESSION
    print(f"🗜️ Running Incremental PCA to compress 1920-dim curves into {N_COMPONENTS} coefficients...")
    # We use IncrementalPCA so it processes in batches without blowing up your unified memory
    pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=10000)
    theta_pca_features = pca.fit_transform(pca_input)
    
    variance_kept = sum(pca.explained_variance_ratio_) * 100
    print(f"✨ PCA Complete! These {N_COMPONENTS} coefficients capture {variance_kept:.2f}% of the future pipeline shape.")

    # 5. ALIGN AND BUILD THE NEW DATASET
    print("🏗️ Building the final trajectory-encoded dataset...")
    # The number of valid windows is M = N - Window_Size + 1
    M = theta_pca_features.shape[0]
    
    # We take X and U exactly at the *start* of the window
    x_data = df[x_cols].iloc[:M].reset_index(drop=True)
    u_data = df[u_cols].iloc[:M].reset_index(drop=True)
    
    # Create the new Theta DataFrame
    theta_pca_df = pd.DataFrame(
        theta_pca_features, 
        columns=[f'PCA_Coefficient_{i+1}' for i in range(N_COMPONENTS)]
    )

    # Combine them all together
    final_df = pd.concat([x_data, u_data, theta_pca_df], axis=1)

    # 6. EXPORT EVERYTHING
    print(f"💾 Saving processed dataset to {OUTPUT_PATH}...")
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    # Save the models so we can reverse the process during inference
    joblib.dump(theta_scaler, os.path.join(CHECKPOINT_DIR, "theta_base_scaler.pkl"))
    joblib.dump(pca, os.path.join(CHECKPOINT_DIR, "trajectory_pca_model.pkl"))

    elapsed = (time.time() - start_time) / 60
    print(f"✅ Preprocessing finished successfully in {elapsed:.2f} minutes!")
    print(f"   Original rows: {len(df):,}")
    print(f"   Final rows:    {len(final_df):,}")

if __name__ == "__main__":
    main()