import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib

def main():
    print("🚀 Starting Clean SCADA Trajectory Preprocessing...")
    start_time = time.time()

    # --- CONFIGURATION ---
    INPUT_PATH = "data/raw/DataAllParts.parquet" 
    OUTPUT_PATH = "data/processed/DataAllParts_PCA.parquet"
    CHECKPOINT_DIR = "outputs/checkpoints"
    
    WINDOW_SIZE = 14400   # 4 hours
    DOWNSAMPLE_RATE = 60  # Compress trajectory to 1-minute intervals
    N_COMPONENTS = 12     # 12 shapes to capture >90% variance

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. DEFINE COLUMNS
    x_cols = ['COMP_Suction_Pressure', 'COMP_Suction_Drum_Temperature', 'KPI_Fuel_Gas_Lower_Heating_Value']
    u_cols = ['Turbine_SHAFT_SPEED', 'UK_14PDCV-504_H-SEL', 'SEAL_GAS_SUP_DE']
    theta_cols = [
        'SEAL_GAS_FLTR_DP', 'LUBE_OIL_LVL_XMTR_HI/LO_TNK', 
        'KPI_Turbine_Overall_Thermal_Cycle_Efficiency', 'KPI_Gas_COMP_Isentropic_Efficiency', 
        'COMP_Discharge_Pressure', 'COMP_Discharge_Temp', 'Exhaust_Temp_Spread_1', 'KPI_Turbine_Heat_Rate'
    ]

    # 2. LOAD DATA
    print(f"📦 Loading raw data from {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        try:
            df = pd.read_excel(INPUT_PATH.replace('.parquet', '.xlsx'))
        except:
            print(f"Failed to load {INPUT_PATH}. Please verify the file exists.")
            return
    else:
        df = pd.read_parquet(INPUT_PATH)

    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)

    # Clean extreme artifacts and fill gaps
    print("🧹 Purging NaNs, Infs, and Flatlined Sensors...")
    for col in x_cols + u_cols + theta_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill().fillna(0.0)

    # Inject micro-noise to flatlined SCADA sensors
    for col in theta_cols:
        if df[col].std() < 1e-6:
            print(f"   ⚠️ Warning: Sensor '{col}' is flatlined! Injecting micro-noise to protect math engine.")
            df[col] += np.random.normal(0, 1e-5, size=len(df))

    # --- CRITICAL FIX: Splitting chronologically FIRST to prevent data leakage ---
    M_total = len(df) - WINDOW_SIZE + 1
    train_end = int(M_total * 0.70)
    val_end = int(M_total * 0.85)

    # 3. SCALING X AND U
    print("⚖️ Applying Z-Score Normalization to Condition Variables (X and U) [Fitted on TRAIN only]...")
    x_scaler = StandardScaler()
    u_scaler = StandardScaler()
    
    x_array = df[x_cols].values
    u_array = df[u_cols].values
    
    x_scaler.fit(x_array[:train_end])
    u_scaler.fit(u_array[:train_end])
    
    x_scaled = x_scaler.transform(x_array)
    u_scaled = u_scaler.transform(u_array)

    # 4. ROBUST SCALING FOR THETA TARGET
    print("⚖️ Applying Robust Scaling to isolate true physical dynamics [Fitted on TRAIN only]...")
    theta_array = df[theta_cols].values
    theta_scaler = RobustScaler()
    
    # We include WINDOW_SIZE because the training curves reach all the way to train_end + WINDOW_SIZE
    theta_scaler.fit(theta_array[:train_end + WINDOW_SIZE])
    
    theta_scaled = theta_scaler.transform(theta_array)
    theta_scaled = np.clip(theta_scaled, -20.0, 20.0)

    # 5. VIRTUAL WINDOWING FOR SCENARIOS
    print(f"🪟 Slicing data into {WINDOW_SIZE}-step curves...")
    windows = np.lib.stride_tricks.sliding_window_view(theta_scaled, window_shape=WINDOW_SIZE, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    
    pca_input = windows[:, ::DOWNSAMPLE_RATE, :].reshape(windows.shape[0], -1)
    M = pca_input.shape[0]

    # 6. FAST PCA COMPRESSION
    print(f"🗜️ Fitting PCA on a random 20,000 sample subset from the TRAINING split...")
    rng = np.random.default_rng(42)
    fit_indices = rng.choice(train_end, size=min(20000, train_end), replace=False)
    
    # --- CRITICAL FIX: np.float64 for Apple Silicon M4 BLAS computation bug ---
    fit_batch = np.ascontiguousarray(pca_input[fit_indices], dtype=np.float64)
    
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(fit_batch)
    
    variance_kept = sum(pca.explained_variance_ratio_) * 100
    print(f"✨ PCA Fit Complete! Captured {variance_kept:.2f}% of future variance.")

    # 7. SAFE BATCH TRANSFORM
    print("🔄 Compressing all curves...")
    theta_pca_features = np.zeros((M, N_COMPONENTS), dtype=np.float64)
    BATCH_SIZE = 5000
    for i in range(0, M, BATCH_SIZE):
        end = min(i + BATCH_SIZE, M)
        batch = np.ascontiguousarray(pca_input[i:end], dtype=np.float64)
        theta_pca_features[i:end] = pca.transform(batch)

    # 8. EXPORT
    print("🏗️ Saving final artifacts...")
    # NOTE: We grab the scaled matrices directly.
    x_df_scaled = pd.DataFrame(x_scaled[:M], columns=x_cols)
    u_df_scaled = pd.DataFrame(u_scaled[:M], columns=u_cols)
    
    theta_pca_df = pd.DataFrame(
        theta_pca_features, 
        columns=[f'PCA_Coefficient_{i+1}' for i in range(N_COMPONENTS)]
    )

    final_df = pd.concat([x_df_scaled, u_df_scaled, theta_pca_df], axis=1)
    
    # --- CRITICAL FIX: Split into Train / Val / Test to avoid loading the whole dataset later ---
    out_dir = os.path.dirname(OUTPUT_PATH)
    final_df.iloc[:train_end].to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    final_df.iloc[train_end:val_end].to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    final_df.iloc[val_end:].to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    
    # Keeping the original joined one just in case downstream code assumes it exists for now
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    joblib.dump(x_scaler, os.path.join(CHECKPOINT_DIR, "x_scaler.pkl"))
    joblib.dump(u_scaler, os.path.join(CHECKPOINT_DIR, "u_scaler.pkl"))
    joblib.dump(theta_scaler, os.path.join(CHECKPOINT_DIR, "theta_base_scaler.pkl"))
    joblib.dump(pca, os.path.join(CHECKPOINT_DIR, "trajectory_pca_model.pkl"))

    elapsed = (time.time() - start_time) / 60
    print(f"✅ Clean pipeline finished in {elapsed:.2f} minutes!")

if __name__ == "__main__":
    main()