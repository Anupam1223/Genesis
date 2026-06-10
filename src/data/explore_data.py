import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Safe for background plotting
import matplotlib.pyplot as plt

def main():
    print("--- Generating Presentation EDA Plots ---")

    # --- CONFIGURATION ---
    INPUT_PATH = "data/raw/DataAllParts.parquet"
    OUTPUT_DIR = "outputs/eda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. LOAD DATA
    print(f"\nLoading data from {INPUT_PATH}...")
    if not os.path.exists(INPUT_PATH):
        try:
            df = pd.read_excel(INPUT_PATH.replace('.parquet', '.xlsx'))
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
    else:
        df = pd.read_parquet(INPUT_PATH)

    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)

    def _to_col(col_name):
        return re.sub(r'\s+', '_', col_name.strip())

    # Drop duplicate columns (SCADA exports often repeat columns; df[col] would
    # return a DataFrame instead of a Series and break pd.to_numeric)
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. OUR CHOSEN VARIABLES
    u_cols_raw = [
        'KPI Fuel Gas Lower Heating Value ',
        'Part4 UK 1',
        'UK 14PDCV-504  H-SEL',
        'UK V-1412 HPOMR FR E-1411  '
    ]
    x_cols_raw = [
        'COMP AXL DSCHRG PRESS', 'COMP Discharge Pressure', 'COMP Suction Pressure',
        'Turbine SHAFT SPEED', 'COMP Discharge FlowDP', 'COMP Discharge Temp',
        'COMP Suction Drum Temperature', 'TBN TEMP 1 STG FWD INR', 'Exhaust Temp Spread 1',
        'SHAFT BAL PISTON', 'LUBE OIL LVL XMTR HI/LO TNK', 'DRIVE MOTOR Oil Press',
        'SEAL GAS SUP DE', 'SEAL Primary DE Pressure'
    ]
    theta_cols_raw = [
        'KPI Turbine Overall Thermal Cycle Efficiency',
        'KPI Gas COMP Isentropic Efficiency',
        'KPI Turbine Heat Rate'
    ]

    u_cols_all = [_to_col(c) for c in u_cols_raw]
    x_cols_all = [_to_col(c) for c in x_cols_raw]
    theta_cols_all = [_to_col(c) for c in theta_cols_raw]

    # Filter only columns that actually exist
    x_cols = [c for c in x_cols_all if c in df.columns]
    u_cols = [c for c in u_cols_all if c in df.columns]
    theta_cols = [c for c in theta_cols_all if c in df.columns]

    missing_u_cols = [c for c in u_cols_all if c not in df.columns]
    if missing_u_cols:
        print("\nWarning: Missing U columns after normalization:")
        for col in missing_u_cols:
            print(f"  - {col}")

    # --- CRITICAL FIX: Force Numeric Types ---
    # SCADA data often has "Bad", "Comm Error", or empty string artifacts.
    # This coerces those strings to NaN so pandas math/sorting doesn't crash.
    print("\nCleaning data types (coercing SCADA string artifacts to NaN)...")
    for col in x_cols + u_cols + theta_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==========================================
    # NARRATIVE OUTPUT (For your speaker notes)
    # ==========================================
    total_rows = len(df)
    total_raw_cols = len(df.columns)
    chosen_cols = len(x_cols) + len(u_cols) + len(theta_cols)

    print("\n" + "="*60)
    print("DATASET OVERVIEW (Use this for your pitch!)")
    print("="*60)
    print(f"Total Rows in Dataset: {total_rows:,}")
    print(f"Total Raw Columns:     {total_raw_cols:,}")
    print("\nNARRATIVE:")
    print(f"\"The raw dataset was massive—over {total_rows:,} rows and {total_raw_cols} columns of noisy sensor data.")
    print(f"To build a focused Proof of Concept, we stripped out the irrelevant noise and selected exactly {chosen_cols} critical columns:")
    print(f"  • 14 'X' variables (The machine's current state: pressures, temps)")
    print(f"  • 4 'U' variables (The boundaries: control valves, fuel heating value)")
    print(f"  • 3 'Theta' variables (The targets we want to predict: efficiency KPIs)\"")
    
    print("\n" + "="*60)
    print("DATA RANGES (Min, Mean, Max)")
    print("="*60)
    pd.options.display.float_format = '{:,.2f}'.format
    if theta_cols: print("\n--- THETA (Targets) ---\n", df[theta_cols].describe().T[['min', 'mean', 'max']])
    if x_cols: print("\n--- X (State Sensors) ---\n", df[x_cols].describe().T[['min', 'mean', 'max']])
    if u_cols: print("\n--- U (Boundaries) ---\n", df[u_cols].describe().T[['min', 'mean', 'max']])

    # ==========================================
    # PLOT 1: The Target KPIs (Theta)
    # ==========================================
    if theta_cols:
        print("\nGenerating Theta (Target) histograms...")
        fig, axes = plt.subplots(1, len(theta_cols), figsize=(14, 4))
        if len(theta_cols) == 1: axes = [axes]

        for i, col in enumerate(theta_cols):
            # Drop NaNs and clip extreme 1% outliers to make histograms readable
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                lower, upper = clean_data.quantile(0.01), clean_data.quantile(0.99)
                clean_data = clean_data[(clean_data >= lower) & (clean_data <= upper)]

                axes[i].hist(clean_data, bins=40, color='#8b5cf6', edgecolor='black', alpha=0.8)
            clean_title = col.replace('KPI_', '').replace('_', ' ')
            axes[i].set_title(clean_title, fontsize=10, fontweight='bold')
            axes[i].set_ylabel("Frequency")
            axes[i].grid(axis='y', linestyle='--', alpha=0.5)

        plt.suptitle("The Spread of Target KPIs (Theta)\nReal-world, scattered distributions", fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plot_theta_path = os.path.join(OUTPUT_DIR, "slide_theta_histograms.png")
        plt.savefig(plot_theta_path, dpi=200, bbox_inches='tight')
        plt.close()

    # ==========================================
    # PLOT 2: The State Sensors (X)
    # ==========================================
    if x_cols:
        print("Generating X (State) histograms...")
        # 14 plots fits well in a 4x4 grid (16 slots, 2 empty)
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i, col in enumerate(x_cols):
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                lower, upper = clean_data.quantile(0.01), clean_data.quantile(0.99)
                clean_data = clean_data[(clean_data >= lower) & (clean_data <= upper)]

                axes[i].hist(clean_data, bins=30, color='#38bdf8', edgecolor='black', alpha=0.8)
            
            clean_title = col.replace('COMP_', '').replace('Turbine_', '').replace('_', ' ')
            # Truncate long titles
            if len(clean_title) > 25: clean_title = clean_title[:22] + "..."
            axes[i].set_title(clean_title, fontsize=9, fontweight='bold')
            axes[i].tick_params(axis='x', labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)
            axes[i].grid(axis='y', linestyle='--', alpha=0.4)

        # Hide the remaining empty subplots (14 and 15)
        for j in range(len(x_cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"The Spread of Current State Sensors (X)\nProof of Concept focuses on {len(x_cols)} critical physical states", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_x_path = os.path.join(OUTPUT_DIR, "slide_x_histograms.png")
        plt.savefig(plot_x_path, dpi=200, bbox_inches='tight')
        plt.close()

    # ==========================================
    # PLOT 3: The Boundary Conditions (U)
    # ==========================================
    if u_cols:
        print("Generating U (Boundary) histograms...")
        grid_size = 2 if len(u_cols) > 1 else 1
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 8))
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, col in enumerate(u_cols):
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                lower, upper = clean_data.quantile(0.01), clean_data.quantile(0.99)
                clean_data = clean_data[(clean_data >= lower) & (clean_data <= upper)]

                axes[i].hist(clean_data, bins=30, color='#f59e0b', edgecolor='black', alpha=0.8)

            clean_title = col.replace('_', ' ')
            if len(clean_title) > 25:
                clean_title = clean_title[:22] + "..."
            axes[i].set_title(clean_title, fontsize=9, fontweight='bold')
            axes[i].tick_params(axis='x', labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)
            axes[i].grid(axis='y', linestyle='--', alpha=0.4)

        for j in range(len(u_cols), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"The Spread of Boundary Conditions (U)\nProof of Concept uses {len(u_cols)} critical exogenous inputs", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_u_path = os.path.join(OUTPUT_DIR, "slide_u_histograms.png")
        plt.savefig(plot_u_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder for your presentation images.")

if __name__ == "__main__":
    main()