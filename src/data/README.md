# 🗃️ Genesis Data Pipeline (Phase II)

This directory contains the data engineering pipeline for the Genesis SCADA Normalizing Flow.

Per the architecture outlined in the DOE research grant (Section 5.2 and 8.3), this pipeline transitions the problem from predicting instantaneous states to predicting 4-hour continuous trajectories using sliding windows and Functional Principal Component Analysis (PCA).

## 🏗️ Pipeline Architecture

The data flow consists of two primary scripts that must be executed in order:

### 1. `preprocessing.py` (The Trajectory Encoder)

This script is run offline, exactly once. It bridges the gap between raw historian SCADA data and the low-dimensional trajectory coefficients required by the Normalizing Flow.

**What it does:**
* Reads the raw 1-million row Parquet/Excel SCADA export.
* Slides a 4-hour window (14,400 seconds) across the timeline, shifting by exactly 1 row (1 second) at a time to preserve the massive dataset size.
* Downsamples the 4-hour window from 1-second resolution to 1-minute resolution to drastically optimize compute while retaining physical wave shapes.
* Standard-Scales the target ($\theta$) trajectories to ensure pressures and temperatures are weighted equally.
* Compresses the 1,920-dimension curve into `N_COMPONENTS` (default: 4) Functional PCA coefficients.
* Outputs the finalized training dataset alongside the PCA and Scaler `.pkl` files.

### 2. `dataset.py` (The PyTorch Dataloader)

This module is consumed directly by the training loop (`train.py`) and is executed at runtime.

**What it does:**
* Reads the optimized `DataAllParts_PCA.parquet` file.
* **Chronological Splitting**: Slices the data chronologically (e.g., first 80% for training, next 20% for validation) to respect time-series continuity.
* **Applies Leakage-Free Normalization**: Calculates Z-score statistics ($mean, std$) exclusively on the training split and saves them to `outputs/checkpoints/scaler.pt`. Validation splits and Edge deployments are strictly forced to load this file to prevent future data leakage.
* Yields PyTorch tensors (`theta` and `condition`) to the M4 Max GPU.

## 📦 Output Artifacts Explained

After running `preprocessing.py`, you will notice three new files. They are strictly required for training and evaluating the model:

| Artifact Path | Type | Description |
|---|---|---|
| `data/processed/DataAllParts_PCA.parquet` | Dataset | The optimized training table. Contains instantaneous $X$ (Context), instantaneous $U$ (Controls), and the newly generated PCA coefficients representing the future $\theta$ (Targets). |
| `outputs/checkpoints/trajectory_pca_model.pkl` | Scikit-Learn Model | The "Decoder". Used during inference to inverse-transform the predicted 4 PCA coefficients back into the full 1,920-point multi-variable pipeline curve. |
| `outputs/checkpoints/theta_base_scaler.pkl` | Scikit-Learn Model | The "Unit Converter". Used during inference to un-scale the decoded curve from Z-scores back into true physical units (PSI, RPM, Celsius). |

## ⚠️ Important Notes on Adding New Data

If you receive a new SCADA export from Emerson or the pipeline operator:

1. Place the raw file in `data/raw/`.
2. Delete the old `data/processed/DataAllParts_PCA.parquet` to avoid caching issues.
3. Re-run `python src/data/preprocessing.py`.
4. Check the terminal output: Ensure the "Variance Kept" percentage printed by the PCA step is high (ideally >90%). If it is too low, open `preprocessing.py` and increase `N_COMPONENTS`.