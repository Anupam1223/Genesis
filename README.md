# 🧬 Genesis: Physics-Informed Stochastic MPC with Normalizing Flows

Genesis is a sophisticated machine learning project designed for Stochastic Model Predictive Control (S-MPC). It uses Conditional Normalizing Flows to model the uncertain dynamics of complex industrial systems (like Emerson SCADA pipelines).

Unlike standard regression models that predict a single instantaneous value, Genesis models the entire probability distribution of future states over a 4-hour trajectory. This allows operators to not only see what is likely to happen but also the risk (variance) associated with those predictions.

## 🏗 Architecture Overview

* **Core Model**: A Conditional RealNVP (Affine Coupling) architecture.
* **Physics-Informed Trajectories**: Ingests instantaneous sensor context ($x$) and control inputs ($u$) to predict Functional PCA coefficients representing a 4-hour future trajectory ($\theta$).
* **Hardware Optimized**: Built specifically for Apple Silicon (M4 Max/Ultra) using PyTorch `mps` (Metal Performance Shaders) for blazing-fast in-memory processing.

## 🛠 Setup & Installation

Follow these steps to get your environment ready for training and inference.

### 1. Environment Initialization

We use a dedicated virtual environment to handle specific dependency versions for Apple Silicon.

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Weights & Biases (W&B) Dashboard

Genesis is deeply integrated with W&B for real-time training visualization, gradient tracking, and hardware monitoring.

```bash
wandb login
```

## 📊 Data Preparation (Crucial)

Before running the project, you must configure exactly which sensors and control variables you want to track. Because we predict 4-hour trajectories instead of instantaneous points, we use a Sliding Window + Functional PCA approach to compress SCADA curves into a few scalar numbers.

### 🧠 Feature Selection

The "Brain" of the model depends on which columns you choose. Open `src/data/preprocessing.py` and modify these three lists to match your specific SCADA export:

* `x_cols` (Context): Measured values you cannot control right now (e.g., Suction Pressure, Ambient Temp).
* `u_cols` (Controls): Variables you can change right now (e.g., Shaft Speed, Valve Positions).
* `theta_cols` (Targets): The variables you want the AI to predict the 4-hour future trajectory for (e.g., Discharge Pressure, Thermal Efficiency).

## 🚀 Running the Project

### Step 1: Preprocess the Data (Run Once)

> ⚠️ **You must run this step if it is your first time running the project, or if you have added a new raw SCADA dataset.**

This script reads your raw Parquet/Excel file, slices it into 4-hour sliding windows, performs Functional PCA compression on the target variables, and saves the optimized training file.

```bash
python src/data/preprocessing.py
```


### Step 2: Start Training

Execute the main training script. This will initialize the Normalizing Flow, start the Apple Silicon GPU-accelerated training loop, and log granular batch-level results to your Weights & Biases dashboard.

```bash
python scripts/train.py
```

### Step 3: Evaluate the Model

Once trained, run the evaluation script to sample 4-hour PCA trajectories from the flow and un-scale them to visualize how well the predicted probability distributions match the real SCADA physics.

```bash
python scripts/evaluate.py
```

## 📂 Project Structure

* `src/data/`: Pipeline for trajectory windowing, PCA compression (`preprocessing.py`), and scaling (`dataset.py`).
* `src/models/`: Contains the `PipelineConditionalFlow` and Coupling layers.
* `src/training/`: The `SMPCTrainer` which handles the log-likelihood loss and gradient clipping.
* `data/raw/`: Place your raw 1-million row Emerson SCADA exports here.
* `data/processed/`: Where the PCA-compressed trajectory dataset is saved.
* `outputs/checkpoints/`: Where your trained `.pt` models, data scalers, and PCA objects are saved.
* `scripts/`: Entry points for training and evaluation.