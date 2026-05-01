# 🧬 Genesis: Physics-Informed Stochastic MPC with Normalizing Flows

Genesis is a sophisticated machine learning project designed for **Stochastic Model Predictive Control (S-MPC)**. It uses **Conditional Normalizing Flows** to model the uncertain dynamics of complex industrial systems (like Emerson SCADA pipelines).

Unlike standard regression models that predict a single value, Genesis models the entire **probability distribution** of future states. This allows operators to not only see what is *likely* to happen but also the *risk* (variance) associated with those predictions.

### 🏗 Architecture Overview
- **Core Model:** A Conditional RealNVP (Affine Coupling) architecture.
- **Physics-Informed:** Designed to ingest sensor context ($x$) and control inputs ($u$) to warp a simple Gaussian distribution into a complex "Physical State" distribution ($\theta$).
- **Hardware Optimized:** Built specifically for **Apple Silicon (M4 Max/Ultra)** using PyTorch `mps` (Metal Performance Shaders) for blazing-fast in-memory processing.

---

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

---

## 📊 Data Preparation (Crucial)

Before running the project, you **must** configure exactly which sensors and control variables you want to track.

### 📍 Setting your Dataset Path
Open `scripts/train.py` and update the `DATA_PATH` variable to point to your local Excel or Parquet file:
```python
# scripts/train.py
DATA_PATH = "data/raw/Your_Custom_Export.xlsx" 
```

### 🧠 Feature Selection
The "Brain" of the model depends on which columns you choose. Open `src/data/dataset.py` and modify these three lists to match your specific SCADA export:

1.  **`self.x_cols` (Context):** Measured values you cannot control (Suction Pressure, Ambient Temp).
2.  **`self.u_cols` (Controls):** Variables you *can* change (Shaft Speed, Valve Positions).
3.  **`self.theta_cols` (Targets):** The variables you want the AI to predict/simulate the uncertainty for (Discharge Pressure, Thermal Efficiency).

---

## 🚀 Running the Project

### Step 1: Verify the Data Pipeline
Run the dataset script standalone to ensure your columns are being read correctly and that the scaling math is working. This will generate a visualization in `outputs/images/`.

```bash
python src/data/dataset.py
```

### Step 2: Start Training
Execute the main training script. This will:
- Initialize the Normalizing Flow.
- Start the Apple Silicon GPU-accelerated training loop.
- Log granular batch-level results to your Weights & Biases dashboard.

```bash
python scripts/train.py
```

---

## 📂 Project Structure

- `src/models/`: Contains the `PipelineConditionalFlow` and Coupling layers.
- `src/training/`: The `SMPCTrainer` which handles the log-likelihood loss and gradient clipping.
- `src/data/`: Pipeline for scaling, cleaning, and caching SCADA data.
- `outputs/checkpoints/`: Where your trained `.pt` models and scalers are saved.
- `scripts/`: Entry points for training and edge-export.