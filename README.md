# Genesis
# Stochastic MPC with Normalizing Flows (S-MPC)

This repository contains an edge-deployable, physics-informed Normalizing Flow architecture designed to simulate and optimize complex pipeline dynamics using Emerson SCADA data.

The system is highly optimized for in-memory training on **Apple Silicon (M4 Max/Ultra)** using PyTorch Metal Performance Shaders (MPS), leveraging massive unified memory for instant data-loading without lazy-fetching bottlenecks.

## 🚀 Quick Start: Environment Setup

We strongly recommend using a virtual environment to prevent dependency conflicts across your machine learning projects.

### 1. Create a Virtual Environment

Open your terminal at the root of this project and run the following commands:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate
```

*(You will know it is activated when you see `(venv)` at the start of your terminal prompt).*

### 2. Install Dependencies

With the virtual environment activated, upgrade `pip` and install the required packages. The `requirements.txt` is already optimized for Apple Silicon natively.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Log in to Weights & Biases (Optional but Recommended)

To enable cloud dashboard tracking for your data transformations, pipeline plots, and training loss:

```bash
wandb login
```

## 📂 Project Structure

```text
smpc-flow-project/
├── configs/            # Hydra YAML configuration files (Hyperparameters, Paths)
├── data/               # Local SCADA data (ignored by git to protect proprietary info)
│   ├── raw/            # Raw Emerson SCADA exports
│   └── processed/      # Compressed functional PCA data
├── outputs/            # Local checkpoints, plots, and tensor logs
│   └── images/         # Transformed data charts and loss graphs
├── src/                # Core Python modules
│   ├── data/           # Data loaders and preprocessing (dataset.py)
│   ├── models/         # Coupling layers, ResidualMLP, and flow wrappers
│   └── training/       # Training loops and optimizers
├── scripts/            # Entry points for running the code
└── tests/              # Unit tests for ensuring gradient stability
```

## 🧪 Testing the Data Pipeline

To verify your M4 Max environment is set up correctly and your unified memory is ingesting the tensors, you can run the standalone Dataset script. This will generate a dummy SCADA wave, preprocess it, and save a high-res visualization directly to `outputs/images/`.

```bash
# Make sure your virtual environment is activated
python src/data/dataset.py
```

Navigate to the `outputs/images/` directory to see the resulting `data_transformation_step.png` chart, confirming that your math transformations and local artifact storage are working perfectly!