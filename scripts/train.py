import os
import sys
import torch
from torch.utils.data import DataLoader
import wandb

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Enable fallback to CPU for unsupported ops on MPS

# Add the root directory to path so python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow
from src.training.trainer import SMPCTrainer

def main():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    DATA_PATH = "data/processed" # Directory containing your train/val/test parquets
    
    # Training Hyperparameters
    BATCH_SIZE = 1024      # Optimized for Apple Silicon MPS
    EPOCHS = 50
    LEARNING_RATE = 5e-4   # TIGHTENED: Splines need a lower LR than Affine flows (was 1e-3)
    LOG_WANDB = True
    
    # Phase III Neural Spline Flow Architecture
    NUM_LAYERS = 6         # Number of coupling layers (Relay race steps)
    HIDDEN_DIM = 128       # Residual MLP size
    NUM_BINS = 8           # SPLINE: Number of Rational-Quadratic bins per variable
    BOUND = 5.0            # SPLINE: The boundary constraint [-5.0, 5.0] matching RobustScaler
    # ==========================================
    
    # Enable Apple Silicon Acceleration natively
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if LOG_WANDB:
        wandb.init(
            project="smpc-spline-flow", 
            config={
                "epochs": EPOCHS, 
                "batch_size": BATCH_SIZE, 
                "lr": LEARNING_RATE, 
                "device": device,
                "num_bins": NUM_BINS,
                "bound": BOUND,
                "num_layers": NUM_LAYERS
            }
        )

    # 1. Load the Datasets (Train, Val, Test Splits)
    print("📦 Initializing Datasets...")
    train_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='train', log_to_wandb=LOG_WANDB)
    val_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='val', log_to_wandb=False)
    test_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='test', log_to_wandb=False)
    
    # Optimized DataLoader settings for Apple Silicon
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": True
    }

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    # 2. Dynamically size the Neural Network based on the data
    sample = train_dataset[0]
    dim_theta = sample['theta'].shape[0]
    dim_condition = sample['condition'].shape[0]

    # 3. Build the Normalizing Flow (Phase III Splines)
    print(f"🧠 Building Neural Spline Flow (Theta: {dim_theta}, Cond: {dim_condition}, Bins: {NUM_BINS})...")
    model = PipelineConditionalFlow(
        dim_theta=dim_theta, 
        dim_condition=dim_condition, 
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        num_bins=NUM_BINS,
        bound=BOUND
    )

    # 4. Ignite the Training Loop
    trainer = SMPCTrainer(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        learning_rate=LEARNING_RATE, 
        epochs=EPOCHS, 
        device=device,
        log_to_wandb=LOG_WANDB
    )
    
    trainer.train()
    
    if LOG_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()