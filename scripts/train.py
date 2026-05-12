import os
import sys
import torch
from torch.utils.data import DataLoader
import wandb

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Enable fallback to CPU for unsupported ops on MPS

# Add the root directory to path so python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow
from src.training.trainer import SMPCTrainer

def main():
    # ==========================================
    # --- CONFIGURATION ---
    # ==========================================
    DATA_PATH = "data/processed" # Directory containing your train/val/test parquets
    
    # Training Hyperparameters — tuned for M4 Max (128GB unified memory)
    BATCH_SIZE = 4096      # M4 Max can handle 4x larger batches — better GPU utilization
    EPOCHS = 50
    LEARNING_RATE = 2e-4   # LOWERED: larger model (512 dim, 10 layers) needs a smaller LR to stay stable
    LOG_WANDB = True
    USE_BF16 = False       # DISABLED: torch.autocast is not stable on MPS backend — causes NaN gradients
    
    # Phase III Neural Spline Flow Architecture — balanced for M4 Max + dataset size
    NUM_LAYERS = 6         # Reduced from 10: prevents overfitting on ~720k samples
    HIDDEN_DIM = 256       # Reduced from 512: better generalisation, still expressive
    NUM_BINS = 12          # Increased from 8: finer-grained spline segments
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
                "num_layers": NUM_LAYERS,
                "hidden_dim": HIDDEN_DIM,
                "use_bf16": USE_BF16
            }
        )

    # 1. Load the Datasets (Train, Val, Test Splits)
    print("📦 Initializing Datasets...")
    train_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='train', log_to_wandb=LOG_WANDB)
    val_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='val', log_to_wandb=False)
    test_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='test', log_to_wandb=False)
    
    # Optimized DataLoader settings for M4 Max (14-core CPU, 128GB RAM)
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 12,          # M4 Max has 14 CPU cores; use 12 for data pipeline
        "persistent_workers": True,
        "prefetch_factor": 4,       # Buffer more batches — 128GB RAM means no pressure
        "pin_memory": device == "cuda"  # pin_memory is not supported on MPS
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
        log_to_wandb=LOG_WANDB,
        use_bf16=USE_BF16
    )
    
    trainer.train()
    
    if LOG_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()