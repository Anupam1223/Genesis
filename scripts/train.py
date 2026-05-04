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
    # --- CONFIGURATION ---
    DATA_PATH = "data/raw/DataAllParts.xlsx" # TODO: Update to actual Emerson CSV
    BATCH_SIZE = 1024  # Increased from 256 → fewer batches per epoch = faster training
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    LOG_WANDB = True
    
    # Enable Apple Silicon Acceleration natively
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if LOG_WANDB:
        wandb.init(project="smpc-flow", config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "device": device})

    # 1. Load the Datasets (Train, Val, Test Splits)
    print("Initializing Datasets...")
    train_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='train', log_to_wandb=LOG_WANDB)
    val_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='val', log_to_wandb=False)
    test_dataset = SCADAPipelineDataset(data_path=DATA_PATH, split='test', log_to_wandb=False)
    
    # Optimized DataLoader settings for Apple Silicon with ~1M row datasets:
    # - num_workers=8: loads next batch in background while GPU trains on current batch
    # - persistent_workers=True: workers stay alive between epochs (avoids costly respawning)
    # - pin_memory=True: faster CPU→GPU memory transfer
    # - prefetch_factor=2: pre-loads 2 batches ahead so GPU never waits for data
    # - batch_size=1024: reduces batches per epoch from ~2868 to ~717 (4x fewer optimizer steps)
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 8,
        "persistent_workers": True,
        "prefetch_factor": 2
    }

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

    # 2. Dynamically size the Neural Network based on the data
    sample = train_dataset[0]
    dim_theta = sample['theta'].shape[0]
    dim_condition = sample['condition'].shape[0]

    # 3. Build the Normalizing Flow
    print(f"Building Flow (Theta Dim: {dim_theta}, Cond Dim: {dim_condition})...")
    model = PipelineConditionalFlow(dim_theta=dim_theta, dim_condition=dim_condition, num_layers=6)

    # 4. Ignite the Training Loop
    trainer = SMPCTrainer(
        model=model, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, # Pass the new validation loader
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