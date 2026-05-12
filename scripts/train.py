import os
import sys
import torch
import yaml
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
    # --- LOAD CONFIG FROM YAML ---
    # All hyperparameters live in configs/train.yaml
    # Edit that file — nothing needs to change here.
    # ==========================================
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Unpack config sections for readability
    data_cfg   = cfg['data']
    train_cfg  = cfg['training']
    model_cfg  = cfg['model']
    dl_cfg     = cfg['dataloader']
    log_cfg    = cfg['logging']

    # Enable Apple Silicon Acceleration natively
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if log_cfg['wandb']:
        wandb.init(
            project=log_cfg['wandb_project'],
            config={
                **train_cfg,
                **model_cfg,
                "device": device,
            }
        )

    # 1. Load the Datasets (Train, Val, Test Splits)
    print("📦 Initializing Datasets...")
    train_dataset = SCADAPipelineDataset(data_path=data_cfg['path'], split='train', log_to_wandb=log_cfg['wandb'])
    val_dataset   = SCADAPipelineDataset(data_path=data_cfg['path'], split='val',   log_to_wandb=False)
    test_dataset  = SCADAPipelineDataset(data_path=data_cfg['path'], split='test',  log_to_wandb=False)

    dataloader_kwargs = {
        "batch_size":         train_cfg['batch_size'],
        "num_workers":        dl_cfg['num_workers'],
        "persistent_workers": dl_cfg['persistent_workers'],
        "prefetch_factor":    dl_cfg['prefetch_factor'],
        "pin_memory":         device == "cuda",   # pin_memory not supported on MPS
    }

    train_dataloader = DataLoader(train_dataset, shuffle=True,  **dataloader_kwargs)
    val_dataloader   = DataLoader(val_dataset,   shuffle=False, **dataloader_kwargs)
    test_dataloader  = DataLoader(test_dataset,  shuffle=False, **dataloader_kwargs)

    # 2. Dynamically size the model from the data
    sample        = train_dataset[0]
    dim_theta     = sample['theta'].shape[0]
    dim_condition = sample['condition'].shape[0]

    # 3. Build the Normalizing Flow
    print(f"🧠 Building Neural Spline Flow (Theta: {dim_theta}, Cond: {dim_condition}, Bins: {model_cfg['num_bins']})...")
    model = PipelineConditionalFlow(
        dim_theta=dim_theta,
        dim_condition=dim_condition,
        num_layers=model_cfg['num_layers'],
        hidden_dim=model_cfg['hidden_dim'],
        num_bins=model_cfg['num_bins'],
        bound=model_cfg['bound'],
        mlp_layers=model_cfg['mlp_layers'],
        dropout_rate=model_cfg['dropout_rate'],
    )

    # 4. Ignite the Training Loop
    trainer = SMPCTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=train_cfg['learning_rate'],
        epochs=train_cfg['epochs'],
        device=device,
        log_to_wandb=log_cfg['wandb'],
        use_bf16=log_cfg['use_bf16'],
        grad_clip=train_cfg['grad_clip'],
        weight_decay=train_cfg['weight_decay'],
        lr_scheduler_factor=train_cfg['lr_scheduler_factor'],
        lr_scheduler_patience=train_cfg['lr_scheduler_patience'],
        lr_scheduler_min=train_cfg['lr_scheduler_min'],
        early_stopping_patience=train_cfg['early_stopping_patience'],
    )

    trainer.train()

    if log_cfg['wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()