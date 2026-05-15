import os
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow
from src.training.trainer import SMPCTrainer

def main():
    # Initialize wandb run for this specific sweep configuration
    # This automatically connects to the Bayesian engine on the W&B server
    wandb.init()
    config = wandb.config

    print(f"🚀 Starting sweep run: {wandb.run.name}")

    # ==========================================
    # --- LOAD BASE CONFIG FOR DATALOADERS ---
    # ==========================================
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'train.yaml')
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    data_cfg = base_cfg['data']
    dl_cfg = base_cfg['dataloader']
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dataset & Dataloader
    print("📦 Initializing Datasets...")
    train_dataset = SCADAPipelineDataset(data_path=data_cfg['path'], split="train", log_to_wandb=True)
    val_dataset = SCADAPipelineDataset(data_path=data_cfg['path'], split="val", log_to_wandb=False)

    # Use batch_size from sweep config if we decide to tune it later, else fallback to base train.yaml
    batch_size = config.get('batch_size', base_cfg['training']['batch_size'])

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": dl_cfg['num_workers'],
        "pin_memory": device == "cuda",   # pin_memory not supported on MPS
    }
    if dl_cfg['num_workers'] > 0:
        dataloader_kwargs["prefetch_factor"] = dl_cfg['prefetch_factor']
        dataloader_kwargs["persistent_workers"] = dl_cfg['persistent_workers']

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    # 2. Compute Data Dimensions dynamically
    sample_batch = train_dataset[0]
    dim_theta = sample_batch['theta'].shape[0]
    dim_condition = sample_batch['condition'].shape[0]

    # 3. DYNAMIC PARAMETER INJECTION (Complete Decoupling)
    import inspect
    model_sig = inspect.signature(PipelineConditionalFlow.__init__).parameters
    model_kwargs = {k: v for k, v in config.items() if k in model_sig}
    
    # Inject core logic that the engine doesn't know about
    model_kwargs['dim_theta'] = dim_theta
    model_kwargs['dim_condition'] = dim_condition

    model = PipelineConditionalFlow(**model_kwargs)

    # 4. DYNAMIC TRAINER INJECTION (Complete Decoupling)
    trainer_sig = inspect.signature(SMPCTrainer.__init__).parameters
    trainer_kwargs = {k: v for k, v in config.items() if k in trainer_sig}

    # Remove duplicates from trainer_kwargs that we want to manually lock here
    for key in ['model', 'train_dataloader', 'val_dataloader', 'device', 'log_to_wandb', 'use_bf16']:
        trainer_kwargs.pop(key, None)

    trainer = SMPCTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device="mps" if torch.backends.mps.is_available() else "cuda",
        log_to_wandb=True,
        use_bf16=True,
        **trainer_kwargs
    )
    
    # 5. ISOLATED CHECKPOINTING
    trainer.checkpoint_dir = f"HPO/models/{wandb.run.name}"
    os.makedirs(trainer.checkpoint_dir, exist_ok=True)
    
    # 6. RUN THE TRAINING LOOP
    trainer.train()
    
    # 7. SAVE ARTIFACT SPECS (Strictly isolated in HPO/models)
    import json
    model_kwargs['val_loss'] = trainer.best_val_loss
    with open(os.path.join(trainer.checkpoint_dir, "model_config.json"), "w") as f:
        json.dump(model_kwargs, f)

    # 8. TOP 5 PRUNING LOGIC (Keep only the absolute best models!)
    import shutil
    print("\n🧹 Cleaning up... Enforcing Top 5 HPO Rule")
    models_dir = "HPO/models"
    all_runs = []
    
    for folder in os.scandir(models_dir):
        if folder.is_dir():
            info_path = os.path.join(folder.path, "model_config.json")
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
                    all_runs.append((info.get('val_loss', float('inf')), folder.path))
                    
    # Sort runs by lowest validation loss
    all_runs.sort(key=lambda x: x[0])
    
    for rank, (loss, folder_path) in enumerate(all_runs):
        if rank >= 5:
            print(f"   🗑️ Pruning run {os.path.basename(folder_path)} (Loss: {loss:.4f}) - Not in Top 5")
            shutil.rmtree(folder_path)
        else:
            print(f"   🏆 Rank {rank+1}: {os.path.basename(folder_path)} (Loss: {loss:.4f})")

    print(f"✅ Sweep run {wandb.run.name} complete!")

if __name__ == "__main__":
    main()
