import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for background training loops
import matplotlib.pyplot as plt
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class SMPCTrainer:
    def __init__(self, model, train_dataloader, val_dataloader,
                 learning_rate=2e-4, epochs=20, device="mps", log_to_wandb=False,
                 use_bf16=False, grad_clip=0.5, weight_decay=1e-4,
                 lr_scheduler_factor=0.5, lr_scheduler_patience=3, lr_scheduler_min=1e-6,
                 early_stopping_patience=8):
        # Move the model to the Apple Metal GPU (or CUDA)
        self.model = model.to(device)
        self.use_bf16 = use_bf16 and device in ("mps", "cuda")
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device
        self.log_to_wandb = log_to_wandb
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self._epochs_no_improve = 0

        # AdamW — weight decay lowered for Spline Flows (too high can break knot positioning)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # ReduceLROnPlateau — all params driven from configs/train.yaml
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min
        )
        
        # Setup local checkpoint directory
        self.checkpoint_dir = "outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track the best validation loss to save the best model
        self.best_val_loss = float('inf')

    def train(self):
        print(f"\n🚀 Starting Phase III Spline training on device: {self.device.upper()}")
        
        if self.log_to_wandb:
            wandb.watch(self.model, log="all", log_freq=50)
            
        global_step = 0
            
        for epoch in range(1, self.epochs + 1):
            # --- TRAINING PHASE ---
            self.model.train()
            epoch_train_loss = 0.0
            
            # Progress bar for the train batch loop
            pbar_train = tqdm(self.train_dataloader, desc=f"Epoch {epoch:03d}/{self.epochs} [TRAIN]")
            for batch in pbar_train:
                theta = batch['theta'].to(self.device)
                condition = batch['condition'].to(self.device)
                
                self.optimizer.zero_grad()

                # BF16 autocast: safe for Spline Flows — bfloat16 has float32-range exponents
                # so rational-quadratic divisions won't overflow. ~1.5-2x faster on M4 Max MPS.
                with torch.autocast(device_type="cpu" if self.device == "mps" else self.device, 
                                    dtype=self.dtype, enabled=self.use_bf16):
                    loss = self.model.compute_loss(theta, condition)

                # Skip batch if loss is non-finite — can happen with extreme HPO configs.
                # Logging it lets W&B mark this run as degraded without crashing the agent.
                if not torch.isfinite(loss):
                    print(f"   ⚠️  Non-finite loss ({loss.item():.2f}) at step {global_step} — skipping batch.")
                    self.optimizer.zero_grad()
                    global_step += 1
                    continue

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Log every 50 steps to avoid network overhead.
                if self.log_to_wandb and global_step % 50 == 0:
                    log_dict = {"train/batch_loss": loss.item()}

                    # ── Gradient flow figure ──────────────────────────────────────
                    # Group all Linear layers by their parent coupling-layer index.
                    # For each coupling layer build one subplot showing the min / max / avg
                    # gradient magnitude across its Linear weights — one image in WandB
                    # instead of 48 individual metric streams.
                    layer_grads: dict[str, list[float]] = {}
                    for name, module in self.model.named_modules():
                        if not isinstance(module, torch.nn.Linear):
                            continue
                        if module.weight.grad is None:
                            continue
                        parts = name.split(".")  # e.g. ["layers","2","brain","layers","0","0"]
                        if len(parts) >= 2 and parts[0] == "layers":
                            group = f"Layer {parts[1]}"
                        else:
                            group = name
                        layer_grads.setdefault(group, []).append(
                            module.weight.grad.abs().mean().item()
                        )

                    if layer_grads:
                        groups  = sorted(layer_grads.keys())
                        indices = list(range(len(groups)))
                        mins    = [min(layer_grads[g])  for g in groups]
                        maxs    = [max(layer_grads[g])  for g in groups]
                        avgs    = [sum(layer_grads[g]) / len(layer_grads[g]) for g in groups]

                        fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4))
                        ax.fill_between(indices, mins, maxs, alpha=0.25, label="Min–Max range")
                        ax.plot(indices, avgs, marker="o", linewidth=1.5, label="Avg |grad|")
                        ax.set_xticks(indices)
                        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
                        ax.set_ylabel("|grad| (mean abs)")
                        ax.set_title(f"Gradient Flow — Step {global_step}")
                        ax.legend(fontsize=8)
                        fig.tight_layout()
                        log_dict["grad/flow"] = wandb.Image(fig)
                        plt.close(fig)

                    wandb.log(log_dict)
                
                global_step += 1
            
            avg_train_loss = epoch_train_loss / len(self.train_dataloader)
            
            # --- VALIDATION PHASE ---
            avg_val_loss = self.evaluate(epoch, avg_train_loss)
            
            # --- LR SCHEDULING ---
            # Step the scheduler based on val loss
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # --- LOGGING & SAVING ---
            if self.log_to_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": avg_train_loss,
                    "val/epoch_loss": avg_val_loss,
                    "learning_rate": current_lr
                })
                
            # Save the "Best" model based on unseen validation data
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_train_loss, avg_val_loss, is_best=True)
                print(f"   🌟 New best model saved! (Val Loss: {avg_val_loss:.4f})")
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1
                print(f"   ⏳ No val improvement for {self._epochs_no_improve}/{self.early_stopping_patience} epochs.")
                if self._epochs_no_improve >= self.early_stopping_patience:
                    print(f"\n⏹  Early stopping triggered at epoch {epoch}. Best val loss: {self.best_val_loss:.4f}")
                    break
                
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_train_loss, avg_val_loss)
                
        print("\n✅ Training Complete!")

    def evaluate(self, epoch, avg_train_loss):
        """
        Runs the model on unseen data without updating weights.
        """
        self.model.eval()
        epoch_val_loss = 0.0
        
        pbar_val = tqdm(self.val_dataloader, desc=f"Epoch {epoch:03d}/{self.epochs} [VAL]  ")
        
        with torch.no_grad(): # Disable physics gradient tracking to save memory/speed
            for batch in pbar_val:
                theta = batch['theta'].to(self.device)
                condition = batch['condition'].to(self.device)

                with torch.autocast(device_type="cpu" if self.device == "mps" else self.device,
                                    dtype=self.dtype, enabled=self.use_bf16):
                    loss = self.model.compute_loss(theta, condition)

                epoch_val_loss += loss.item()
                pbar_val.set_postfix({"val_loss": f"{loss.item():.4f}"})
                
        avg_val_loss = epoch_val_loss / len(self.val_dataloader)
        
        print(f"   ↳ Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        filename = "model_best.pt" if is_best else f"model_epoch_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, path)