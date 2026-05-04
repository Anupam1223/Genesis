import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class SMPCTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, learning_rate=1e-3, epochs=50, device="mps", log_to_wandb=False):
        # Move the model to the Apple Metal GPU
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device
        self.log_to_wandb = log_to_wandb
        
        # AdamW is the industry standard optimizer for Normalizing Flows
        # weight_decay increased heavily to combat the massive train/val overfitting gap
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)

        # ReduceLROnPlateau: Halves the LR when val_loss stops improving for 3 epochs.
        # This is the #1 fix for the "jumping validation loss" problem in Normalizing Flows.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Setup local checkpoint directory
        self.checkpoint_dir = "outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track the best validation loss to save the best model
        self.best_val_loss = float('inf')

    def train(self):
        print(f"\n🚀 Starting training on device: {self.device.upper()}")
        
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

                # NOTE: torch.autocast float16 is intentionally NOT used here.
                # Normalizing Flows use exp(s) in affine coupling layers which overflows
                # in float16 (max ~65504), causing NaN loss and training collapse.
                # MPS speedup comes from num_workers + batch size instead.
                loss = self.model.compute_loss(theta, condition)

                loss.backward()
                # Strict grad clipping to prevent unstable MLP extrapolation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Only log every 50 steps — wandb.log() has network overhead.
                if self.log_to_wandb and global_step % 50 == 0:
                    wandb.log({
                        "train/batch_loss": loss.item()
                    })
                
                global_step += 1
            
            avg_train_loss = epoch_train_loss / len(self.train_dataloader)
            
            # --- VALIDATION PHASE ---
            # Pass the train loss in so we can print it accurately
            avg_val_loss = self.evaluate(epoch, avg_train_loss)
            
            # --- LR SCHEDULING ---
            # Step the scheduler based on val loss — this is what tames the jumping curve
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
                
            # Save regular checkpoint every 10 epochs
            elif epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_train_loss, avg_val_loss)
                
        print("\n✅ Training Complete!")

    def evaluate(self, epoch, avg_train_loss):
        """
        Runs the model on unseen data without updating weights.
        """
        self.model.eval() # Turn off training features like Dropout
        epoch_val_loss = 0.0
        
        pbar_val = tqdm(self.val_dataloader, desc=f"Epoch {epoch:03d}/{self.epochs} [VAL]  ")
        
        with torch.no_grad(): # Disable physics gradient tracking to save memory/speed
            for batch in pbar_val:
                theta = batch['theta'].to(self.device)
                condition = batch['condition'].to(self.device)

                loss = self.model.compute_loss(theta, condition)

                epoch_val_loss += loss.item()
                pbar_val.set_postfix({"val_loss": f"{loss.item():.4f}"})
                
        avg_val_loss = epoch_val_loss / len(self.val_dataloader)
        
        # --- FIX: PRINT BUG ---
        # Now correctly shows the difference between Train and Val performance
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