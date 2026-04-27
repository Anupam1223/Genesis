import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class SMPCTrainer:
    def __init__(self, model, dataloader, learning_rate=1e-3, epochs=50, device="mps", log_to_wandb=False):
        # Move the model to the Apple Metal GPU
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.log_to_wandb = log_to_wandb
        
        # AdamW is the industry standard optimizer for Normalizing Flows
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Setup local checkpoint directory
        self.checkpoint_dir = "outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        print(f"\n🚀 Starting training on device: {self.device.upper()}")
        self.model.train()
        
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            
            # Progress bar for the batch loop
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch:03d}/{self.epochs}")
            
            for batch in pbar:
                # Move data to Apple Metal GPU
                theta = batch['theta'].to(self.device)
                condition = batch['condition'].to(self.device)
                
                # 1. Zero out old gradients
                self.optimizer.zero_grad()
                
                # 2. Forward Pass & Loss Calculation (from your wrapper!)
                loss = self.model.compute_loss(theta, condition)
                
                # 3. Backward Pass (Calculate the physics gradients)
                loss.backward()
                
                # 4. Update the neural network weights
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # End of epoch tracking
            avg_loss = epoch_loss / len(self.dataloader)
            
            if self.log_to_wandb:
                wandb.log({"epoch": epoch, "loss": avg_loss})
                
            # Save local checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)
                
        print("\n✅ Training Complete!")

    def save_checkpoint(self, epoch, loss):
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)