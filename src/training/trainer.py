import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for background training loops
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as skPCA
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

        # ReduceLROnPlateau — halves LR when val_loss stalls for `patience` epochs.
        # CosineAnnealingLR was tried (lr=2e-4) but caused catastrophic overfitting:
        # train loss → -9.7 while val loss shot to +8.9 by epoch 18.
        # The high fixed starting LR bent splines far past the data distribution.
        # ReduceLROnPlateau is safer: it starts at a conservative lr and only
        # decays when the model genuinely stops improving on unseen data.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min
        )
        
        # Setup local checkpoint directory
        self.checkpoint_dir = "outputs/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track the best validation loss to save the best model
        self.best_val_loss = float('inf')

        # Fixed probe batch for epoch-level W&B visualizations.
        # Randomly sampled from ACROSS the full val set (not just the first sequential
        # batch) so the probe represents diverse operating conditions rather than one
        # narrow time window — avoids the "spiky histogram" artefact.
        all_theta  = torch.cat([b["theta"]     for b in val_dataloader], dim=0)
        all_cond   = torch.cat([b["condition"] for b in val_dataloader], dim=0)
        probe_size = min(500, all_theta.shape[0])
        perm       = torch.randperm(all_theta.shape[0])[:probe_size]
        self._probe_theta = all_theta[perm]
        self._probe_cond  = all_cond[perm]

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
            # ReduceLROnPlateau: pass val loss so it can detect plateaus
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
                self._log_epoch_visuals(epoch)
                
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

    @torch.no_grad()
    def _log_epoch_visuals(self, epoch):
        """
        Logs two W&B images once per epoch:
          viz/latent_space  — PCA-2D projection of z (should gaussianise as training progresses)
          viz/spline_shapes — learned spline curve per coupling layer for one probe sample
        """
        from src.models.components import rational_quadratic_spline
        import torch.nn.functional as F

        self.model.eval()
        theta_probe = self._probe_theta.to(self.device)
        cond_probe  = self._probe_cond.to(self.device)

        # ── 1. LATENT SPACE ─────────────────────────────────────────────────────
        z, _ = self.model.forward(theta_probe, cond_probe)
        z_np = z.cpu().float().numpy()                          # (N, dim_theta)

        pca2d = skPCA(n_components=2)
        z_2d  = pca2d.fit_transform(z_np)                      # (N, 2)

        # Colour by ||z|| (L2 norm across all 8 dims).
        # Under a perfect N(0,I_8) this should be chi-distributed with mean ~2.8.
        # Points far from that value = flow hasn't fully gaussianised yet.
        z_norms = np.linalg.norm(z_np, axis=1)                 # (N,)

        fig_lat, axes_lat = plt.subplots(1, 2, figsize=(12, 5))

        # ── Left: scatter coloured by ||z|| ─────────────────────────────────────
        ax = axes_lat[0]
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=z_norms, cmap='plasma',
                        s=6, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label='‖z‖  (L2 norm, ideal ≈ 2.8 for 8-dim)')
        ax.set_title(f'Latent Space z — Epoch {epoch}\n'
                     'Goal: tight blob, uniform colour (= standard normal)')
        ax.set_xlabel('PCA dim 1')
        ax.set_ylabel('PCA dim 2')
        ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.4)

        # ── Right: histogram of ||z|| vs ideal chi distribution ─────────────────
        ax2 = axes_lat[1]
        ax2.hist(z_norms, bins=40, density=True, alpha=0.7,
                 color='steelblue', label='Observed ‖z‖')
        # Ideal chi(k=8) pdf for reference
        import scipy.stats as _stats
        x_chi = np.linspace(0, z_norms.max() * 1.3, 300)
        ax2.plot(x_chi, _stats.chi.pdf(x_chi, df=8),
                 'r--', lw=1.5, label='Ideal χ(df=8)')
        ax2.set_xlabel('‖z‖')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of ‖z‖\nRed = target if z ~ N(0,I₈)')
        ax2.legend(fontsize=8)

        fig_lat.tight_layout()

        # ── 2. SPLINE SHAPES ────────────────────────────────────────────────────
        # For each coupling layer plot the learned spline for dim-0 of theta_2
        # using the first probe sample as the conditioning context.
        n_layers = len(self.model.layers)
        ncols    = (n_layers + 1) // 2
        fig_spl, axes_spl = plt.subplots(2, ncols, figsize=(ncols * 3, 6))
        axes_spl = axes_spl.flatten()

        bound  = self.model.bound
        x_grid = torch.linspace(-bound * 0.99, bound * 0.99, 300, device=self.device)

        # Walk N_PROBE different samples through the flow simultaneously.
        # Each gets its own personalised W,H,D from the MLP → shows the
        # SPREAD of spline shapes across different operating conditions,
        # not just one arbitrary sample.
        N_PROBE   = min(10, theta_probe.shape[0])
        z_tmp     = theta_probe[:N_PROBE].clone()   # (N_PROBE, 8)
        cond_multi = cond_probe[:N_PROBE]            # (N_PROBE, 18)

        import torch.nn.functional as _F
        DIM_COLORS = ['#4fc3f7', '#ff8a65', '#a5d6a7', '#ce93d8']  # one colour per theta2 dim

        for i, layer in enumerate(self.model.layers):
            half   = layer.half_dim
            z1, z2 = z_tmp[:, :half], z_tmp[:, half:]   # both (N_PROBE, 4)

            # Brain produces params for all N_PROBE samples at once
            raw = layer.brain(torch.cat([z1, cond_multi], dim=-1))
            raw = raw.reshape(N_PROBE, z2.shape[-1], layer.params_per_dim)  # (N_PROBE, 4, 3K-1)

            K      = layer.num_bins
            n_dims = z2.shape[-1]
            ax     = axes_spl[i]

            for d in range(n_dims):
                colour = DIM_COLORS[d % len(DIM_COLORS)]
                for s in range(N_PROBE):
                    W_raw = raw[s, d, :K].unsqueeze(0).expand(300, -1)
                    H_raw = raw[s, d, K:2*K].unsqueeze(0).expand(300, -1)
                    D_raw = raw[s, d, 2*K:].unsqueeze(0).expand(300, -1)

                    y_grid, _ = rational_quadratic_spline(
                        x_grid, W_raw, H_raw, D_raw, inverse=False, bound=bound
                    )
                    # First sample of dim 0 gets a legend label; rest are faint repeats
                    lbl = f'θ{d}' if (s == 0) else None
                    ax.plot(x_grid.cpu().numpy(), y_grid.cpu().numpy(),
                            lw=1.0, alpha=0.35, color=colour, label=lbl)

                # ── Knot dots for sample 0 of dim 0 only (keeps plot readable) ──
                if d == 0:
                    w_norm = _F.softmax(raw[0, d, :K], dim=-1)
                    w_norm = 1e-2 + (1 - 1e-2 * K) * w_norm
                    knot_x = torch.cumsum(w_norm, dim=0)
                    knot_x = (bound * 2.0) * knot_x / knot_x[-1] - bound
                    knot_x = torch.cat([torch.tensor([-bound], device=self.device), knot_x])

                    h_norm = _F.softmax(raw[0, d, K:2*K], dim=-1)
                    h_norm = 1e-2 + (1 - 1e-2 * K) * h_norm
                    knot_y = torch.cumsum(h_norm, dim=0)
                    knot_y = (bound * 2.0) * knot_y / knot_y[-1] - bound
                    knot_y = torch.cat([torch.tensor([-bound], device=self.device), knot_y])

                    ax.scatter(knot_x.cpu().numpy(), knot_y.cpu().numpy(),
                               s=22, zorder=5, color='#888', edgecolors='white', linewidths=0.5)

            # ── Dashed bounding box (exactly like the screenshot) ─────────────
            box_x = [-bound, bound, bound, -bound, -bound]
            box_y = [-bound, -bound, bound, bound, -bound]
            ax.plot(box_x, box_y, 'c--', lw=1.2, alpha=0.8)

            # ── Identity diagonal (dashed) ─────────────────────────────────────
            margin = bound * 0.4
            ax.plot([-bound - margin, -bound], [-bound - margin, -bound],
                    'c--', lw=1.0, alpha=0.5)
            ax.plot([bound, bound + margin], [bound, bound + margin],
                    'c--', lw=1.0, alpha=0.5)
            ax.plot([-bound, bound], [-bound, bound], color='gray', lw=0.8,
                    linestyle='--', alpha=0.4)

            ax.set_xlim(-bound - margin, bound + margin)
            ax.set_ylim(-bound - margin, bound + margin)
            ax.set_title(f'Layer {i}  ({n_dims} dims × {N_PROBE} samples)', fontsize=8)
            ax.set_xlabel('Input Data (x)', fontsize=7)
            ax.set_ylabel('Latent Space (y)', fontsize=7)
            ax.set_aspect('equal')
            ax.grid(False)
            ax.set_facecolor('#0d1117')
            ax.tick_params(labelsize=6)
            if i == 0 and n_dims <= 4:
                ax.legend(fontsize=6, loc='upper left')

            # Advance all N_PROBE samples through this layer + flip (mirrors model.forward)
            z_out, _ = layer(z_tmp, cond_multi)
            z_tmp    = torch.flip(z_out, dims=[-1]).contiguous()

        for j in range(n_layers, len(axes_spl)):
            axes_spl[j].set_visible(False)

        fig_spl.suptitle(
            f'Learned Spline Shapes — Epoch {epoch}\n'
            'Diagonal = identity (untrained). Curves bending = flow learning.',
            fontsize=10
        )
        fig_spl.tight_layout()

        wandb.log({
            "epoch":              epoch,
            "viz/latent_space":   wandb.Image(fig_lat),
            "viz/spline_shapes":  wandb.Image(fig_spl),
        })
        plt.close(fig_lat)
        plt.close(fig_spl)

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