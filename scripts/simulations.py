import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow

def load_config():
    with open("configs/train.yaml") as f:
        return yaml.safe_load(f)

def main():
    print("🚀 Starting Monte Carlo Dataset Generation for Phase III Handoff...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Setup & Load Config
    cfg = load_config()
    output_dir = "outputs/simulation"
    os.makedirs(output_dir, exist_ok=True)
    num_samples = 5000 # The Genesis Mission target

    # 2. Load the Test Dataset to get realistic 'Measured-Now' (x) conditions
    print("📦 Loading baseline conditions from test split...")
    dataset = SCADAPipelineDataset(
        data_path=cfg["data"]["path"], split="test", log_to_wandb=False
    )
    
    # Randomly select 5,000 operating conditions (x) from the test set
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    # Extract the 'condition' tensors and original unscaled data (for context)
    conditions = torch.stack([dataset[int(i)]["condition"] for i in sample_idxs]).to(device)
    
    # 3. Build Model & Load Best Weights
    print("🧠 Loading trained Normalizing Flow weights...")
    dim_theta = dataset[0]["theta"].shape[0]
    dim_condition = dataset[0]["condition"].shape[0]
    
    m = cfg["model"]
    model = PipelineConditionalFlow(
        dim_theta=dim_theta,
        dim_condition=dim_condition,
        num_layers=m["num_layers"],
        hidden_dim=m["hidden_dim"],
        num_bins=m["num_bins"],
        bound=m["bound"],
        mlp_layers=m["mlp_layers"],
        dropout_rate=m["dropout_rate"],
    ).to(device)
    
    ckpt = torch.load("outputs/checkpoints/model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Generate the Futures!
    print(f"🎲 Simulating {len(conditions)} distinct futures based on historical conditions...")
    with torch.no_grad():
        # sample() generates exactly one future per row in conditions.
        # conditions.shape = (5000, dim_condition) → output shape = (5000, dim_theta)
        generated_thetas = model.sample(condition=conditions).cpu().numpy()
        
    conditions_np = conditions.cpu().numpy()

    # 5. Package into a DataFrame for the Physics Team
    print("🏗️ Packaging dataset for Phase III Tensor Network...")
    
    # Reconstruct column names
    x_cols = dataset.x_cols
    theta_cols = dataset.theta_cols
    
    # Combine conditions (x) and generated futures (theta)
    full_data = np.hstack([conditions_np, generated_thetas])
    all_cols = x_cols + theta_cols
    
    df_monte_carlo = pd.DataFrame(full_data, columns=all_cols)
    
    # Save to Parquet
    output_path = os.path.join(output_dir, "phase3_monte_carlo_dataset.parquet")
    df_monte_carlo.to_parquet(output_path, index=False)
    
    print("═" * 60)
    print(f"✅ Handoff Dataset Created Successfully!")
    print(f"   Shape: {df_monte_carlo.shape} (Rows: Scenarios, Cols: x + theta)")
    print(f"   Saved to: {output_path}")
    print("   Ready to send to the Tensor Network physics team!")
    print("═" * 60)

    # 6. VISUALIZE: Generated θ vs Real Test θ
    # Validates §8.4 — marginal histogram reproduction + pairwise correlation structure
    print("📊 Generating validation figures...")
    _visualize(dataset, sample_idxs, generated_thetas, theta_cols, output_dir)

def _visualize(dataset, sample_idxs, generated_thetas, theta_cols, output_dir):
    n_dims = len(theta_cols)

    # --- Collect real theta from the same sampled test indices ---
    real_thetas = np.stack([dataset[int(i)]["theta"].numpy() for i in sample_idxs])

    # ── FIGURE 1: Marginal Histograms (Real vs Generated per PCA coefficient) ──
    ncols = 4
    nrows = (n_dims + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for d in range(n_dims):
        ax = axes[d]
        ax.hist(real_thetas[:, d],      bins=60, density=True, alpha=0.55,
                color='steelblue', label='Real (test)')
        ax.hist(generated_thetas[:, d], bins=60, density=True, alpha=0.55,
                color='darkorange', label='Generated')
        ax.set_title(theta_cols[d], fontsize=8)
        ax.set_xlabel('Scaled value', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6)
        if d == 0:
            ax.legend(fontsize=7)

    for j in range(n_dims, len(axes)):
        axes[j].set_visible(False)

    fig1.suptitle('Marginal Distributions: Real vs Generated θ\n'
                  '(Good model → orange overlaps blue)', fontsize=11)
    fig1.tight_layout()
    path1 = os.path.join(output_dir, "validation_marginals.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"   Saved: {path1}")

    # ── FIGURE 2: Pairwise Correlation Heatmaps (Real vs Generated) ──
    real_corr = np.corrcoef(real_thetas.T)
    gen_corr  = np.corrcoef(generated_thetas.T)
    diff_corr = gen_corr - real_corr

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    short_names = [c.replace('PCA_Coefficient_', 'PC') for c in theta_cols]

    for ax, mat, title, cmap, vmin, vmax in [
        (axes2[0], real_corr,  'Real θ Correlations',       'coolwarm', -1, 1),
        (axes2[1], gen_corr,   'Generated θ Correlations',  'coolwarm', -1, 1),
        (axes2[2], diff_corr,  'Difference (Gen − Real)',   'RdBu_r',  -0.3, 0.3),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_dims)); ax.set_xticklabels(short_names, rotation=45, fontsize=7)
        ax.set_yticks(range(n_dims)); ax.set_yticklabels(short_names, fontsize=7)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n_dims):
            for j in range(n_dims):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                        fontsize=5, color='black')

    fig2.suptitle('Pairwise Correlation Structure: Real vs Generated θ\n'
                  '(Good model → Difference panel near zero everywhere)', fontsize=11)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "validation_correlations.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   Saved: {path2}")

if __name__ == "__main__":
    main()