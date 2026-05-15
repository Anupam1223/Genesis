"""
evaluate.py — Full post-training evaluation for Genesis Neural Spline Flow.

Loads the best checkpoint and the untouched test.parquet split, then produces
a comprehensive suite of diagnostics and visualisations saved to outputs/images/.

Panels generated
────────────────
1.  NLL per-sample histogram       — distribution of how well the model scores every test sample
2.  Latent-space Z scatter grid    — are mapped Z's landing in N(0,1) as expected?
3.  Latent marginal Q-Q plots      — quantile comparison vs true standard normal, per PC
4.  Conditional density overlays   — sampled future vs true value for 6 random timesteps
5.  PC reconstruction error        — |true PCA coeff − flow mean prediction| across all 12 PCs
6.  Coverage calibration chart     — % true values inside k-sigma predicted intervals for k=1..3
7.  Condition–NLL heatmap          — does loss vary with operating point (suction pressure × drum temp)?
8.  Sample diversity strip         — 50 draws from the same condition, shows predicted spread
9.  Marginal histograms            — true test distribution vs model-sampled density, per PC
10. Sampled 4-hour trajectories    — 100 flow samples decoded back to physical sensor time-series
                                     plotted as shaded confidence band vs actual history
"""

import os
import sys
import yaml
import torch
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow

# ── Aesthetics ────────────────────────────────────────────────────────────────
PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
           "#8b5cf6", "#ec4899", "#14b8a6", "#f43f5e",
           "#6366f1", "#84cc16", "#fb923c", "#06b6d4"]
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})

IMAGE_DIR = "outputs/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

PC_NAMES = [f"PC{i+1:02d}" for i in range(12)]

THETA_SENSOR_NAMES = [
    "Seal Gas Filter DP",
    "Lube Oil Level",
    "Thermal Cycle Efficiency",
    "Isentropic Efficiency",
    "Discharge Pressure",
    "Discharge Temp",
    "Exhaust Temp Spread",
    "Turbine Heat Rate",
]

CHECKPOINT_DIR = "outputs/checkpoints"
WINDOW_SIZE    = 14400   # seconds — 4-hour window used in preprocessing
DOWNSAMPLE     = 60      # 1-minute intervals after downsampling

# ─────────────────────────────────────────────────────────────────────────────
def load_config():
    with open("configs/train.yaml") as f:
        return yaml.safe_load(f)

def build_model(cfg, dim_theta, dim_condition, device):
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
    )
    ckpt = torch.load("outputs/checkpoints/model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ Loaded checkpoint — best epoch {ckpt.get('epoch','?')}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.4f}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_nll_and_z(model, dataloader, device):
    """Return per-sample NLL, mapped Z tensors, thetas, and conditions."""
    all_nll, all_z, all_theta, all_cond = [], [], [], []
    for batch in dataloader:
        theta = batch["theta"].to(device)
        cond  = batch["condition"].to(device)
        z, log_det = model.forward(theta, cond)
        bp  = model.get_blueprint()
        nll = -(bp.log_prob(z) + log_det)
        all_nll.append(nll.cpu())
        all_z.append(z.cpu())
        all_theta.append(theta.cpu())
        all_cond.append(cond.cpu())
    return (torch.cat(all_nll), torch.cat(all_z),
            torch.cat(all_theta), torch.cat(all_cond))

# ─────────────────────────────────────────────────────────────────────────────
def panel_nll_histogram(nll):
    vals = nll.numpy()
    q1, q99 = np.percentile(vals, 1), np.percentile(vals, 99)
    clipped = vals[(vals >= q1) & (vals <= q99)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clipped, bins=80, color=PALETTE[0], alpha=0.8, edgecolor="none")
    ax.axvline(vals.mean(),    color="black",    lw=1.5, linestyle="--", label=f"Mean   = {vals.mean():.2f}")
    ax.axvline(np.median(vals), color=PALETTE[1], lw=1.5, linestyle=":",  label=f"Median = {np.median(vals):.2f}")
    ax.set_xlabel("Per-sample NLL  (lower = model assigns higher probability)")
    ax.set_ylabel("Count")
    ax.set_title("Panel 1 — Test Set NLL Distribution")
    ax.legend()
    path = os.path.join(IMAGE_DIR, "p1_nll_histogram.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [1] NLL histogram        → {path}")
    return vals.mean(), np.median(vals)

# ─────────────────────────────────────────────────────────────────────────────
def panel_latent_scatter(z):
    z_np = z.numpy()[:5000]
    dims = min(4, z_np.shape[1])
    fig, axes = plt.subplots(dims, dims, figsize=(10, 10))
    for i in range(dims):
        for j in range(dims):
            ax = axes[i][j]
            if i == j:
                ax.hist(z_np[:, i], bins=50, color=PALETTE[i], alpha=0.8, edgecolor="none")
                ax.set_title(PC_NAMES[i], fontsize=8)
            else:
                ax.scatter(z_np[:, j], z_np[:, i], s=1, alpha=0.15, color=PALETTE[i])
            if i == dims - 1: ax.set_xlabel(PC_NAMES[j], fontsize=7)
            if j == 0:        ax.set_ylabel(PC_NAMES[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Panel 2 — Latent Space Z  (should be symmetric, centred at 0)", fontsize=12)
    path = os.path.join(IMAGE_DIR, "p2_latent_scatter.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [2] Latent scatter       → {path}")

# ─────────────────────────────────────────────────────────────────────────────
def panel_qq_plots(z):
    z_np = z.numpy()
    n_pc = z_np.shape[1]
    cols = 4
    rows = int(np.ceil(n_pc / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    ref = np.linspace(-3.5, 3.5, 200)
    for i in range(n_pc):
        ax = axes[i]
        (osm, osr), _ = stats.probplot(z_np[:, i], dist="norm")
        ax.scatter(osm, osr, s=2, alpha=0.3, color=PALETTE[i % len(PALETTE)])
        ax.plot(ref, ref, "k--", lw=1)
        ax.set_title(PC_NAMES[i], fontsize=9)
        ax.set_xlabel("Theoretical quantiles", fontsize=7)
        ax.set_ylabel("Sample quantiles",      fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n_pc, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Panel 3 — Latent Q-Q Plots  (points on the diagonal = perfect Gaussianisation)", fontsize=12)
    path = os.path.join(IMAGE_DIR, "p3_latent_qq.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [3] Q-Q plots            → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_conditional_density(model, dataset, device, n_panels=6, n_draws=3000):
    rng   = np.random.default_rng(0)
    idxs  = rng.choice(len(dataset), size=n_panels, replace=False)
    pcs   = [0, 1, 2, 3]
    fig, axes = plt.subplots(n_panels, len(pcs), figsize=(14, n_panels * 2.5))
    for row, idx in enumerate(idxs):
        sample  = dataset[int(idx)]
        cond    = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
        true_th = sample["theta"].numpy()
        draws   = model.sample(num_samples=n_draws, condition=cond).cpu().numpy()
        for col, pc in enumerate(pcs):
            ax = axes[row][col]
            ax.hist(draws[:, pc], bins=60, density=True,
                    color=PALETTE[pc], alpha=0.7, edgecolor="none")
            ax.axvline(true_th[pc], color="black", lw=2, linestyle="--", label="True")
            if row == 0: ax.set_title(PC_NAMES[pc], fontsize=9)
            if col == 0: ax.set_ylabel(f"t={idx}", fontsize=8, rotation=0, labelpad=30)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)
    fig.suptitle("Panel 4 — Conditional Density: sampled futures vs true realisation\n"
                 "(black dashed = what actually happened)", fontsize=11)
    path = os.path.join(IMAGE_DIR, "p4_conditional_density.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [4] Conditional density  → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_reconstruction_error(model, dataset, device, n_samples=2000, n_draws=500):
    rng    = np.random.default_rng(1)
    idxs   = rng.choice(len(dataset), size=n_samples, replace=False)
    errors = np.zeros((n_samples, 12))
    for i, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        cond   = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
        true   = sample["theta"].numpy()
        draws  = model.sample(num_samples=n_draws, condition=cond).cpu().numpy()
        errors[i] = np.abs(draws.mean(axis=0) - true)
    mae = errors.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(PC_NAMES, mae, color=PALETTE[:12], alpha=0.85, edgecolor="none")
    for bar, v in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Mean |posterior mean − true|  (scaled units)")
    ax.set_title("Panel 5 — Per-PC Reconstruction Error")
    path = os.path.join(IMAGE_DIR, "p5_reconstruction_error.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [5] Reconstruction error → {path}")
    return mae

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_coverage_calibration(model, dataset, device, n_samples=1000, n_draws=1000):
    rng     = np.random.default_rng(2)
    idxs    = rng.choice(len(dataset), size=n_samples, replace=False)
    sigmas  = np.arange(0.25, 3.26, 0.25)
    coverage = np.zeros(len(sigmas))
    for idx in idxs:
        sample = dataset[int(idx)]
        cond   = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
        true   = sample["theta"].numpy()
        draws  = model.sample(num_samples=n_draws, condition=cond).cpu().numpy()
        mu     = draws.mean(axis=0)
        sigma  = draws.std(axis=0) + 1e-8
        for k_i, k in enumerate(sigmas):
            coverage[k_i] += float(np.all(np.abs(true - mu) <= k * sigma))
    coverage /= n_samples
    nominal  = 2 * stats.norm.cdf(sigmas) - 1
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, nominal,  "k--", lw=1.5, label="Ideal Gaussian")
    ax.plot(sigmas, coverage, color=PALETTE[0], lw=2, marker="o", label="Empirical")
    ax.fill_between(sigmas, nominal, coverage, alpha=0.2, color=PALETTE[1], label="Calibration gap")
    ax.set_xlabel("k  (interval = posterior mean ± k × posterior std)")
    ax.set_ylabel("Fraction of true values inside interval")
    ax.set_title("Panel 6 — Coverage Calibration\n"
                 "Above diagonal = overconfident  |  Below = underconfident")
    ax.legend()
    path = os.path.join(IMAGE_DIR, "p6_coverage_calibration.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [6] Coverage calibration → {path}")

# ─────────────────────────────────────────────────────────────────────────────
def panel_nll_heatmap(nll, cond):
    nll_np = nll.numpy()
    p_np   = cond.numpy()[:, 0]
    t_np   = cond.numpy()[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(p_np, t_np,
                    c=np.clip(nll_np, *np.percentile(nll_np, [2, 98])),
                    cmap="RdYlGn_r", s=1, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, label="NLL  (lower = model more confident)")
    ax.set_xlabel("Suction Pressure  (standardised)")
    ax.set_ylabel("Drum Temperature  (standardised)")
    ax.set_title("Panel 7 — NLL by Operating Point\nGreen = confident  |  Red = uncertain")
    path = os.path.join(IMAGE_DIR, "p7_nll_heatmap.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [7] NLL heatmap          → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_sample_diversity(model, dataset, device, n_draws=50):
    rng   = np.random.default_rng(7)
    idx   = int(rng.integers(0, len(dataset)))
    sample = dataset[idx]
    cond   = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
    true   = sample["theta"].numpy()
    draws  = model.sample(num_samples=n_draws, condition=cond).cpu().numpy()
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    for i in range(12):
        ax = axes[i]
        for d in range(n_draws):
            ax.plot(0, draws[d, i], "o", color=PALETTE[i % len(PALETTE)], alpha=0.3, markersize=4)
        ax.axhline(true[i], color="black", lw=2, linestyle="--", label="True")
        ax.set_title(PC_NAMES[i], fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Coeff (scaled)", fontsize=7)
        if i == 0: ax.legend(fontsize=7)
    fig.suptitle(f"Panel 8 — Sample Diversity at Timestep {idx}\n"
                 "50 draws from identical operating condition  |  black = true value", fontsize=11)
    path = os.path.join(IMAGE_DIR, "p8_sample_diversity.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [8] Sample diversity     → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_marginal_histograms(model, dataset, device, n_draws=5000):
    """
    Panel 9 — Marginal Histograms.

    For each of the 12 PCA coefficients we plot two things on the same axis:
      • BLUE bars  — the true distribution of that coefficient across the entire
                     test set (what the real data looks like)
      • ORANGE KDE — the density curve produced by sampling 5,000 times from the
                     model using random test conditions

    If the model has learned the data distribution well, the orange curve should
    closely hug the blue histogram for every PC.  A wide gap means the model is
    generating coefficients with the wrong range or shape for that PC.
    """
    # Collect all true theta values from the test set
    all_true = dataset.theta_tensor.numpy()   # (N, 12)

    # Sample from the model using a random subset of test conditions
    rng  = np.random.default_rng(3)
    idxs = rng.choice(len(dataset), size=n_draws, replace=False)
    conds = torch.stack([dataset[int(i)]["condition"] for i in idxs]).to(device)
    samples = model.sample(num_samples=n_draws, condition=conds).cpu().numpy()  # (n_draws, 12)

    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        lo = min(all_true[:, i].min(), samples[:, i].min())
        hi = max(all_true[:, i].max(), samples[:, i].max())
        bins = np.linspace(lo, hi, 60)

        # True distribution (histogram, normalised to density)
        ax.hist(all_true[:, i], bins=bins, density=True,
                color=PALETTE[0], alpha=0.5, edgecolor="none", label="True (test set)")

        # Model-sampled distribution (smooth KDE curve)
        kde = gaussian_kde(samples[:, i])
        xs  = np.linspace(lo, hi, 300)
        ax.plot(xs, kde(xs), color=PALETTE[1], lw=2, label="Model samples")

        ax.set_title(PC_NAMES[i], fontsize=9, pad=4)
        ax.set_xlabel("Coefficient value (scaled)", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        # Legend only on first subplot to avoid clutter
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Panel 9 — Marginal Histograms\n"
        "Blue = true test distribution  |  Orange = model-generated density\n"
        "Overlap means the model reproduces the right range and shape for each PC",
        fontsize=10, y=1.01
    )
    path = os.path.join(IMAGE_DIR, "p9_marginal_histograms.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [9] Marginal histograms  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_sampled_trajectories(model, dataset, device, n_draws=100, n_windows=3):
    """
    Panel 10 — Sampled 4-Hour Trajectories.

    This is the most physically meaningful panel.  It shows what the model
    actually produces in sensor-space (not PCA-space).

    Pipeline:
      1. Pick a random test window (a real 4-hour historical period).
      2. Feed its measured-now condition x into the flow 100 times →
         100 different sampled theta coefficient vectors.
      3. Decode each theta vector back to physical sensor units by reversing
         the full preprocessing chain:
             scaled coefficients
             → inverse StandardScaler (pca_coeff_scaler)
             → inverse PCA            (trajectory_pca_model)
             → reshape to (240 timesteps × 8 sensors)
             → inverse RobustScaler   (theta_base_scaler)
      4. Plot the 100 reconstructed trajectories as a shaded 5–95 % confidence
         band (light colour) and the median trajectory (solid line).
      5. Overlay the actual historical 4-hour sensor trace (black dashed).

    What you are looking at:
      • The BLACK dashed line  = what the sensors actually recorded.
      • The COLOURED band      = the range of futures the model considers
                                 plausible given the current operating state.
      • If the black line sits INSIDE the band → model is well-calibrated.
      • If the band is VERY WIDE  → model is uncertain (expected early in training).
      • If the band MISSES the black line → model is biased or miscalibrated.
    """
    # ── Load inverse-transform artefacts saved by preprocessing ──────────────
    pca         = joblib.load(os.path.join(CHECKPOINT_DIR, "trajectory_pca_model.pkl"))
    coeff_scaler = joblib.load(os.path.join(CHECKPOINT_DIR, "pca_coeff_scaler.pkl"))
    base_scaler  = joblib.load(os.path.join(CHECKPOINT_DIR, "theta_base_scaler.pkl"))

    n_sensors    = len(THETA_SENSOR_NAMES)          # 8
    n_timesteps  = WINDOW_SIZE // DOWNSAMPLE        # 240  (one point per minute)
    time_axis    = np.arange(n_timesteps) / 60.0    # hours (0 → 4)

    rng  = np.random.default_rng(5)
    idxs = rng.choice(len(dataset), size=n_windows, replace=False)

    fig, axes = plt.subplots(n_windows, n_sensors,
                             figsize=(n_sensors * 3, n_windows * 3.5))
    if n_windows == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing

    for row, idx in enumerate(idxs):
        sample   = dataset[int(idx)]
        true_pca = sample["theta"].numpy()          # (12,) — scaled PCA coeffs
        cond     = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)

        # ── Sample 100 theta vectors from the flow ────────────────────────────
        sampled_pca_scaled = model.sample(
            num_samples=n_draws, condition=cond
        ).cpu().numpy()                              # (100, 12)

        # ── Decode samples: scaled PCA → physical sensor trajectories ─────────
        # Step 1 — undo StandardScaler on PCA coefficients
        sampled_pca = coeff_scaler.inverse_transform(sampled_pca_scaled)   # (100, 12)
        true_pca_unscaled = coeff_scaler.inverse_transform(true_pca[np.newaxis])[0]

        # Step 2 — inverse PCA projection → (n_draws, 240*8) flattened windows
        # (n_draws, 12) @ (12, 1920) + (1920,) = (n_draws, 1920)
        sampled_flat = sampled_pca @ pca.components_ + pca.mean_
        true_flat    = (true_pca_unscaled[np.newaxis] @ pca.components_) + pca.mean_

        # Step 3 — reshape to (100, 240, 8)
        sampled_windows = sampled_flat.reshape(n_draws,   n_timesteps, n_sensors)
        true_window     = true_flat.reshape(1, n_timesteps, n_sensors)[0]

        # Step 4 — undo RobustScaler on each sensor column
        #          RobustScaler operates per-column; we apply it timestep-by-timestep
        for t in range(n_timesteps):
            sampled_windows[:, t, :] = base_scaler.inverse_transform(
                sampled_windows[:, t, :]
            )
            if t == 0:
                true_window[t] = base_scaler.inverse_transform(
                    true_window[t][np.newaxis]
                )[0]
        # Undo remaining true timesteps
        true_phys = np.stack([
            base_scaler.inverse_transform(true_window[t][np.newaxis])[0]
            for t in range(n_timesteps)
        ])                                           # (240, 8)

        # ── Plot ──────────────────────────────────────────────────────────────
        p5  = np.percentile(sampled_windows, 5,  axis=0)   # (240, 8)
        p50 = np.percentile(sampled_windows, 50, axis=0)
        p95 = np.percentile(sampled_windows, 95, axis=0)

        for col in range(n_sensors):
            ax = axes[row][col]
            ax.fill_between(time_axis, p5[:, col], p95[:, col],
                            alpha=0.25, color=PALETTE[col],
                            label="5–95% predicted range")
            ax.plot(time_axis, p50[:, col],
                    color=PALETTE[col], lw=1.5, label="Predicted median")
            ax.plot(time_axis, true_phys[:, col],
                    color="black", lw=1.5, linestyle="--", label="Actual")

            if row == 0:
                ax.set_title(THETA_SENSOR_NAMES[col], fontsize=8, pad=4)
            if col == 0:
                ax.set_ylabel(f"Window {idx}\n(physical units)", fontsize=7)
            if row == n_windows - 1:
                ax.set_xlabel("Hours ahead", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Panel 10 — Sampled 4-Hour Sensor Trajectories\n"
        "Shaded band = 5–95% of 100 flow samples  |  Black dashed = actual history\n"
        "Band containing the black line = model is well-calibrated for that sensor",
        fontsize=10, y=1.01
    )
    path = os.path.join(IMAGE_DIR, "p10_sampled_trajectories.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [10] Sampled trajectories → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def summary_table(mean_nll, median_nll, mae):
    print("\n" + "═" * 52)
    print("  EVALUATION SUMMARY")
    print("═" * 52)
    print(f"  Mean NLL   (test set) : {mean_nll:>10.4f}  nats")
    print(f"  Median NLL (test set) : {median_nll:>10.4f}  nats")
    print(f"  {'─' * 30}")
    print(f"  {'PC':<6}  MAE (scaled units)")
    print(f"  {'─' * 30}")
    for i, v in enumerate(mae):
        print(f"  {PC_NAMES[i]:<6}  {v:.4f}")
    print("═" * 52 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🚀 Genesis Evaluation — device: {device.upper()}")
    print("─" * 52)

    cfg = load_config()

    print("📦 Loading test.parquet  (untouched holdout split)...")
    test_dataset = SCADAPipelineDataset(
        data_path=cfg["data"]["path"], split="test", log_to_wandb=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2048, shuffle=False,
        num_workers=0,   # MPS + macOS: shared-memory spawn crashes with num_workers > 0
    )

    sample0 = test_dataset[0]
    model = build_model(cfg,
                        dim_theta=sample0["theta"].shape[0],
                        dim_condition=sample0["condition"].shape[0],
                        device=device)

    print("⚙️  Running forward pass on entire test set...")
    nll, z, theta, cond = compute_nll_and_z(model, test_loader, device)
    print(f"   Processed {len(nll):,} test samples.")

    print("\n📊 Generating evaluation panels...")
    mean_nll, median_nll = panel_nll_histogram(nll)
    panel_latent_scatter(z)
    panel_qq_plots(z)
    panel_conditional_density(model, test_dataset, device)
    mae = panel_reconstruction_error(model, test_dataset, device)
    panel_coverage_calibration(model, test_dataset, device)
    panel_nll_heatmap(nll, cond)
    panel_sample_diversity(model, test_dataset, device)
    panel_marginal_histograms(model, test_dataset, device)
    panel_sampled_trajectories(model, test_dataset, device)

    summary_table(mean_nll, median_nll, mae)
    print(f"✅ All panels saved to  {IMAGE_DIR}/\n")

if __name__ == "__main__":
    main()