"""
evaluate.py — Post-training evaluation for Genesis Neural Spline Flow.

Panels generated
────────────────
1.  NLL histogram          — how confidently the model scores every test sample
5.  PC reconstruction error — |true PCA coeff − flow mean prediction| per PC
6.  Coverage calibration   — are the model's uncertainty intervals honest?
9.  Marginal histograms    — true vs model-generated PCA coefficient distributions
10. Sampled trajectories   — predicted 4-hour sensor bands vs actual history
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
           "#8b5cf6", "#ec4899", "#14b8a6", "#f43f5e"]
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})

IMAGE_DIR      = "outputs/images"
CHECKPOINT_DIR = "outputs/checkpoints"
WINDOW_SIZE    = 14400   # seconds — 4-hour window used in preprocessing
DOWNSAMPLE     = 60      # 1-minute intervals after downsampling

os.makedirs(IMAGE_DIR, exist_ok=True)

PC_NAMES = [f"PC{i+1:02d}" for i in range(8)]

THETA_SENSOR_NAMES = [
    "Thermal Cycle Efficiency",
    "Isentropic Efficiency",
    "Turbine Heat Rate",
]

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
def compute_nll(model, dataloader, device):
    """Return per-sample NLL for the full test set."""
    all_nll = []
    for batch in dataloader:
        theta = batch["theta"].to(device)
        cond  = batch["condition"].to(device)
        z, log_det = model.forward(theta, cond)
        bp  = model.get_blueprint()
        nll = -(bp.log_prob(z) + log_det)
        all_nll.append(nll.cpu())
    return torch.cat(all_nll)

# ─────────────────────────────────────────────────────────────────────────────
def panel_nll_histogram(nll):
    """
    Panel 1 — NLL Histogram.
    Shows how confidently the model scores every test sample.
    A distribution centred well to the left (low NLL) means the model
    assigns high probability to what actually happened. A long right tail
    reveals timesteps the model struggles with.
    """
    vals = nll.numpy()
    q1, q99 = np.percentile(vals, 1), np.percentile(vals, 99)
    clipped = vals[(vals >= q1) & (vals <= q99)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clipped, bins=80, color=PALETTE[0], alpha=0.8, edgecolor="none")
    ax.axvline(vals.mean(),     color="black",    lw=1.5, linestyle="--",
               label=f"Mean   = {vals.mean():.2f}")
    ax.axvline(np.median(vals), color=PALETTE[1], lw=1.5, linestyle=":",
               label=f"Median = {np.median(vals):.2f}")
    ax.set_xlabel("Per-sample NLL  (lower = model assigns higher probability)")
    ax.set_ylabel("Count")
    ax.set_title("Panel 1 — Test Set NLL Distribution\n"
                 "Left-skewed = consistently confident  |  Long right tail = hard timesteps")
    ax.legend()
    path = os.path.join(IMAGE_DIR, "p1_nll_histogram.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [1] NLL histogram          → {path}")
    return vals.mean(), np.median(vals)

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_reconstruction_error(model, dataset, device, n_samples=2000, n_draws=500):
    """
    Panel 5 — Per-PC Reconstruction Error.
    For each of the 8 PCA components, measures how far the model's mean
    prediction is from the true value (MAE in scaled units).
    Short bars = the model predicts that PC accurately.
    Tall bars  = the model struggles to pin down that shape of the trajectory.
    """
    rng    = np.random.default_rng(1)
    idxs   = rng.choice(len(dataset), size=n_samples, replace=False)
    errors = np.zeros((n_samples, 8))
    for i, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        cond   = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
        true   = sample["theta"].numpy()
        draws  = model.sample(condition=cond).cpu().numpy()
        errors[i] = np.abs(draws.mean(axis=0) - true)
    mae = errors.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(PC_NAMES, mae, color=PALETTE[:8], alpha=0.85, edgecolor="none")
    for bar, v in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean |posterior mean − true|  (scaled units)")
    ax.set_title("Panel 5 — Per-PC Reconstruction Error\n"
                 "Short bars = accurate  |  Tall bars = model uncertain about that trajectory shape")
    path = os.path.join(IMAGE_DIR, "p5_reconstruction_error.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [5] Reconstruction error   → {path}")
    return mae

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_coverage_calibration(model, dataset, device, n_samples=1000, n_draws=1000):
    """
    Panel 6 — Coverage Calibration.
    Checks whether the model's uncertainty intervals are honest.
    The x-axis is the interval width in standard deviations (k).
    The y-axis is the fraction of true values that actually fall inside.
    - On the black dashed line = perfectly calibrated
    - Above the line           = overconfident (intervals too narrow)
    - Below the line           = underconfident (intervals too wide)
    """
    rng      = np.random.default_rng(2)
    idxs     = rng.choice(len(dataset), size=n_samples, replace=False)
    sigmas   = np.arange(0.25, 3.26, 0.25)
    coverage = np.zeros(len(sigmas))
    for idx in idxs:
        sample = dataset[int(idx)]
        cond   = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)
        true   = sample["theta"].numpy()
        draws  = model.sample(condition=cond).cpu().numpy()
        mu     = draws.mean(axis=0)
        sigma  = draws.std(axis=0) + 1e-8
        for k_i, k in enumerate(sigmas):
            coverage[k_i] += float(np.all(np.abs(true - mu) <= k * sigma))
    coverage /= n_samples
    nominal  = 2 * stats.norm.cdf(sigmas) - 1
    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, nominal,  "k--", lw=1.5, label="Ideal (perfectly calibrated)")
    ax.plot(sigmas, coverage, color=PALETTE[0], lw=2, marker="o", label="Model (empirical)")
    ax.fill_between(sigmas, nominal, coverage, alpha=0.2, color=PALETTE[1],
                    label="Calibration gap")
    ax.set_xlabel("k  (interval = posterior mean ± k × posterior std)")
    ax.set_ylabel("Fraction of true values inside interval")
    ax.set_title("Panel 6 — Coverage Calibration\n"
                 "On the line = honest uncertainty  |  Above = overconfident  |  Below = underconfident")
    ax.legend()
    path = os.path.join(IMAGE_DIR, "p6_coverage_calibration.png")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  [6] Coverage calibration   → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_marginal_histograms(model, dataset, device, n_draws=5000):
    """
    Panel 9 — Marginal Histograms.
    For each of the 8 PCA coefficients:
      BLUE bars   = true distribution across the entire test set
      ORANGE line = density of what the model generates
    If orange closely hugs blue → model reproduces the right range and shape.
    A gap means the model generates coefficients that are too wide, too narrow,
    or shifted relative to reality.
    """
    all_true = dataset.theta_tensor.numpy()   # (N, 8)
    rng  = np.random.default_rng(3)
    idxs = rng.choice(len(dataset), size=n_draws, replace=False)
    conds = torch.stack([dataset[int(i)]["condition"] for i in idxs]).to(device)
    samples = model.sample(condition=conds).cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for i in range(8):
        ax   = axes[i]
        lo   = min(all_true[:, i].min(), samples[:, i].min())
        hi   = max(all_true[:, i].max(), samples[:, i].max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(all_true[:, i], bins=bins, density=True,
                color=PALETTE[0], alpha=0.5, edgecolor="none", label="True (test set)")
        kde = gaussian_kde(samples[:, i])
        xs  = np.linspace(lo, hi, 300)
        ax.plot(xs, kde(xs), color=PALETTE[1], lw=2, label="Model samples")
        ax.set_title(PC_NAMES[i], fontsize=9, pad=4)
        ax.set_xlabel("Coefficient value (scaled)", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Panel 9 — Marginal Histograms\n"
        "Blue = true test distribution  |  Orange = model-generated density\n"
        "Orange hugging blue = model reproduces the right range and shape per PC",
        fontsize=10, y=1.01
    )
    path = os.path.join(IMAGE_DIR, "p9_marginal_histograms.png")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  [9] Marginal histograms    → {path}")

# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_pairwise_correlation(model, dataset, device, n_draws=3000):
    """
    Panel 11 — Pairwise Correlation Structure.
    For each pair of PCA coefficients:
      BLUE scatter  = true joint distribution from the test set
      ORANGE scatter = samples drawn from the flow
    If orange overlaps blue → the flow has learned not just marginals but
    the correlations and joint shape between every pair of PCs.
    A mismatch reveals spurious or missing correlations in the model.
    """
    all_true = dataset.theta_tensor.numpy()          # (N, 8)
    rng  = np.random.default_rng(11)
    idxs = rng.choice(len(dataset), size=n_draws, replace=False)
    conds = torch.stack([dataset[int(i)]["condition"] for i in idxs]).to(device)
    samples = model.sample(condition=conds).cpu().numpy()  # (n_draws, 8)
    true_sub = all_true[idxs]                        # same n_draws rows for fair comparison

    n_pcs = 8
    fig, axes = plt.subplots(n_pcs, n_pcs, figsize=(18, 18))
    for i in range(n_pcs):
        for j in range(n_pcs):
            ax = axes[i][j]
            if i == j:
                # Diagonal — marginal density overlay
                lo = min(true_sub[:, i].min(), samples[:, i].min())
                hi = max(true_sub[:, i].max(), samples[:, i].max())
                xs = np.linspace(lo, hi, 200)
                ax.fill_between(xs, gaussian_kde(true_sub[:, i])(xs),
                                alpha=0.4, color=PALETTE[0])
                ax.plot(xs, gaussian_kde(samples[:, i])(xs),
                        color=PALETTE[1], lw=1.2)
                ax.set_yticks([])
            elif i < j:
                # Upper triangle — true data
                ax.scatter(true_sub[:, j], true_sub[:, i],
                           s=1.5, alpha=0.15, color=PALETTE[0], rasterized=True)
            else:
                # Lower triangle — model samples
                ax.scatter(samples[:, j], samples[:, i],
                           s=1.5, alpha=0.15, color=PALETTE[1], rasterized=True)
            if i == n_pcs - 1:
                ax.set_xlabel(PC_NAMES[j], fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(PC_NAMES[i], fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)

    fig.suptitle(
        "Panel 11 — Pairwise Correlation Structure\n"
        "Upper triangle = true test data (blue)  |  Lower triangle = flow samples (orange)  |  Diagonal = marginals\n"
        "Orange matching blue shape = flow has learned the correct joint structure between PCs",
        fontsize=10, y=1.005
    )
    path = os.path.join(IMAGE_DIR, "p11_pairwise_correlation.png")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", dpi=120); plt.close(fig)
    print(f"  [11] Pairwise correlation  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def panel_sampled_trajectories(model, dataset, device, n_draws=100, n_windows=3):
    """
    Panel 10 — Sampled 4-Hour Sensor Trajectories.
    The most physically meaningful panel.

    For each test window:
      - BLACK dashed line = what the sensors actually recorded
      - COLOURED band     = 5–95% range of 100 futures the model considers plausible
      - SOLID line        = model's median prediction

    Black line inside the band = model is well-calibrated for that sensor.
    Band missing the black line = model is biased or miscalibrated.
    Very wide band              = model is uncertain (normal early in training).
    """
    pca          = joblib.load(os.path.join(CHECKPOINT_DIR, "trajectory_pca_model.pkl"))
    coeff_scaler = joblib.load(os.path.join(CHECKPOINT_DIR, "pca_coeff_scaler.pkl"))
    base_scaler  = joblib.load(os.path.join(CHECKPOINT_DIR, "theta_base_scaler.pkl"))

    n_sensors   = len(THETA_SENSOR_NAMES)
    n_timesteps = WINDOW_SIZE // DOWNSAMPLE        # 240 (one point per minute)
    time_axis   = np.arange(n_timesteps) / 60.0   # hours (0 → 4)

    rng  = np.random.default_rng(5)
    idxs = rng.choice(len(dataset), size=n_windows, replace=False)

    fig, axes = plt.subplots(n_windows, n_sensors,
                             figsize=(n_sensors * 3, n_windows * 3.5))
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(idxs):
        sample   = dataset[int(idx)]
        true_pca = sample["theta"].numpy()
        cond     = sample["condition"].unsqueeze(0).repeat(n_draws, 1).to(device)

        sampled_pca_scaled = model.sample(
            condition=cond
        ).cpu().numpy()                                         # (100, 8)

        # Decode: scaled PCA coeffs → physical sensor trajectories
        sampled_pca       = coeff_scaler.inverse_transform(sampled_pca_scaled)
        true_pca_unscaled = coeff_scaler.inverse_transform(true_pca[np.newaxis])[0]

        sampled_flat = sampled_pca @ pca.components_ + pca.mean_
        true_flat    = (true_pca_unscaled[np.newaxis] @ pca.components_) + pca.mean_

        sampled_windows = sampled_flat.reshape(n_draws, n_timesteps, n_sensors)
        true_window     = true_flat.reshape(1, n_timesteps, n_sensors)[0]

        # Vectorised inverse — reshape to (n_draws*240, 3), invert, reshape back
        sampled_windows = base_scaler.inverse_transform(
            sampled_windows.reshape(-1, n_sensors)
        ).reshape(n_draws, n_timesteps, n_sensors)

        # true_window is (240, 3) in RobustScaled space — invert in one call
        true_phys = base_scaler.inverse_transform(true_window)  # (240, n_sensors)

        p5  = np.percentile(sampled_windows, 5,  axis=0)
        p50 = np.percentile(sampled_windows, 50, axis=0)
        p95 = np.percentile(sampled_windows, 95, axis=0)

        for col in range(n_sensors):
            ax = axes[row][col]
            ax.fill_between(time_axis, p5[:, col], p95[:, col],
                            alpha=0.25, color=PALETTE[col], label="5–95% predicted range")
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
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  [10] Sampled trajectories  → {path}")

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
        test_dataset, batch_size=2048, shuffle=False, num_workers=0,
    )

    sample0 = test_dataset[0]
    model = build_model(cfg,
                        dim_theta=sample0["theta"].shape[0],
                        dim_condition=sample0["condition"].shape[0],
                        device=device)

    print("⚙️  Running forward pass on entire test set...")
    nll = compute_nll(model, test_loader, device)
    print(f"   Processed {len(nll):,} test samples.")

    print("\n📊 Generating evaluation panels...")
    mean_nll, median_nll = panel_nll_histogram(nll)
    mae = panel_reconstruction_error(model, test_dataset, device)
    panel_coverage_calibration(model, test_dataset, device)
    panel_marginal_histograms(model, test_dataset, device)
    panel_pairwise_correlation(model, test_dataset, device)
    panel_sampled_trajectories(model, test_dataset, device)

    summary_table(mean_nll, median_nll, mae)
    print(f"✅ All panels saved to  {IMAGE_DIR}/\n")

if __name__ == "__main__":
    main()
