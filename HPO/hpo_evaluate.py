import os
import sys
import json
import torch
import glob
import yaml
from torch.utils.data import DataLoader

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import SCADAPipelineDataset
from src.models.flow_model import PipelineConditionalFlow

# Import the existing evaluation logic instead of rewriting everything!
import scripts.evaluate as ev

def main():
    print("🔍 Scanning for best HPO models...")
    models_dir = "HPO/models"
    results_dir = "HPO/results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    # List all trained model runs
    run_folders = [f.path for f in os.scandir(models_dir) if f.is_dir()]
    
    if not run_folders:
        print("No models found. Run the HPO sweep first!")
        return

    print(f"Found {len(run_folders)} run directories. Evaluating...")

    # ── STEP 1: RANK ALL RUNS BY VAL LOSS ────────────────────────────────────
    ranked_runs = []
    for folder in run_folders:
        config_path = os.path.join(folder, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                info = json.load(f)
            ranked_runs.append((info.get('val_loss', float('inf')), folder))
    ranked_runs.sort(key=lambda x: x[0])

    # ── STEP 2: RENAME MODEL FOLDERS WITH RANK LABELS ────────────────────────
    import shutil
    renamed_folders = []
    for rank, (val_loss, folder) in enumerate(ranked_runs, start=1):
        run_name = os.path.basename(folder)
        # Avoid double-labelling if we already renamed this in a previous run
        if f"-best-{rank}" not in run_name:
            # Strip any existing rank suffix first (e.g. from a previous evaluate run)
            base_name = run_name.rsplit("-best-", 1)[0]
            new_name = f"{base_name}-best-{rank}"
        else:
            new_name = run_name
            base_name = run_name.rsplit("-best-", 1)[0]
        new_folder = os.path.join(models_dir, new_name)
        if folder != new_folder:
            os.rename(folder, new_folder)
            print(f"   📝 Renamed: {run_name}  →  {new_name}  (val_loss={val_loss:.4f})")
        renamed_folders.append((rank, val_loss, new_name, new_folder))

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load dataset only once across evaluations to save memory
    with open("configs/train.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
        
    print("📦 Loading test.parquet...")
    test_dataset = SCADAPipelineDataset(data_path=base_cfg['data']['path'], split="test", log_to_wandb=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=4)

    for rank, val_loss, run_name, folder in renamed_folders:
        best_model_path = os.path.join(folder, "model_best.pt")
        config_path = os.path.join(folder, "model_config.json")
        
        if os.path.exists(best_model_path) and os.path.exists(config_path):
            run_result_dir = os.path.join(results_dir, run_name)
            os.makedirs(run_result_dir, exist_ok=True)
            
            # Redirect the evaluate module's save path for this specific run
            ev.IMAGE_DIR = run_result_dir
            
            print(f"\n=======================================================")
            print(f"🏆 Rank {rank} | val_loss={val_loss:.4f}")
            print(f"🔬 Evaluating {run_name} into {run_result_dir}...")
            
            with open(config_path, "r") as f:
                model_kwargs = json.load(f)
            
            # Remove injected val_loss parameter before building the model
            model_kwargs.pop('val_loss', None)
                
            # 2. Build the model specifically for this run's architecture
            model = PipelineConditionalFlow(**model_kwargs)
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            
            # 3. Call the imported evaluation routines directly!
            nll, z, theta, cond = ev.compute_nll_and_z(model, test_loader, device)
            mean_nll, median_nll = ev.panel_nll_histogram(nll)
            ev.panel_latent_scatter(z)
            ev.panel_qq_plots(z)
            ev.panel_conditional_density(model, test_dataset, device)
            mae = ev.panel_reconstruction_error(model, test_dataset, device)
            ev.panel_coverage_calibration(model, test_dataset, device)
            ev.panel_nll_heatmap(nll, cond)
            ev.panel_sample_diversity(model, test_dataset, device)

            ev.summary_table(mean_nll, median_nll, mae)
            
    print("✅ All HPO models evaluated.")

if __name__ == "__main__":
    main()