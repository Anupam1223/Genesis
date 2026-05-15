# Genesis Hyperparameter Optimization (HPO)

This folder contains a fully isolated, automated Hyperparameter Optimization (HPO) pipeline powered by **Weights & Biases (W&B) Sweeps**. 

The goal of this module is to intelligently discover the best combination of macro-architecture parameters (like number of spline layers) and micro-architecture parameters (like hidden dimensions and dropout) to prevent model overfitting without you having to manually tweak `configs/train.yaml`.

## 📂 Folder Structure

- **`sweep_config.yaml`**: The definition of the search space. It lists all the parameters (learning rate, weight decay, layers, bins, etc.) the Bayesian engine is allowed to tweak.
- **`hpo_train.py`**: The dynamic training wrapper. It asks W&B for a set of parameters, automatically detects if your base model accepts them, and runs the training loop. It enforces a strict **Top 5** rule, automatically pruning bad models from your disk to save space.
- **`hpo_evaluate.py`**: The post-sweep crawler. Once you finish your sweep, this script goes through your saved Top 5 models, dynamically rebuilds their custom architectures, and runs the full visual evaluation pipeline for each.
- **`models/`**: (Auto-generated) Where the top 5 checkpoint files and their specific `model_config.json` blueprints are saved.
- **`results/`**: (Auto-generated) Where the visual evaluation panels for your top 5 models are saved.

---

## 🚀 How to Run the HPO Sweep

You can run the entire automated Hyperparameter Optimization pipeline—creating the sweep, running trials, enforcing the top 5 models rule, and evaluating them—with a single command from the project root:

```bash
./HPO/run.sh
```

*(If you want to run more or fewer than 15 trials, simply edit `HPO/run.sh` and change the `--count 15` parameter).*

### What happens automatically:
1. **Sweep Creation**: Reads `sweep_config.yaml` and starts a Bayesian optimization sweep on W&B.
2. **Intelligent Training**: The W&B agent automatically runs the requested number of trials using `hpo_train.py`.
3. **Top 5 Pruning**: After every run, the script aggressively deletes underperforming iterations from your disk, leaving only the 5 lowest-loss models inside `HPO/models/`.
4. **Evaluation**: Immediately upon finishing, `hpo_evaluate.py` generates the complete evaluation dashboard (all 8 figures) for your top 5 models, saving them neatly in `HPO/results/<run_name>/`.