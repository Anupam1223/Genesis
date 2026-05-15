#!/bin/zsh

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
sweep_file="$script_dir/sweep_config.yaml"
evaluate_script="$script_dir/hpo_evaluate.py"

cd "$project_root"

# 1. Initialize the W&B Sweep and extract the Agent ID
echo "🚀 Initializing W&B Sweep..."
wandb_output=$(wandb sweep "$sweep_file" 2>&1) || {
    echo "$wandb_output"
    echo "❌ Failed to create sweep. Most likely cause: W&B is not logged in or the sweep file path is wrong."
    echo "   Fix: run \`wandb login\` once, then retry \`./HPO/run.sh\`."
    exit 1
}

AGENT_CMD=$(printf '%s' "$wandb_output" | grep -o 'wandb agent [^ ]*' | head -n 1)

if [ -z "$AGENT_CMD" ]; then
    echo "$wandb_output"
    echo "❌ Could not parse the W&B agent command from sweep creation output."
    exit 1
fi

echo "🎯 Sweep created! Running agent command: $AGENT_CMD"

# 2. Run the agent for 15 trials (you can change this number)
echo "🏃 Running 15 HPO trials..."
eval "$AGENT_CMD --count 15"

# 3. Evaluate the top 5 models remaining
echo "📊 Evaluating top 5 models..."
python "$evaluate_script"

echo "✅ HPO Pipeline Complete! Check HPO/results for the charts."