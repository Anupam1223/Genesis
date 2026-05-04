# Git Branch Tracking

This document keeps track of active branches, experiments, and our naming conventions to avoid confusion as the project scales.

## 🌳 Current Active Branches

| Branch Name | Status | Description |
| :--- | :--- | :--- |
| `main` | Stable | The primary, working version of the codebase. |
| `initial_affine_coupling` | Active | Baseline architecture implementation using affine coupling layers. |
| `desktop_run` | Temporary | Created to run on the Mac Tower with the MPS fallback. (See "Hardware Setup" below for plans to deprecate this). |

## 🚀 Planned / Future Branches

When starting a new architecture change or experiment, add it here before creating the branch.

| Branch Name | Type | Description |
| :--- | :--- | :--- |
| (Example) `feat/transformer-flow` | Feature | Swapping coupling layers for a transformer-based flow. |
| (Example) `exp/gelu-activation` | Experiment | Testing GELU vs ReLU in the hidden layers. |

## 📖 Branch Naming Conventions

To keep things organized, use the following prefixes when creating new branches:

- **feat/...** (Feature): For new architectural changes, models, or major additions. (e.g., `feat/residual-blocks`)
- **exp/...** (Experiment): For quick tests or hyperparameter tuning that might be discarded. (e.g., `exp/high-learning-rate`)
- **fix/...** (Bug Fix): For patching errors in the code. (e.g., `fix/nan-loss-bug`)
- **docs/...** (Documentation): Updates to README or logging tracking.

## 💻 Hardware Setup (Deprecating `desktop_run`)

Currently, `desktop_run` exists solely to include the MPS CPU fallback for the Mac Tower:

```python
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

**Goal:** To prevent merging headaches between architectures and hardware, we will phase out the `desktop_run` branch.

### Steps to Unify

1. Keep development strictly to feature branches (like `initial_affine_coupling`).
2. Do not commit the `os.environ` code directly to the repository.
3. Instead, rely on local `.env` files on the Mac Tower (which are ignored by `.gitignore`) to handle the hardware-specific fallbacks automatically, allowing the exact same branch to run on both the M4 laptop and the desktop.
