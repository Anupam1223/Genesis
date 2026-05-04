# Git Branch Tracking

This document keeps track of active branches, experiments, and our naming conventions to avoid confusion as the project scales.

## 🌳 Current Active Branches

| Branch Name | Status | Description |
| :--- | :--- | :--- |
| `main` | Stable | The primary, working version of the codebase. |
| `initial_affine_coupling` | Active | Baseline architecture implementation using affine coupling layers. |

## 🚀 Planned / Future Branches

When starting a new architecture change or experiment, add it here before creating the branch.

| Branch Name | Type | Description |
| :--- | :--- | :--- |
| 'pca/coefficient_extra_change1' | Feature | Slide a 4-hour window across the timeline. Take the 4-hour curves of the target variables and compress them into a few scalar numbers using Functional PCA. |

## 📖 Branch Naming Conventions

To keep things organized, use the following prefixes when creating new branches:

- **change(1,2,3,4....)** (Naming): For gradual change to new feature, for example change1 is first step after sucessfully created affine coupling layer, after that change2 will be immidiate feature that we will work on
