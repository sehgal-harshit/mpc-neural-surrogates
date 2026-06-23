# Neural-Network Surrogate Models for Real-Time Model Predictive Control

Master's thesis — **M.Sc. Automation & Robotics, TU Dortmund** (supervised by Niklas Kemmerling).

> 🧹 **Before pushing:** the code is yours to publish. Just keep large/raw datasets and trained weights out of the repo (the included `.gitignore` handles this) and don't commit the reference paper PDF.

## What it does

Model Predictive Control (MPC) is powerful but expensive: every control step re-solves an optimization over a physical model, and high-fidelity first-principles models are often too slow for real-time use. This thesis develops a **generalizable framework that replaces the expensive model with a learned neural-network surrogate** embedded directly in the MPC loop — keeping control performance while cutting solve times by orders of magnitude.

**Application:** robust MPC of KDP crystallization in a Continuous Oscillatory Baffled Reactor (COBR).

## Approach

The work progresses through increasingly capable surrogates:

1. **Single-Step NARX (baseline)** — autoregressive network predicting one step ahead.
2. **SS-AE-NARX** — adds an autoencoder for model-order reduction, which makes MPC-embedded rollout far faster and more stable.
3. **Multi-Step-Ahead NARX (MSA-NARX)** — trained for stable multi-step rollout, the regime MPC actually needs.
4. **TiDE + quantile regression + Conformalized Quantile Regression (CQR)** — adds calibrated uncertainty estimates, enabling **robust** MPC that tightens constraints under model uncertainty.

The full pipeline is covered: data generation from plant simulations → surrogate training → multi-step rollout stability → direct embedding in the optimizer.

## Results (headline)

- Surrogate-based MPC meets the real-time control cycle where the full model cannot (solve time reduced by orders of magnitude).
- Autoencoder-based reduction is key to keeping MPC-embedded rollout within the cycle budget.
- Conformal prediction provides calibrated uncertainty for constraint tightening in robust MPC.

## Tech stack

`Python` · `PyTorch` · `PyTorch Lightning` · `do-mpc` · `CasADi` · `l4casadi` (PyTorch-in-CasADi) · `scikit-learn` · `NumPy / SciPy / pandas` · `h5py` · `TensorBoard`

## Repository layout (high level)

```
common/            # shared model, MPC utilities, plant interface
SS_NARX_MPC/       # single-step (incl. autoencoder) training + MPC
MSA_NARX_MPC/      # multi-step-ahead NARX training + MPC
TiDE_MPC/          # TiDE + quantile/CQR training + robust MPC
Data_Sampling/     # dataset generation & sampling configs
```

## Notes
This repository showcases my thesis work. Large raw datasets and trained model weights are excluded from version control for size.

<!-- Fill in once cleared: link to thesis PDF, a results figure (rollout vs ground truth, solve-time comparison), and a citation. -->
