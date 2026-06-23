Always refer to the following directory for notes - 
"C:\Users\Harshit\Documents\TU DO\Thesis\Theory\Thesis_Ideas" and "C:\Users\Harshit\Documents\TU DO\Thesis\Theory\Model_Notes"

---

## Project Overview

**Thesis**: Surrogate models for Model Predictive Control (MPC) of KDP crystallization in a Continuous Oscillatory Baffled Reactor (COBR).
**Supervisor**: Niklas Kemmerling (TU Dortmund)
**Thesis arc**: Single-Step NARX (baseline) → MSA-NARX (multi-step ahead) → TiDE with quantile regression + CQR → Robust MPC

**Current status (16 Jun 2026)**: SS-NARX (ver_11), SS-AE-NARX (ver_12), MSA-NARX (ver_3), and TiDE (ver_0 one-shot CQR, ver_1 AR-calibrated CQR) are all complete. Working MPC notebooks exist for SS-AE-NARX and MSA-NARX. **Current frontier: the robust-MPC comparison** (nominal MSA vs constrained TiDE-median vs CQR-tightened robust TiDE).

Key reference paper: `TUAMPCoCMAZC_Kemmerling.pdf` — Kemmerling et al., "Towards Uncertainty-Aware MPC of Continuous Microwave-Assisted Zeolite Crystallization". Key findings:
- dt=250s, 20-hour sims, steady-state warmup before dynamic phase
- Dataset: 1M training + 800k calibration + 200k validation + 500 test trajectories
- Architecture: eNARX with autoencoder-based MOR (dimensionality reduction) — outperforms standard NARX for MPC-embedded rollout
- MPC: autoencoder-based robust MPC achieves 5% mean tracking error, 100% quality satisfaction; standard eNARX solve times reach 919s (fails real-time 250s cycle); AE-based: max 68s

---

## Key Files

> **Repo reorganised (branch `repo-reorg`).** `Run_1/` has been replaced by a monorepo layout.
> Shared code lives in `common/`; each pipeline is self-contained under its own folder.

| File | Purpose |
|------|---------|
| `SS_NARX_MPC/training/SO_NARX_pipeline.ipynb` | SS-AE-NARX training + evaluation notebook |
| `MSA_NARX_MPC/training/MSA_NARX_pipeline.ipynb` | MSA-NARX training + evaluation notebook |
| `TiDE_MPC/training/TiDE_pipeline.ipynb` | TiDE quantile + CQR training/calibration/evaluation notebook |
| `TiDE_MPC/training/TiDE_implementation.md` | Full persistent reference code for `TiDE_helpers.py` + notebook |
| `SS_NARX_MPC/mpc/MPC_SO_NARX.ipynb` | SS-AE-NARX MPC notebook |
| `MSA_NARX_MPC/mpc/MPC_MSA_NARX.ipynb` | MSA-NARX MPC notebook |
| `MSA_NARX_MPC/mpc/mpc_msa_utils.py` | MSA-only MPC solver (L-BFGS, warmstart) |
| `common/shared_helpers/helpers_MSA.py` | Shared MSA utilities (scaling, trainer, `build_msa_dataset`) — single source of truth |
| `common/shared_helpers/helper_classes_MSA.py` | Shared network base class (`pytorch_lightning_standard_network`, `GeLU`) |
| `common/mpc_common.py` | Shared MPC utilities (`NARXWindowManager`, scalers) — used by SS + MSA MPC |
| `TiDE_MPC/training/helpers/TiDE_helpers.py` | TiDE model, PinballLoss, CQR (`calibrate_cqr`, `calibrate_cqr_ar`), prob. metrics, rollout |
| `SS_NARX_MPC/training/helpers/helpers.py`, `helper_classes.py` | SS-only helpers (MLP, NARX_AE) |
| `common/plant/base_cobr_model.py` | Full do-mpc/CasADi symbolic COBR model with PI controller |
| `Data_Sampling/sampling/narx_subsampler_dataset.py` | Subsamples raw HDF5 at dt_source with stride to produce effective dt_target |
| `Data_Sampling/sampling/configs/thermal_narx_dataset_config.yaml` | Feature vector layout, lag config, input delays |
| `Data_Sampling/sampling/configs/thermal_cobr_sampling_config.yaml` | Simulation sampling parameters (dt, t_final, warmup, bounds) |
| `Data_Sampling/datasets/` | All HDF5 datasets (raw + NARX-format) |
| `SS_NARX_MPC/training/Models/` | SS model versions (incl. version_12) |
| `MSA_NARX_MPC/training/Models_MSA/` | MSA model versions (incl. version_3) |
| `TiDE_MPC/training/Models_TiDE/` | TiDE model versions (0/1/2) |
| `Theory/Model_Notes/Single_Step_NARX.md` | SS-NARX version table and per-version analysis notes |
| `Theory/Model_Notes/MSA_NARX.md`, `MSA_NARX + MPC.md` | MSA-NARX model + MPC notes |
| `Theory/Model_Notes/SS_AE_NARX + MPC.md` | SS-AE-NARX MPC notes |
| `Theory/Model_Notes/TiDE_plan.md` | TiDE goals, CQR plan, AR-calibrated CQR (ver_1), robust-MPC formulation |
| `Theory/16_06_Update.md` | Latest consolidated progress snapshot |

---

## Feature Vector Layout (ver_11 current: 3796-D input, 25 outputs, 146 lags)

| Indices | Variable | Type | Width |
|---------|----------|------|-------|
| [0:1314] | T_reactor_meas | feature | 146 lags × 9 zones |
| [1314:2482] | T_thermostat_meas | feature | 146 lags × 8 zones |
| [2482:2628] | flow_inlet | input (delay=1) | 146 lags × 1 |
| [2628:3796] | T_setpoint_thermostats | input (delay=1) | 146 lags × 8 |

Labels (25 outputs): T_reactor_meas (9), T_thermostat_meas (8), heating_power_avg (8).
**integral_term is fully dropped** (ver_11+) — following Kemmerling's approach. heating_power_avg is retained as an auxiliary output (needed for energy constraint in MPC, per Kemmerling). Neither is in the feature window — they are prediction targets only, never fed back as lags.

Control inputs (9 total): flow_inlet (1) + T_setpoint_thermostats (8 zones, stacked).

Lookback: 146 steps × 15s = 2190s ≈ reactor residence time.

<details>
<summary>Legacy layout (ver_3–ver_5): 189-D, 9 lags, 4 zones</summary>

| Indices | Variable | Width |
|---------|----------|-------|
| [0:36] | T_reactor_meas | 9 lags × 4 zones |
| [36:72] | T_thermostat_meas | 9 lags × 4 zones |
| [72:108] | heating_power_avg | 9 lags × 4 zones |
| [108:144] | integral_term | 9 lags × 4 zones |
| [144:189] | u-window (5 inputs) | 5 × 9 lags |

</details>

---

## Superseded / Legacy Notes

<details>
<summary>Both items below are no longer central but kept for reference. The delay bug only affected the legacy rollout path (ver_3–ver_8); the PI co-simulator was a ver_9 plan that ver_11 made moot by dropping integral_term entirely.</summary>

### Known Bug: u-window Delay Alignment in Rollout (Cell 15, legacy path)

**Affects**: ver_3–ver_8 legacy rollout path (`_rollout_legacy`).
**Status**: ver_9+ uses `_rollout_with_meta` which correctly handles delay via metadata — not affected.

Training config specifies `delay: 1` for all inputs. The legacy rollout u-window index is **one step behind**, causing systematic trajectory divergence from step 1.

```python
# WRONG — currently in legacy code:
_u_idx = np.clip(np.arange(_k - N_PAST_MODEL, _k), 0, _T - 1)

# CORRECT — must fix:
_u_idx = np.clip(np.arange(_k - N_PAST_MODEL + 1, _k + 1), 0, _T - 1)
```

The `_rollout_with_meta` function (ver_9) computes the input window correctly:
```python
win_i = np.clip(np.arange(k - e['n_past'] - delay + 1, k - delay + 1), 0, T_sim - 1)
```

### PI Co-Simulator (base_cobr_model.py)

`integral_term` and `heating_power_avg` are physically self-contained integrating states driven by the PI controller. They can be computed exactly in numpy alongside the NARX model, bypassing ML prediction entirely (eliminates exposure bias and the poor integral_term R²=0.61–0.73 seen in ver_9). **Superseded**: ver_11 dropped integral_term from the outputs entirely, so this co-simulator was never needed.

**ver_9 architecture is already set up for this**: integral_term and heating_power_avg are not in the feature window, so the rollout loop has a natural slot to inject PI co-simulator values instead of model predictions.

**PI dynamics** (from `base_cobr_model.py`):
```python
# Parameters: K_p=9000, K_i=0.45, anti_windup=0.99999, tau=180s
error = T_setpoint - T_thermostat
heating = clip(K_p * error + K_i * integral_term, min_val, max_val)
integral_term_next = integral_term + dt * anti_windup * error        # RK4 preferred
heating_power_avg_next = heating_power_avg + dt * (heating - heating_power_avg) / tau
```

Use RK4 (not Euler) at dt=15s (ver_9) — dt/tau = 15/180 = 0.083, Euler is acceptable here but RK4 is still cleaner. At dt=60s the error was more significant (dt/tau=0.33).

</details>

---

## Sampling Config — ver_9/ver_11 (same dataset, current)

| Parameter | ver_4 (prev best) | ver_9 (current) |
|-----------|-------------------|-----------------|
| dt | 60s | 15s |
| t_final | 9000s | 36000s |
| warmup | 2400s | 3600s |
| n_sims | 100 | 499 |
| setpoint bounds | [292, 365] | [292, 365] (8 zones) |
| h_reactor | constant 2200 | N(2200, 440) per sim |
| h_loss | constant 12 | N(12, 2.4) per sim |
| h_jacket | constant 1200 | constant 1200 |
| stride | 1 | 1 |
| NARX samples | 14,300 | **1,124,247** |

Data volume gap vs Kemmerling is now closed (~1.1M vs 1M training samples).

---

## Model Version Performance Summary

| Version | Warmup | dt | n_sims | Samples | reactor R² | therm R² | heating R² | integral R² | Epochs | AR Rollout |
|---------|--------|----|--------|---------|-----------|---------|-----------|------------|--------|------------|
| ver_3 | 30s | 60s | 100 | 14,300 | 0.992–0.997 | 0.977–0.988 | 0.993–0.996 | 0.997–0.999 | — | diverges |
| ver_4 | 2400s | 60s | 100 | 14,300 | 0.990–0.999 | 0.983–0.986 | 0.992–0.996 | 0.999 | — | diverges |
| ver_5 | 2400s | 60s (stride=2) | 100 | 29,000 | 0.994–0.998 | 0.989–0.992 | 0.991–0.995 | 0.999 | — | diverges |
| ver_9 | 3600s | 15s | 499 | 1,124,247 | 0.9988–0.9995 | 0.9978–0.9984 | 0.9991–0.9992 | 0.61–0.73 | 66 | spike ~step 2000 |
| **ver_11** | **3600s** | **15s** | **499** | **1,124,247** | **0.9988–0.9995** | **0.9976–0.9984** | **0.9991–0.9994** | **dropped** | **384** | **✓ stable 2500 steps** |
| ver_12 | 3600s | 15s | 499 | 1,124,247 | 0.9989 | 0.9979 | 0.9992 | dropped | 217 | mild drift (recon-loss norm issue) |

\* ver_9 Mean RMSE=40.74 dominated by integral_term RMSE (152–170). The poor integral_term R² is structural — the model cannot infer PI internal state from temperatures alone.

\*\* ver_11 val loss at epoch 384: **0.000521** (vs ver_9 plateau at epoch 66).

**ver_9 architecture**: MLP 3796→128→128→64→33 (GeLU), 512k params, 66 epochs. integral_term R²=0.61–0.73 (structural, not capacity). AR rollout diverges at ~step 2000.

**ver_11 architecture (SS-NARX baseline)**: MLP 3796→512→256→128→25 (GeLU). Changes vs ver_9: (1) integral_term dropped from outputs entirely, (2) hidden_dims [512,256,128], (3) noise injection σ=0.05 in training_step, (4) StagnationEarlyStopping relaxed (patience=25, min_delta=0.001). Trained 384 epochs. AR rollout **stable through all 2500 steps** across all 4 test simulations.

**ver_12 architecture (SS-AE-NARX)**: joint autoencoder + predictor. Encoder 3796→[512→256]→25 (latent z), decoder 25→[256→512]→3796 (reconstruction), pred head 25→[64]→26 (adds T_reactor_max, R²=0.854, for MPC safety constraint). Joint loss α=0.3 (recon) + β=1.0 (pred), 217 epochs, val_loss=0.007750. Latent z runs once per MPC step outside the CasADi graph → reduces symbolic state from 3796-D to 25-D. Known issue: recon loss over 3796 dims can overwhelm pred loss → mild AR drift; candidate fix z=50.

---

## MSA-NARX (Multi-Step Ahead) — `Models_MSA/`

One-shot M-step forecaster: a single forward pass returns the full M-step trajectory (no rollout-error compounding inside the horizon). **Requires future covariates** — `u(k+1:k+14)` (14×9=126-D) appended to the 3796-D past window → **3922-D input**, M×O output. Without them the MPC optimiser has no gradient path through the model. M=15 (225s horizon at dt=15s). Built via `build_msa_dataset`.

- **ver_0** (375-D out, 25 outputs): 499 epochs, val_loss=0.000313 (40% below ver_11). R² flat ~0.9996 h=1→h=15. Single-step RMSE 3–5× better than ver_11 (future covariates + multi-step loss). Stable AR rollout.
- **ver_1 / ver_2**: failed — gradient explosion at lr=1e-3 (clipping alone insufficient); AR diverged.
- **ver_3** (390-D out, 26 outputs incl. T_reactor_max) — **MSA baseline**: lr=1e-4 + warm start from ver_0. 211 epochs, val_loss=0.002646, R² flat 0.9954–0.9956 across horizon, acceptable AR drift (~2.5K over 2500 steps). The gap vs ver_0 is the cost of the harder T_reactor_max output.

---

## TiDE (Calibrated Uncertainty) — `Models_TiDE/`

One-shot M-step **quantile** forecaster replacing MSA-NARX as the MPC surrogate. Same 3922-D input; output **(M, 26, 3)** = lower/median/upper at τ=[0.05, 0.50, 0.95] (90% PI). Pinball loss; monotonic softplus quantile head (no crossing). Architecture: per-step covariate projection → dense residual encoder → decoder → per-horizon temporal decoder + global linear skip. 4-way simulation split (train 0.6 / val 0.1 / calib 0.15 / test rest) so CQR coverage guarantees hold. Full code in `TiDE_implementation.md`.

- **ver_0** — one-shot CQR. `calibrate_cqr` computes `q̂` of shape (M, O) from the held-out calib split (model sees a **true** past window). val_loss≈0.0031 (best ep ~434). Metrics: per-horizon median R²/RMSE, coverage, sharpness, CRPS.
- **ver_1** — **AR-Calibrated CQR** (commit `2b0c985`, 16 Jun). Reuses ver_0 weights (no retraining, identical median). `calibrate_cqr_ar` runs a full autoregressive rollout per calibration sim, collects the horizon-1 nonconformity score `s_k = max(lower−y, y−upper)` only at steps `k > burn_in=150` (after the 146-lag window is fully model-driven), pools over all steps/sims → a **constant (O,) threshold** `q_hat_ar`. **Motivation**: one-shot CQR calibrates on the true-window regime and systematically *under-covers* the compounding-error regime the surrogate actually faces in MPC rollout. The constant (O,) band drops cleanly into the robust-MPC constraint tightening (`ŷ_upper + q_hat_ar ≤ T_max`). Artifacts: `cqr_qhat.npy` (one-shot) + `cqr_qhat_ar.npy` + `ar_calibration_config.yml`. Full per-version analysis in `Theory/Model_Notes/TiDE_Models.md`.
  - **Headline (AR rollout, k>150, n=2088 of 2239, one test sim)**: raw quantiles cov **0.864** (under-covers despite over-covering one-shot); one-shot CQR cov **0.731** (*worse* — its q̂ is negative because raw over-covers the easy regime, so it shrinks the band); **AR-CQR cov 0.901** (on target). Clean validation: calibrating on the wrong regime is worse than not calibrating.
  - **Caveat for robust MPC**: 0.901 is *marginal* (pooled). Per-output reactor coverage is heterogeneous — `T_reactor_meas[7]`=0.302, [4]=0.780, `T_reactor[0]`=0.817 vs [5]/[8]=1.000; thermostats 0.88–0.92 and heating_power 0.89–0.96 are fine. Reactor bands are tiny (~0.06 K, R²=1.0000) so the dominant rollout error is slow per-zone drift, which a constant-in-horizon per-zone band can't track. Confounders: single test sim (high per-output variance), horizon-constant band. Fix before/within robust MPC: pool over several test sims, horizon-dependent band, or max-over-reactor-zones band.

`get_simulation_split_dataloaders_4way` now also returns `split_info` (per-split sim ids + per-sim sample counts in contiguous order) — needed for the per-sim AR rollout in `calibrate_cqr_ar`.

---

## MPC — `Run_1/mpc/`

**MSA-NARX MPC** (single-shooting: one forward pass → full N-step trajectory; optimiser differentiates w.r.t. the future control sequence directly — no multiple-shooting). ver_0: mean |err| 1.02 K, 90% within ±2K, 0/100 safety violations. Beats SS-NARX MPC because the multi-step model is accurate across the whole horizon without rollout compounding.

**SS-AE-NARX MPC**: encoder E(3796→25) runs once per step outside CasADi; only latent transition f and output head g are inside the optimiser. Tuned through ver_3 (N_HORIZON=25, R_du=2.0 for smoother inputs). Cost in physical space (per-zone MSE tracking + energy); ver_0 had a sign bug in the energy term (fixed in ver_1).

**Robust-MPC target** (next): constraint tightening `ŷ_upper_quantile + q_hat_ar ≤ T_max` using the TiDE CQR tube. Optimisation stack (per Chen et al.): PyTorch autograd + L-BFGS + Augmented Lagrangian penalties, warm-started from the previous step.

---

## Key Technical Concepts to Remember

- **dt-locking**: NARX models learn per-step corrections calibrated to training dt. Cannot generalise across sampling frequencies. dt is not a free hyperparameter.
- **Near-persistence predictor**: Fine-dt models (dt=1s) learn T(t+1) ≈ T(t) + tiny correction. High single-step R² but drift over long horizons.
- **Exposure bias on integrating states**: `integral_term` is a cumulative integrator. Small per-step prediction bias compounds linearly during rollout → primary cause of T_reactor divergence in earlier versions.
- **Integral_term structural problem (ver_9, resolved in ver_11)**: Even with integral_term excluded from the feature window, predicting it from temperature observations alone yields R²=0.61–0.73. This is not a capacity issue — it's a fundamental information problem. **Fix (ver_11): drop integral_term from outputs entirely**, following Kemmerling who never predicts it. The 146-lag T_thermostat window implicitly encodes PI state for temperature prediction purposes.
- **Autoencoder MOR**: Kemmerling's key insight — autoencoder-based dimensionality reduction of NARX feature vector required for MPC-embedded rollout stability. Plan for multi-step NARX phase.
- **Warmup**: 3600s warmup in ver_9 (vs 2400s before) ensures all initial states lie on physically realistic steady-state manifold.
- **Wide bounds [292, 365]**: Allows CoupledRandomWalkSampler to explore full operating range. Tight zone-specific bounds (ver_0) caused near-constant inputs in later trajectory steps.
- **Physical parameter perturbation**: h_reactor ~ N(2200, 440), h_loss ~ N(12, 2.4) per simulation in ver_9. Makes model robust to heat transfer uncertainty across the full realistic range [946–3474] for h_reactor.
- **CoupledRandomWalkSampler**: Flow inlet changes trigger correlated setpoint boosts (boost_factor=2.0) that decay over 1 timestep. Change probability: flow=5%, setpoints=15% each.

---

## Immediate Next Steps

**SS-NARX, SS-AE-NARX, MSA-NARX, and TiDE (one-shot + AR-calibrated CQR) are all complete.** Surrogate-modelling phase is essentially done; the work is now MPC and write-up.

1. **Run TiDE ver_1 Cell 15 and fill the headline numbers**: one-shot CQR coverage (expected < 0.90, under-covers) vs AR-CQR coverage (expected ≈ 0.90) on the AR rollout, k>150. Record in `TiDE_plan.md` and `16_06_Update.md`.
2. **Robust-MPC comparison experiment** (current frontier): run three variants on the same COBR test trajectory and report constraint-violation rate + tracking RMSE — (a) Nominal MPC = MSA-NARX median, hard constraints; (b) Constrained MPC = TiDE median, no tightening; (c) Robust MPC = TiDE + CQR-tightened tube (`ŷ_upper + q_hat_ar ≤ T_max`) via Augmented Lagrangian + L-BFGS. Replicates Table 2 of Chen et al. for the COBR.
3. **Thesis write-up**: baseline chapter is the ver_9→ver_11 SS-NARX progression (dropping integral_term + noise injection + wider net → stable rollout); then MSA-NARX horizon efficiency; then TiDE calibrated uncertainty with the AR-CQR contribution beyond Chen et al.
4. **Dataset**: existing `21_05_2026` (1,124,247 samples; `thermal_narx_dataset_3.h5` for MSA/TiDE) is sufficient for all remaining phases — no new sampling needed.

