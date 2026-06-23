# TiDE Implementation Reference (Goals 1–4)

Persistent reference for the TiDE surrogate build. **The user types this code themselves** (learning in parallel). This file holds the full, exact code for both deliverables so context survives conversation compaction.

- **File 1:** `Run_1/TiDE_helpers.py` — classes + helper functions
- **File 2:** `Run_1/TiDE_pipeline.ipynb` — training/CQR/evaluation notebook

Design decisions (confirmed): Option A (raw 3922-D end-to-end), τ=[0.05,0.5,0.95] (90% PI), 26 base outputs, 4-way sim split (0.6/0.1/0.15/0.15), CQR per-output-per-horizon (q̂ shape (M,26)), monotonic softplus quantile head.

Data path reused from MSA: `data_sets/21_05_2026/narx/thermal_narx_dataset_3.h5`, `build_msa_dataset` → 3922-D input (3796 past + 14×9 future cov) + M×26 labels, M=15.

---

# File 1 — `Run_1/TiDE_helpers.py`

## Section A — header, imports, reuse

```python
"""
TiDE (Time-series Dense Encoder) helpers for the COBR surrogate.

This module adds the TiDE-specific pieces on top of the shared MSA utilities:
    Classes
        ResidualBlock   - canonical TiDE residual MLP block (skip + LayerNorm)
        TiDE            - one-shot M-step quantile forecaster (Lightning module)
        PinballLoss     - multi-quantile (pinball) loss

    Functions
        get_simulation_split_dataloaders_4way - train/val/calib/test sim split
        evaluate_tide_on_test_set             - per-horizon R2/RMSE on the median
        calibrate_cqr                         - conformal correction threshold q_hat (one-shot, (M,O))
        calibrate_cqr_ar                      - AR-rollout conformal threshold q_hat ((O,), compounding regime)
        apply_cqr                             - widen [lower, upper] by q_hat
        probabilistic_metrics                 - coverage / sharpness / CRPS table
        recursive_tide_rollout                - autoregressive rollout with tube

Everything else (scaling, trainer, metadata I/O, build_msa_dataset) is imported
from helpers.helpers_MSA so there is a single source of truth.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .helper_classes_MSA import pytorch_lightning_standard_network
```

> **Import note:** this file lives in `Run_1/helpers/`, so the import is **relative**
> (`from .helper_classes_MSA import ...`), exactly like `helpers_MSA.py`. The notebook
> imports the module as `from helpers.TiDE_helpers import ...`.

## Section B — ResidualBlock

```python
# ── Building block ──────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    TiDE residual block:  Linear -> act -> dropout -> Linear,  plus a skip
    connection (projected when in/out dims differ), followed by LayerNorm.

    out = LayerNorm( MLP(x) + skip(x) )
    """

    def __init__(self, in_dim, out_dim, hidden_dim, dropout, activation):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.act = activation
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.lin2(self.drop(self.act(self.lin1(x))))
        return self.norm(h + self.skip(x))
```

## Section C — TiDE model

```python
# ── TiDE model ──────────────────────────────────────────────────────────────────

class TiDE(pytorch_lightning_standard_network):
    """
    One-shot multi-step quantile forecaster.

    Input  x : (B, past_dim + future_cov_dim)   -- same 3922-D vector as MSA-NARX
    Output   : (B, M, base_output_dim, n_quantiles)

    network_hyperparameters keys:
        past_dim, future_cov_dim, M, base_output_dim,
        hidden_dim, decoder_output_dim, temporal_width,
        num_encoder_layers, num_decoder_layers, dropout,
        activation (nn.Module instance), quantiles (list[float]), noise_sigma
    """

    def __init__(self, network_hyperparameters, training_hyperparameters):
        super().__init__(**training_hyperparameters)
        self.save_hyperparameters()

        hp = network_hyperparameters
        self.past_dim = hp['past_dim']
        self.future_cov_dim = hp['future_cov_dim']
        self.M = hp['M']
        self.base_output_dim = hp['base_output_dim']
        self.quantiles = list(hp['quantiles'])
        self.n_quantiles = len(self.quantiles)
        self.noise_sigma = hp.get('noise_sigma', 0.0)

        hidden_dim = hp['hidden_dim']
        decoder_output_dim = hp['decoder_output_dim']
        temporal_width = hp['temporal_width']
        dropout = hp['dropout']
        act = hp['activation']

        # future covariates arrive flat as (M-1) steps x n_ctrl. Recover n_ctrl.
        self.n_steps_cov = self.M - 1
        self.n_ctrl = self.future_cov_dim // self.n_steps_cov

        # (1) Per-step covariate projection — shared weights across horizon steps.
        self.cov_proj = nn.Linear(self.n_ctrl, temporal_width)

        # (2) Dense encoder: [past || projected covariates over M steps] -> latent.
        enc_in = self.past_dim + self.M * temporal_width
        enc_layers = [ResidualBlock(enc_in, hidden_dim, hidden_dim, dropout, act)]
        for _ in range(hp['num_encoder_layers'] - 1):
            enc_layers.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout, act))
        self.encoder = nn.ModuleList(enc_layers)

        # (3) Dense decoder: latent -> (M x decoder_output_dim).
        dec_layers = []
        for _ in range(hp['num_decoder_layers'] - 1):
            dec_layers.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout, act))
        self.decoder = nn.ModuleList(dec_layers)
        self.decoder_head = nn.Linear(hidden_dim, self.M * decoder_output_dim)
        self.decoder_output_dim = decoder_output_dim

        # (4) Temporal decoder: per-horizon [decoded slice || covariate] -> O*Q raw.
        td_in = decoder_output_dim + temporal_width
        self.temporal_decoder = ResidualBlock(
            td_in, self.base_output_dim * self.n_quantiles, temporal_width, dropout, act
        )

        # (5) Global linear residual skip: past window -> (M x base_output_dim).
        self.global_skip = nn.Linear(self.past_dim, self.M * self.base_output_dim)

    def _project_covariates(self, future_cov, batch_size):
        """future_cov (B, (M-1)*n_ctrl) -> projected (B, M, temporal_width).
        Horizon 1 has no future covariate (it's already in the past window), so
        we prepend a zero row to align horizon h with covariate for step k+h."""
        cov = future_cov.view(batch_size, self.n_steps_cov, self.n_ctrl)
        cov = self.cov_proj(cov)                               # (B, M-1, temporal_width)
        pad = torch.zeros(batch_size, 1, cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return torch.cat([pad, cov], dim=1)                    # (B, M, temporal_width)

    def forward(self, x):
        B = x.shape[0]
        past = x[:, :self.past_dim]
        future_cov = x[:, self.past_dim:]

        cov_proj = self._project_covariates(future_cov, B)     # (B, M, temporal_width)

        # Encoder
        e = torch.cat([past, cov_proj.reshape(B, -1)], dim=1)
        for block in self.encoder:
            e = block(e)

        # Decoder
        d = e
        for block in self.decoder:
            d = block(d)
        d = self.decoder_head(d).view(B, self.M, self.decoder_output_dim)

        # Temporal decoder (apply the shared block across all horizon steps at once)
        td_in = torch.cat([d, cov_proj], dim=-1)               # (B, M, dec_out+temp_w)
        raw = self.temporal_decoder(td_in.reshape(B * self.M, -1))
        raw = raw.view(B, self.M, self.base_output_dim, self.n_quantiles)

        # Global residual skip added to the median channel
        skip = self.global_skip(past).view(B, self.M, self.base_output_dim)

        # Monotonic quantile head: build quantiles outward from the median so that
        # lower <= median <= upper by construction (no quantile crossing).
        mid = self.n_quantiles // 2
        out = torch.empty_like(raw)
        out[..., mid] = raw[..., mid] + skip
        for q in range(mid - 1, -1, -1):
            out[..., q] = out[..., q + 1] - F.softplus(raw[..., q])
        for q in range(mid + 1, self.n_quantiles):
            out[..., q] = out[..., q - 1] + F.softplus(raw[..., q])
        return out                                             # (B, M, O, Q)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.noise_sigma > 0.0:
            x = x + torch.randn_like(x) * self.noise_sigma
        loss = self.loss_function(self(x), y)
        self.log('train_loss', loss)
        return loss

    def predict_trajectory(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)
```

## Section D — PinballLoss

```python
# ── Quantile (pinball) loss ──────────────────────────────────────────────────────

class PinballLoss(nn.Module):
    """
    Multi-quantile pinball loss.

    pred   : (B, M, O, Q)  -- quantile predictions
    target : (B, M*O)      -- flat MSA labels (reshaped internally to (B, M, O))

    rho_tau(e) = max(tau * e, (tau - 1) * e),   e = target - pred_tau
    """

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('taus', torch.tensor(list(quantiles), dtype=torch.float32))

    def forward(self, pred, target):
        B, M, O, Q = pred.shape
        target = target.view(B, M, O, 1)               # broadcast over Q
        err = target - pred                            # (B, M, O, Q)
        taus = self.taus.view(1, 1, 1, Q)
        loss = torch.maximum(taus * err, (taus - 1.0) * err)
        return loss.mean()
```

## Section E — 4-way simulation split

```python
# ── Train / val / calib / test split (by whole simulations) ──────────────────────

def get_simulation_split_dataloaders_4way(
        features, labels, sim_sample_counts,
        train_frac=0.6, val_frac=0.1, calib_frac=0.15,
        batch_size=512, seed=42,
        multiprocessing=True, cpu_count=mp.cpu_count()):
    """
    Split into train/val/calib/test DataLoaders by whole simulation trajectories.
    The calibration split is held fully out of training so CQR coverage guarantees
    hold. Test is whatever remains after train+val+calib.
    """
    if train_frac + val_frac + calib_frac >= 1.0:
        raise ValueError(
            f'train+val+calib must be < 1.0, got {train_frac + val_frac + calib_frac}')

    sim_sample_counts = np.asarray(sim_sample_counts, dtype=np.int64)
    n_sims = len(sim_sample_counts)

    rng = np.random.default_rng(seed)
    sim_order = rng.permutation(n_sims)

    n_train = int(train_frac * n_sims)
    n_val = int(val_frac * n_sims)
    n_calib = int(calib_frac * n_sims)

    train_ids = sim_order[:n_train]
    val_ids = sim_order[n_train:n_train + n_val]
    calib_ids = sim_order[n_train + n_val:n_train + n_val + n_calib]
    test_ids = sim_order[n_train + n_val + n_calib:]

    offsets = np.concatenate([[0], np.cumsum(sim_sample_counts)])

    def _gather(ids):
        parts = [np.arange(offsets[s], offsets[s + 1]) for s in sorted(ids)]
        return np.concatenate(parts).astype(np.int64)

    train_idx, val_idx = _gather(train_ids), _gather(val_ids)
    calib_idx, test_idx = _gather(calib_ids), _gather(test_ids)

    def _loader(idx, shuffle, workers):
        ds = TensorDataset(features[idx], labels[idx])
        kw = dict(batch_size=batch_size)
        if multiprocessing and workers > 0:
            kw.update(persistent_workers=True, pin_memory=True)
        return DataLoader(ds, shuffle=shuffle, num_workers=workers, **kw)

    train_loader = _loader(train_idx, True, max(1, cpu_count // 2) if multiprocessing else 0)
    val_loader = _loader(val_idx, False, 0)
    calib_loader = _loader(calib_idx, False, 0)
    test_loader = _loader(test_idx, False, 0)

    # Per-split sim boundaries — needed for per-sim AR rollout on the calib split.
    # _gather iterates sorted(ids), so these counts are in the same contiguous order
    # the samples appear inside X_calib / X_test.
    split_info = {
        'calib_ids':        np.array(sorted(calib_ids)),
        'test_ids':         np.array(sorted(test_ids)),
        'calib_sim_counts': sim_sample_counts[sorted(calib_ids)],
        'test_sim_counts':  sim_sample_counts[sorted(test_ids)],
    }

    print(f'Sim split (seed={seed}): '
          f'{n_train} train ({len(train_idx):,}) | {n_val} val ({len(val_idx):,}) | '
          f'{n_calib} calib ({len(calib_idx):,}) | {len(test_ids)} test ({len(test_idx):,})')
    return train_loader, val_loader, calib_loader, test_loader, split_info
```

> **Signature change:** this now returns a 5th value `split_info`. Cell 5 must unpack it.
> `calib_sim_counts` gives the per-sim length of each calibration trajectory **in the
> contiguous order `X_calib` was assembled**, which `calibrate_cqr_ar` relies on.

## Section F — evaluation (median R²/RMSE)

```python
# ── Forward / evaluation ─────────────────────────────────────────────────────────

def _forward_quantiles(model, X, device='cpu', batch_size=1000):
    """Run the model in batches. Returns scaled quantile preds (N, M, O, Q)."""
    model = model.to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].to(device)
            out.append(model(xb).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def evaluate_tide_on_test_set(model, X_test, y_test, label_scaler_params,
                              M=15, base_output_dim=26, quantiles=None,
                              device='cpu'):
    """
    Per-horizon R2/RMSE/MAE on the MEDIAN prediction, plus inverse-scaled arrays.
    Returns dict: predictions (N,M,O,Q), ground_truth (N,M,O),
                  median/lower/upper (N,M,O), metrics_df.
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    mid = len(quantiles) // 2
    lmean = np.array(label_scaler_params['mean'], dtype=np.float64)
    lstd = np.array(label_scaler_params['std'], dtype=np.float64)

    pred_sc = _forward_quantiles(model, X_test, device)                 # (N,M,O,Q)
    pred_phys = pred_sc * lstd[None, None, :, None] + lmean[None, None, :, None]

    y_sc = y_test.numpy().astype(np.float64).reshape(-1, M, base_output_dim)
    y_phys = y_sc * lstd[None, None, :] + lmean[None, None, :]

    median = pred_phys[..., mid]                                        # (N,M,O)

    metrics = []
    for h in range(M):
        ph, gh = median[:, h, :], y_phys[:, h, :]
        metrics.append({'Horizon': h + 1,
                        'R2': r2_score(gh, ph, multioutput='uniform_average'),
                        'RMSE': np.sqrt(mean_squared_error(gh, ph)),
                        'MAE': mean_absolute_error(gh, ph)})
    metrics_df = pd.DataFrame(metrics)
    print('\n--- TiDE Median Evaluation (per horizon) ---')
    print(metrics_df.to_string(index=False))

    return {'predictions': pred_phys, 'ground_truth': y_phys,
            'median': median, 'lower': pred_phys[..., 0], 'upper': pred_phys[..., -1],
            'metrics_df': metrics_df}
```

## Section G — CQR

```python
# ── Conformalized Quantile Regression ────────────────────────────────────────────

def calibrate_cqr(model, X_calib, y_calib, label_scaler_params,
                  M=15, base_output_dim=26, quantiles=None,
                  device='cpu', alpha=0.10):
    """
    Compute the conformal correction threshold q_hat from the calibration set.
        s_i = max(lower - y_i, y_i - upper)          (per sample, per horizon, per output)
        q_hat = ceil((n+1)(1-alpha))-th smallest s    (finite-sample quantile)
    Returns q_hat of shape (M, O).
    """
    lmean = np.array(label_scaler_params['mean'], dtype=np.float64)
    lstd = np.array(label_scaler_params['std'], dtype=np.float64)

    pred_sc = _forward_quantiles(model, X_calib, device)
    pred_phys = pred_sc * lstd[None, None, :, None] + lmean[None, None, :, None]
    lower, upper = pred_phys[..., 0], pred_phys[..., -1]               # (N,M,O)

    y_phys = (y_calib.numpy().astype(np.float64).reshape(-1, M, base_output_dim)
              * lstd[None, None, :] + lmean[None, None, :])

    s = np.maximum(lower - y_phys, y_phys - upper)                     # (N,M,O)
    n = s.shape[0]
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))                      # finite-sample
    q_hat = np.quantile(s, level, axis=0, method='higher')            # (M,O)
    print(f'CQR: n_calib={n}, level={level:.5f}, '
          f'q_hat range [{q_hat.min():.3f}, {q_hat.max():.3f}]')
    return q_hat


def calibrate_cqr_ar(model, X_calib, y_calib, calib_sim_counts,
                     feature_scaler_params, label_scaler_params,
                     state_group_dims, n_ctrl_dims,
                     M=15, base_output_dim=26, quantiles=None,
                     device='cpu', alpha=0.10, burn_in=150,
                     max_steps_per_sim=None):
    """
    AR-calibrated CQR — sizes the tube for the *compounding* error regime the
    surrogate faces in MPC / long-horizon rollout (where the lag window fills with
    the model's own predictions), which one-shot calibrate_cqr under-covers.

    For each calibration simulation, run a full autoregressive rollout
    (recursive_tide_rollout). Collect the horizon-1 nonconformity score
        s_k = max(lower_k - y_k, y_k - upper_k)     (per output, per rollout step)
    at each step k > burn_in -- after transients, once the lag window is fully
    populated by model predictions (N_LAGS=146, so burn_in=150 leaves a margin).
    Pool over all steps & sims; q_hat = finite-sample (1-alpha) quantile per output.

    calib_sim_counts       : per-sim sample counts WITHIN X_calib, in the contiguous
                             order the 4-way split laid them out (split_info).
    feature_scaler_params  : the FULL MSA feature scaler (msa_feat_scaler) — rollout
                             un/re-scales the whole 3922-D window.
    Returns q_hat of shape (O,).
    """
    offsets = np.concatenate([[0], np.cumsum(np.asarray(calib_sim_counts, dtype=np.int64))])
    scores = []
    for s in range(len(calib_sim_counts)):
        start, count = int(offsets[s]), int(calib_sim_counts[s])
        n_roll = count if max_steps_per_sim is None else min(count, max_steps_per_sim)
        if n_roll <= burn_in:
            continue
        roll = recursive_tide_rollout(
            model, X_calib, y_calib, feature_scaler_params, label_scaler_params,
            N_rollout=n_roll, start_idx=start, device=device,
            state_group_dims=state_group_dims, n_ctrl_dims=n_ctrl_dims,
            M=M, base_output_dim=base_output_dim, quantiles=quantiles)
        s_sim = np.maximum(roll['lower'] - roll['ground_truth'],
                           roll['ground_truth'] - roll['upper'])      # (steps, O)
        scores.append(s_sim[burn_in:])
    S = np.concatenate(scores, axis=0)                                # (Ntot, O)
    n = S.shape[0]
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    q_hat = np.quantile(S, level, axis=0, method='higher')            # (O,)
    print(f'AR-CQR: {len(scores)} sims, n_scores={n} (k>{burn_in}), '
          f'level={level:.5f}, q_hat range [{q_hat.min():.3f}, {q_hat.max():.3f}]')
    return q_hat


def apply_cqr(lower, upper, q_hat):
    """Widen the interval: [lower - q_hat, upper + q_hat].
    Broadcasts (M,O) (one-shot) or (O,) (AR) over the leading axes."""
    return lower - q_hat[None, ...], upper + q_hat[None, ...]
```

> **Note:** `apply_cqr` is unchanged — `q_hat[None, ...]` broadcasts a `(M,O)` one-shot
> threshold over `(N,M,O)`, and an `(O,)` AR threshold over `(steps,O)` rollout arrays.

## Section H — probabilistic metrics + AR rollout

```python
# ── Probabilistic metrics ────────────────────────────────────────────────────────

def probabilistic_metrics(lower, upper, y_true, label_names,
                          quantile_preds=None, quantiles=None):
    """
    Per-output probabilistic metrics. lower/upper/y_true: (N, M, O).
        coverage   : fraction of y_true inside [lower, upper]
        mean_width : mean interval width (sharpness; narrower is better)
        CRPS       : quantile approx (2/K) * mean_k pinball_k  (if quantile_preds given)
    Returns a DataFrame, one row per output.
    """
    O = y_true.shape[-1]
    inside = (y_true >= lower) & (y_true <= upper)
    coverage = inside.reshape(-1, O).mean(axis=0)
    width = (upper - lower).reshape(-1, O).mean(axis=0)

    rows = {'output': label_names, 'coverage': coverage, 'mean_width': width}
    if quantile_preds is not None and quantiles is not None:
        taus = np.array(quantiles)
        err = y_true[..., None] - quantile_preds                       # (N,M,O,Q)
        pin = np.maximum(taus * err, (taus - 1) * err)
        rows['CRPS'] = (2.0 / len(taus)) * pin.mean(axis=(0, 1, 3))     # (O,)
    return pd.DataFrame(rows)


# ── Autoregressive rollout (with uncertainty tube) ───────────────────────────────

def recursive_tide_rollout(model, X_test_scaled, y_test_scaled,
                           feature_scaler_params, label_scaler_params,
                           N_rollout=2225, start_idx=0, device='cpu',
                           state_group_dims=None, n_ctrl_dims=None,
                           M=15, base_output_dim=26, quantiles=None):
    """
    Autoregressive rollout. The model is called every step; the MEDIAN horizon-1
    prediction advances the AR state window (decode -> roll -> insert -> re-encode),
    while horizon-1 lower/median/upper are recorded for tube-coverage checks.
    Index math mirrors recursive_msa_narx_rollout in helpers_MSA.
    """
    model = model.to(device)
    model.eval()

    feat_mean = np.array(feature_scaler_params['mean'], dtype=np.float64)
    feat_std = np.array(feature_scaler_params['std'], dtype=np.float64)
    lab_mean = np.array(label_scaler_params['mean'], dtype=np.float64)
    lab_std = np.array(label_scaler_params['std'], dtype=np.float64)
    mid = len(quantiles) // 2

    _cum = [0]
    for d in state_group_dims:
        _cum.append(_cum[-1] + d)
    total_state_dims = _cum[-1]

    N_LAGS = (feat_mean.shape[0] - (M - 1) * n_ctrl_dims) // (total_state_dims + n_ctrl_dims)
    ctrl_start = total_state_dims * N_LAGS

    N_test = X_test_scaled.shape[0]
    max_steps = min(N_rollout, N_test - start_idx)
    x = X_test_scaled[start_idx].numpy().astype(np.float64).copy()

    median = np.zeros((max_steps, base_output_dim))
    lower = np.zeros((max_steps, base_output_dim))
    upper = np.zeros((max_steps, base_output_dim))
    ground_truth = np.zeros((max_steps, base_output_dim))

    with torch.no_grad():
        for k in range(max_steps):
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(xt).squeeze(0).cpu().numpy().astype(np.float64)   # (M,O,Q)

            med_h1 = pred[0, :, mid] * lab_std + lab_mean
            low_h1 = pred[0, :, 0] * lab_std + lab_mean
            up_h1 = pred[0, :, -1] * lab_std + lab_mean
            gt_h1 = (y_test_scaled[start_idx + k].numpy()[:base_output_dim].astype(np.float64)
                     * lab_std + lab_mean)

            median[k], lower[k], upper[k], ground_truth[k] = med_h1, low_h1, up_h1, gt_h1

            for v, dims_v in enumerate(state_group_dims):
                s, e = _cum[v] * N_LAGS, _cum[v + 1] * N_LAGS
                win = (x[s:e] * feat_std[s:e] + feat_mean[s:e]).reshape(N_LAGS, dims_v)
                win = np.roll(win, -1, axis=0)
                win[-1] = med_h1[_cum[v]:_cum[v + 1]]
                x[s:e] = (win.flatten() - feat_mean[s:e]) / feat_std[s:e]

            next_idx = start_idx + k + 1
            if next_idx < N_test:
                x[ctrl_start:] = X_test_scaled[next_idx].numpy().astype(np.float64)[ctrl_start:]

    return {'median': median, 'lower': lower, 'upper': upper,
            'ground_truth': ground_truth, 'abs_error': np.abs(median - ground_truth)}
```

---

# File 2 — `Run_1/TiDE_pipeline.ipynb`

16 cells. Notebook imports helpers via `from helpers.TiDE_helpers import ...`. Model is
saved as `tide_model_full.pt` (distinct from MSA's `narx_model_full.pt`). Outputs →
`Models_TiDE/version_X/`, logs → `logs/TiDE/`.

**ver_1 (AR-calibrated) path:** set `CALIB_FROM_VERSION = 'version_0'` in Cell 2. The notebook
then loads ver_0 weights (Cell 6), **skips training** (Cell 7), creates `version_1` and copies the
ver_0 weights/scalers/metadata into it (Cells 8–9), and Cell 11 additionally computes the
AR-rollout threshold `cqr_qhat_ar.npy` ((O,)) alongside the one-shot `cqr_qhat.npy` ((M,O)).
Cell 15 reports rollout coverage under raw / one-shot-CQR / AR-CQR bands. Set
`CALIB_FROM_VERSION = None` to train a fresh model as before.

## Cell 0 — markdown

```markdown
# TiDE Training Pipeline — Calibrated Uncertainty Surrogate

One-shot **M-step quantile forecaster** for the COBR. Same 3922-D input as MSA-NARX
(3796 past window + 14×9 future covariates), but outputs a **(M, 26, 3)** tensor:
lower / median / upper quantile per output per horizon.

- **Quantile (pinball) loss** at τ = [0.05, 0.50, 0.95] → 90% prediction interval
- **CQR** (Conformalized Quantile Regression) post-hoc corrects coverage using a held-out
  calibration split → finite-sample coverage guarantee (the gap Chen et al. leave open)
- **4-way simulation split**: train / val / calibration / test
- Evaluation: per-horizon median R²/RMSE, coverage, sharpness, CRPS, AR-rollout tube coverage

Outputs → `Models_TiDE/version_X/`, logs → `logs/TiDE/`.
```

## Cell 1 — imports

```python
import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')
sys.path.insert(0, os.path.abspath('.'))

from helpers.helper_classes_MSA import GeLU
from helpers.helpers_MSA import (
    load_narx_dataset_with_metadata,
    load_scaler_params, save_scaler_params,
    scale_data,
    get_standard_trainer,
    save_model_metadata,
    get_latest_version_dir, create_next_version_dir,
    visualize_training_logs,
    filter_narx_data_by_vars,
    build_msa_dataset,
)
from helpers.TiDE_helpers import (
    TiDE, PinballLoss,
    get_simulation_split_dataloaders_4way,
    evaluate_tide_on_test_set,
    calibrate_cqr, calibrate_cqr_ar, apply_cqr,
    probabilistic_metrics,
    recursive_tide_rollout,
)
print(f'PyTorch  : {torch.__version__}')
print(f'Lightning: {pl.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
```

## Cell 2 — config

```python
DATASET_PATH = 'data_sets/21_05_2026/narx/thermal_narx_dataset_3.h5'
RAW_HDF5     = 'data_sets/21_05_2026/thermal_cobr_raw_data.h5'

M = 15
INCLUDE_HP_INT = True            # True -> 26 base outputs (incl. T_reactor max)

# Quantiles & conformal level
TAU   = [0.05, 0.50, 0.95]       # 90% prediction interval (median is the middle)
ALPHA = 0.10                     # CQR target miscoverage -> 1-ALPHA = 90% coverage
BURN_IN = 150                    # AR-CQR: skip first 150 rollout steps (transient; window
                                 # not yet fully model-driven, N_LAGS=146) before pooling residuals

# Calibration-only path (ver_1 = AR-calibrated variant of ver_0, no retraining).
# Set to a version dir ('version_0') to LOAD those weights and SKIP training;
# set to None to train a fresh model as before.
CALIB_FROM_VERSION = 'version_0'

# Feature window layout (must match dataset config)
STATE_GROUP_DIMS = [9, 8]
N_CTRL_DIMS      = 9

# 4-way simulation split (test = remainder = 0.15)
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.10
CALIB_FRAC = 0.15
SPLIT_SEED = 42

# TiDE architecture
HIDDEN_DIM         = 512
DECODER_OUTPUT_DIM = 32
TEMPORAL_WIDTH     = 16
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT            = 0.1
NOISE_SIGMA        = 0.05

# Training
LEARNING_RATE = 1e-4
MAX_EPOCHS    = 500
BATCH_SIZE    = 8192

GRADIENT_CLIP_VAL           = 0.5
GRADIENT_CLIP_ALGORITHM     = 'norm'
EARLY_STOPPING_PATIENCE     = 25
EARLY_STOPPING_MIN_DELTA    = 0.001
USE_STANDARD_EARLY_STOPPING = False

LOG_DIR     = 'logs/TiDE'
EXPERIMENT  = 'TiDE'
MODELS_BASE = 'Models_TiDE'
RESUME      = False

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
ACCELERATOR = 'gpu'  if DEVICE == 'cuda' else 'cpu'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_BASE, exist_ok=True)
print(f'Device : {DEVICE} | M={M} | quantiles={TAU} | alpha={ALPHA}')
print(f'Outputs: {26 if INCLUDE_HP_INT else 17} base | split {TRAIN_FRAC}/{VAL_FRAC}/{CALIB_FRAC}/rest')
```

## Cell 3 — load dataset

```python
data, metadata = load_narx_dataset_with_metadata(DATASET_PATH)
print('Data arrays:')
for k, v in data.items():
    print(f'  {k}: {tuple(v.shape)}')

label_names = []
if 'labels' in metadata:
    for vm in metadata['labels']:
        name = vm['name']; nd = vm.get('selected_dims', 1)
        label_names.extend([f'{name}[{i}]' for i in range(nd)] if nd > 1 else [name])
print(f'\nLabel columns ({len(label_names)}): {label_names[:3]} ... {label_names[-3:]}')
```

## Cell 4 — scale features / labels / inputs

```python
features_t       = data['narx_state_features'].float()
labels_t         = data['labels'].float()
input_features_t = data['input_features'].float()

_EXCLUDE = [] if INCLUDE_HP_INT else ['heating_power_avg']
features_t, labels_t, label_names, active_metadata = filter_narx_data_by_vars(
    features_t, labels_t, metadata, exclude_var_names=_EXCLUDE)
print(f'Feature dim: {features_t.shape[1]} | Label dim: {labels_t.shape[1]} | Input dim: {input_features_t.shape[1]}')

feat_scaler_path       = os.path.join(LOG_DIR, 'feature_scaler.yml')
label_scaler_path      = os.path.join(LOG_DIR, 'label_scaler.yml')
input_feat_scaler_path = os.path.join(LOG_DIR, 'input_feature_scaler.yml')

features_scaled,       feat_scaler       = scale_data(features_t.numpy(),       save_path=feat_scaler_path)
labels_scaled,         label_scaler      = scale_data(labels_t.numpy(),         save_path=label_scaler_path)
input_features_scaled, input_feat_scaler = scale_data(input_features_t.numpy(), save_path=input_feat_scaler_path)

print(f'NARX feats {tuple(features_scaled.shape)} | labels {tuple(labels_scaled.shape)} | inputs {tuple(input_features_scaled.shape)}')
```

## Cell 5 — build MSA dataset + 4-way split

```python
import h5py
with h5py.File(DATASET_PATH, 'r') as _f:
    sim_sample_counts = np.array(_f['sim_sample_counts'], dtype=np.int64)

msa_features_np, msa_labels_np, msa_sim_counts = build_msa_dataset(
    features_scaled.numpy(), labels_scaled.numpy(), input_features_scaled.numpy(),
    sim_sample_counts, M=M)

# Combined MSA feature scaler (analytic: tile input scaler M-1 times — no re-fit)
msa_feat_scaler = {
    'mean': np.concatenate([feat_scaler['mean'], np.tile(input_feat_scaler['mean'], M - 1)]),
    'std':  np.concatenate([feat_scaler['std'],  np.tile(input_feat_scaler['std'],  M - 1)]),
}
save_scaler_params(os.path.join(LOG_DIR, 'msa_feature_scaler.yml'), msa_feat_scaler)

msa_features_t = torch.from_numpy(msa_features_np).float()
msa_labels_t   = torch.from_numpy(msa_labels_np).float()
n_outputs = labels_scaled.shape[1]   # 26

print(f'MSA features {tuple(msa_features_t.shape)} | labels {tuple(msa_labels_t.shape)}')
print(f'Expected feat dim {features_scaled.shape[1] + (M-1)*N_CTRL_DIMS}, label dim {n_outputs*M}')

train_loader, val_loader, calib_loader, test_loader, split_info = get_simulation_split_dataloaders_4way(
    msa_features_t, msa_labels_t, sim_sample_counts=msa_sim_counts,
    train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, calib_frac=CALIB_FRAC,
    batch_size=BATCH_SIZE, seed=SPLIT_SEED, multiprocessing=False)

# calib/test loaders use shuffle=False, so torch.cat preserves the contiguous per-sim
# order in split_info — required for AR-CQR per-sim rollout alignment.
X_test  = torch.cat([xb for xb, _ in test_loader]);  y_test  = torch.cat([yb for _, yb in test_loader])
X_calib = torch.cat([xb for xb, _ in calib_loader]); y_calib = torch.cat([yb for _, yb in calib_loader])
print(f'Test  tensors: X={tuple(X_test.shape)}  y={tuple(y_test.shape)}')
print(f'Calib tensors: X={tuple(X_calib.shape)} y={tuple(y_calib.shape)}')

# sanity: calib_sim_counts must sum to len(X_calib)
assert split_info['calib_sim_counts'].sum() == X_calib.shape[0], 'calib sim-count mismatch'
print(f"Calib sims: {len(split_info['calib_sim_counts'])} | "
      f"Test sims: {len(split_info['test_sim_counts'])}")
```

## Cell 6 — build TiDE model (+ shape & monotonicity check)

```python
past_dim       = features_scaled.shape[1]      # 3796
future_cov_dim = (M - 1) * N_CTRL_DIMS         # 126

network_hp = {
    'past_dim':           past_dim,
    'future_cov_dim':     future_cov_dim,
    'M':                  M,
    'base_output_dim':    n_outputs,
    'hidden_dim':         HIDDEN_DIM,
    'decoder_output_dim': DECODER_OUTPUT_DIM,
    'temporal_width':     TEMPORAL_WIDTH,
    'num_encoder_layers': NUM_ENCODER_LAYERS,
    'num_decoder_layers': NUM_DECODER_LAYERS,
    'dropout':            DROPOUT,
    'activation':         GeLU(),
    'quantiles':          TAU,
    'noise_sigma':        NOISE_SIGMA,
}
training_hp = {
    'loss_function':    PinballLoss(TAU),
    'optimizer_class':  torch.optim.Adam,
    'optimizer_kwargs': {'lr': LEARNING_RATE},
    'scheduler_class':  torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_kwargs': {'factor': 0.5, 'patience': 10},
}

if CALIB_FROM_VERSION is None:
    model = TiDE(network_hp, training_hp)
else:
    # ver_1 = AR-calibrated variant of ver_0: reuse the trained weights, no retraining.
    _load_pt = os.path.join(MODELS_BASE, CALIB_FROM_VERSION, 'tide_model_full.pt')
    model = torch.load(_load_pt, weights_only=False)
    print(f'Loaded {CALIB_FROM_VERSION} weights from {_load_pt} (calibration-only, no retraining)')

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f'\nTrainable parameters: {n_params:,}')

# quick sanity check on a small batch
_out = model(X_test[:4])
print(f'Output shape: {tuple(_out.shape)}  (expect (4, {M}, {n_outputs}, {len(TAU)}))')
assert (_out[..., 0] <= _out[..., 1]).all() and (_out[..., 1] <= _out[..., 2]).all(), 'quantile crossing!'
print('Monotonicity OK: lower <= median <= upper')
print(f'PinballLoss on batch: {PinballLoss(TAU)(_out, y_test[:4]).item():.4f}')
```

## Cell 7 — train

```python
if CALIB_FROM_VERSION is not None:
    print(f'CALIB_FROM_VERSION={CALIB_FROM_VERSION} — skipping training, reusing loaded weights.')
else:
    logger = TensorBoardLogger(save_dir=LOG_DIR, name=EXPERIMENT)

    ckpt_path = None
    if RESUME:
        exp_dir = os.path.join(LOG_DIR, EXPERIMENT)
        if os.path.isdir(exp_dir):
            try:
                vdir = get_latest_version_dir(exp_dir)
                last = os.path.join(vdir, 'checkpoints', 'last.ckpt')
                if os.path.exists(last):
                    ckpt_path = last; print(f'Resuming from {last}')
            except Exception:
                print('No prior version — fresh start')

    trainer = get_standard_trainer(
        logger=logger, max_epochs=MAX_EPOCHS, accelerator=ACCELERATOR,
        gradient_clip_val=GRADIENT_CLIP_VAL, gradient_clip_algorithm=GRADIENT_CLIP_ALGORITHM,
        patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA,
        use_standard_early_stopping=USE_STANDARD_EARLY_STOPPING)

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
```

## Cell 8 — save model + scalers

```python
MODEL_SAVE_DIR = create_next_version_dir(MODELS_BASE)
_scaler_files = ('feature_scaler.yml', 'label_scaler.yml',
                 'input_feature_scaler.yml', 'msa_feature_scaler.yml')

if CALIB_FROM_VERSION is None:
    pt_path = os.path.join(MODEL_SAVE_DIR, 'tide_model_full.pt')
    torch.save(model, pt_path)
    print(f'Model saved: {pt_path}')
    _scaler_src_dir = LOG_DIR
else:
    # ver_1: copy ver_0 weights unchanged (no retraining) + its scalers
    _src_pt = os.path.join(MODELS_BASE, CALIB_FROM_VERSION, 'tide_model_full.pt')
    shutil.copy2(_src_pt, os.path.join(MODEL_SAVE_DIR, 'tide_model_full.pt'))
    print(f'Copied {CALIB_FROM_VERSION} weights -> {MODEL_SAVE_DIR}')
    _scaler_src_dir = os.path.join(MODELS_BASE, CALIB_FROM_VERSION)

for sc in _scaler_files:
    src = os.path.join(_scaler_src_dir, sc)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(MODEL_SAVE_DIR, sc))
print(f'Scalers copied to {MODEL_SAVE_DIR}')
```

## Cell 9 — save metadata

```python
model_config = {
    'network_hyperparameters': {k: (type(v).__name__ if k == 'activation' else v)
                                for k, v in network_hp.items()},
    'training_hyperparameters': {
        'loss_function': type(training_hp['loss_function']).__name__,
        'optimizer': training_hp['optimizer_class'].__name__,
        'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS,
        'scheduler': training_hp['scheduler_class'].__name__,
    },
    'data_split': {'method': 'simulation-wise-4way',
                   'train': TRAIN_FRAC, 'val': VAL_FRAC, 'calib': CALIB_FRAC, 'seed': SPLIT_SEED},
    'quantiles': TAU, 'alpha': ALPHA, 'burn_in': BURN_IN,
    'calib_from_version': CALIB_FROM_VERSION,
    'include_hp_int': INCLUDE_HP_INT, 'label_names': label_names,
    'M': M, 'base_output_dim': n_outputs,
    'n_ctrl_dims': N_CTRL_DIMS, 'state_group_dims': STATE_GROUP_DIMS,
}
if CALIB_FROM_VERSION is None:
    version_dir = get_latest_version_dir(os.path.join(LOG_DIR, EXPERIMENT))
    save_model_metadata(version_dir, active_metadata, model_config, DATASET_PATH)
    for fname in ('model_metadata.yml', 'metrics.csv'):
        src = os.path.join(version_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODEL_SAVE_DIR, fname))
else:
    # ver_1: inherit ver_0's metadata/metrics; record the AR-calibration config separately
    for fname in ('model_metadata.yml', 'metrics.csv'):
        src = os.path.join(MODELS_BASE, CALIB_FROM_VERSION, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODEL_SAVE_DIR, fname))
    import yaml
    with open(os.path.join(MODEL_SAVE_DIR, 'ar_calibration_config.yml'), 'w') as _f:
        yaml.safe_dump({'calib_from_version': CALIB_FROM_VERSION,
                        'quantiles': TAU, 'alpha': ALPHA, 'burn_in': BURN_IN,
                        'method': 'AR-rollout CQR (constant per-output q_hat)'}, _f)
print(f'Metadata saved & copied to {MODEL_SAVE_DIR}')
```

## Cell 10 — per-horizon median evaluation

```python
results = evaluate_tide_on_test_set(
    model, X_test, y_test, label_scaler_params=label_scaler,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE)
metrics_df = results['metrics_df']

for horizon in [1, M]:
    hi = horizon - 1
    ph = results['median'][:, hi, :]; gh = results['ground_truth'][:, hi, :]
    ss_res = np.sum((gh - ph) ** 2, axis=0); ss_tot = np.sum((gh - gh.mean(0)) ** 2, axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-12); rmse = np.sqrt(np.mean((ph - gh) ** 2, axis=0))
    print(f'\n--- Horizon {horizon} per-output (median) ---')
    print(f"{'Output':<28}{'RMSE':>9}{'R2':>9}"); print('-' * 46)
    for i, nm in enumerate(label_names):
        print(f'{nm:<28}{rmse[i]:9.4f}{r2[i]:9.4f}')

_res_path = os.path.join(MODEL_SAVE_DIR, 'Results.txt')
with open(_res_path, 'w') as f:
    f.write(f'TiDE  M={M}  base_output_dim={n_outputs}  quantiles={TAU}  alpha={ALPHA}\n')
    f.write(f'Dataset: {DATASET_PATH}\n\nPer-horizon median summary:\n')
    f.write(metrics_df.to_string(index=False)); f.write('\n')
print(f'\nResults saved: {_res_path}')
```

## Cell 11 — CQR calibration

```python
q_hat = calibrate_cqr(
    model, X_calib, y_calib, label_scaler_params=label_scaler,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE, alpha=ALPHA)
np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat.npy'), q_hat)
print(f'q_hat shape {q_hat.shape} saved to {MODEL_SAVE_DIR}/cqr_qhat.npy')

# AR-calibrated CQR — sized for the compounding rollout regime the MPC faces.
q_hat_ar = calibrate_cqr_ar(
    model, X_calib, y_calib, calib_sim_counts=split_info['calib_sim_counts'],
    feature_scaler_params=msa_feat_scaler, label_scaler_params=label_scaler,
    state_group_dims=STATE_GROUP_DIMS, n_ctrl_dims=N_CTRL_DIMS,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE,
    alpha=ALPHA, burn_in=BURN_IN)
np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat_ar.npy'), q_hat_ar)
print(f'q_hat_ar shape {q_hat_ar.shape} saved to {MODEL_SAVE_DIR}/cqr_qhat_ar.npy')

# AR inflation vs one-shot (horizon-1): AR-CQR should be wider on most outputs.
print(f"\n{'Output':<28}{'q_hat[h1]':>12}{'q_hat_ar':>12}{'ratio':>9}")
for i, nm in enumerate(label_names):
    print(f'{nm:<28}{q_hat[0, i]:12.4f}{q_hat_ar[i]:12.4f}'
          f'{q_hat_ar[i] / (q_hat[0, i] + 1e-12):9.2f}')
```

## Cell 12 — probabilistic metrics: raw vs CQR

```python
median    = results['median']; lower_raw = results['lower']; upper_raw = results['upper']
y_phys    = results['ground_truth']; quant_preds = results['predictions']

lower_cqr, upper_cqr = apply_cqr(lower_raw, upper_raw, q_hat)

df_raw = probabilistic_metrics(lower_raw, upper_raw, y_phys, label_names,
                               quantile_preds=quant_preds, quantiles=TAU)
df_cqr = probabilistic_metrics(lower_cqr, upper_cqr, y_phys, label_names,
                               quantile_preds=quant_preds, quantiles=TAU)

cmp = df_raw[['output', 'coverage', 'mean_width']].rename(
        columns={'coverage': 'cov_raw', 'mean_width': 'width_raw'})
cmp['cov_cqr']   = df_cqr['coverage'].values
cmp['width_cqr'] = df_cqr['mean_width'].values
cmp['CRPS']      = df_raw['CRPS'].values
print(cmp.to_string(index=False))
print(f"\nMean coverage  raw={cmp['cov_raw'].mean():.3f}  cqr={cmp['cov_cqr'].mean():.3f}  (target {1-ALPHA:.2f})")
print(f"Mean width     raw={cmp['width_raw'].mean():.3f}  cqr={cmp['width_cqr'].mean():.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
x = np.arange(len(label_names))
ax1.bar(x - 0.2, cmp['cov_raw'], 0.4, label='raw', color='lightcoral')
ax1.bar(x + 0.2, cmp['cov_cqr'], 0.4, label='CQR', color='steelblue')
ax1.axhline(1 - ALPHA, ls='--', color='k', label=f'target {1-ALPHA:.2f}')
ax1.set_xticks(x); ax1.set_xticklabels(label_names, rotation=90, fontsize=7)
ax1.set_ylabel('Coverage'); ax1.set_title('Coverage: raw vs CQR'); ax1.legend()
ax2.bar(x - 0.2, cmp['width_raw'], 0.4, label='raw', color='lightcoral')
ax2.bar(x + 0.2, cmp['width_cqr'], 0.4, label='CQR', color='steelblue')
ax2.set_xticks(x); ax2.set_xticklabels(label_names, rotation=90, fontsize=7)
ax2.set_ylabel('Mean interval width'); ax2.set_title('Sharpness: raw vs CQR'); ax2.legend()
fig.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'cqr_coverage_sharpness.png'), dpi=150)
plt.show()
```

## Cell 13 — prediction + tube plots (horizons 1 and M)

```python
for plot_h in [1, M]:
    hi = plot_h - 1
    med_h = median[:, hi, :]; lo_h = lower_cqr[:, hi, :]; up_h = upper_cqr[:, hi, :]; gt_h = y_phys[:, hi, :]
    n_plot = min(2400, med_h.shape[0]); t = np.arange(n_plot)
    n_cols = 2; n_rows = (n_outputs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=True)
    axes = axes.flatten()
    for i in range(n_outputs):
        ax = axes[i]
        ax.fill_between(t, lo_h[:n_plot, i], up_h[:n_plot, i], color='steelblue', alpha=0.25, label='90% CQR tube')
        ax.plot(t, gt_h[:n_plot, i],  color='black',   lw=1.0, label='Actual')
        ax.plot(t, med_h[:n_plot, i], color='crimson', lw=0.9, ls='--', label='Median')
        ax.set_title(label_names[i], fontsize=9); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=7)
    for ax in axes[n_outputs:]: ax.set_visible(False)
    fig.suptitle(f'TiDE horizon={plot_h}: median + CQR tube vs actual (first {n_plot} test samples)', fontsize=13)
    fig.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'test_tube_h{plot_h}.png'), dpi=150)
    plt.show()
```

## Cell 14 — training curves

```python
EVAL_VERSION = None  # None = current session | 'version_0' = reload from Models_TiDE/
if EVAL_VERSION is None:
    _vd = get_latest_version_dir(os.path.join(LOG_DIR, EXPERIMENT))
    _title = f'Training curves — {EXPERIMENT} (current session)'
else:
    _vd = os.path.join(MODELS_BASE, EVAL_VERSION)
    _title = f'Training curves — {EVAL_VERSION}'
_metrics = os.path.join(_vd, 'metrics.csv')
if not os.path.exists(_metrics):
    print(f'metrics.csv not found at {_metrics}')
else:
    visualize_training_logs(_vd); plt.title(_title); plt.tight_layout(); plt.show()
```

## Cell 15 — AR rollout + tube coverage

```python
N_ROLLOUT = int(split_info['test_sim_counts'][0]); START_IDX = 0   # clean single-sim rollout
_msa_fs = load_scaler_params(os.path.join(MODEL_SAVE_DIR, 'msa_feature_scaler.yml'))

rollout = recursive_tide_rollout(
    model, X_test, y_test, feature_scaler_params=_msa_fs, label_scaler_params=label_scaler,
    N_rollout=N_ROLLOUT, start_idx=START_IDX, device=DEVICE,
    state_group_dims=STATE_GROUP_DIMS, n_ctrl_dims=N_CTRL_DIMS,
    M=M, base_output_dim=n_outputs, quantiles=TAU)

gt_roll = rollout['ground_truth']; med_roll = rollout['median']
steps = med_roll.shape[0]

# Three bands: raw model quantiles, one-shot CQR (horizon-1), AR-CQR.
# q_hat[0] and q_hat_ar are (O,) -> broadcast over (steps, O).
bands = {
    'raw':         (rollout['lower'],            rollout['upper']),
    'CQR(1-shot)': (rollout['lower'] - q_hat[0], rollout['upper'] + q_hat[0]),
    'AR-CQR':      (rollout['lower'] - q_hat_ar, rollout['upper'] + q_hat_ar),
}

# Coverage over the post-transient region only (apples-to-apples with calibration).
mask = np.arange(steps) > BURN_IN
print(f'AR rollout coverage over steps k>{BURN_IN} (n={int(mask.sum())} of {steps}):')
print(f"  {'band':<14}{'overall cov':>12}{'mean width':>12}")
inside_by_band = {}
for name, (lo, up) in bands.items():
    inside = (gt_roll >= lo) & (gt_roll <= up)
    inside_by_band[name] = inside
    print(f'  {name:<14}{inside[mask].mean():12.3f}{(up - lo)[mask].mean():12.3f}')
print(f'  (target {1-ALPHA:.2f})  --  expect CQR(1-shot) < target, AR-CQR ~= target')

print(f'\nAR-CQR per-output coverage (k>{BURN_IN}):')
_inside_ar = inside_by_band['AR-CQR']
for i, nm in enumerate(label_names):
    print(f'  {nm:<28} {_inside_ar[mask, i].mean():.3f}')

# plot the AR-CQR tube
lo_roll, up_roll = bands['AR-CQR']

from collections import OrderedDict
t = np.arange(steps)
groups = OrderedDict()
for i, nm in enumerate(label_names):
    groups.setdefault(nm.rsplit('[', 1)[0].rstrip('_'), []).append(i)

fig, axes = plt.subplots(len(groups), 1, figsize=(13, 3.2 * len(groups)), sharex=True)
if len(groups) == 1: axes = [axes]
for ax, (gname, idxs) in zip(axes, groups.items()):
    i0 = idxs[0]
    ax.fill_between(t, lo_roll[:, i0], up_roll[:, i0], color='steelblue', alpha=0.2)
    for i in idxs:
        ax.plot(t, gt_roll[:, i],  lw=0.8)
        ax.plot(t, med_roll[:, i], lw=0.8, ls='--')
    ax.set_title(f'{gname} — AR rollout (median dashed, GT solid, tube=ch{i0})'); ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Rollout step')
fig.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'tide_ar_rollout.png'), dpi=150)
plt.show()
```

---

# ver_2 — AR-CQR Evaluation Enhancements (current)

Pools AR calibration **and** evaluation over **all** sims (not one), adds a **step-adaptive** band, a **Mondrian-max reactor safety** band, group-conditional reporting, and a coverage-vs-step diagnostic. Calibration/eval only — no retraining. New artifacts → `Models_TiDE/version_2/`: `cqr_qhat_ar_stepwise.npy` (n_bins,O), `cqr_qhat_ar_safety.npy` (O,), `ar_eval_config.yml`, `ar_coverage_per_output.csv`, `ar_coverage_vs_step.png`.

## File 1 additions — `helpers/TiDE_helpers.py`

`calibrate_cqr_ar` is refactored to call `collect_ar_rollouts` then take the pooled `(O,)` quantile (numerically identical `q_hat_ar`). New functions appended after `recursive_tide_rollout`:

```python
# Pooled per-sim AR rollouts + step-adaptive / safety CQR ----------------------------

def collect_ar_rollouts(model, X, y, sim_counts,
                        feature_scaler_params, label_scaler_params,
                        state_group_dims, n_ctrl_dims,
                        M=15, base_output_dim=26, quantiles=None,
                        device='cpu', burn_in=150, max_steps_per_sim=None):
    """
    Run a per-sim AR rollout over every simulation in `sim_counts` (the contiguous
    per-sim sample counts from split_info), keep steps k >= burn_in, and pool the
    rows across sims. Reused by AR calibration (calib split) and AR evaluation
    (test split) so the rollout logic lives in one place.

    Returns dict of pooled arrays:
        lower, median, upper, ground_truth : (Ntot, O)
        step_idx : (Ntot,)  absolute rollout step k of each pooled row
        sim_idx  : (Ntot,)  index (0..n_sims-1) of the sim each row came from
        n_sims_used : int
    """
    offsets = np.concatenate([[0], np.cumsum(np.asarray(sim_counts, dtype=np.int64))])
    lo, md, up, gt, sidx, simidx = [], [], [], [], [], []
    n_used = 0
    for s in range(len(sim_counts)):
        start, count = int(offsets[s]), int(sim_counts[s])
        n_roll = count if max_steps_per_sim is None else min(count, max_steps_per_sim)
        if n_roll <= burn_in:
            continue
        roll = recursive_tide_rollout(
            model, X, y, feature_scaler_params, label_scaler_params,
            N_rollout=n_roll, start_idx=start, device=device,
            state_group_dims=state_group_dims, n_ctrl_dims=n_ctrl_dims,
            M=M, base_output_dim=base_output_dim, quantiles=quantiles)
        steps = roll['median'].shape[0]
        keep = np.arange(steps) >= burn_in
        lo.append(roll['lower'][keep]);        md.append(roll['median'][keep])
        up.append(roll['upper'][keep]);        gt.append(roll['ground_truth'][keep])
        sidx.append(np.arange(steps)[keep])
        simidx.append(np.full(int(keep.sum()), s, dtype=np.int64))
        n_used += 1
    return {
        'lower':  np.concatenate(lo),  'median':       np.concatenate(md),
        'upper':  np.concatenate(up),  'ground_truth': np.concatenate(gt),
        'step_idx': np.concatenate(sidx), 'sim_idx': np.concatenate(simidx),
        'n_sims_used': n_used,
    }


def step_bin_ids(step_idx, n_bins, bin_width=250, burn_in=150):
    """Fixed-width absolute-step bin index for each step, clipped to the last bin."""
    return np.clip((np.asarray(step_idx) - burn_in) // bin_width, 0, n_bins - 1).astype(np.int64)


def make_step_bins(step_idx, bin_width=250, burn_in=150):
    """Fixed-width absolute-step bins starting at burn_in; last bin open-ended.
    Returns (bin_edges, bin_ids). bin_edges[i] is the lower edge of bin i."""
    max_step = int(np.max(step_idx))
    bin_edges = np.arange(burn_in, max_step + 1, bin_width)
    n_bins = len(bin_edges)
    return bin_edges, step_bin_ids(step_idx, n_bins, bin_width, burn_in)


def cqr_qhat_stepwise(scores, bin_ids, n_bins, alpha=0.10):
    """Finite-sample (1-alpha) nonconformity quantile within each (step-bin, output).
    scores (N,O), bin_ids (N,). Returns (n_bins, O). Empty bins fall back to the
    pooled-over-all-steps quantile so the band is always defined."""
    O = scores.shape[1]
    q = np.zeros((n_bins, O))
    n_all = scores.shape[0]
    lvl_all = min(1.0, (1 - alpha) * (1 + 1.0 / n_all))
    q_pool = np.quantile(scores, lvl_all, axis=0, method='higher')
    for b in range(n_bins):
        m = bin_ids == b
        nb = int(m.sum())
        if nb == 0:
            q[b] = q_pool
            continue
        lvl = min(1.0, (1 - alpha) * (1 + 1.0 / nb))
        q[b] = np.quantile(scores[m], lvl, axis=0, method='higher')
    return q


def apply_cqr_stepwise(lower, upper, q_hat_stepwise, bin_ids):
    """Widen each row by its bin's q_hat. lower/upper (N,O); q_hat_stepwise (n_bins,O);
    bin_ids (N,). Returns (lower_adj, upper_adj)."""
    qb = q_hat_stepwise[bin_ids]                 # (N, O)
    return lower - qb, upper + qb


def cqr_qhat_safety(q_hat_per_output, safety_channels):
    """Mondrian-max safety band: set the given channels to their shared max q_hat,
    so every channel in the safety group meets target coverage. Returns (O,)."""
    q = np.asarray(q_hat_per_output, dtype=np.float64).copy()
    sc = list(safety_channels)
    q[sc] = q[sc].max()
    return q
```

## File 2 changes — `TiDE_pipeline.ipynb`

**Cell 1 imports** add: `collect_ar_rollouts, make_step_bins, step_bin_ids, cqr_qhat_stepwise, apply_cqr_stepwise, cqr_qhat_safety`.

**Cell 2 config** adds:
```python
BIN_WIDTH = 250          # AR step-bin width (absolute rollout steps)
GROUPS = {'reactor': slice(0, 9), 'thermostat': slice(9, 17),
          'heating': slice(17, 25), 'T_reactor_max': slice(25, 26)}
SAFETY_CHANNELS = list(range(9)) + [25]   # reactor zones + T_reactor_max
```

**Cell 11 — calibration (one-shot + AR constant/stepwise/safety + config):**
```python
# Cell 11 -- CQR Calibration (one-shot + AR: constant / step-adaptive / safety)
import yaml

q_hat = calibrate_cqr(
    model, X_calib, y_calib, label_scaler_params=label_scaler,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE, alpha=ALPHA)
np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat.npy'), q_hat)

# ONE AR calib-rollout pass over all calib sims -> derive every AR band from it.
coll_cal = collect_ar_rollouts(
    model, X_calib, y_calib, split_info['calib_sim_counts'],
    feature_scaler_params=msa_feat_scaler, label_scaler_params=label_scaler,
    state_group_dims=STATE_GROUP_DIMS, n_ctrl_dims=N_CTRL_DIMS,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE, burn_in=BURN_IN)
S_cal = np.maximum(coll_cal['lower'] - coll_cal['ground_truth'],
                   coll_cal['ground_truth'] - coll_cal['upper'])

n_cal = S_cal.shape[0]
_lvl = min(1.0, (1 - ALPHA) * (1 + 1.0 / n_cal))
q_hat_ar = np.quantile(S_cal, _lvl, axis=0, method='higher')         # (O,)

bin_edges, bin_ids_cal = make_step_bins(coll_cal['step_idx'], bin_width=BIN_WIDTH, burn_in=BURN_IN)
n_bins = len(bin_edges)
q_hat_ar_stepwise = cqr_qhat_stepwise(S_cal, bin_ids_cal, n_bins, alpha=ALPHA)   # (n_bins, O)
q_hat_ar_safety   = cqr_qhat_safety(q_hat_ar, SAFETY_CHANNELS)                   # (O,)

np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat_ar.npy'),          q_hat_ar)
np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat_ar_stepwise.npy'), q_hat_ar_stepwise)
np.save(os.path.join(MODEL_SAVE_DIR, 'cqr_qhat_ar_safety.npy'),   q_hat_ar_safety)

_ar_cfg = {
    'alpha': ALPHA, 'burn_in': BURN_IN, 'bin_width': BIN_WIDTH,
    'n_bins': int(n_bins), 'bin_edges': [int(e) for e in bin_edges],
    'safety_channels': SAFETY_CHANNELS,
    'groups': {k: [int(v.start), int(v.stop)] for k, v in GROUPS.items()},
    'seed': SPLIT_SEED, 'calib_sims': int(coll_cal['n_sims_used']),
    'n_calib_scores': int(n_cal), 'source_version': CALIB_FROM_VERSION,
}
with open(os.path.join(MODEL_SAVE_DIR, 'ar_eval_config.yml'), 'w') as _f:
    yaml.safe_dump(_ar_cfg, _f, sort_keys=False)
```

**Cell 15 — eval pooled over ALL test sims (5 bands + group-conditional + per-sim min/max):**
```python
# Cell 15 -- AR rollout evaluation POOLED OVER ALL TEST SIMS (5 bands)
import pandas as pd
_msa_fs = load_scaler_params(os.path.join(MODEL_SAVE_DIR, 'msa_feature_scaler.yml'))

coll_te = collect_ar_rollouts(
    model, X_test, y_test, split_info['test_sim_counts'],
    feature_scaler_params=_msa_fs, label_scaler_params=label_scaler,
    state_group_dims=STATE_GROUP_DIMS, n_ctrl_dims=N_CTRL_DIMS,
    M=M, base_output_dim=n_outputs, quantiles=TAU, device=DEVICE, burn_in=BURN_IN)

lo0, up0 = coll_te['lower'], coll_te['upper']
med, gt  = coll_te['median'], coll_te['ground_truth']
sim_idx, step_idx = coll_te['sim_idx'], coll_te['step_idx']
n_test_sims = coll_te['n_sims_used']

bin_ids_te = step_bin_ids(step_idx, n_bins, bin_width=BIN_WIDTH, burn_in=BURN_IN)
lo_step, up_step = apply_cqr_stepwise(lo0, up0, q_hat_ar_stepwise, bin_ids_te)

bands = {
    'raw':          (lo0,                    up0),
    'CQR(1-shot)':  (lo0 - q_hat[0],         up0 + q_hat[0]),
    'AR-CQR':       (lo0 - q_hat_ar,         up0 + q_hat_ar),
    'AR-CQR(step)': (lo_step,                up_step),
    'safety':       (lo0 - q_hat_ar_safety,  up0 + q_hat_ar_safety),
}
inside_by_band = {nm: ((gt >= lo) & (gt <= up)) for nm, (lo, up) in bands.items()}

# overall + group-conditional coverage per band
group_idx = {g: list(range(sl.start, sl.stop)) for g, sl in GROUPS.items()}
for name, (lo, up) in bands.items():
    inside = inside_by_band[name]
    # ... print overall inside.mean(), width, and inside[:, idxs].mean() per group ...

# per-output coverage = mean/min/max across test sims (via sim_idx) -> ar_coverage_per_output.csv
def per_sim_cov(inside):
    out = np.zeros((n_test_sims, inside.shape[1]))
    for s in range(n_test_sims):
        out[s] = inside[sim_idx == s].mean(axis=0)
    return out
# representative sim-0 tube plot -> tide_ar_rollout.png
```
(Full cell text is in the notebook; only the structure is shown here.)

**Cell 16 — coverage-vs-step diagnostic (NEW):**
```python
# Cell 16 -- Coverage-vs-step diagnostic (pooled over test sims, per group)
bin_centers = bin_edges + BIN_WIDTH / 2.0
diag_bands = ['raw', 'AR-CQR', 'AR-CQR(step)']
fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.3 * len(GROUPS), 4), sharey=True)
if len(GROUPS) == 1: axes = [axes]
for ax, (gname, sl) in zip(axes, GROUPS.items()):
    gidx = list(range(sl.start, sl.stop))
    for bname in diag_bands:
        inside_g = inside_by_band[bname][:, gidx]
        cov_by_bin = np.array([
            inside_g[bin_ids_te == b].mean() if np.any(bin_ids_te == b) else np.nan
            for b in range(n_bins)])
        ax.plot(bin_centers, cov_by_bin, marker='o', ms=3, label=bname)
    ax.axhline(1 - ALPHA, ls='--', color='k', lw=1)
    ax.set_title(gname); ax.set_xlabel('rollout step'); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.02)
axes[0].set_ylabel('empirical coverage'); axes[0].legend(fontsize=8, loc='lower left')
fig.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'ar_coverage_vs_step.png'), dpi=150)
plt.show()
```
