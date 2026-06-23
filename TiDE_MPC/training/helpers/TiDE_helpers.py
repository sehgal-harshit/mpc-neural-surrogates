"""
TiDE (Time-series Dense Encoder) helpers for the COBR surrogate.

This module adds the TiDE-specific pieces on top of the shared MSA utilities:
    Classes
        ResidualBlock   - canonical TiDE residual MLP block (skip + LayerNorm)
        TiDE            - one-shot M-step quantile forecaster (Lightning module)
        PinballLoss     - multi-quantile (pinball) loss

    Functions
        get_simulation_split_dataloaders_4way - train/val/calib/test sim split (+split_info)
        evaluate_tide_on_test_set             - per-horizon R2/RMSE on the median
        calibrate_cqr                         - one-shot conformal threshold q_hat (M,O)
        calibrate_cqr_ar                      - AR-calibrated constant band q_hat (O,)
        apply_cqr                             - widen [lower, upper] by q_hat
        probabilistic_metrics                 - coverage / sharpness / CRPS table
        recursive_tide_rollout                - autoregressive rollout with tube
        collect_ar_rollouts                   - pool per-sim AR rollouts (calib/test)
        make_step_bins / step_bin_ids         - fixed-width absolute-step bins
        cqr_qhat_stepwise / apply_cqr_stepwise- step-adaptive band per (bin, output)
        cqr_qhat_safety                       - Mondrian-max conservative group band

Everything else (scaling, trainer, metadata I/O, build_msa_dataset) is imported
from helpers_MSA so there is a single source of truth.
"""

import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.shared_helpers.helper_classes_MSA import pytorch_lightning_standard_network


# Building block ------

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

    def forward(self,x):
        h = self.lin2(self.drop(self.act(self.lin1(x))))
        return self.norm(h + self.skip(x))


# TiDE Model -----

class TiDE(pytorch_lightning_standard_network):
    """
    One-shot multiple quantile forecaster
    
    Input x : (B, past_dim + future_cov_dim) -- same input as MSA_NARX
    Output : (B, M, base_output_dim, n_quantiles)
    
    network_hyperparameters keys:
    past_dim, future_cov_dim, M, base_output_dim,
    hidden_dim, decoder_output_dim, temportal_width,
    num_encoder_layers, num_decoder_layers, dropout,
    activation, quantiles (list[float]), noise_sigma
    """

    def __init__(self, network_hyperparameters, training_hyperparameters):
        super().__init__(**training_hyperparameters)
        self.save_hyperparameters()

        hp                   = network_hyperparameters
        self.past_dim        = hp['past_dim']
        self.future_cov_dim  = hp['future_cov_dim']
        self.M               = hp['M']
        self.base_output_dim = hp['base_output_dim']
        self.quantiles       = list(hp['quantiles'])
        self.n_quantiles     = len(self.quantiles)
        self.noise_sigma     = hp.get('noise_sigma', 0.0)

        hidden_dim         = hp['hidden_dim']
        decoder_output_dim = hp['decoder_output_dim']
        temporal_width     = hp['temporal_width']
        dropout            = hp['dropout']
        act                = hp['activation']

        # Future covariates arrive flat as (M-1) steps * x n_ctrl, recover n_ctrl
        self.n_steps_cov = self.M - 1
        self.n_ctrl      = self.future_cov_dim // self.n_steps_cov

        # (1) Per-step covariate projection -- shared weights across horizon steps, dim down to "t"
        self.cov_proj = nn.Linear(self.n_ctrl, temporal_width)

        # (2) Dense encoder: [past || projected cov over M steps] -> latent
        enc_in      = self.past_dim + self.M * temporal_width
        enc_layers  = [ResidualBlock(enc_in, hidden_dim, hidden_dim, dropout, act)]
        for _ in range(hp['num_encoder_layers'] - 1):
            enc_layers.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout, act))
        self.encoder = nn.ModuleList(enc_layers)

        # (3) Dense Decoder: latent -> (M x decoder_output_dim)
        dec_layers = []
        for _ in range(hp['num_decoder_layers'] - 1):
            dec_layers.append(ResidualBlock(hidden_dim, hidden_dim, hidden_dim, dropout, act))
        self.decoder            = nn.ModuleList(dec_layers)
        self.decoder_head       = nn.Linear(hidden_dim, self.M*decoder_output_dim)
        self.decoder_output_dim = decoder_output_dim
        
        #(4) Temporal Decoder: per-horizon [decoded slice || covariate]
        td_in = decoder_output_dim + temporal_width
        self.temporal_decoder = ResidualBlock(
            td_in, self.base_output_dim*self.n_quantiles, temporal_width, dropout, act
            )
        
        #(5) Global linear residual skip ---
        self.global_skip = nn.Linear(self.past_dim, self.M*self.base_output_dim)

    def _project_covariates(self, future_cov, batch_size):
        """future_cov (B, (M-1)*n_ctrl) -> projected (B, M, temporal_width).
        Horizon 1 has no future covariate (it's already in the past window), so
        we prepend a zero row to align horizon h with covariate for step k+h."""
        cov = future_cov.view(batch_size, self.n_steps_cov, self.n_ctrl)
        cov = self.cov_proj(cov)
        pad = torch.zeros(batch_size, 1, cov.shape[-1], device=cov.device, dtype=cov.dtype)
        return torch.cat([pad, cov], dim=1)
    
    def forward(self, x):
        B = x.shape[0]
        past = x[:, :self.past_dim]
        future_cov = x[:, self.past_dim:]
        cov_proj = self._project_covariates(future_cov, B)

        # Encoder - 
        e = torch.cat([past, cov_proj.reshape(B, -1)], dim=1)
        for block in self.encoder:
            e = block(e)
        
        # Decoder - 
        d = e
        for block in self.decoder:
            d = block(d)
        d = self.decoder_head(d).view(B, self.M, self.decoder_output_dim)

        # Temporal Decoder --
        td_in = torch.cat([d, cov_proj], dim=-1)
        raw = self.temporal_decoder(td_in.reshape(B*self.M, -1))
        raw = raw.view(B, self.M, self.base_output_dim, self.n_quantiles)

        # Global residual skip ---
        skip = self.global_skip(past).view(B, self.M, self.base_output_dim)

        # Monotonic quantile head -- 
        mid = self.n_quantiles // 2
        out = torch.empty_like(raw)
        out[..., mid] = raw[..., mid] + skip
        for q in range(mid-1, -1, -1):
            out[..., q] = out[..., q + 1] - F.softplus(raw[..., q])
        for q in range(mid + 1, self.n_quantiles):
            out[..., q] = out[..., q - 1] + F.softplus(raw[..., q])
        return out
    
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
        

# Quantile (pinball) loss -----

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


# Train / val / calib / test split -- simulation wise -- 

def get_simulation_split_dataloaders_4way(
        features, labels, sim_sample_counts,
        train_frac=0.6, val_frac=0.1, calib_frac=0.15,
        batch_size=512, seed=42,
        multiprocessing=True, cpu_count=mp.cpu_count()):
    """
    Split into train/val/calib/test DataLoaders by whole simulation trajectories.
    The calibration split is held fully out of training so CQR coverage guarantees
    hold. Test is whatever remains after train+val+calib.

    Also returns split_info with per-split sim ids and per-sim sample counts (in the
    contiguous order the samples appear inside X_calib / X_test) -- needed for the
    per-sim AR rollout in calibrate_cqr_ar.
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

    # Per-split sim boundaries -- needed for per-sim AR rollout on the calib split.
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


# Forward / Eval ---

def _forward_quantiles(model, X, device='cpu', batch_size=1000):
    """Run the model in batches, retun scaled quantile predictions (N, M, O, Q)"""
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


# CQR ---

def calibrate_cqr(model, X_calib, y_calib, label_scaler_params,
                  M=15, base_output_dim=26, quantiles=None, device='cpu', alpha=0.10):
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
    lower, upper = pred_phys[..., 0], pred_phys[..., -1]

    y_phys = (y_calib.numpy().astype(np.float64).reshape(-1, M, base_output_dim)
              * lstd[None, None, :] + lmean[None, None, :])

    s = np.maximum(lower - y_phys, y_phys - upper)
    n = s.shape[0]
    level = min(1.0, (1-alpha)*(1 + 1.0/n))

    q_hat = np.quantile(s, level, axis=0, method='higher')

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
    AR-calibrated CQR -- sizes the tube for the *compounding* error regime the
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
    feature_scaler_params  : the FULL MSA feature scaler (msa_feat_scaler) -- rollout
                             un/re-scales the whole 3922-D window.
    Returns q_hat of shape (O,).
    """
    coll = collect_ar_rollouts(
        model, X_calib, y_calib, calib_sim_counts,
        feature_scaler_params, label_scaler_params,
        state_group_dims, n_ctrl_dims,
        M=M, base_output_dim=base_output_dim, quantiles=quantiles,
        device=device, burn_in=burn_in, max_steps_per_sim=max_steps_per_sim)
    S = np.maximum(coll['lower'] - coll['ground_truth'],
                   coll['ground_truth'] - coll['upper'])              # (Ntot, O)
    n = S.shape[0]
    level = min(1.0, (1 - alpha) * (1 + 1.0 / n))
    q_hat = np.quantile(S, level, axis=0, method='higher')            # (O,)
    print(f'AR-CQR: {coll["n_sims_used"]} sims, n_scores={n} (k>={burn_in}), '
          f'level={level:.5f}, q_hat range [{q_hat.min():.3f}, {q_hat.max():.3f}]')
    return q_hat

def apply_cqr(lower, upper, q_hat):
    """Widen the interval: [lower - q_hat, upper + q_hat].
    Broadcasts (M,O) (one-shot) or (O,) (AR) over the leading axes."""
    return lower - q_hat[None, ...], upper + q_hat[None, ...]


# Probabilistic metrics --- 

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


# Autoregressive rollout (with uncertainty tube) ----

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
