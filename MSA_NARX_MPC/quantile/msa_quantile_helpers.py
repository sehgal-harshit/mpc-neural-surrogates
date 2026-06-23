"""
Quantile (triple-head) MSA-NARX: point MLP_MSA -> monotonic 3-quantile head.
Mirrors the TiDE quantile construction so the model feeds the common CQR pipeline.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from common.shared_helpers.helper_classes_MSA import pytorch_lightning_standard_network

class MLP_MSA_Quantile(pytorch_lightning_standard_network):
    """
    MSA-NARX with a monotonic multi-quantile head.

    Body == MLP_MSA ([Linear->act]*hidden); head emits raw (B, M, O, Q), then quantiles
    are built outward from the median via softplus so lower <= median <= upper by
    construction (no quantile crossing) -- identical to the TiDE head.

    network_hyperparameters keys:
        input_dim, hidden_dims, base_output_dim, M,
        quantiles (list[float]), activation (nn.Module), noise_sigma
    Output: (B, M, base_output_dim, n_quantiles)
    """

    def __init__(self, network_hyperparameters, training_hyperparameters):
        super().__init__(**training_hyperparameters)
        
        hp = network_hyperparameters
        input_dim = hp['input_dim']
        hidden_dims = hp['hidden_dims']
        self.base_output_dim = hp['base_output_dim']
        self.M = hp['M']
        self.quantiles = list(hp['quantiles'])
        self.n_quantiles = len(self.quantiles)
        activation = hp['activation']
        self.noise_sigma = hp.get('noise_sigma', 0.0)

        layers, in_features = [], input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
        
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, self.M*self.base_output_dim*self.n_quantiles)

    def forward(self, x):
        B = x.shape[0]
        raw = self.head(self.body(x)).view(B, self.M, self.base_output_dim, self.n_quantiles)
        mid = self.n_quantiles // 2
        out = torch.empty_like(raw)
        out[..., mid] = raw[..., mid]
        for q in range(mid - 1, -1, -1):
            out[..., q] = out[..., q + 1] - F.softplus(raw[..., q])
        for q in range(mid + 1, self.n_quantiles):
            out[..., q] = out[..., q - 1] + F.softplus(raw[..., q])
        return out                                            # (B, M, O, Q)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.noise_sigma > 0.0:
            x = x + torch.randn_like(x) * self.noise_sigma
        loss = self.loss_function(self(x), y)                 # PinballLoss: y is (B, M*O)
        self.log('train_loss', loss)
        return loss

    def predict_trajectory(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

def warmstart_quantile_from_point(qmodel, point_model, init_half_width=0.1):
    """
    Warm-start the quantile model from a point MLP_MSA (ver_3):
      - body hidden Linears                    -> qmodel.body hidden Linears (verbatim)
      - point output layer (rows row-major M,O)-> head MEDIAN quantile rows
      - lower/upper head rows: weights zeroed; bias set so softplus(bias)=init_half_width
    => the median equals ver_3 exactly at init; bands start at a small constant half-width.
    """
    M, O, Q = qmodel.M, qmodel.base_output_dim, qmodel.n_quantiles
    mid = Q // 2

    src_lins = [m for m in point_model.model if isinstance(m, nn.Linear)]      # [h0,h1,h2,out]
    dst_lins = [m for m in qmodel.body       if isinstance(m, nn.Linear)]      # [h0,h1,h2]
    assert len(src_lins) - 1 == len(dst_lins), 'hidden-layer count mismatch vs ver_3'
    for s, d in zip(src_lins[:-1], dst_lins):
        d.weight.data.copy_(s.weight.data)
        d.bias.data.copy_(s.bias.data)

    out_lin = src_lins[-1]                                                      # (M*O, in)
    in_f = out_lin.weight.shape[1]
    w_src = out_lin.weight.data.view(M, O, in_f)
    b_src = out_lin.bias.data.view(M, O)

    hw = qmodel.head.weight.data.view(M, O, Q, in_f)
    hb = qmodel.head.bias.data.view(M, O, Q)
    hw.zero_(); hb.zero_()
    hw[:, :, mid, :] = w_src
    hb[:, :, mid]    = b_src
    b_init = math.log(math.expm1(init_half_width))            # softplus(b_init) = init_half_width
    for q in range(Q):
        if q != mid:
            hb[:, :, q] = b_init                              # offset weights stay 0 -> constant band
    return qmodel

def evaluate_msa_quantile_on_test_set(model, X_test, y_test, label_scaler_params,
                                      M=15, base_output_dim=26, quantiles=None,
                                      device='cpu', batch_size=1000):
    """
    Per-horizon median R2/RMSE/MAE + per-horizon RAW coverage (fraction of y in [q_lo, q_hi]).
    Returns dict: predictions (N,M,O,Q), ground_truth/median/lower/upper (N,M,O), metrics_df.
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    mid = len(quantiles) // 2
    lmean = np.array(label_scaler_params['mean'], dtype=np.float64)
    lstd  = np.array(label_scaler_params['std'],  dtype=np.float64)

    model = model.to(device); model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            xb = X_test[i:i + batch_size].to(device)
            chunks.append(model(xb).cpu().numpy().astype(np.float64))
    pred_sc = np.concatenate(chunks, axis=0)                                  # (N,M,O,Q)
    pred = pred_sc * lstd[None, None, :, None] + lmean[None, None, :, None]
    y = (y_test.numpy().astype(np.float64).reshape(-1, M, base_output_dim)
         * lstd[None, None, :] + lmean[None, None, :])

    median, lower, upper = pred[..., mid], pred[..., 0], pred[..., -1]
    rows = []
    for h in range(M):
        ph, gh = median[:, h, :], y[:, h, :]
        inside = (gh >= lower[:, h, :]) & (gh <= upper[:, h, :])
        rows.append({'Horizon': h + 1,
                     'R2':   r2_score(gh, ph, multioutput='uniform_average'),
                     'RMSE': np.sqrt(mean_squared_error(gh, ph)),
                     'MAE':  mean_absolute_error(gh, ph),
                     'coverage': inside.mean()})
    metrics_df = pd.DataFrame(rows)
    print('\n--- MSA Quantile: per-horizon median metrics + raw coverage ---')
    print(metrics_df.to_string(index=False))
    return {'predictions': pred, 'ground_truth': y,
            'median': median, 'lower': lower, 'upper': upper, 'metrics_df': metrics_df}
