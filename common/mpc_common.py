import numpy as np, yaml, torch

class NARXWindowManager:
    """Maintains rolling lag window for the NARX model.
    
    Window updated after each prediction:
    - feature vars (T_reactor_meas, T_thermostat_meas): filled from 
    predictions
    - input vars  (flow_inlet, T_setpoint): filled from applied 
    controls (delay=1)"""
    def __init__(self, metadata: dict):
        self.entries = metadata['feature_groups']['narx_state_features']
        # Build ordered list of (nsame, type, n_past, delay, narx_type, n_cols)

        self._groups = []
        for e in self.entries:
            self._groups.append(e)

        # Compute total window dim and per-group slices
        self._slices = {}
        offset = 0
        for e in self._groups:
            n_cols = e['selected_dims']
            size = e['n_past'] * n_cols
            self._slices[e['name']] = (slice(offset, offset + size),
                                        e['n_past'], n_cols)
            offset += size
        self.window_dim = offset # ideally 3796
        self.window = np.zeros(self.window_dim, dtype=np.float64)

    def init_from_obs(self, obs_dict: dict, feat_scaler: dict):
        """Warm-start from a history of observations
        (n_past x n_channels for each group)"""
        for e in self._groups:
            sl, n_past, n_cols = self._slices[e['name']]
            hist = obs_dict[e['name']]
            self.window[sl] = ((hist - feat_scaler['mean'][sl].reshape(n_past, n_cols)) / 
                               feat_scaler['std'][sl].reshape(n_past, n_cols)).flatten()
    
    def get_window(self) -> np.ndarray:
        return self.window.copy()
    
    def update(self, y_pred_scaled: np.ndarray, u_scaled: np.ndarray,
               flow_scaled: float, label_metadata: list):
        """Roll window one step and insert new values.
        
        y_pred_scaled: (26,) — scaled predictions from NARX
        u_scaled:      (8,)  — scaled T_setpoint (delay=1 → goes into input window)
        flow_scaled:   float — scaled flow_inlet  (delay=1)
        label_metadata: from metadata['labels'], same order as y_pred
        """
        # Build label name -- col slice in y_pred
        lbl_col = {}
        off = 0
        for lbl in label_metadata:
            lbl_col[lbl['name']] = slice(off, off + lbl['selected_dims'])
            off += lbl['selected_dims']

        for e in self._groups:
            sl, n_past, n_cols = self._slices[e['name']]
            buf = self.window[sl].reshape(n_past, n_cols)
            buf = np.roll(buf, shift=-1, axis=0) # oldest shifted out

            if e['narx_type'] == 'feature':
                # fill latest row with predictions
                name = e['name']
                if name in lbl_col:
                    buf[-1] = y_pred_scaled[lbl_col[name]]
            else:
                if e['name'] == 'flow_inlet':
                        buf[-1] = flow_scaled
                elif e['name'] == 'T_setpoint_thermostats':
                        buf[-1] = u_scaled

            self.window[sl] = buf.flatten()

    def push_true_measurement(self, true_meas, u_apply, feat_scaler=None):
        """TRUE-measurement feedback: scale+roll. Injects ONLY windowed channels
        (T_reactor->T_reactor_meas, T_thermostat->T_thermostat_meas) + applied controls.
        heating_power/T_reactor_max are never windowed."""
        fs = feat_scaler if feat_scaler is not None else getattr(self, 'feat_scaler', None)
        if fs is None:
            raise ValueError("feat_scaler required (set window_manager.feat_scaler or pass it)")
        key_map = {'T_reactor_meas': 'T_reactor', 'T_thermostat_meas': 'T_thermostat'}
        for e in self._groups:
            name = e['name']; sl, n_past, n_cols = self._slices[name]
            buf = np.roll(self.window[sl].reshape(n_past, n_cols), -1, axis=0)
            mean = fs['mean'][sl].reshape(n_past, n_cols)[-1]
            std = fs['std'][sl].reshape(n_past, n_cols)[-1]
            if e['narx_type'] == 'feature' and name in key_map:
                buf[-1] = (np.asarray(true_meas[key_map[name]], float).reshape(-1) - mean) / std
            elif name == 'flow_inlet':
                buf[-1] = (float(u_apply['flow_inlet']) - mean) / std
            elif name == 'T_setpoint_thermostats':
                buf[-1] = (np.asarray(u_apply['T_setpoint'], float).reshape(-1) - mean) / std
            self.window[sl] = buf.flatten()

def load_scaler(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {'mean': np.array(raw['mean']), 'std': np.array(raw['std'])}

def scale(x, scaler):
    return (x - scaler['mean']) / scaler['std']

def unscale(x, scaler):
    return x * scaler['std'] + scaler['mean']

def export_narx_ae_parts(narx_ae_model, device='cpu'):
    """
    Export encoder, pred_head (and optionally decoder) separately.
    Returns l4casadi callables for each sub-network.
    
    Usage:
        enc_fn, pred_fn, dec_fn = export_narx_ae_parts(model)
        z_cas = enc_fn(x_cas)          # CasADi: 3796 → 25
        y_cas = pred_fn(z_cas)         # CasADi: 25 → 26
        x_rec = dec_fn(z_cas)          # CasADi: 25 → 3796  (for window reconstruction)
    """
    import l4casadi as l4c

    class _Enc(torch.nn.Module):
        def __init__(self, ae):
            super().__init__()
            self.enc = ae.encoder

        def forward(self, x):
            return self.enc(x)

    class _Pred(torch.nn.Module):
        def __init__(self, ae):
            super().__init__()
            self.ph = ae.pred_head

        def forward(self, z):
            return self.ph(z)
        
    class _Dec(torch.nn.Module):
        def __init__(self, ae):
            super().__init__()
            self.dec = ae.decoder
        
        def forward(self, z):
            return self.dec(z)
    
    m = narx_ae_model.to(device).eval()
    enc_fn = l4c.L4CasADi(_Enc(m), batched=True, device=device, name='narx_ae_enc')
    pred_fn = l4c.L4CasADi(_Pred(m), batched=True, device=device, name='narx_ae_pred')
    dec_fn = l4c.L4CasADi(_Dec(m), batched=True, device=device, name='narx_ae_dec')

    return enc_fn, pred_fn, dec_fn

def export_narx_ae_compact(narx_ae_model, device='cpu'):
    """
    Export AE-NARX as two compact l4casadi functions for do-mpc embedding.

    The decode → roll → encode cycle lives entirely inside PyTorch, so CasADi
    only sees two small black-box functions instead of the full 3796-dim
    intermediate state (which causes OOM during symbolic differentiation).

    Returns:
        transition_fn  (batch, 34) [z:25 | u:8 | tvp:1] → (batch, 25)  z_next
        pred_fn        (batch, 25) [z]                   → (batch, 26)  y_pred

    Usage in do-mpc Cell 2:
        z_u_tvp = ca.vertcat(z, u, tvp).T         # (1, 34)
        z_next  = transition_fn(z_u_tvp).T        # (25, 1)
        y_pred  = pred_fn(z.T).T                  # (26, 1)
    """
    import l4casadi as l4c
    BUILD_DIR = '/tmp/l4c_narx_ae'

    class _Transition(torch.nn.Module):
        def __init__(self, ae: torch.nn.Module) -> None:
            super().__init__()
            self.encoder  = ae.encoder
            self.decoder  = ae.decoder
            self.pred_head = ae.pred_head

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            z   = x_in[:, :25]
            u   = x_in[:, 25:33]
            tvp = x_in[:, 33:34]

            x_rec  = self.decoder(z)    # (batch, 3796)
            y_pred = self.pred_head(z)  # (batch, 26)

            # Roll each feature group in the flat buffer:
            # drop the oldest n_cols values (first n_cols elements of that group),
            # keep the rest, append the new values — no view/reshape needed.
            #   T_reactor  [0:1314]    146×9:  drop 9  → cat y_pred[0:9]
            #   T_therm    [1314:2482] 146×8:  drop 8  → cat y_pred[9:17]
            #   flow       [2482:2628] 146×1:  drop 1  → cat tvp
            #   setpt      [2628:3796] 146×8:  drop 8  → cat u
            t_react_new = torch.cat([x_rec[:, 9:1314],    y_pred[:, 0:9]],  dim=1)  # (batch,1314)
            t_therm_new = torch.cat([x_rec[:, 1322:2482], y_pred[:, 9:17]], dim=1)  # (batch,1168)
            flow_new    = torch.cat([x_rec[:, 2483:2628], tvp],             dim=1)  # (batch,146)
            setpt_new   = torch.cat([x_rec[:, 2636:3796], u],               dim=1)  # (batch,1168)

            x_next = torch.cat([t_react_new, t_therm_new, flow_new, setpt_new], dim=1)  # (batch,3796)
            return self.encoder(x_next)  # (batch, 25)

    class _Pred(torch.nn.Module):
        def __init__(self, ae: torch.nn.Module) -> None:
            super().__init__()
            self.ph = ae.pred_head

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.ph(z)

    # NOTE: generate_adj1=False / generate_jac_adj1=False.
    # l4casadi's auto-generated reverse-mode adjoint (adj1) mis-wires the
    # GELU backward inside the decoder->roll->encoder composite (a 25-vs-3796
    # tensor-shape crash at solve time). The forward jacobian codegen is
    # correct, so we disable adj1 and let CasADi derive gradients from the
    # forward jacobian instead. This requires the MPC to use IPOPT with
    # hessian_approximation='limited-memory' (first-order info only).
    m = narx_ae_model.to(device).eval()
    transition_fn = l4c.L4CasADi(_Transition(m), batched=True, device=device,
                                   name='narx_ae_transition', build_dir=BUILD_DIR,
                                   generate_jac=True, generate_adj1=False,
                                   generate_jac_adj1=False)
    pred_fn       = l4c.L4CasADi(_Pred(m),        batched=True, device=device,
                                   name='narx_ae_pred',       build_dir=BUILD_DIR,
                                   generate_jac=True, generate_adj1=False,
                                   generate_jac_adj1=False)
    return transition_fn, pred_fn


def export_msa_model(msa_model, device='cpu'):
    """Export Full MSA model as l4casadi"""
    import l4casadi as l4c
    m = msa_model.to(device).eval()
    return l4c.L4CasADi(m, batched=True, device=device)

