"""EKF-style covariance propagation for SS-AE-NARX (ver_12) over the MPC horizon.

Latent state z (25) evolves via the single-step transition f(z,u,tvp) -> z_next
(decode -> roll the flat NARX window -> re-encode), identical to
common.mpc_common.export_narx_ae_compact._Transition. We propagate latent
covariance Sigma with an EKF:

    Sigma_{h+1} = Jf Sigma_h Jf^T + Q,     Q = Jg^+ diag(sigma^2) Jg^+^T
    Sigma_y(h)  = Jg Sigma_h Jg^T          (output covariance)

where Jf = d z_next / d z, Jg = d pred_head / d z, sigma^2 = exp(variance head),
and Q maps the per-output predictive variance into latent space via the pred-head
pseudo-inverse. All Jacobians via torch.func.jacrev (CPU-cheap on 25-D).

Everything is in SCALED label space until the final multiply by label_std.
"""

import numpy as np
import torch
from torch.func import jacrev


def latent_transition(base, z, u, tvp):
    """z(25), u(8), tvp(1) -> z_next(25). Pure torch (decode -> roll -> encode).

    Roll offsets MUST match export_narx_ae_compact._Transition exactly:
        T_reactor [0:1314]    146x9 : drop oldest 9  -> append y_pred[0:9]
        T_therm   [1314:2482] 146x8 : drop oldest 8  -> append y_pred[9:17]
        flow      [2482:2628] 146x1 : drop oldest 1  -> append tvp
        setpt     [2628:3796] 146x8 : drop oldest 8  -> append u
    Only y_pred[0:9] (T_reactor) and [9:17] (T_thermostat) feed back; heating /
    T_reactor_max (y_pred[17:26]) are outputs only and never lagged.
    """
    x_rec = base.decoder(z.unsqueeze(0)).squeeze(0)       # (3796,)
    y = base.pred_head(z.unsqueeze(0)).squeeze(0)         # (26,)
    t_react = torch.cat([x_rec[9:1314],    y[0:9]])       # (1314,)
    t_therm = torch.cat([x_rec[1322:2482], y[9:17]])      # (1168,)
    flow    = torch.cat([x_rec[2483:2628], tvp])          # (146,)
    setpt   = torch.cat([x_rec[2636:3796], u])            # (1168,)
    x_next = torch.cat([t_react, t_therm, flow, setpt])   # (3796,)
    return base.encoder(x_next.unsqueeze(0)).squeeze(0)   # (25,)


def jac_f(base, z, u, tvp):
    """(25,25) = d z_next / d z at fixed (u, tvp)."""
    return jacrev(lambda zz: latent_transition(base, zz, u, tvp))(z)


def jac_g(base, z):
    """(26,25) = d pred_head / d z."""
    return jacrev(lambda zz: base.pred_head(zz.unsqueeze(0)).squeeze(0))(z)


@torch.no_grad()
def ar_rollout_latent(base, window0_scaled, u_traj, tvp_traj):
    """Autoregressive latent rollout from a scaled feature window under given controls.

    window0_scaled: (3796,) feature-scaled. u_traj: (M,8), tvp_traj: (M,1) feature-scaled.
    Returns z_traj: (M+1, 25)  = [z_0, ..., z_M].
    """
    z = base.encoder(window0_scaled.unsqueeze(0)).squeeze(0)
    zs = [z]
    for h in range(len(u_traj)):
        z = latent_transition(base, z, u_traj[h], tvp_traj[h])
        zs.append(z)
    return torch.stack(zs)


@torch.no_grad()
def propagate_cov(base, var_head, z_traj, u_traj, tvp_traj, label_std, jitter=1e-8):
    """EKF covariance propagation -> per-horizon output std in PHYSICAL units.

    z_traj: (M+1,25) latent rollout. u_traj: (M,8), tvp_traj: (M,1).
    label_std: (26,) label scaler std (scaled -> physical).
    Returns sigma_phys: (M,26) std of the output at horizon steps 1..M.
    The band the harness consumes is z_score * gamma * sigma_phys.
    """
    M = len(u_traj)
    label_std = torch.as_tensor(np.asarray(label_std), dtype=torch.float32)
    Sigma = torch.zeros(25, 25)                       # cov of z_0 (known window) = 0
    out = []
    for h in range(M):
        z_dep = z_traj[h]                             # departure state for transition h -> h+1
        Jg_dep = jac_g(base, z_dep)                   # (26,25)
        s2 = torch.exp(var_head(z_dep.unsqueeze(0)).squeeze(0))   # (26,) scaled variance
        Jg_pinv = torch.linalg.pinv(Jg_dep)           # (25,26)
        Q = Jg_pinv @ torch.diag(s2) @ Jg_pinv.T      # (25,25) latent process noise
        Jf = jac_f(base, z_dep, u_traj[h], tvp_traj[h])
        Sigma = Jf @ Sigma @ Jf.T + Q                 # cov of z_{h+1}
        Jg_arr = jac_g(base, z_traj[h + 1])           # output map at the predicted state
        Sy = Jg_arr @ Sigma @ Jg_arr.T + jitter * torch.eye(26)
        sigma_scaled = torch.sqrt(torch.clamp(torch.diag(Sy), min=0.0))
        out.append((sigma_scaled * label_std).numpy())
    return np.stack(out)                              # (M,26) physical
