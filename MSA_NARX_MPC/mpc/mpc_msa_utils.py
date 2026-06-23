import numpy as np
import yaml
import torch
from scipy.optimize import minimize


def load_scaler(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {'mean': np.array(raw['mean']), 'std': np.array(raw['std'])}


def warmstart_shift(U_prev: np.ndarray, n_steps: int = 14, n_setpt: int = 8) -> np.ndarray:
    """
    Shift previous optimal control sequence forward by one step.
    Drop the first step (just applied), keep steps 1..n_steps-1, repeat the last.

    U_prev: (n_steps * n_setpt,) flattened
    Returns: (n_steps * n_setpt,) warm-start for the next MPC solve
    """
    U = U_prev.reshape(n_steps, n_setpt)
    U_shift = np.concatenate([U[1:], U[-1:]], axis=0)
    return U_shift.flatten()


def solve_msa_step(
    msa_model: torch.nn.Module,
    x_current_sc: np.ndarray,       # (3796,) scaled NARX window
    flow_future_sc: np.ndarray,      # (M-1,) = (14,) scaled flow for steps k+1..k+M-1
    feat_scaler: dict,               # 3922-D msa_feature_scaler
    label_scaler: dict,              # 26-D
    inp_feat_scaler: dict,           # 9-D: [flow_inlet, T_setpt×8]
    T_sp: float = 330.0,            # reactor temperature setpoint [K]
    Q_track: float = 10.0,
    R_energy: float = 1.0,
    R_du: float = 1.0,
    U_warm: np.ndarray = None,       # (112,) warm-start; None → midpoint of bounds
    M: int = 15,
    max_iter: int = 100,
) -> tuple:
    """
    Solve one MSA-NARX MPC step using scipy L-BFGS-B + PyTorch autograd.

    The MSA model predicts all M horizon steps in a single forward pass
    (single-shooting MPC). Decision variable is only the future T_setpoint
    sequence — flow is fixed as a TVP provided in flow_future_sc.

    Objective (all in scaled space):
        Σ_{i=0}^{M-1}  Q/n_react * ||T_react_i - T_sp_sc||²
                      + R/n_heat  * ||heat_i||²
        + R_du / (n_steps*n_setpt) * Σ ||u_{k+1} - u_k||²   (rate penalty)

    Returns:
        u_setpt_sc_next: (8,)  scaled T_setpoints to apply at step k+1
        U_opt_sc:        (112,) full optimal sequence (pass as U_warm next step)
        obj_val:         scalar objective at optimum
    """
    n_steps = M - 1    # 14 future steps
    n_setpt = 8        # T_setpoint zones
    n_ctrl  = 9        # 1 flow + 8 setpoints per step

    # ── Bounds on scaled T_setpoints ─────────────────────────────────────────
    T_min_sc = (292.0 - inp_feat_scaler['mean'][1:9]) / inp_feat_scaler['std'][1:9]
    T_max_sc = (365.0 - inp_feat_scaler['mean'][1:9]) / inp_feat_scaler['std'][1:9]
    bounds = [(float(T_min_sc[j % n_setpt]), float(T_max_sc[j % n_setpt]))
              for j in range(n_steps * n_setpt)]

    # ── Scaled setpoint vector ────────────────────────────────────────────────
    T_sp_sc = torch.tensor(
        (T_sp - label_scaler['mean'][:9]) / label_scaler['std'][:9],
        dtype=torch.float32)   # (9,)

    # ── Fixed tensors (not part of the decision variable) ────────────────────
    x_t    = torch.tensor(x_current_sc.reshape(1, 3796), dtype=torch.float32)
    flow_t = torch.tensor(flow_future_sc.reshape(1, n_steps, 1), dtype=torch.float32)

    # ── Warm-start ────────────────────────────────────────────────────────────
    if U_warm is None:
        mid = (T_min_sc + T_max_sc) / 2.0
        U_warm = np.tile(mid, n_steps)   # (112,)

    n_react, n_heat = 9, 8

    def obj_and_grad(U_np: np.ndarray):
        # U_np: (n_steps * n_setpt,) = (112,) numpy — scipy passes this
        U = torch.tensor(U_np.reshape(1, n_steps, n_setpt),
                         dtype=torch.float32, requires_grad=True)

        # Assemble full covariate block: interleave flow (fixed TVP) + T_setpoints
        U_full = torch.cat([flow_t, U], dim=2)          # (1, 14, 9)
        U_flat = U_full.reshape(1, n_steps * n_ctrl)    # (1, 126)
        msa_in = torch.cat([x_t, U_flat], dim=1)        # (1, 3922)

        with torch.enable_grad():
            Y = msa_model(msa_in).reshape(1, M, 26)     # (1, M, 26) scaled outputs

            T_react = Y[0, :, 0:9]                      # (M, 9) T_reactor scaled
            heat    = Y[0, :, 17:25]                    # (M, 8) heating_power scaled

            track  = Q_track  * (T_react - T_sp_sc).pow(2).sum() / n_react
            energy = R_energy * heat.pow(2).sum() / n_heat

            # Control-rate penalty (penalises large changes between consecutive steps)
            U_mat = U[0]                                 # (14, 8)
            du    = U_mat[1:] - U_mat[:-1]              # (13, 8)
            rate  = R_du * du.pow(2).sum() / (n_steps * n_setpt)

            loss = track + energy + rate
            loss.backward()

        return float(loss.detach().numpy()), U.grad.numpy().flatten().astype(np.float64)

    result = minimize(
        obj_and_grad, U_warm,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-7, 'gtol': 1e-5},
    )

    U_opt_sc    = result.x            # (112,)
    u_next_sc   = U_opt_sc[:n_setpt]  # (8,) — apply to system at k+1

    return u_next_sc, U_opt_sc, float(result.fun)
