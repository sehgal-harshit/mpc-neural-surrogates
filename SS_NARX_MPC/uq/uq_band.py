"""Part 2 — build the static per-horizon EKF band + validate coverage.

Runs EKF covariance propagation along autoregressive latent rollouts on held-out
test simulations (the ver_12 seed=42 test split), calibrates a single scalar
gamma so the propagated 90% band hits ~0.90 empirical coverage, builds a static
(M,26) physical band array, and exposes `make_static_uq_band` -> uq_band(h, idx)
for the closed-loop harness.

Run:  uv run SS_NARX_MPC/uq/uq_band.py
"""

import os
import sys
import yaml
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "SS_NARX_MPC", "training"))  # `helpers.helper_classes` pickle

from SS_NARX_MPC.uq.variance_head import VarianceHead  # noqa: E402
from SS_NARX_MPC.uq.covariance import ar_rollout_latent, propagate_cov  # noqa: E402

import h5py  # noqa: E402

MODEL_DIR = os.path.join(REPO, "SS_NARX_MPC/training/Models/version_12")
OUT_DIR = os.path.join(REPO, "SS_NARX_MPC/training/Models/version_12_uq")
DATASET = os.path.join(REPO, "Data_Sampling/datasets/data_sets/21_05_2026/narx/thermal_narx_dataset_3.h5")

TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.7, 0.1, 42
M = 15                      # MPC horizon
Z_SCORE = 1.645             # ~90% one-sided Gaussian
N_VAL_SIMS = 2              # held-out test sims used for coverage
STARTS_PER_SIM = 60         # start indices per sim (strided) — keeps CPU runtime modest
DEVICE = "cpu"              # band is precomputed; keep CPU-runnable

# Flat-window slice positions of the newest input lag (feature-scaled), see covariance.py
FLOW_NEWEST = 2627          # last element of flow block [2482:2628]
SETPT_NEWEST = slice(3788, 3796)   # last row of setpoint block [2628:3796]


def load_scaler(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return np.asarray(raw["mean"], np.float64), np.asarray(raw["std"], np.float64)


def test_sim_ids(sim_counts):
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(sim_counts))
    ntr, nv = int(TRAIN_FRAC * len(sim_counts)), int(VAL_FRAC * len(sim_counts))
    return np.sort(order[ntr + nv:])


def collect(base, var_head, fmean, fstd, lmean, lstd):
    """Run EKF along AR rollouts on held-out sims. Returns (errs, sigmas): (N,M,26) each
    (absolute physical error and propagated physical std, pre-gamma)."""
    fmean_t = torch.tensor(fmean, dtype=torch.float32)
    fstd_t = torch.tensor(fstd, dtype=torch.float32)
    errs, sigs = [], []
    with h5py.File(DATASET, "r") as f:
        sim_counts = np.asarray(f["sim_sample_counts"], np.int64)
        offsets = np.concatenate([[0], np.cumsum(sim_counts)])
        sims = test_sim_ids(sim_counts)[:N_VAL_SIMS]
        for s in sims:
            a, b = int(offsets[s]), int(offsets[s + 1])
            Xphys = f["narx_state_features"][a:b, :].astype(np.float64)
            Yphys = f["labels"][a:b, :].astype(np.float64)
            T = len(Xphys)
            Xsc = torch.tensor((Xphys - fmean) / fstd, dtype=torch.float32)
            starts = np.linspace(0, T - M - 2, STARTS_PER_SIM).astype(int)
            for k in range(0, len(starts)):
                k0 = int(starts[k])
                # controls advancing k0+h -> k0+h+1 = newest input row of window k0+h+1
                u_traj, tvp_traj = [], []
                for h in range(M):
                    w = Xsc[k0 + h + 1]
                    u_traj.append(w[SETPT_NEWEST])
                    tvp_traj.append(w[FLOW_NEWEST:FLOW_NEWEST + 1])
                u_traj = torch.stack(u_traj); tvp_traj = torch.stack(tvp_traj)
                z_traj = ar_rollout_latent(base, Xsc[k0], u_traj, tvp_traj)   # (M+1,25)
                sigma_phys = propagate_cov(base, var_head, z_traj, u_traj, tvp_traj, lstd)  # (M,26)
                # mean predictions at horizon steps 1..M (physical)
                with torch.no_grad():
                    mu_sc = base.pred_head(z_traj[1:M + 1])                  # (M,26) scaled
                mu_phys = mu_sc.numpy() * lstd + lmean
                y_true = Yphys[k0 + 1:k0 + 1 + M, :]                         # (M,26) physical
                errs.append(np.abs(y_true - mu_phys))
                sigs.append(sigma_phys)
    return np.stack(errs), np.stack(sigs)                                    # (N,M,26)


def _bisect_gamma(e, s, target):
    """Smallest gamma with pooled coverage(|e| <= Z*gamma*s) >= target (monotonic in gamma)."""
    s = np.maximum(s, 1e-9)
    lo, hi = 1e-3, 200.0
    for _ in range(60):
        g = 0.5 * (lo + hi)
        if (e <= Z_SCORE * g * s).mean() < target:
            lo = g
        else:
            hi = g
    return 0.5 * (lo + hi)


def calibrate_gamma(errs, sigs, target=0.90):
    """Single global scalar gamma (reference / fallback)."""
    return _bisect_gamma(errs, sigs, target)


def calibrate_gamma_per_horizon(errs, sigs, target=0.90):
    """Per-horizon gamma_h (pooled over outputs) so each horizon hits ~target.

    The EKF covariance compounds super-linearly with horizon, so a single global
    scalar cannot give uniform per-horizon coverage; gamma_h scales each horizon's
    band level while the EKF still sets the relative per-output widths within h.
    Returns gammas: (M,).
    """
    M = errs.shape[1]
    return np.array([_bisect_gamma(errs[:, h, :], sigs[:, h, :], target) for h in range(M)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fmean, fstd = load_scaler(os.path.join(MODEL_DIR, "feature_scaler.yml"))
    lmean, lstd = load_scaler(os.path.join(MODEL_DIR, "label_scaler.yml"))

    base = torch.load(os.path.join(MODEL_DIR, "narx_model_full.pt"),
                      map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    ck = torch.load(os.path.join(OUT_DIR, "var_head.pt"), map_location=DEVICE)
    var_head = VarianceHead(**ck["config"]).to(DEVICE)
    var_head.load_state_dict(ck["state_dict"]); var_head.eval()

    print(f"collecting EKF rollouts on {N_VAL_SIMS} held-out sims x {STARTS_PER_SIM} starts ...")
    errs, sigs = collect(base, var_head, fmean, fstd, lmean, lstd)
    print(f"collected {errs.shape[0]} rollouts, shape {errs.shape}")

    gamma_global = calibrate_gamma(errs, sigs, target=0.90)    # reference scalar
    gammas = calibrate_gamma_per_horizon(errs, sigs, target=0.90)  # (M,) per-horizon
    print(f"global gamma = {gamma_global:.4f}  |  per-horizon gamma range "
          f"[{gammas.min():.3f}, {gammas.max():.3f}]")

    # Static band: EKF mean sigma per (h,output) scaled by the per-horizon gamma_h
    sigma_mean = sigs.mean(axis=0)                             # (M,26)
    band_array = Z_SCORE * gammas[:, None] * sigma_mean       # (M,26) physical
    np.save(os.path.join(OUT_DIR, "uq_band_array.npy"), band_array)
    with open(os.path.join(OUT_DIR, "gamma.txt"), "w") as f:
        f.write(f"global_gamma={gamma_global:.6f}\n")
        f.write("per_horizon_gamma=" + ",".join(f"{g:.6f}" for g in gammas) + "\n")

    # Per-horizon coverage with the per-horizon gammas (pooled over all outputs)
    s = np.maximum(sigs, 1e-9)
    covered = errs <= Z_SCORE * gammas[None, :, None] * s     # (N,M,26)
    per_h = covered.reshape(errs.shape[0], M, -1).mean(axis=(0, 2))
    react = covered[:, :, 0:9].mean(axis=(0, 2))
    tmax = covered[:, :, 25].mean(axis=0)
    print("\nper-horizon coverage (gamma-calibrated, target 0.90):")
    print(" h:  all   reactor  Tmax")
    for h in range(M):
        print(f"{h+1:2d}: {per_h[h]:.3f}   {react[h]:.3f}   {tmax[h]:.3f}")
    print(f"\npooled coverage all={covered.mean():.3f} | reactor={covered[:,:,0:9].mean():.3f} "
          f"| Tmax={covered[:,:,25].mean():.3f}")
    print(f"saved uq_band_array.npy (shape {band_array.shape}) + gamma.txt to {OUT_DIR}")

    # Smoke: harness-consumable closure
    band = make_static_uq_band(band_array)
    print(f"smoke: band(0,25)={band(0,25):.3f} K  band(14,25)={band(14,25):.3f} K  "
          f"band(0,0)={band(0,0):.3f} K  band(14,0)={band(14,0):.3f} K")
    assert band(14, 0) >= band(0, 0) - 1e-6, "band should grow (or hold) over the horizon"


def make_static_uq_band(band_array):
    """band_array: (M,26) physical. Returns uq_band(h, idx) for the closed-loop harness."""
    M = len(band_array)
    return lambda h, idx: float(band_array[min(int(h), M - 1), int(idx)])


if __name__ == "__main__":
    main()
