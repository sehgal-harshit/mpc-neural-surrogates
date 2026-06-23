"""Part 1 — train the SS-AE-NARX (ver_12) variance head with Gaussian NLL.

Freezes encoder/decoder/pred_head (mean stays bit-identical to ver_12); trains
ONLY the new variance head. Uses ver_12's saved scalers and the same seed=42
simulation split, subsampling train-sim rows from HDF5 to stay memory-light
(the head is tiny). Saves to Models/version_12_uq without touching version_12.

Run:  uv run SS_NARX_MPC/uq/train_variance_head.py
"""

import os
import sys
import time
import yaml
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "SS_NARX_MPC", "training"))  # for `helpers.helper_classes` pickle

from SS_NARX_MPC.uq.variance_head import VarianceHead, AE_NARX_UQ, gaussian_nll  # noqa: E402

import h5py  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(REPO, "SS_NARX_MPC/training/Models/version_12")
OUT_DIR = os.path.join(REPO, "SS_NARX_MPC/training/Models/version_12_uq")
DATASET = os.path.join(REPO, "Data_Sampling/datasets/data_sets/21_05_2026/narx/thermal_narx_dataset_3.h5")

TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.7, 0.1, 42
N_TRAIN, N_VAL = 200_000, 40_000          # subsample (variance head is tiny)
BATCH, LR, MAX_EPOCHS, PATIENCE = 4096, 1e-3, 40, 6
SAMPLE_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_scaler(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return np.asarray(raw["mean"], np.float64), np.asarray(raw["std"], np.float64)


def split_sim_ids(sim_counts):
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(sim_counts))
    ntr, nv = int(TRAIN_FRAC * len(sim_counts)), int(VAL_FRAC * len(sim_counts))
    return order[:ntr], order[ntr:ntr + nv], order[ntr + nv:]


def sample_rows(offsets, sim_ids, n, seed):
    """Random global row indices within the given sims (sorted unique for h5py)."""
    ranges = [np.arange(offsets[s], offsets[s + 1]) for s in sorted(sim_ids)]
    pool = np.concatenate(ranges)
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
    return np.sort(idx)


def read_scaled(h5, idx, fmean, fstd, lmean, lstd):
    X = h5["narx_state_features"][idx, :].astype(np.float64)
    y = h5["labels"][idx, :].astype(np.float64)
    X = (X - fmean) / fstd
    y = (y - lmean) / lstd
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fmean, fstd = load_scaler(os.path.join(MODEL_DIR, "feature_scaler.yml"))
    lmean, lstd = load_scaler(os.path.join(MODEL_DIR, "label_scaler.yml"))

    base = torch.load(os.path.join(MODEL_DIR, "narx_model_full.pt"),
                      map_location=DEVICE, weights_only=False).eval()

    with h5py.File(DATASET, "r") as f:
        sim_counts = np.asarray(f["sim_sample_counts"], np.int64)
        offsets = np.concatenate([[0], np.cumsum(sim_counts)])
        tr_ids, va_ids, te_ids = split_sim_ids(sim_counts)
        print(f"split: {len(tr_ids)} train / {len(va_ids)} val / {len(te_ids)} test sims")
        tr_idx = sample_rows(offsets, tr_ids, N_TRAIN, SAMPLE_SEED)
        va_idx = sample_rows(offsets, va_ids, N_VAL, SAMPLE_SEED + 1)
        Xtr, ytr = read_scaled(f, tr_idx, fmean, fstd, lmean, lstd)
        Xva, yva = read_scaled(f, va_idx, fmean, fstd, lmean, lstd)
    print(f"train rows {len(Xtr):,} | val rows {len(Xva):,} | device {DEVICE}")

    var_head = VarianceHead(latent_dim=25, hidden=(64,), output_dim=26).to(DEVICE)
    uq = AE_NARX_UQ(base, var_head).to(DEVICE)
    opt = torch.optim.Adam(var_head.parameters(), lr=LR)

    @torch.no_grad()
    def latents(X):                              # precompute frozen z + mu in chunks
        zs, mus = [], []
        for i in range(0, len(X), BATCH):
            xb = X[i:i + BATCH].to(DEVICE)
            z, _, mu = base(xb)
            zs.append(z.cpu()); mus.append(mu.cpu())
        return torch.cat(zs), torch.cat(mus)

    Ztr, Mtr = latents(Xtr)
    Zva, Mva = latents(Xva)

    def epoch_nll(Z, M, Y, train):
        order = torch.randperm(len(Z)) if train else torch.arange(len(Z))
        tot, nb = 0.0, 0
        for i in range(0, len(Z), BATCH):
            sl = order[i:i + BATCH]
            zb, mb, yb = Z[sl].to(DEVICE), M[sl].to(DEVICE), Y[sl].to(DEVICE)
            logvar = var_head(zb)
            loss = gaussian_nll(mb, logvar, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(sl); nb += len(sl)
        return tot / nb

    best, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tr = epoch_nll(Ztr, Mtr, ytr, True)
        with torch.no_grad():
            va = epoch_nll(Zva, Mva, yva, False)
        hist.append((ep, tr, va))
        print(f"[ep {ep:02d}] train_nll={tr:.4f}  val_nll={va:.4f}  ({time.time()-t0:.1f}s)")
        if va < best - 1e-4:
            best, best_state, wait = va, {k: v.cpu().clone() for k, v in var_head.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"early stop at epoch {ep} (best val_nll={best:.4f})")
                break

    var_head.load_state_dict(best_state)

    # Per-output 1-step coverage on val (sanity): |y-mu| <= 1.645*sigma
    with torch.no_grad():
        logv = var_head(Zva.to(DEVICE)).cpu()
        sigma = torch.exp(0.5 * logv)
        cov = ((yva - Mva).abs() <= 1.645 * sigma).float().mean(0).numpy()

    torch.save({"state_dict": var_head.state_dict(),
                "config": {"latent_dim": 25, "hidden": [64], "output_dim": 26,
                           "logvar_clamp": [-10.0, 5.0]}},
               os.path.join(OUT_DIR, "var_head.pt"))
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"best_val_nll={best:.6f}\n")
        f.write("epoch,train_nll,val_nll\n")
        for ep, tr, va in hist:
            f.write(f"{ep},{tr:.6f},{va:.6f}\n")
        f.write("\nper_output_1step_val_coverage (target ~0.90):\n")
        for o, c in enumerate(cov):
            f.write(f"  out[{o:2d}]={c:.3f}\n")
    print(f"saved var_head.pt + metrics to {OUT_DIR}")
    print(f"1-step val coverage: mean={cov.mean():.3f}  reactor[0:9]={cov[:9].mean():.3f}  Tmax[25]={cov[25]:.3f}")


if __name__ == "__main__":
    main()
