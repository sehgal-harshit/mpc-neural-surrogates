"PCA Script to analyse dimensionality of dataset"
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from helpers.helpers import load_narx_dataset_with_metadata

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH = pathlib.Path(__file__).parent / "data_sets/21_05_2026/narx/thermal_narx_dataset_2.h5"
N_SAMPLE  = 50_000   # rows to draw for PCA (sufficient for convergence)
N_COMPS   = 200      # max components for full-vector PCA
SEED      = 42
OUT_DIR   = pathlib.Path(__file__).parent / "pca_results"
OUT_DIR.mkdir(exist_ok=True)

# Feature group slices (3796-D vector)
GROUPS = {
    "T_reactor_meas":         slice(0,    1314),   # 146 lags x 9 channels
    "T_thermostat_meas":      slice(1314, 2482),   # 146 lags x 8 channels
    "flow_inlet":             slice(2482, 2628),   # 146 lags x 1 channel
    "T_setpoint_thermostats": slice(2628, 3796),   # 146 lags x 8 channels
}
GROUP_CHANNELS = {"T_reactor_meas": 9, "T_thermostat_meas": 8, "flow_inlet": 1, "T_setpoint_thermostats": 8}
N_LAGS = 146

# ── Load & subsample ──────────────────────────────────────────────────────────
data, _ = load_narx_dataset_with_metadata(HDF5_PATH)
X = data["narx_state_features"].numpy().astype(np.float32) # (1124247, 3796)

# Small sub-sample for quick results
rng   = np.random.default_rng(SEED) # rng from SEED = 42 for reproducibility 
idx   = rng.choice(len(X), size=N_SAMPLE, replace=False) # random N_SAMPLE samples from the dataset without replacement
X_sub = X[idx] # (N_SAMPLE, 3796)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_sub) # X_scaled has zero mean and unit variance

# ── Helper ────────────────────────────────────────────────────────────────────
def fit_pca(data_2d, n_components=None): #PCA function
    n_components = min(n_components or data_2d.shape[1], data_2d.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
    pca.fit(data_2d)
    return pca

def threshold_components(evr_cumulative, thresholds=(0.95, 0.99)):
    return {t: int(np.searchsorted(evr_cumulative, t) + 1) for t in thresholds}


# ── 1. Full-vector PCA ────────────────────────────────────────────────────────

print("Full-vector PCA ...")
pca_full = fit_pca(X_scaled, n_components=N_COMPS) # PCA for 200 principal components
evr_full = pca_full.explained_variance_ratio_ # Array of explained-variance for 200 components
cumevr   = np.cumsum(evr_full) # Cumalative arry of the explained-variances
thresh   = threshold_components(cumevr) # Looks thru cumevr for how many components needed for 95% and 99%

# Plotting PCA
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.bar(range(1, len(evr_full) + 1), evr_full, alpha=0.5, color="steelblue", label="Individual EVR")
ax2.plot(range(1, len(cumevr) + 1), cumevr, color="tomato", lw=2, label="Cumulative EVR")
for t, n in thresh.items():
    ax2.axhline(t, ls="--", color="gray", lw=0.8)
    ax2.axvline(n, ls="--", color="gray", lw=0.8)
    ax2.text(n + 1, t + 0.003, f"{int(t*100)}% @ {n}", fontsize=8, color="gray")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Individual Explained Variance Ratio", color="steelblue")
ax2.set_ylabel("Cumulative EVR", color="tomato")
ax2.set_ylim(0, 1.05)
ax1.set_title("Full feature vector (3796-D) -- PCA scree")
fig.tight_layout()
fig.savefig(OUT_DIR / "pca_full_scree.png", dpi=150)
plt.close(fig)


# ── 2. Per-variable-group PCA ─────────────────────────────────────────────────

group_results = {}
for name, sl in tqdm(GROUPS.items(), desc="Per-variable-group PCA", unit="group"):
    Xg    = StandardScaler().fit_transform(X_sub[:, sl]) # Take slice, & scale
    pca_g = fit_pca(Xg, n_components=min(Xg.shape[1], N_COMPS)) # PCA -- group-wise
    evr_g = np.cumsum(pca_g.explained_variance_ratio_) # Ordered cumalative explained variance
    group_results[name] = {"pca": pca_g, "cumevr": evr_g, "thresh": threshold_components(evr_g)} # Arrange into new dict

# Plotting Group-wise PCA
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, res) in zip(axes.flat, group_results.items()):
    ce = res["cumevr"]
    ax.plot(range(1, len(ce) + 1), ce, lw=2)
    for t, n in res["thresh"].items():
        ax.axhline(t, ls="--", color="gray", lw=0.8)
        ax.axvline(n, ls="--", color="gray", lw=0.8)
        ax.text(n + 0.5, t + 0.01, f"{int(t*100)}%@{n}", fontsize=8, color="gray")
    ax.set_title(name)
    ax.set_xlabel("# Components")
    ax.set_ylabel("Cumulative EVR")
    ax.set_ylim(0, 1.05)
fig.suptitle("Per-variable-group PCA", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "pca_groups_cumulative.png", dpi=150)
plt.close(fig)


# ── 3. Lag-importance heuristic ───────────────────────────────────────────────
# Sum |loadings| across top n_95 components, reshape to (n_lags, n_channels),
# average over channels per group, then average over groups -> (146,) curve.
print("Lag importance ...")

n_95       = thresh[0.95]
comps      = pca_full.components_[:n_95]          # (n_95, 3796)
importance = np.abs(comps).sum(axis=0)            # (3796,) 1D array of importance of each feature

lag_imp = np.zeros(N_LAGS)
for name, sl in GROUPS.items():
    n_ch = GROUP_CHANNELS[name]
    block = importance[sl].reshape(N_LAGS, n_ch)  # lags are outer dim
    lag_imp += block.mean(axis=1)
    print(f"Group: {name}, shape: {block.shape}")
    print(block)
lag_imp /= len(GROUPS)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(N_LAGS), lag_imp, color="steelblue", alpha=0.8)
ax.set_xlabel("Lag index (0 = most recent)")
ax.set_ylabel("Mean |loading| (top-n_95 PCs)")
ax.set_title(f"Lag importance across full feature vector (top {n_95} PCs, 95% variance)")
fig.tight_layout()
fig.savefig(OUT_DIR / "pca_lag_importance.png", dpi=150)
plt.close(fig)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n-- PCA Summary ---------------------------------------------------")
print(f"{'Variable group':<28} {'n_95':>5}  {'n_99':>5}  {'dim':>6}")
print(f"{'Full vector (3796-D)':<28} {thresh[0.95]:>5}  {thresh[0.99]:>5}  {3796:>6}")
for name, res in group_results.items():
    dim = GROUPS[name].stop - GROUPS[name].start
    print(f"{name:<28} {res['thresh'][0.95]:>5}  {res['thresh'][0.99]:>5}  {dim:>6}")
print(f"\nPlots saved to: {OUT_DIR}")
