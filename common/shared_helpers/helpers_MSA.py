"""
Utility functions for SO-NARX training in Run_1.

Contents:
    Data scaling
        scale_data               - Fit StandardScaler, return tensor + params
        unscale_data             - Inverse-transform using saved params
        scale_data_with_scaler   - Transform (no fit) using saved params
        save_scaler_params       - Persist scaler params to YAML
        load_scaler_params       - Load scaler params from YAML

    DataLoaders
        get_train_val_dataloaders - Split dataset, return train/val DataLoaders

    Trainer
        get_standard_trainer     - Build pl.Trainer with checkpointing + early stopping

    Checkpoint / log utilities
        get_latest_log_and_checkpoint_path - Find latest version dir + checkpoint file
        find_latest_checkpoint             - Find latest .ckpt in a directory
        get_latest_version_dir             - Find latest version_X directory
        visualize_training_logs            - Plot train/val loss from metrics.csv

    Dataset I/O
        load_narx_dataset_with_metadata    - Load NARX HDF5 → tensors + metadata dict

    Metadata helpers
        create_feature_structure_from_metadata       - narx_state_features → structure dict
        create_input_feature_structure_from_metadata - input_features → structure dict
        create_label_structure_from_metadata         - labels → structure dict
        filter_narx_data_by_vars                     - Drop named state vars from features & labels
        save_model_metadata                          - Write model_metadata.yml to log dir
        load_model_metadata                          - Read model_metadata.yml from log dir

    Train/val/test split
        get_train_val_test_dataloaders    - 70/20/10 sample-level split returning three DataLoaders
        get_simulation_split_dataloaders  - 80/20 simulation-wise split keeping trajectories intact

    Evaluation
        evaluate_on_test_set - Run model on test set, return metrics + inverse-scaled arrays
"""

import copy
import os
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from .helper_classes_MSA import MLP_MSA, MLP, StagnationEarlyStopping


# ── Data scaling ──────────────────────────────────────────────────────────────

def scale_data(data: np.ndarray, _torch: bool = True,
               dtype=torch.float32, save_path: Optional[str] = None,
               chunk_size: int = 10_000):
    """
    Fit standard scaling on `data`, return (scaled_data, {'mean': ..., 'std': ...}).

    Bypasses sklearn to avoid its internal float64 full-array allocation.
    Mean/std are computed with chunked float64 accumulators (tiny RAM), then
    data is scaled **in-place** — peak extra RAM is one chunk (~300 MB at default).
    The input array is modified; callers that need the originals should copy first.
    """
    n, d = data.shape

    # Pass 1: accumulate sum and sum-of-squares in float64 (one chunk at a time)
    sum_  = np.zeros(d, dtype=np.float64)
    sum2_ = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk_size):
        chunk = data[start:start + chunk_size].astype(np.float64)
        sum_  += chunk.sum(axis=0)
        sum2_ += (chunk * chunk).sum(axis=0)
        del chunk

    mean = (sum_ / n).astype(np.float32)
    std  = np.sqrt(np.maximum(sum2_ / n - (sum_ / n) ** 2, 0.0)).astype(np.float32)
    std[std < 1e-8] = 1.0

    # Pass 2: scale in-place (no second full copy of data)
    data -= mean
    data /= std

    params = {'mean': mean.astype(np.float64), 'std': std.astype(np.float64)}

    if save_path is not None:
        save_scaler_params(save_path, params)

    if _torch:
        return torch.from_numpy(data).type(dtype), params

    return data, params


def unscale_data(data, scaling_params: dict, _torch: bool = True,
                 dtype=torch.float32):
    """Inverse-transform `data` using previously saved scaler params."""
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaling_params['mean'])
    scaler.scale_ = np.array(scaling_params['std'])

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    restored = scaler.inverse_transform(data)
    if _torch:
        return torch.from_numpy(restored).type(dtype)
    return restored


def scale_data_with_scaler(data, scaling_params: dict, _torch: bool = True,
                           dtype=torch.float32):
    """Transform `data` with a pre-fitted scaler (no re-fitting)."""
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaling_params['mean'])
    scaler.scale_ = np.array(scaling_params['std'])

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    scaled = scaler.transform(data)
    if _torch:
        return torch.from_numpy(scaled).type(dtype)
    return scaled


def save_scaler_params(save_path: str, scaling_params: dict):
    """Save mean/std arrays to a YAML file."""
    payload = {
        'mean': scaling_params['mean'].tolist(),
        'std': scaling_params['std'].tolist(),
    }
    with open(save_path, 'w') as f:
        yaml.dump(payload, f)


def load_scaler_params(file_path: Union[str, Path]) -> dict:
    """Load scaler params from YAML, returning {'mean': np.array, 'std': np.array}."""
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)

    if isinstance(params, dict) and 'mean' in params and 'std' in params:
        return {'mean': np.array(params['mean']), 'std': np.array(params['std'])}
    return params


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_train_val_dataloaders(feature_data: torch.Tensor, labels_data: torch.Tensor,
                              batch_size: int = 1024,
                              train_size_fraction: float = 0.8,
                              multiprocessing: bool = True,
                              cpu_count: int = mp.cpu_count()):
    """
    Split feature/label tensors into train and val DataLoaders.

    Args:
        feature_data: (N, F) float tensor
        labels_data:  (N, L) float tensor
        batch_size:   mini-batch size (default 1024)
        train_size_fraction: fraction of data used for training (default 0.8)
        multiprocessing: enable multi-worker loading (disable on Windows if issues arise)
        cpu_count: number of CPUs available for worker threads
    """
    n = feature_data.shape[0]
    dataset = TensorDataset(feature_data, labels_data)
    train_size = int(train_size_fraction * n)
    val_size = n - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    loader_kwargs = dict(batch_size=batch_size)
    if multiprocessing:
        loader_kwargs.update(
            num_workers=max(1, int(cpu_count / 2)),
            persistent_workers=True,
            pin_memory=True,
        )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False,
                            **{**loader_kwargs,
                               'num_workers': max(1, int(cpu_count / 3))} if multiprocessing else loader_kwargs)
    return train_loader, val_loader


# ── Trainer ───────────────────────────────────────────────────────────────────

def get_standard_trainer(logger, max_epochs: int, accelerator: str = 'auto',
                         checkpoint_callback: bool = True, 
                         gradient_clip_val: Optional[float] = 1.0,
                         gradient_clip_algorithm: Optional[str] = "norm",
                        patience: int = 25,
                        min_delta: float = 0.001,
                        use_standard_early_stopping: bool = False,
                        **kwargs) -> pl.Trainer:
    """
    Build a pl.Trainer with:
      - Configurable Early Stopping
      - LearningRateMonitor
      - ModelCheckpoint (top-3 by val_loss) if checkpoint_callback=True
      - CSVLogger alongside TensorBoardLogger so metrics.csv is always written
      - Gradient clipping configuration
    """
    from pytorch_lightning.loggers import CSVLogger
    # Force version resolution on the primary logger so both loggers share the same version dir
    log_dir = logger.log_dir
    csv_logger = CSVLogger(save_dir=logger.save_dir, name=logger.name, version=logger.version)
    loggers = [logger, csv_logger]

    if use_standard_early_stopping:
        early_stop = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            mode='min',
            verbose=True,
        )
    else:
        early_stop = StagnationEarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            mode='min',
            verbose = 'True',
        )

    callbacks = [
        early_stop,
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
    ]

    if checkpoint_callback:
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=log_dir + '/checkpoints',
            filename='narx-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
        )
        callbacks.append(ckpt_cb)

    return pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator=accelerator,
        logger=loggers,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        **kwargs,
    )


# ── Checkpoint / log utilities ────────────────────────────────────────────────

def get_latest_log_and_checkpoint_path(log_path: str) -> Tuple[str, str]:
    """
    Find the highest-numbered version_X directory under `log_path` and the
    last checkpoint file inside it.

    Returns:
        (version_log_path, checkpoint_file_path)
    """
    version_nums = []
    for name in os.listdir(log_path):
        if name.startswith('version_'):
            try:
                version_nums.append(int(name.split('_')[-1]))
            except ValueError:
                pass

    if not version_nums:
        raise ValueError(f"No version_X directories found in {log_path}")

    latest = max(version_nums)
    version_log_path = os.path.join(log_path, f'version_{latest}')
    checkpoint_dir = os.path.join(version_log_path, 'checkpoints')
    files = os.listdir(checkpoint_dir)
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")

    return version_log_path, os.path.join(checkpoint_dir, files[-1])


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the most recently modified .ckpt file in `checkpoint_dir`, or None."""
    if not os.path.exists(checkpoint_dir):
        return None
    ckpts = [os.path.join(checkpoint_dir, f)
             for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def get_latest_version_dir(base_path: str) -> str:
    """Return the path of the highest-numbered version_X directory."""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")
    dirs = [d for d in os.listdir(base_path)
            if d.startswith('version_') and os.path.isdir(os.path.join(base_path, d))]
    if not dirs:
        raise FileNotFoundError(f"No version directories in: {base_path}")
    dirs.sort(key=lambda x: int(x.split('_')[1]))
    return os.path.join(base_path, dirs[-1])


def create_next_version_dir(base_path: str) -> str:
    """
    Create and return the next version_X directory under `base_path`.
    Scans for existing version_X dirs and increments by one.
    """
    os.makedirs(base_path, exist_ok=True)
    existing = [d for d in os.listdir(base_path) if d.startswith('version_')]
    nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    next_version = max(nums) + 1 if nums else 0
    version_dir = os.path.join(base_path, f'version_{next_version}')
    os.makedirs(version_dir, exist_ok=True)
    return version_dir


def visualize_training_logs(logger_directory: str):
    """Plot train_loss and val_loss from TensorBoard CSV metrics."""
    metrics = pd.read_csv(os.path.join(logger_directory, 'metrics.csv'), index_col='epoch')
    try:
        metrics['train_loss'].dropna().plot(label='Train')
        metrics['val_loss'].dropna().plot(label='Validation')
    except Exception:
        metrics = metrics.dropna()
        metrics['train_loss'].abs().plot(label='Train')
        metrics['val_loss'].abs().plot(label='Validation')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


# ── Dataset I/O ───────────────────────────────────────────────────────────────

def load_narx_dataset_with_metadata(dataset_path: Union[str, Path]):
    """
    Load a NARX HDF5 dataset produced by NARXDatasetCreator.

    Returns:
        data     : dict mapping group names → float64 torch tensors
        metadata : dict with keys 'config', 'feature_groups', 'labels',
                   'simulation_metadata' (where present)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = {}
    metadata = {}

    with h5py.File(dataset_path, 'r') as f:
        for key in f.keys():
            if key != 'metadata':
                data[key] = torch.tensor(f[key][:], dtype=torch.float32)

        if 'metadata' not in f:
            print("Warning: no metadata found in dataset file")
            return data, metadata

        meta = f['metadata']

        if 'config' in meta.attrs:
            metadata['config'] = json.loads(meta.attrs['config'])
        if 'source_file' in meta.attrs:
            metadata['source_file'] = meta.attrs['source_file']

        if 'feature_groups' in meta:
            metadata['feature_groups'] = {
                k: json.loads(meta['feature_groups'].attrs[k])
                for k in meta['feature_groups'].attrs
            }

        if 'labels' in meta:
            metadata['labels'] = json.loads(meta['labels'].attrs['labels'])

        if 'simulation_metadata' in meta:
            sim_meta = meta['simulation_metadata']
            sim_dict = {}
            for attr in sim_meta.attrs:
                try:
                    sim_dict[attr] = json.loads(sim_meta.attrs[attr])
                except (json.JSONDecodeError, TypeError):
                    sim_dict[attr] = sim_meta.attrs[attr]
            for ds_name in sim_meta.keys():
                try:
                    item = sim_meta[ds_name]
                    if isinstance(item, h5py.Dataset):
                        val = item[()]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        sim_dict[ds_name] = val
                    elif isinstance(item, h5py.Group):
                        sim_dict[ds_name] = {
                            k: json.loads(item.attrs[k]) if isinstance(item.attrs[k], str) else item.attrs[k]
                            for k in item.attrs
                        }
                except Exception as e:
                    sim_dict[ds_name] = f"Error: {e}"
            metadata['simulation_metadata'] = sim_dict

    return data, metadata


# ── Metadata helpers ──────────────────────────────────────────────────────────

def create_feature_structure_from_metadata(metadata: dict) -> dict:
    """Map narx_state_features metadata → {var_name: {type, shape, n_past, ...}} dict."""
    structure = {}
    for var_meta in metadata.get('feature_groups', {}).get('narx_state_features', []):
        name = var_meta['name']
        structure[name] = {
            'type': 'measured_state' if var_meta['narx_type'] == 'state' else var_meta['narx_type'],
            'shape': var_meta['selected_dims'],
            'n_past': var_meta['n_past'],
            'delay': var_meta['delay'],
            'indices': var_meta['indices'],
            'original_dims': var_meta['original_dims'],
            'data_type': var_meta['type'],
        }
    return structure


def create_input_feature_structure_from_metadata(metadata: dict) -> dict:
    """Map input_features metadata → {var_name: {type, shape, ...}} dict."""
    structure = {}
    for var_meta in metadata.get('feature_groups', {}).get('input_features', []):
        name = var_meta['name']
        structure[name] = {
            'type': 'input',
            'shape': var_meta['selected_dims'],
            'n_past': var_meta.get('n_past', 0),
            'delay': var_meta.get('delay', 0),
            'indices': var_meta['indices'],
            'original_dims': var_meta['original_dims'],
            'data_type': var_meta['type'],
        }
    return structure


def create_label_structure_from_metadata(metadata: dict) -> dict:
    """Map labels metadata → {var_name: {type, shape, ...}} dict."""
    structure = {}
    for var_meta in metadata.get('labels', []):
        name = var_meta['name']
        structure[name] = {
            'type': 'predicted_state',
            'shape': var_meta['selected_dims'],
            'indices': var_meta['indices'],
            'original_dims': var_meta['original_dims'],
            'data_type': var_meta['type'],
            'narx_type': var_meta.get('narx_type', 'unknown'),
        }
    return structure


def filter_narx_data_by_vars(
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
    exclude_var_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], dict]:
    """
    Drop named state variables from the feature matrix and label matrix.

    Input entries (narx_type == 'input') in narx_state_features are never excluded.
    Pass exclude_var_names=[] to return everything unchanged.

    Returns:
        features_filtered : (N, n_feat_kept)
        labels_filtered   : (N, n_label_kept)
        label_names       : list of 'var[i]' strings for kept labels
        active_metadata   : deep-copy of metadata with excluded entries removed
    """
    exclude = set(exclude_var_names)

    # Feature columns: walk narx_state_features, accumulate offset
    feat_keep: List[int] = []
    offset = 0
    active_state_features = []
    for entry in metadata.get('feature_groups', {}).get('narx_state_features', []):
        n_cols = entry['selected_dims'] * entry.get('n_past', 1)
        is_input = entry.get('narx_type') == 'input'
        if is_input or entry['name'] not in exclude:
            feat_keep.extend(range(offset, offset + n_cols))
            active_state_features.append(entry)
        offset += n_cols

    # Label columns: walk labels list, accumulate offset
    label_keep: List[int] = []
    label_names: List[str] = []
    active_labels = []
    offset = 0
    for entry in metadata.get('labels', []):
        n_cols = entry['selected_dims']
        if entry['name'] not in exclude:
            label_keep.extend(range(offset, offset + n_cols))
            label_names.extend([f'{entry["name"]}[{i}]' for i in range(n_cols)])
            active_labels.append(entry)
        offset += n_cols

    feat_idx  = torch.tensor(feat_keep,  dtype=torch.long)
    label_idx = torch.tensor(label_keep, dtype=torch.long)

    active_metadata = copy.deepcopy(metadata)
    active_metadata['feature_groups']['narx_state_features'] = active_state_features
    active_metadata['labels'] = active_labels

    return features[:, feat_idx], labels[:, label_idx], label_names, active_metadata


def save_model_metadata(log_dir: str, narx_metadata: dict,
                        model_config: dict, dataset_path: Union[str, Path]):
    """
    Save a model_metadata.yml file to `log_dir` capturing dataset + model config.
    """
    n_past = max(
        (var.get('n_past', 0)
         for group in narx_metadata.get('feature_groups', {}).values()
         for var in group),
        default=0,
    )

    model_metadata = {
        'dataset_metadata': narx_metadata,
        'feature_structure': create_feature_structure_from_metadata(narx_metadata),
        'input_feature_structure': create_input_feature_structure_from_metadata(narx_metadata),
        'label_structure': create_label_structure_from_metadata(narx_metadata),
        'n_past': n_past,
        'model_config': model_config,
        'training_date': str(datetime.now()),
        'dataset_path': str(dataset_path),
    }

    metadata_path = os.path.join(log_dir, 'model_metadata.yml')
    with open(metadata_path, 'w') as f:
        yaml.dump(model_metadata, f, default_flow_style=False, sort_keys=False)

    print(f"Saved model metadata to: {metadata_path}")
    return model_metadata


def load_model_metadata(log_dir: Union[str, Path]) -> dict:
    """Load model_metadata.yml from a training log directory."""
    path = Path(log_dir) / 'model_metadata.yml'
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_narx_model(log_dir: Union[str, Path]) -> Tuple[MLP, dict, dict]:
    """
    Load a trained SO-NARX model together with its scaler params and metadata.

    Returns:
        model         : MLP instance loaded from the best checkpoint
        scaler_params : {'narx_features': {...}, 'labels': {...}}
        metadata      : dict from model_metadata.yml
    """
    log_dir = Path(log_dir)
    metadata = load_model_metadata(log_dir)

    ckpt_dir = log_dir / 'checkpoints'
    ckpt = find_latest_checkpoint(str(ckpt_dir))
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model = MLP.load_from_checkpoint(ckpt)
    model.eval()

    scaler_params = {}
    for key, fname in [('narx_features', 'narx_feature_scaler_params.yml'),
                       ('labels', 'label_scaler_params.yml')]:
        p = log_dir / fname
        if p.exists():
            scaler_params[key] = load_scaler_params(p)

    return model, scaler_params, metadata


# ── Train / val / test split ──────────────────────────────────────────────────

def get_train_val_test_dataloaders(features: torch.Tensor, labels: torch.Tensor,
                                   train_frac: float = 0.7,
                                   val_frac: float = 0.2,
                                   test_frac: float = 0.1,
                                   batch_size: int = 1024,
                                   multiprocessing: bool = True,
                                   cpu_count: int = mp.cpu_count()):
    """
    Split feature/label tensors into train, val, and test DataLoaders.

    Args:
        features:    (N, F) float tensor
        labels:      (N, L) float tensor
        train_frac:  fraction for training (default 0.7)
        val_frac:    fraction for validation (default 0.2)
        test_frac:   fraction for test (default 0.1)
        batch_size:  mini-batch size
        multiprocessing: enable multi-worker loading
        cpu_count:   available CPUs

    Returns:
        (train_loader, val_loader, test_loader)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"

    n = features.shape[0]
    train_n = int(train_frac * n)
    val_n = int(val_frac * n)
    test_n = n - train_n - val_n

    dataset = TensorDataset(features, labels)
    train_ds, val_ds, test_ds = random_split(dataset, [train_n, val_n, test_n])

    common_kwargs = dict(batch_size=batch_size)
    if multiprocessing:
        common_kwargs.update(persistent_workers=True, pin_memory=True)

    train_loader = DataLoader(
        train_ds, shuffle=True,
        num_workers=max(1, int(cpu_count / 2)) if multiprocessing else 0,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False,
        num_workers=max(1, int(cpu_count / 3)) if multiprocessing else 0,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_ds, shuffle=False,
        num_workers=0,
        batch_size=batch_size,
    )

    print(f"Split: train={train_n}, val={val_n}, test={test_n} samples")
    return train_loader, val_loader, test_loader


def get_simulation_split_dataloaders(
        features: torch.Tensor,
        labels: torch.Tensor,
        sim_sample_counts: np.ndarray,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        batch_size: int = 512,
        seed: int = 42,
        multiprocessing: bool = True,
        cpu_count: int = mp.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split into train/val/test DataLoaders by whole simulation trajectories.

    Shuffles simulation IDs (reproducible via seed), assigns the first
    train_frac fraction to train, the next val_frac to val, and the remainder
    to test.  Every timestep of each simulation lands in exactly one split,
    so no trajectory is split across splits.

    Args:
        features:          (N, F) float tensor — all samples concatenated
        labels:            (N, L) float tensor — aligned with features
        sim_sample_counts: 1-D int array, number of samples per simulation
                           (must sum to N)
        train_frac:        fraction of simulations assigned to train (default 0.7)
        val_frac:          fraction of simulations assigned to val   (default 0.1)
        batch_size:        mini-batch size
        seed:              RNG seed for reproducible simulation shuffle
        multiprocessing:   enable multi-worker DataLoader
        cpu_count:         CPUs available for worker threads

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(f'train_frac + val_frac must be < 1.0, got {train_frac + val_frac}')

    sim_sample_counts = np.asarray(sim_sample_counts, dtype=np.int64)
    n_sims = len(sim_sample_counts)

    rng = np.random.default_rng(seed)
    sim_order = rng.permutation(n_sims)

    n_train_sims = int(train_frac * n_sims)
    n_val_sims   = int(val_frac   * n_sims)
    train_sim_ids = sim_order[:n_train_sims]
    val_sim_ids   = sim_order[n_train_sims:n_train_sims + n_val_sims]
    test_sim_ids  = sim_order[n_train_sims + n_val_sims:]

    offsets = np.concatenate([[0], np.cumsum(sim_sample_counts)])

    def _gather_indices(sim_ids: np.ndarray) -> np.ndarray:
        parts = [np.arange(offsets[s], offsets[s + 1]) for s in sorted(sim_ids)]
        return np.concatenate(parts).astype(np.int64)

    train_idx = _gather_indices(train_sim_ids)
    val_idx   = _gather_indices(val_sim_ids)
    test_idx  = _gather_indices(test_sim_ids)

    train_ds = TensorDataset(features[train_idx], labels[train_idx])
    val_ds   = TensorDataset(features[val_idx],   labels[val_idx])
    test_ds  = TensorDataset(features[test_idx],  labels[test_idx])

    common_kw = dict(batch_size=batch_size)
    if multiprocessing:
        common_kw.update(persistent_workers=True, pin_memory=True)

    train_loader = DataLoader(
        train_ds, shuffle=True,
        num_workers=max(1, cpu_count // 2) if multiprocessing else 0,
        **common_kw,
    )
    val_loader = DataLoader(
        val_ds, shuffle=False,
        num_workers=0,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        test_ds, shuffle=False,
        num_workers=0,
        batch_size=batch_size,
    )

    n_test_sims = n_sims - n_train_sims - n_val_sims
    print(
        f'Simulation split (seed={seed}): '
        f'{n_train_sims} train sims ({len(train_idx):,} samples) | '
        f'{n_val_sims} val sims ({len(val_idx):,} samples) | '
        f'{n_test_sims} test sims ({len(test_idx):,} samples)'
    )
    return train_loader, val_loader, test_loader


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_on_test_set(model, test_loader, label_scaler_params: dict,
                         device: str = 'cuda') -> dict:
    """
    Evaluate a trained model on the test DataLoader.

    Args:
        model:               trained MLP (or any nn.Module)
        test_loader:         DataLoader for the test split
        label_scaler_params: {'mean': np.array, 'std': np.array} for inverse-transform
        device:              'cuda' or 'cpu'

    Returns:
        dict with keys:
            'mse'         - per-output MSE (np.array, shape L)
            'rmse'        - per-output RMSE
            'mae'         - per-output MAE
            'r2'          - per-output R²
            'mean_rmse'   - scalar mean RMSE across outputs
            'predictions' - inverse-scaled predictions (np.array, N×L)
            'targets'     - inverse-scaled targets     (np.array, N×L)
    """
    model = model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu()
            all_preds.append(pred)
            all_targets.append(y)

    preds_scaled = torch.cat(all_preds, dim=0).numpy().astype(np.float64)
    targets_scaled = torch.cat(all_targets, dim=0).numpy().astype(np.float64)

    # Inverse-transform to physical units
    preds = unscale_data(preds_scaled, label_scaler_params, _torch=False)
    targets = unscale_data(targets_scaled, label_scaler_params, _torch=False)

    # Per-output metrics
    mse = np.mean((preds - targets) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets), axis=0)

    ss_res = np.sum((targets - preds) ** 2, axis=0)
    ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_rmse': float(rmse.mean()),
        'predictions': preds,
        'targets': targets,
    }


# ── Recursive Rollout ──────────────────────────────────────────────────────────

def recursive_narx_rollout(
    model,
    X_test_scaled: torch.Tensor,
    y_test_scaled: torch.Tensor,
    feature_scaler_params: dict,
    label_scaler_params: dict,
    N_rollout: int = 50,
    start_idx: int = 0,
    device: str = 'cpu',
    state_group_dims: list = None,
    n_ctrl_dims: int = None,
) -> dict:
    """
    Run the NARX model recursively for N_rollout steps, feeding predictions back
    as state inputs while using ground-truth control sequences.

    state_group_dims: dim sizes for each autoregressive state group in the feature window,
        e.g. [9, 8] for T_reactor_meas(9) + T_thermostat_meas(8).
        Defaults to backward-compatible inference: n_out//4 groups × 4 dims, n_ctrl_dims=5.
    n_ctrl_dims: total dims of all control inputs in the feature window.
        Must be set when state_group_dims is set.
    """
    model = model.to(device)
    model.eval()

    feat_mean = np.array(feature_scaler_params['mean'], dtype=np.float64)
    feat_std  = np.array(feature_scaler_params['std'],  dtype=np.float64)
    lab_mean  = np.array(label_scaler_params['mean'],   dtype=np.float64)
    lab_std   = np.array(label_scaler_params['std'],    dtype=np.float64)

    n_out = lab_mean.shape[0]

    if state_group_dims is None or n_ctrl_dims is None:
        # Backward-compatible: assume 4 dims per group, 5 control dims
        _n_vars = n_out // 4
        state_group_dims = [4] * _n_vars
        n_ctrl_dims = 5

    # Cumulative feature offsets per state group
    _cum = [0]
    for d in state_group_dims:
        _cum.append(_cum[-1] + d)
    total_state_dims = _cum[-1]

    N_LAGS     = feat_mean.shape[0] // (total_state_dims + n_ctrl_dims)
    ctrl_start = total_state_dims * N_LAGS
    ctrl_end   = feat_mean.shape[0]

    N_test    = X_test_scaled.shape[0]
    max_steps = min(N_rollout, N_test - start_idx)

    x = X_test_scaled[start_idx].numpy().astype(np.float64).copy()

    predictions  = np.zeros((max_steps, n_out))
    ground_truth = np.zeros((max_steps, n_out))

    with torch.no_grad():
        for k in range(max_steps):
            x_tensor    = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float64)

            pred_phys = pred_scaled * lab_std + lab_mean
            gt_phys   = y_test_scaled[start_idx + k].numpy().astype(np.float64) * lab_std + lab_mean

            predictions[k]  = pred_phys
            ground_truth[k] = gt_phys

            # Update each state variable window (decode → shift → insert → re-encode)
            for v, dims_v in enumerate(state_group_dims):
                s   = _cum[v] * N_LAGS
                e   = _cum[v + 1] * N_LAGS
                win = (x[s:e] * feat_std[s:e] + feat_mean[s:e]).reshape(N_LAGS, dims_v)
                win = np.roll(win, -1, axis=0)
                win[-1] = pred_phys[_cum[v]:_cum[v + 1]]
                x[s:e] = (win.flatten() - feat_mean[s:e]) / feat_std[s:e]

            # Control window: take from ground-truth at next step
            next_idx = start_idx + k + 1
            if next_idx < N_test:
                x[ctrl_start:ctrl_end] = X_test_scaled[next_idx].numpy()[ctrl_start:ctrl_end]

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'abs_error': np.abs(predictions - ground_truth),
    }


def recursive_narx_rollout_partial(
    model,
    X_test_scaled: torch.Tensor,
    y_test_scaled: torch.Tensor,
    feature_scaler_params: dict,
    label_scaler_params: dict,
    N_rollout: int = 50,
    start_idx: int = 0,
    n_autoregressive_vars: int = 2,
    device: str = 'cpu',
    state_group_dims: list = None,
    n_ctrl_dims: int = None,
) -> dict:
    """
    Partial-autoregressive recursive rollout.

    The first ``n_autoregressive_vars`` state variable groups (from state_group_dims)
    are fed back from model predictions; the remaining groups use ground-truth values.

    state_group_dims / n_ctrl_dims: same semantics as recursive_narx_rollout.
    For new models where heating_power_avg / integral_term are not in the feature
    window, pass state_group_dims=[9,8], n_ctrl_dims=9 and n_autoregressive_vars=2
    (both groups are autoregressive — equivalent to full AR).
    """
    model = model.to(device)
    model.eval()

    feat_mean = np.array(feature_scaler_params['mean'], dtype=np.float64)
    feat_std  = np.array(feature_scaler_params['std'],  dtype=np.float64)
    lab_mean  = np.array(label_scaler_params['mean'],   dtype=np.float64)
    lab_std   = np.array(label_scaler_params['std'],    dtype=np.float64)

    n_out = lab_mean.shape[0]

    if state_group_dims is None or n_ctrl_dims is None:
        _n_vars = n_out // 4
        state_group_dims = [4] * _n_vars
        n_ctrl_dims = 5

    if n_autoregressive_vars > len(state_group_dims):
        raise ValueError(
            f'n_autoregressive_vars ({n_autoregressive_vars}) > '
            f'len(state_group_dims) ({len(state_group_dims)})'
        )

    _cum = [0]
    for d in state_group_dims:
        _cum.append(_cum[-1] + d)
    total_state_dims = _cum[-1]

    N_LAGS     = feat_mean.shape[0] // (total_state_dims + n_ctrl_dims)
    ctrl_start = total_state_dims * N_LAGS
    ctrl_end   = feat_mean.shape[0]

    N_test    = X_test_scaled.shape[0]
    max_steps = min(N_rollout, N_test - start_idx)

    x = X_test_scaled[start_idx].numpy().astype(np.float64).copy()

    predictions  = np.zeros((max_steps, n_out))
    ground_truth = np.zeros((max_steps, n_out))

    with torch.no_grad():
        for k in range(max_steps):
            x_tensor    = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float64)

            pred_phys = pred_scaled * lab_std + lab_mean
            gt_phys   = y_test_scaled[start_idx + k].numpy().astype(np.float64) * lab_std + lab_mean

            predictions[k]  = pred_phys
            ground_truth[k] = gt_phys

            for v, dims_v in enumerate(state_group_dims):
                s   = _cum[v] * N_LAGS
                e   = _cum[v + 1] * N_LAGS
                win = (x[s:e] * feat_std[s:e] + feat_mean[s:e]).reshape(N_LAGS, dims_v)
                win = np.roll(win, -1, axis=0)
                if v < n_autoregressive_vars:
                    win[-1] = pred_phys[_cum[v]:_cum[v + 1]]
                else:
                    gt_next_idx = min(start_idx + k + 1, N_test - 1)
                    gt_scaled   = y_test_scaled[gt_next_idx].numpy()[_cum[v]:_cum[v + 1]]
                    win[-1]     = gt_scaled * lab_std[_cum[v]:_cum[v + 1]] \
                                  + lab_mean[_cum[v]:_cum[v + 1]]
                x[s:e] = (win.flatten() - feat_mean[s:e]) / feat_std[s:e]

            next_idx = start_idx + k + 1
            if next_idx < N_test:
                x[ctrl_start:ctrl_end] = X_test_scaled[next_idx].numpy()[ctrl_start:ctrl_end]

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'abs_error': np.abs(predictions - ground_truth),
    }


def build_msa_dataset(
    narx_features_scaled,     # (N, 3796) np.ndarray
    labels_scaled,            # (N, 26) np.ndarray
    input_features_scaled,    # (N, 9) np.ndarray
    sim_sample_counts,        # (n_sims,) np.ndarray
    M=15,
):
    import numpy as np
    valid_indices = []
    msa_sim_sample_counts = []

    start_idx = 0
    for s_count in sim_sample_counts:
        valid_samples = s_count - M + 1
        if valid_samples > 0:
            msa_sim_sample_counts.append(valid_samples)
            valid_indices.extend(range(start_idx, start_idx + valid_samples))
        start_idx += s_count

    valid_indices = np.array(valid_indices)
    N_prime = len(valid_indices)

    feat_narx = narx_features_scaled[valid_indices]

    if M > 1:
        offsets = np.arange(1, M)
        covariate_indices = valid_indices[:, None] + offsets[None, :]
        feat_cov = input_features_scaled[covariate_indices] # (N_prime, M-1, 9)
        feat_cov = feat_cov.reshape(N_prime, -1) # (N_prime, (M-1)*9)
        msa_features = np.concatenate([feat_narx, feat_cov], axis=1)
    else:
        msa_features = feat_narx

    offsets_labels = np.arange(0, M)
    label_indices = valid_indices[:, None] + offsets_labels[None, :]
    msa_labels = labels_scaled[label_indices] # (N_prime, M, 26)
    msa_labels = msa_labels.reshape(N_prime, -1) # (N_prime, M*26)

    msa_sim_sample_counts = np.array(msa_sim_sample_counts)

    return msa_features, msa_labels, msa_sim_sample_counts


def evaluate_msa_on_test_set(
    model,
    X_test_scaled,
    y_test_scaled,
    label_scaler_params,
    M=15,
    base_output_dim=25,
    device='cpu',
):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd
    import torch

    model.to(device)
    model.eval()

    lab_mean = np.array(label_scaler_params['mean'], dtype=np.float64)
    lab_std = np.array(label_scaler_params['std'], dtype=np.float64)

    # MSA labels unscaling: tile the 26-D scaler M times
    lab_mean_msa = np.tile(lab_mean, M)
    lab_std_msa = np.tile(lab_std, M)

    predictions = []
    targets = []

    batch_size = 1000
    with torch.no_grad():
        for i in range(0, X_test_scaled.shape[0], batch_size):
            x_batch = X_test_scaled[i:i + batch_size].to(device)
            pred_scaled = model(x_batch).cpu().numpy().astype(np.float64)
            y_batch_scaled = y_test_scaled[i:i + batch_size].numpy().astype(np.float64)

            pred_phys = pred_scaled * lab_std_msa + lab_mean_msa
            gt_phys = y_batch_scaled * lab_std_msa + lab_mean_msa

            predictions.append(pred_phys)
            targets.append(gt_phys)

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Reshape to (N, M, base_output_dim)
    predictions_msa = predictions.reshape(-1, M, base_output_dim)
    targets_msa = targets.reshape(-1, M, base_output_dim)

    # Calculate metrics for each horizon
    metrics_list = []
    for h in range(M):
        pred_h = predictions_msa[:, h, :]
        gt_h = targets_msa[:, h, :]

        r2 = r2_score(gt_h, pred_h, multioutput='uniform_average')
        rmse = np.sqrt(mean_squared_error(gt_h, pred_h))
        mae = mean_absolute_error(gt_h, pred_h)
        metrics_list.append({'Horizon': h+1, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    metrics_df = pd.DataFrame(metrics_list)
    print("\n--- MSA Evaluation Metrics ---")
    print(metrics_df.to_string(index=False))

    return {
        'metrics_df': metrics_df,
        'predictions': predictions_msa,
        'ground_truth': targets_msa,
    }


def recursive_msa_narx_rollout(
    model,
    X_test_scaled: torch.Tensor,
    y_test_scaled: torch.Tensor,
    feature_scaler_params: dict,
    label_scaler_params: dict,
    N_rollout: int = 50,
    start_idx: int = 0,
    device: str = 'cpu',
    state_group_dims: list = None,
    n_ctrl_dims: int = None,
    M: int = 15,
    base_output_dim: int = 25
) -> dict:
    """
    Autoregressive rollout where the model is called at every step.

    At each step k the model produces predictions for horizons k+1 … k+M in one
    forward pass.  Only horizon-1 (pred[0]) is used to advance the AR state
    window; the remaining M-1 predictions are discarded.  Future covariates
    u[k+2]…u[k+M] are still fed as model inputs (taken from the ground-truth
    test sample at k+1), matching MPC deployment where the optimizer supplies
    the full control trajectory.
    """
    import numpy as np
    import torch

    model = model.to(device)
    model.eval()

    feat_mean = np.array(feature_scaler_params['mean'], dtype=np.float64)
    feat_std  = np.array(feature_scaler_params['std'],  dtype=np.float64)
    lab_mean  = np.array(label_scaler_params['mean'],   dtype=np.float64)
    lab_std   = np.array(label_scaler_params['std'],    dtype=np.float64)

    if state_group_dims is None or n_ctrl_dims is None:
        _n_vars = base_output_dim // 4
        state_group_dims = [4] * _n_vars
        n_ctrl_dims = 5

    _cum = [0]
    for d in state_group_dims:
        _cum.append(_cum[-1] + d)
    total_state_dims = _cum[-1]

    # feat_mean covers: NARX features ((total_state_dims + n_ctrl_dims) * N_LAGS)
    # plus future covariates ((M-1) * n_ctrl_dims). Solve for N_LAGS from the NARX part only.
    N_LAGS = (feat_mean.shape[0] - (M - 1) * n_ctrl_dims) // (total_state_dims + n_ctrl_dims)
    ctrl_start = total_state_dims * N_LAGS  # x[ctrl_start:] covers ctrl block + future covariates

    N_test    = X_test_scaled.shape[0]
    max_steps = min(N_rollout, N_test - start_idx)

    x = X_test_scaled[start_idx].numpy().astype(np.float64).copy()

    predictions  = np.zeros((max_steps, base_output_dim))
    ground_truth = np.zeros((max_steps, base_output_dim))

    with torch.no_grad():
        for k in range(max_steps):
            x_tensor    = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(x_tensor).squeeze(0).cpu().numpy().astype(np.float64)

            # Unscale only horizon-1 (first base_output_dim elements)
            pred_h1 = pred_scaled[:base_output_dim] * lab_std + lab_mean

            # Ground truth for this step: first horizon of MSA label at k (= state at k+1)
            gt_h1 = (y_test_scaled[start_idx + k].numpy()[:base_output_dim].astype(np.float64)
                     * lab_std + lab_mean)

            predictions[k]  = pred_h1
            ground_truth[k] = gt_h1

            # Advance state window by 1 step using horizon-1 prediction
            for v, dims_v in enumerate(state_group_dims):
                s   = _cum[v] * N_LAGS
                e   = _cum[v + 1] * N_LAGS
                win = (x[s:e] * feat_std[s:e] + feat_mean[s:e]).reshape(N_LAGS, dims_v)
                win = np.roll(win, -1, axis=0)
                win[-1] = pred_h1[_cum[v]:_cum[v + 1]]
                x[s:e] = (win.flatten() - feat_mean[s:e]) / feat_std[s:e]

            # Update ctrl block + future covariates from ground-truth at k+1.
            # X_test_scaled[k+1] carries u[k+2]…u[k+M] as its future-covariate block,
            # which is exactly what the model needs for the next call.
            next_idx = start_idx + k + 1
            if next_idx < N_test:
                x[ctrl_start:] = X_test_scaled[next_idx].numpy().astype(np.float64)[ctrl_start:]

    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'abs_error': np.abs(predictions - ground_truth),
    }

