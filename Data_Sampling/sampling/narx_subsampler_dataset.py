"""
Subsample a raw COBR simulation HDF5 (sampled at dt_source) down to dt_target,
then build the NARX windowed dataset using the standard NARXDatasetCreator pipeline.

Usage:
    python narx_subsampler_dataset.py                    # uses defaults below
    python narx_subsampler_dataset.py --dt-target 120    # override target dt (s)
"""
import sys
import argparse
import tempfile
import h5py
import numpy as np
import yaml
from pathlib import Path

# Allow imports from Run_1/ regardless of working directory
_RUN1_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _RUN1_ROOT.parent
sys.path.insert(0, str(_RUN1_ROOT))

from narx_dataset_creator import NARXDatasetCreator


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def detect_dt(source_path: str) -> float:
    """Read the time step from the first simulation in the HDF5 file."""
    with h5py.File(source_path, 'r') as f:
        sim_ids = sorted([k for k in f['simulations'].keys() if k.isdigit()], key=int)
        time = f['simulations'][sim_ids[0]]['time'][:]
    return float(time[1] - time[0])


def subsample_h5(source_path: str, dest_path: str, stride: int) -> int:
    """Write a stride-subsampled copy of source_path to dest_path.

    Copies every ``stride``-th timestep for all variables in every simulation.
    Top-level metadata is preserved unchanged.

    Returns the number of simulations copied.
    """
    with h5py.File(source_path, 'r') as src, h5py.File(dest_path, 'w') as dst:
        sim_ids = sorted([k for k in src['simulations'].keys() if k.isdigit()], key=int)

        dst_sims = dst.create_group('simulations')
        for sim_id in sim_ids:
            src_sim = src['simulations'][sim_id]
            dst_sim = dst_sims.create_group(sim_id)

            dst_sim.create_dataset('time', data=src_sim['time'][::stride])

            for group_name in ['x', 'u', 'y', 'z', 'aux', 'tvp', 'p']:
                if group_name not in src_sim:
                    continue
                dst_grp = dst_sim.create_group(group_name)
                for var_name, var_data in src_sim[group_name].items():
                    arr = var_data[()]
                    # Scalar or 0-D datasets are not time series — copy unchanged
                    if arr.ndim == 0:
                        dst_grp.create_dataset(var_name, data=arr)
                    else:
                        dst_grp.create_dataset(var_name, data=arr[::stride])

        if 'metadata' in src:
            src.copy('metadata', dst)

    return len(sim_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None,
                        help='Path to NARX dataset config YAML (default: configs/thermal_narx_dataset_config.yaml)')
    parser.add_argument('--dt-target', type=float, default=60.0,
                        help='Target sampling interval in seconds (default: 60.0)')
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else _RUN1_ROOT / 'configs' / 'thermal_narx_dataset_config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve source path relative to repo root if it is a relative path
    source_path = Path(config['source_file'])
    if not source_path.is_absolute():
        source_path = _REPO_ROOT / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    dt_source = detect_dt(str(source_path))
    dt_target = args.dt_target

    if dt_target % dt_source != 0:
        raise ValueError(
            f"dt_target ({dt_target}s) must be an integer multiple of dt_source ({dt_source}s)"
        )
    stride = int(round(dt_target / dt_source))

    print(f"Source:    {source_path}")
    print(f"dt_source: {dt_source}s  →  dt_target: {dt_target}s  (stride={stride})")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h5', prefix='cobr_subsampled_')
    import os; os.close(tmp_fd)

    try:
        n_sims = subsample_h5(str(source_path), tmp_path, stride)
        print(f"Subsampled {n_sims} simulations → temp file")

        # Patch config: point source at the subsampled temp file; keep output_file as-is.
        # Store subsampling provenance so downstream consumers (notebooks) can read it.
        config['source_file'] = tmp_path
        config['subsampling'] = {
            'dt_source': dt_source,
            'dt_target': dt_target,
            'stride': stride,
        }

        output_path = Path(config['output_file'])
        if not output_path.is_absolute():
            output_path = _REPO_ROOT / output_path
            config['output_file'] = str(output_path)

        creator = NARXDatasetCreator(config)
        creator.create_dataset()

    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
