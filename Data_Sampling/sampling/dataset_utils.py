"""
Shared utilities for dataset creation (NARX and Multi-Step Ahead).
Atomic functions extracted for reuse across different dataset types.
"""
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# BASIC ARRAY OPERATIONS
# ============================================================================

def create_sliding_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    """Create sliding windows from time-series data."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    shape = (data.shape[0] - window_size + 1, window_size) + data.shape[1:]
    strides = (data.strides[0],) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def apply_variable_indices(data: np.ndarray, indices: Optional[List[int]]) -> np.ndarray:
    """Apply index selection to variable data."""
    if indices is not None:
        data = data[:, indices]
    return data


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_simulation_data(sim_data: Dict) -> bool:
    """Validate that simulation data has required structure."""
    if sim_data is None or 'time' not in sim_data:
        return False
    if len(sim_data.get('time', [])) < 2:
        return False
    return True


# ============================================================================
# DATA EXTRACTION FUNCTIONS
# ============================================================================

def extract_simulation_data(h5_file: str, sim_id: int) -> Optional[Dict]:
    """Extract all data for one simulation."""
    try:
        with h5py.File(h5_file, 'r') as f:
            sim_group = f['simulations'][str(sim_id)]
            sim_data = {'time': np.array(sim_group['time'])}
            
            for data_type in ['x', 'u', 'z', 'aux', 'tvp', 'p']:
                if data_type in sim_group:
                    sim_data[data_type] = {}
                    for var_name in sim_group[data_type].keys():
                        sim_data[data_type][var_name] = np.array(sim_group[data_type][var_name])
            
            return sim_data
    except Exception as e:
        return None


def extract_simulation_metadata(h5_file: str) -> Dict[str, Any]:
    """Extract all metadata from a simulation HDF5 file."""
    metadata = {}
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'metadata' in f:
                metadata_group = f['metadata']
                
                for key in metadata_group.keys():
                    if isinstance(metadata_group[key], h5py.Group):
                        metadata[key] = {}
                        for subkey in metadata_group[key].keys():
                            if isinstance(metadata_group[key][subkey], h5py.Dataset):
                                data = metadata_group[key][subkey][()]
                                if isinstance(data, bytes):
                                    metadata[key][subkey] = data.decode('utf-8')
                                else:
                                    metadata[key][subkey] = data
                    elif isinstance(metadata_group[key], h5py.Dataset):
                        data = metadata_group[key][()]
                        if isinstance(data, bytes):
                            metadata[key] = data.decode('utf-8')
                        else:
                            metadata[key] = data
    except Exception as e:
        pass
    
    return metadata


def extract_variable_data(sim_data: Dict, data_type: str, var_name: str) -> Optional[np.ndarray]:
    """Extract and reshape variable data from simulation data.
    
    For parameters (data_type='p'), scalar values are broadcast to match time dimension.
    """
    if data_type not in sim_data or var_name not in sim_data[data_type]:
        return None
    
    raw_data = sim_data[data_type][var_name]
    
    # Handle scalar parameters - broadcast to time dimension
    if raw_data.ndim == 0:
        n_timesteps = len(sim_data['time'])
        raw_data = np.full((n_timesteps, 1), raw_data.item())
    elif raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)
    
    return raw_data


# ============================================================================
# TIMING PARAMETER CALCULATION
# ============================================================================

def calculate_timing_parameters(config: Dict) -> Tuple[int, int]:
    """Calculate max_n_past and max_delay from configuration."""
    max_n_past = max((var.get('n_past', 0) 
                     for group in config.get('feature_groups', {}).values() 
                     for var in group), default=0)
    max_delay = max((var.get('delay', 0) 
                    for group in config.get('feature_groups', {}).values() 
                    for var in group), default=0)
    return max_n_past, max_delay


def calculate_valid_samples_narx(sim_data: Dict, max_n_past: int, max_delay: int) -> int:
    """Calculate number of valid samples for NARX dataset."""
    total_timesteps = len(sim_data['time'])
    required_history = max_n_past + max_delay
    return total_timesteps - required_history - 1


def calculate_valid_samples_msa(sim_data: Dict, max_n_past: int, max_delay: int, horizon: int) -> int:
    """Calculate number of valid samples for multi-step ahead dataset."""
    total_timesteps = len(sim_data['time'])
    required_history = max_n_past + max_delay
    return total_timesteps - required_history - horizon


# ============================================================================
# VARIABLE METADATA CREATION
# ============================================================================

def create_variable_metadata(var_conf: Dict, original_shape: int, 
                            selected_dims: int, n_past: int = 0) -> Dict:
    """Create metadata dictionary for a variable."""
    metadata = {
        'type': var_conf['type'],
        'name': var_conf['name'],
        'n_past': n_past,
        'delay': var_conf.get('delay', 0),
        'indices': var_conf.get('indices'),
        'original_dims': original_shape,
        'selected_dims': selected_dims,
        'narx_type': var_conf.get('narx_type', 'unknown')
    }
    
    if n_past > 0:
        metadata['windowed'] = True
    
    metadata['feature_dims'] = selected_dims * (n_past if n_past > 0 else 1)
    return metadata


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def save_complete_metadata(h5_file: h5py.File, config: Dict, source_file: str, 
                          source_metadata: Dict, processing_metadata: Dict) -> None:
    """Save complete metadata chain to HDF5 file."""
    import yaml
    import json
    from datetime import datetime
    
    meta_group = h5_file.create_group('metadata')
    
    # Save dataset config
    meta_group.attrs['narx_config'] = json.dumps(config)
    meta_group.attrs['source_file'] = str(source_file)
    
    # Save original simulation metadata
    if source_metadata:
        sim_meta_group = meta_group.create_group('simulation_metadata')
        
        for key, value in source_metadata.items():
            if isinstance(value, dict):
                subgroup = sim_meta_group.create_group(key)
                for sub_key, sub_value in value.items():
                    try:
                        subgroup.attrs[sub_key] = json.dumps(sub_value) if isinstance(sub_value, (dict, list)) else sub_value
                    except:
                        subgroup.attrs[sub_key] = str(sub_value)
            else:
                try:
                    sim_meta_group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
                except:
                    sim_meta_group.attrs[key] = str(value)
    
    # Save processing metadata (NARX or MSA specific)
    for group_type in ['feature_groups', 'labels', 'future_inputs', 'trajectory_labels']:
        if group_type in processing_metadata:
            meta_subgroup = meta_group.create_group(group_type)
            
            # Handle different metadata structures
            if group_type == 'feature_groups':
                # Dict of group_name -> metadata_list
                for key, metadata_list in processing_metadata[group_type].items():
                    meta_subgroup.attrs[key] = json.dumps(metadata_list)
            elif isinstance(processing_metadata[group_type], dict):
                # Dict structure (for backwards compatibility)
                for key, metadata_list in processing_metadata[group_type].items():
                    meta_subgroup.attrs[key] = json.dumps(metadata_list)
            else:
                # List structure (for labels, future_inputs, trajectory_labels)
                meta_subgroup.attrs[group_type] = json.dumps(processing_metadata[group_type])
    
    # Save creation timestamp
    meta_group.attrs['created_at'] = str(datetime.now())


def load_complete_metadata(h5_file_path: str) -> Dict[str, Any]:
    """Load complete metadata from HDF5 file."""
    import json
    
    metadata = {}
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'metadata' not in f:
                return metadata
            
            meta_group = f['metadata']
            
            # Load config
            if 'narx_config' in meta_group.attrs:
                metadata['narx_config'] = json.loads(meta_group.attrs['narx_config'])
            
            # Load source file info
            if 'source_file' in meta_group.attrs:
                metadata['source_file'] = meta_group.attrs['source_file']
                if isinstance(metadata['source_file'], bytes):
                    metadata['source_file'] = metadata['source_file'].decode('utf-8')
            
            # Load simulation metadata
            if 'simulation_metadata' in meta_group:
                metadata['simulation_metadata'] = {}
                sim_meta = meta_group['simulation_metadata']
                
                for attr_name in sim_meta.attrs:
                    try:
                        metadata['simulation_metadata'][attr_name] = json.loads(sim_meta.attrs[attr_name])
                    except (json.JSONDecodeError, TypeError):
                        metadata['simulation_metadata'][attr_name] = sim_meta.attrs[attr_name]
                
                for subgroup_name in sim_meta.keys():
                    try:
                        subgroup_item = sim_meta[subgroup_name]
                        if isinstance(subgroup_item, h5py.Group):
                            metadata['simulation_metadata'][subgroup_name] = {}
                            for attr_name in subgroup_item.attrs:
                                try:
                                    metadata['simulation_metadata'][subgroup_name][attr_name] = json.loads(subgroup_item.attrs[attr_name])
                                except (json.JSONDecodeError, TypeError):
                                    metadata['simulation_metadata'][subgroup_name][attr_name] = subgroup_item.attrs[attr_name]
                        elif isinstance(subgroup_item, h5py.Dataset):
                            dataset_value = subgroup_item[()]
                            if isinstance(dataset_value, bytes):
                                dataset_value = dataset_value.decode('utf-8')
                            try:
                                metadata['simulation_metadata'][subgroup_name] = json.loads(dataset_value)
                            except (json.JSONDecodeError, TypeError):
                                metadata['simulation_metadata'][subgroup_name] = dataset_value
                    except Exception as e:
                        metadata['simulation_metadata'][subgroup_name] = f"Error loading: {str(e)}"
            
            # Load processing-specific metadata (NARX or MSA)
            for group_type in ['feature_groups', 'labels', 'future_inputs', 'trajectory_labels']:
                if group_type in meta_group:
                    try:
                        # Check if it's stored as a dict (multiple groups) or list (single group)
                        group_attrs = meta_group[group_type].attrs
                        
                        if group_type in group_attrs:
                            # Single list stored with group_type as key
                            metadata[group_type] = json.loads(group_attrs[group_type])
                        else:
                            # Multiple groups stored as separate keys
                            metadata[group_type] = {}
                            for key in group_attrs:
                                metadata[group_type][key] = json.loads(group_attrs[key])
                    except Exception as e:
                        metadata[group_type] = {}
            
            # Load timestamp
            if 'created_at' in meta_group.attrs:
                metadata['created_at'] = meta_group.attrs['created_at']
                if isinstance(metadata['created_at'], bytes):
                    metadata['created_at'] = metadata['created_at'].decode('utf-8')
    
    except Exception as e:
        pass
    
    return metadata


# ============================================================================
# SLICING FUNCTIONS
# ============================================================================

def slice_features(features: np.ndarray, metadata: List[Dict]) -> Dict[str, np.ndarray]:
    """Slice concatenated features back into individual variables."""
    sliced = {}
    current_idx = 0
    
    for var_meta in metadata:
        var_name = var_meta['name']
        feature_dims = var_meta['feature_dims']
        
        var_data = features[:, current_idx:current_idx + feature_dims]
        
        if var_meta.get('windowed', False) and var_meta['n_past'] > 0:
            n_past = var_meta['n_past']
            selected_dims = var_meta['selected_dims']
            var_data = var_data.reshape(-1, n_past, selected_dims)
        
        sliced[var_name] = var_data
        current_idx += feature_dims
    
    return sliced


def slice_labels(labels: np.ndarray, metadata: List[Dict]) -> Dict[str, np.ndarray]:
    """Slice concatenated labels back into individual variables."""
    sliced = {}
    current_idx = 0
    
    for var_meta in metadata:
        var_name = var_meta['name']
        selected_dims = var_meta['selected_dims']
        
        sliced[var_name] = labels[:, current_idx:current_idx + selected_dims]
        current_idx += selected_dims
    
    return sliced
