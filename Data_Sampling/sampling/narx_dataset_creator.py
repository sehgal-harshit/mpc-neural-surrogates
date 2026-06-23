"""
NARX Dataset Creator - Transforms simulation data into structured datasets for sequence models.
Refactored version with atomic functions and improved metadata management.
"""
import numpy as np
import h5py
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from tqdm import tqdm

try:
    from .dataset_utils import (
        create_sliding_windows,
        apply_variable_indices,
        validate_simulation_data,
        extract_simulation_data,
        extract_simulation_metadata,
        extract_variable_data,
        calculate_timing_parameters,
        calculate_valid_samples_narx,
        create_variable_metadata,
        save_complete_metadata,
        load_complete_metadata,
        slice_features,
        slice_labels,
    )
except ImportError:
    from dataset_utils import (
        create_sliding_windows,
        apply_variable_indices,
        validate_simulation_data,
        extract_simulation_data,
        extract_simulation_metadata,
        extract_variable_data,
        calculate_timing_parameters,
        calculate_valid_samples_narx,
        create_variable_metadata,
        save_complete_metadata,
        load_complete_metadata,
        slice_features,
        slice_labels,
    )


# ============================================================================
# VARIABLE PROCESSING FUNCTIONS
# ============================================================================

def process_windowed_variable(raw_data: np.ndarray, n_past: int, delay: int,
                             max_n_past: int, max_delay: int, n_samples: int) -> np.ndarray:
    """Process a windowed variable with delay alignment."""
    # Apply delay
    if delay > 0:
        delayed_data = raw_data[:-delay]
    else:
        delayed_data = raw_data
    
    # Create windows
    windows = create_sliding_windows(delayed_data, n_past)
    
    # Align to common time reference
    start_idx = max_n_past - n_past + max_delay - delay
    aligned_data = windows[start_idx:start_idx + n_samples]
    
    # Flatten for feature vector
    return aligned_data.reshape(n_samples, -1)


def process_non_windowed_variable(raw_data: np.ndarray, delay: int,
                                 max_n_past: int, max_delay: int, n_samples: int) -> np.ndarray:
    """Process a non-windowed variable with delay alignment."""
    if delay > 0:
        raw_data = raw_data[:-delay]
    feature_time_idx = max_n_past + max_delay - 1 - delay
    return raw_data[feature_time_idx:feature_time_idx + n_samples]


def process_variable(sim_data: Dict, var_conf: Dict, max_n_past: int, 
                    max_delay: int, n_samples: int) -> Tuple[np.ndarray, Dict]:
    """Process a single variable according to its configuration."""
    # Extract raw data
    raw_data = extract_variable_data(sim_data, var_conf['type'], var_conf['name'])
    if raw_data is None:
        return None, None
    
    # Store original shape
    original_shape = raw_data.shape[1]
    
    # Apply indices if specified
    raw_data = apply_variable_indices(raw_data, var_conf.get('indices'))
    
    # Get processing parameters
    n_past = var_conf.get('n_past', 0)
    delay = var_conf.get('delay', 0)
    
    # Process based on window type
    if n_past > 0:
        aligned_data = process_windowed_variable(
            raw_data, n_past, delay, max_n_past, max_delay, n_samples
        )
    else:
        aligned_data = process_non_windowed_variable(
            raw_data, delay, max_n_past, max_delay, n_samples
        )
    
    # Create metadata
    metadata = create_variable_metadata(
        var_conf, original_shape, raw_data.shape[1], n_past
    )
    
    return aligned_data, metadata


def process_label_variable(sim_data: Dict, var_conf: Dict, 
                          max_n_past: int, max_delay: int, n_samples: int) -> Tuple[np.ndarray, Dict]:
    """Process a label variable (one timestep ahead)."""
    # Extract raw data
    raw_data = extract_variable_data(sim_data, var_conf['type'], var_conf['name'])
    if raw_data is None:
        return None, None
    
    original_shape = raw_data.shape[1]
    raw_data = apply_variable_indices(raw_data, var_conf.get('indices'))
    
    reduce_op = var_conf.get('reduce')
    if reduce_op == 'max':
        raw_data = raw_data.max(axis=1, keepdims=True)

    # Labels are one timestep ahead
    label_start_idx = max_n_past + max_delay
    aligned_labels = raw_data[label_start_idx:label_start_idx + n_samples]
    
    # Create metadata
    metadata = {
        'type': var_conf['type'],
        'name': var_conf['name'],
        'indices': var_conf.get('indices'),
        'reduce' : var_conf.get('reduce'),
        'original_dims': original_shape,
        'selected_dims': aligned_labels.shape[1],
        'narx_type': var_conf.get('narx_type', 'unknown')
    }
    
    return aligned_labels, metadata


# ============================================================================
# WORKER FUNCTIONS
# ============================================================================

def process_feature_group(sim_data: Dict, group_vars: List[Dict], 
                         max_n_past: int, max_delay: int, n_samples: int) -> Tuple[np.ndarray, List[Dict]]:
    """Process a group of feature variables."""
    feature_arrays = []
    group_metadata = []
    
    for var_conf in group_vars:
        data, metadata = process_variable(
            sim_data, var_conf, max_n_past, max_delay, n_samples
        )
        if data is None:
            return None, None
        feature_arrays.append(data)
        group_metadata.append(metadata)
    
    if feature_arrays:
        return np.hstack(feature_arrays), group_metadata
    return None, None


def process_labels_group(sim_data: Dict, label_vars: List[Dict],
                        max_n_past: int, max_delay: int, n_samples: int) -> Tuple[np.ndarray, List[Dict]]:
    """Process all label variables."""
    label_arrays = []
    label_metadata = []
    
    for var_conf in label_vars:
        data, metadata = process_label_variable(
            sim_data, var_conf, max_n_past, max_delay, n_samples
        )
        if data is None:
            return None, None
        label_arrays.append(data)
        label_metadata.append(metadata)
    
    if label_arrays:
        return np.hstack(label_arrays), label_metadata
    return None, None


def process_simulation_worker(args: tuple) -> Optional[Dict]:
    """Process a single simulation into feature/label sets with metadata."""
    sim_id, h5_file, config = args
    
    # Extract simulation data
    sim_data = extract_simulation_data(h5_file, sim_id)
    if not validate_simulation_data(sim_data):
        return None
    
    # Calculate timing parameters
    max_n_past, max_delay = calculate_timing_parameters(config)
    
    # Validate data length
    total_time_points = len(sim_data['time'])
    required_points = max_n_past + max_delay + 1
    
    if total_time_points < required_points:
        return None
    
    n_samples = total_time_points - max_n_past - max_delay
    
    # Initialize result
    result = {'metadata': {'feature_groups': {}, 'labels': {}}}
    
    # Process feature groups
    for group_name, group_vars in config.get('feature_groups', {}).items():
        features, metadata = process_feature_group(
            sim_data, group_vars, max_n_past, max_delay, n_samples
        )
        if features is None:
            return None
        result[group_name] = features
        result['metadata']['feature_groups'][group_name] = metadata
    
    # Process labels
    labels, metadata = process_labels_group(
        sim_data, config.get('labels', []), max_n_past, max_delay, n_samples
    )
    if labels is not None:
        result['labels'] = labels
        result['metadata']['labels'] = metadata
    
    return result


# ============================================================================
# MAIN CLASS
# ============================================================================

class NARXDatasetCreator:
    def __init__(self, config_path = None):
        if isinstance(config_path, str):
            self.config_path = Path(config_path)
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            raise ValueError("Invalid config_path type. Must be str or dict.")
        if 'source_file' not in self.config or 'output_file' not in self.config:
            raise ValueError("Config must contain 'source_file' and 'output_file' keys.")
        self.h5_input_path = Path(self.config['source_file'])
        self.h5_output_path = Path(self.config['output_file'])
        self.n_workers = self.config.get('n_workers', 1)
        
        if not self.h5_input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.h5_input_path}")
        
        # Get number of simulations
        with h5py.File(self.h5_input_path, 'r') as f:
            self.n_simulations = len([k for k in f.get('simulations', {}).keys() if k.isdigit()])
        
        # Extract source metadata
        self.source_metadata = extract_simulation_metadata(str(self.h5_input_path))
    
    def create_dataset(self):
        """Create NARX dataset from simulations."""
        self.h5_output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.h5_output_path.exists():
            self.h5_output_path.unlink()

        worker_args = [(i, str(self.h5_input_path), self.config)
                      for i in range(self.n_simulations)]

        # Stream results directly into HDF5 (one sim at a time) to avoid
        # accumulating the full dataset (~32 GB float64) in RAM.
        combined_metadata = None
        sim_sample_counts_list = []

        with h5py.File(self.h5_output_path, 'w') as f:
            hdf_datasets = {}  # key -> h5py.Dataset (resizable)
            hdf_offsets  = {}  # key -> next write row

            def _write_result(result):
                nonlocal combined_metadata
                if result is None:
                    return
                metadata = result.pop('metadata')
                if combined_metadata is None:
                    combined_metadata = metadata

                first_arr = next(iter(result.values()))
                sim_sample_counts_list.append(first_arr.shape[0])

                for key, data in result.items():
                    arr = np.asarray(data, dtype=np.float32)
                    n = arr.shape[0]
                    if key not in hdf_datasets:
                        chunk = (min(1024, n),) + arr.shape[1:]
                        ds = f.create_dataset(
                            key, data=arr,
                            maxshape=(None,) + arr.shape[1:],
                            chunks=chunk,
                        )
                        hdf_datasets[key] = ds
                        hdf_offsets[key]  = n
                    else:
                        ds = hdf_datasets[key]
                        old = hdf_offsets[key]
                        ds.resize(old + n, axis=0)
                        ds[old:old + n] = arr
                        hdf_offsets[key] = old + n

            if self.n_workers <= 1:
                for args in tqdm(worker_args, desc="Processing"):
                    _write_result(process_simulation_worker(args))
            else:
                with mp.Pool(self.n_workers) as pool:
                    for result in tqdm(
                        pool.imap(process_simulation_worker, worker_args),
                        total=len(worker_args), desc="Processing"
                    ):
                        _write_result(result)

            if not hdf_datasets:
                raise ValueError("No valid data processed.")

            sim_sample_counts = np.array(sim_sample_counts_list, dtype=np.int64)
            f.create_dataset('sim_sample_counts', data=sim_sample_counts)

            for key, ds in hdf_datasets.items():
                print(f"Saved '{key}': {ds.shape}")
            print(f"Saved 'sim_sample_counts': {sim_sample_counts.shape}  "
                  f"(total={sim_sample_counts.sum():,})")

            save_complete_metadata(
                f, self.config, str(self.h5_input_path),
                self.source_metadata, combined_metadata
            )

        print(f"Dataset created with complete metadata chain: {self.h5_output_path}")
    
    def load_dataset(self, include_metadata: bool = True) -> Dict:
        """Load dataset with optional metadata."""
        if not self.h5_output_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.h5_output_path}")
        
        data = {}
        with h5py.File(self.h5_output_path, 'r') as f:
            # Load data
            for key in f.keys():
                if key != 'metadata':
                    data[key] = np.array(f[key])
            
            # Load metadata if requested
            if include_metadata:
                data['metadata'] = load_complete_metadata(str(self.h5_output_path))
        
        return data
    
    def get_simulation_data(self, sim_id: int, include_raw: bool = True, 
                       include_features: bool = True, include_labels: bool = True) -> Dict:
        """Get raw data, features, and labels for a specific simulation."""
        result = {}
        
        # Get raw simulation data
        if include_raw:
            raw_data = extract_simulation_data(str(self.h5_input_path), sim_id)
            if raw_data is None:
                raise ValueError(f"Simulation {sim_id} not found or invalid")
            result['raw_data'] = raw_data
        
        # Process features and labels if requested
        if include_features or include_labels:
            processed_data = process_simulation_worker((sim_id, str(self.h5_input_path), self.config))
            if processed_data is None:
                raise ValueError(f"Could not process simulation {sim_id}")
            
            if include_features:
                features = {}
                for group_name in self.config.get('feature_groups', {}):
                    if group_name in processed_data:
                        features[group_name] = processed_data[group_name]
                result['features'] = features
                result['feature_metadata'] = processed_data['metadata']['feature_groups']
            
            if include_labels:
                if 'labels' in processed_data:
                    result['labels'] = processed_data['labels']
                    result['label_metadata'] = processed_data['metadata']['labels']
        
        return result
    
    def slice_features(self, features: np.ndarray, group_name: str, 
                      metadata: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Slice features using stored or provided metadata."""
        if metadata is None:
            dataset = self.load_dataset(include_metadata=True)
            metadata = dataset['metadata']
        
        group_metadata = metadata['feature_groups'][group_name]
        return slice_features(features, group_metadata)
    
    def slice_labels(self, labels: np.ndarray, 
                    metadata: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Slice labels using stored or provided metadata."""
        if metadata is None:
            dataset = self.load_dataset(include_metadata=True)
            metadata = dataset['metadata']
        
        label_metadata = metadata['labels']
        return slice_labels(labels, label_metadata)
    
    def get_feature_dimension(self, group_name: Optional[str] = None) -> int:
        """Get the total feature dimension for a specific group or all groups."""
        if group_name:
            if group_name not in self.config.get('feature_groups', {}):
                raise ValueError(f"Feature group '{group_name}' not found")
            
            group_vars = self.config['feature_groups'][group_name]
            total_dim = 0
            
            for var_conf in group_vars:
                n_past = var_conf.get('n_past', 0)
                indices = var_conf.get('indices')
                
                # Determine dimension of this variable
                if indices:
                    var_dim = len(indices)
                else:
                    # Would need model info to know exact dimension
                    # For now, assume scalar if not specified
                    var_dim = 1
                
                if n_past > 0:
                    total_dim += n_past * var_dim
                else:
                    total_dim += var_dim
            
            return total_dim
        else:
            # Calculate total dimension across all groups
            total_dim = 0
            for group_name in self.config.get('feature_groups', {}):
                total_dim += self.get_feature_dimension(group_name)
            return total_dim
    
    # Keep other methods unchanged...
    def extract_features_from_simulator(self, simulator, parameters: Optional[Dict] = None) -> Optional[Dict]:
        """Extract NARX features from a live do-mpc simulator - unchanged."""
        # [Keep original implementation]
        max_n_past, max_delay = calculate_timing_parameters(self.config)
        required_history = max_n_past + max_delay + 1
        
        current_timesteps = len(simulator.data['_time'])
        if current_timesteps < required_history:
            return None
        
        # Build simulation data dict from simulator
        sim_data = {'time': simulator.data['_time'].flatten()}
        
        # Extract states
        if simulator.model.n_x > 0:
            sim_data['x'] = {}
            for state_name in simulator.model._x.keys():
                if state_name != 'default':
                    sim_data['x'][state_name] = simulator.data['_x', state_name]
        
        # Extract inputs
        if simulator.model.n_u > 0:
            sim_data['u'] = {}
            for input_name in simulator.model._u.keys():
                if input_name != 'default':
                    sim_data['u'][input_name] = simulator.data['_u', input_name]
        
        # [Rest of original implementation...]
        # Extract algebraic states
        if simulator.model.n_z > 0:
            sim_data['z'] = {}
            for z_name in simulator.model._z.keys():
                if z_name != 'default':
                    sim_data['z'][z_name] = simulator.data['_z', z_name]
        
        # Extract auxiliary expressions
        if simulator.model.n_aux > 0:
            sim_data['aux'] = {}
            for aux_name in simulator.model._aux.keys():
                if aux_name != 'default':
                    sim_data['aux'][aux_name] = simulator.data['_aux', aux_name]
        
        # Extract time-varying parameters
        if simulator.model.n_tvp > 0:
            sim_data['tvp'] = {}
            for tvp_name in simulator.model._tvp.keys():
                if tvp_name != 'default':
                    sim_data['tvp'][tvp_name] = simulator.data['_tvp', tvp_name]
        
        # Extract or use provided parameters
        if parameters is None:
            sim_data['p'] = {}
            if simulator.model.n_p > 0:
                p_num = simulator.p_fun(simulator.t0)
                for i, p_name in enumerate(simulator.model._p.keys()):
                    if p_name != 'default':
                        sim_data['p'][p_name] = p_num[p_name]
        else:
            sim_data['p'] = parameters
        
        # Process features for the latest timestep
        features = {}
        
        for group_name, group_vars in self.config.get('feature_groups', {}).items():
            feature_arrays = []
            
            for var_conf in group_vars:
                data_type = var_conf['type']
                var_name = var_conf['name']
                
                if data_type not in sim_data or var_name not in sim_data[data_type]:
                    return None
                
                raw_data = sim_data[data_type][var_name]
                if raw_data.ndim == 1:
                    raw_data = raw_data.reshape(-1, 1)
                
                if 'indices' in var_conf:
                    raw_data = raw_data[:, var_conf['indices']]
                
                n_past = var_conf.get('n_past', 0)
                delay = var_conf.get('delay', 0)
                
                if n_past > 0:
                    end_idx = current_timesteps - delay
                    start_idx = end_idx - n_past
                    
                    if start_idx < 0 or end_idx > current_timesteps:
                        return None
                    
                    window = raw_data[start_idx:end_idx]
                    feature = window.flatten()
                else:
                    feature_idx = current_timesteps - 1 - delay
                    if feature_idx < 0 or feature_idx >= current_timesteps:
                        return None
                    feature = raw_data[feature_idx].flatten()
                
                feature_arrays.append(feature)
            
            if feature_arrays:
                features[group_name] = np.hstack(feature_arrays)
        
        return features
    
    @staticmethod
    def extend_existing_dataset(existing_narx_file: str, new_raw_file: str, 
                               output_file: Optional[str] = None, n_workers: int = 1) -> str:
        """
        Convenience method to extend an existing NARX dataset with new raw data.
        
        This is a static method that can be called without creating a NARXDatasetCreator instance.
        
        Args:
            existing_narx_file: Path to existing NARX dataset
            new_raw_file: Path to new raw simulation data
            output_file: Path for extended dataset (optional)
            n_workers: Number of workers for parallel processing
            
        Returns:
            Path to the extended dataset file
            
        Example:
            extended_file = NARXDatasetCreator.extend_existing_dataset(
                "path/to/existing_dataset.h5",
                "path/to/new_raw_data.h5",
                "path/to/extended_dataset.h5"
            )
        """
        return extend_narx_dataset(existing_narx_file, new_raw_file, output_file, n_workers)


# ============================================================================
# DATASET EXTENSION FUNCTIONS
# ============================================================================

def load_existing_narx_dataset(narx_file: str) -> Tuple[Dict, Dict, Dict]:
    """Load existing NARX dataset and extract data, metadata, and config."""
    if not Path(narx_file).exists():
        raise FileNotFoundError(f"NARX dataset not found: {narx_file}")
    
    data = {}
    metadata = {}
    config = {}
    
    with h5py.File(narx_file, 'r') as f:
        # Load data arrays
        for key in f.keys():
            if key != 'metadata':
                data[key] = np.array(f[key])
        
        # Load metadata
        if 'metadata' in f:
            meta_group = f['metadata']
            
            # Load NARX config
            if 'narx_config' in meta_group.attrs:
                config = json.loads(meta_group.attrs['narx_config'])
            
            # Load NARX metadata
            if 'narx_metadata' in meta_group.attrs:
                metadata = json.loads(meta_group.attrs['narx_metadata'])
    
    return data, metadata, config


def validate_extension_compatibility(existing_config: Dict, new_raw_file: str) -> bool:
    """Validate that new raw data is compatible with existing NARX config."""
    try:
        # Extract a sample simulation to check structure
        sample_data = extract_simulation_data(new_raw_file, 0)
        if sample_data is None:
            return False
        
        # Check that all required variables exist in new data
        for group_name, group_vars in existing_config.get('feature_groups', {}).items():
            for var_conf in group_vars:
                data_type = var_conf['type']
                var_name = var_conf['name']
                
                if data_type not in sample_data or var_name not in sample_data[data_type]:
                    print(f"Missing variable: {data_type}.{var_name}")
                    return False
        
        # Check labels
        for var_conf in existing_config.get('labels', []):
            data_type = var_conf['type']
            var_name = var_conf['name']
            
            if data_type not in sample_data or var_name not in sample_data[data_type]:
                print(f"Missing label variable: {data_type}.{var_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error validating compatibility: {e}")
        return False


def extend_narx_dataset(existing_narx_file: str, new_raw_file: str, 
                       output_file: Optional[str] = None, n_workers: int = 1) -> str:
    """
    Extend an existing NARX dataset with new raw simulation data.
    
    Args:
        existing_narx_file: Path to existing NARX dataset
        new_raw_file: Path to new raw simulation data file  
        output_file: Path for extended dataset (optional, defaults to modifying existing file)
        n_workers: Number of workers for parallel processing
    
    Returns:
        Path to the extended dataset file
    """
    print(f"Extending NARX dataset: {existing_narx_file}")
    print(f"With new raw data: {new_raw_file}")
    
    # Load existing metadata and config from the file
    existing_metadata = load_complete_metadata(existing_narx_file)
    existing_config = existing_metadata.get('narx_config', {})
    
    if not existing_config:
        raise ValueError("Could not load NARX configuration from existing file")
    
    print(f"Loaded existing configuration")
    
    # Validate new raw data compatibility
    if not validate_extension_compatibility(existing_config, new_raw_file):
        raise ValueError("New raw data is not compatible with existing NARX configuration")
    
    # Create temporary config for processing new data
    temp_config = existing_config.copy()
    temp_config['source_file'] = str(new_raw_file)
    temp_config['output_file'] = str(Path(existing_narx_file).parent / 'temp_new_data.h5')
    temp_config['n_workers'] = n_workers
    
    # Process new raw data using existing configuration
    print("Processing new raw data...")
    temp_creator = NARXDatasetCreator(temp_config)
    temp_creator.create_dataset()
    
    # Load new processed data
    new_data = {}
    with h5py.File(temp_config['output_file'], 'r') as f:
        for key in f.keys():
            if key != 'metadata':
                new_data[key] = np.array(f[key])
    
    # Determine output file - use existing file if no output specified
    if output_file is None:
        output_file = existing_narx_file
    
    # Resize datasets in existing file or create new file
    print("Extending datasets...")
    
    # First, load all existing data and metadata into memory
    existing_data = {}
    existing_shapes = {}
    existing_metadata = None
    
    with h5py.File(existing_narx_file, 'r') as f_existing:
        # Get existing data info and load data
        for key in f_existing.keys():
            if key != 'metadata':
                existing_shapes[key] = f_existing[key].shape
                existing_data[key] = np.array(f_existing[key])
        
        # Load metadata
        if 'metadata' in f_existing:
            existing_metadata = f_existing['metadata']
    
    # Now create output file (existing file is now closed)
    with h5py.File(output_file, 'w') as f_out:
        # Extend each dataset
        for key in existing_shapes.keys():
            if key in new_data:
                # Calculate new shape
                existing_shape = existing_shapes[key]
                new_shape = new_data[key].shape
                combined_shape = (existing_shape[0] + new_shape[0],) + existing_shape[1:]
                
                # Create new dataset with combined size
                combined_dataset = f_out.create_dataset(key, shape=combined_shape, dtype=existing_data[key].dtype)
                
                # Copy existing data
                combined_dataset[:existing_shape[0]] = existing_data[key]
                
                # Add new data
                combined_dataset[existing_shape[0]:] = new_data[key]
                
                print(f"Extended '{key}': {existing_shape} + {new_shape} = {combined_shape}")
            else:
                # Just copy existing data if no new data for this key
                f_out.create_dataset(key, data=existing_data[key])
                print(f"Kept existing '{key}': {existing_shapes[key]}")
        
        # Add any completely new datasets
        for key in new_data.keys():
            if key not in existing_shapes:
                f_out.create_dataset(key, data=new_data[key])
                print(f"Added new '{key}': {new_data[key].shape}")
        
        # Copy metadata from existing file
        if existing_metadata is not None:
            # Re-open existing file to copy metadata properly
            with h5py.File(existing_narx_file, 'r') as f_existing:
                if 'metadata' in f_existing:
                    f_existing.copy('metadata', f_out)
    
    # Clean up temporary file
    if Path(temp_config['output_file']).exists():
        Path(temp_config['output_file']).unlink()
    
    print(f"Successfully extended dataset saved to: {output_file}")
    return output_file


class NARXDatasetExtender:
    """
    A helper class for extending existing NARX datasets with new raw simulation data.
    """
    
    def __init__(self, existing_narx_file: str):
        """
        Initialize the extender with an existing NARX dataset.
        
        Args:
            existing_narx_file: Path to the existing NARX dataset file
        """
        self.existing_narx_file = Path(existing_narx_file)
        if not self.existing_narx_file.exists():
            raise FileNotFoundError(f"NARX dataset not found: {existing_narx_file}")
        
        # Load existing dataset information
        self.existing_data, self.existing_metadata, self.existing_config = load_existing_narx_dataset(str(self.existing_narx_file))
        
        print(f"Loaded existing NARX dataset:")
        print(f"  File: {self.existing_narx_file}")
        print(f"  Data groups: {list(self.existing_data.keys())}")
        if self.existing_data:
            sample_key = list(self.existing_data.keys())[0]
            print(f"  Existing samples: {self.existing_data[sample_key].shape[0]}")
    
    def extend_with_raw_data(self, new_raw_file: str, output_file: Optional[str] = None, 
                            n_workers: int = 1) -> str:
        """
        Extend the dataset with new raw simulation data.
        
        Args:
            new_raw_file: Path to new raw simulation data file
            output_file: Path for extended dataset (optional)
            n_workers: Number of workers for parallel processing
        
        Returns:
            Path to the extended dataset file
        """
        return extend_narx_dataset(
            str(self.existing_narx_file), new_raw_file, output_file, n_workers
        )
    
    def preview_extension(self, new_raw_file: str) -> Dict[str, Any]:
        """
        Preview what the extension would look like without actually creating it.
        
        Args:
            new_raw_file: Path to new raw simulation data file
            
        Returns:
            Dictionary with preview information
        """
        if not Path(new_raw_file).exists():
            raise FileNotFoundError(f"New raw data file not found: {new_raw_file}")
        
        # Check compatibility
        is_compatible = validate_extension_compatibility(self.existing_config, new_raw_file)
        
        # Get number of new simulations
        new_simulations = 0
        try:
            with h5py.File(new_raw_file, 'r') as f:
                new_simulations = len([k for k in f.get('simulations', {}).keys() if k.isdigit()])
        except Exception as e:
            print(f"Error reading new raw file: {e}")
        
        # Estimate new samples (rough estimate)
        estimated_new_samples = 0
        if new_simulations > 0:
            try:
                sample_sim = extract_simulation_data(new_raw_file, 0)
                if sample_sim and 'time' in sample_sim:
                    time_points = len(sample_sim['time'])
                    max_n_past, max_delay = calculate_timing_parameters(self.existing_config)
                    samples_per_sim = max(0, time_points - max_n_past - max_delay)
                    estimated_new_samples = samples_per_sim * new_simulations
            except Exception as e:
                print(f"Error estimating samples: {e}")
        
        existing_samples = 0
        if self.existing_data:
            sample_key = list(self.existing_data.keys())[0]
            existing_samples = self.existing_data[sample_key].shape[0]
        
        preview = {
            'compatible': is_compatible,
            'new_raw_file': str(new_raw_file),
            'new_simulations': new_simulations,
            'existing_samples': existing_samples,
            'estimated_new_samples': estimated_new_samples,
            'estimated_total_samples': existing_samples + estimated_new_samples,
            'existing_config': self.existing_config
        }
        
        return preview
    
    def get_existing_info(self) -> Dict[str, Any]:
        """Get information about the existing dataset."""
        info = {
            'file_path': str(self.existing_narx_file),
            'data_groups': list(self.existing_data.keys()),
            'config': self.existing_config,
            'metadata': self.existing_metadata
        }
        
        if self.existing_data:
            sample_key = list(self.existing_data.keys())[0]
            info['samples'] = self.existing_data[sample_key].shape[0]
            info['data_shapes'] = {key: data.shape for key, data in self.existing_data.items()}
        
        return info


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a NARX dataset from raw simulation HDF5.')
    parser.add_argument('config', help='Path to NARX dataset config YAML')
    args = parser.parse_args()

    creator = NARXDatasetCreator(args.config)
    creator.create_dataset()
