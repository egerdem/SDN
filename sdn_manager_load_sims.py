import os
import json
# import pickle
import numpy as np
# import spatial_analysis as sa
# import matplotlib.pyplot as plt
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
from datetime import datetime
import hashlib
# import importlib
import sys
# from pathlib import Path
import webbrowser
from threading import Timer

# Import modules from the main codebase
import geometry
import plot_room as pp
import EchoDensity as ned
import analysis as an
from sdn_core import DelayNetwork


class Room:
    """Class to manage room-specific data and associated experiments."""
    
    def __init__(self, name, parameters):
        """
        Initialize a room with its parameters.
        
        Args:
            name (str): Unique identifier for the room (e.g. 'room_aes')
            parameters (dict): Room parameters including dimensions, absorption, etc.
        """
        self.name = name
        self.parameters = parameters
        # Store additional metadata for better display
        self.parameters['room_name'] = name
        self.experiments = {}  # experiment_id -> SDNExperiment
        self.experiments_by_position = {}  # (source_pos, mic_pos) -> list of experiments
        # Use the provided name as display name
        self.display_name = name
        
    def _get_position_key(self, source_pos, mic_pos):
        """Create a tuple key for source-mic position."""
        return (tuple(source_pos), tuple(mic_pos))
        
    @property
    def dimensions_str(self):
        """Get formatted string of room dimensions."""
        return f"{self.parameters['width']}x{self.parameters['depth']}x{self.parameters['height']}m"
    
    @property
    def absorption_str(self):
        """Get formatted absorption coefficient."""
        return f"{self.parameters['absorption']:.2f}"
    
    @property
    def source_mic_pairs(self):
        """Get list of unique source-mic pairs."""
        return list(self.experiments_by_position.keys())
        
    def add_experiment(self, experiment):
        """Add an experiment to this room."""
        # First check if experiment with this ID already exists - don't duplicate
        if experiment.experiment_id in self.experiments:
            # Just update the experiment if it already exists
            self.experiments[experiment.experiment_id] = experiment
            # Update in position-based dictionary if it exists there
            for pos_list in self.experiments_by_position.values():
                for i, exp in enumerate(pos_list):
                    if exp.experiment_id == experiment.experiment_id:
                        pos_list[i] = experiment
            return
            
        # Add to main experiments dictionary
        self.experiments[experiment.experiment_id] = experiment
        
        # Get source and mic positions from config
        room_params = experiment.config.get('room_parameters', {})
        
        # For batch processing, check if positions are in receiver info
        receiver_info = experiment.config.get('receiver', {})
        if receiver_info and 'position' in receiver_info:
            mic_pos = receiver_info['position']
        else:
            # Use standard room parameters
            mic_pos = [
                room_params.get('mic x', 0),
                room_params.get('mic y', 0),
                room_params.get('mic z', 0)
            ]
            
        # Similarly for source position
        source_info = experiment.config.get('source', {})
        if source_info and 'position' in source_info:
            source_pos = source_info['position']
        else:
            source_pos = [
                room_params.get('source x', 0),
                room_params.get('source y', 0),
                room_params.get('source z', 0)
            ]
        
        # Add to position-based dictionary
        pos_key = self._get_position_key(source_pos, mic_pos)
        if pos_key not in self.experiments_by_position:
            self.experiments_by_position[pos_key] = []
        
        # Check if experiment is already in the list for this position (prevent duplicates)
        if not any(exp.experiment_id == experiment.experiment_id for exp in self.experiments_by_position[pos_key]):
            self.experiments_by_position[pos_key].append(experiment)
    
    def get_experiments_for_position(self, pos_idx):
        """Get all experiments for a given source-mic position index."""
        if not self.source_mic_pairs:
            return []
        
        pos_key = self.source_mic_pairs[pos_idx % len(self.source_mic_pairs)]
        return self.experiments_by_position[pos_key]
    
    def get_position_info(self, pos_idx):
        """Get formatted string describing the source-mic position."""
        if not self.source_mic_pairs:
            return "No source-mic pairs"
        
        pos_key = self.source_mic_pairs[pos_idx % len(self.source_mic_pairs)]
        source_pos, mic_pos = pos_key
        return f"Source: ({source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f}), Mic: ({mic_pos[0]:.1f}, {mic_pos[1]:.1f}, {mic_pos[2]:.1f})"
    
    @property
    def theoretical_rt_str(self):
        """Get formatted string of theoretical RT values."""
        room_dim = np.array([self.parameters['width'], 
                           self.parameters['depth'], 
                           self.parameters['height']])
        rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, self.parameters['absorption'])
        return f"RT sabine={rt60_sabine:.1f}s eyring={rt60_eyring:.1f}s"
    
    def get_header_info(self):
        """Get room information for display header."""
        return {
            'name': self.name,
            'dimensions': self.dimensions_str,
            'absorption': self.absorption_str,
            'rt_values': self.theoretical_rt_str,
            'n_experiments': len(self.experiments),
            'n_source_mic_pairs': len(self.source_mic_pairs)
        }
    
    def matches_parameters(self, parameters):
        """Check if given parameters match this room's parameters."""
        for key in ['width', 'depth', 'height', 'absorption']:
            if abs(self.parameters[key] - parameters[key]) > 1e-6:
                return False
        return True

    def to_dict(self):
        """Convert room to a dictionary for serialization."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'parameters': self.parameters
        }


class SDNExperiment:
    """Class to store and manage acoustic simulation experiment data and metadata."""
    
    def __init__(self, config, rir, fs=44100, duration=0.5, experiment_id=None, skip_metrics=False):
        """
        Initialize an acoustic simulation experiment.
        
        Args:
            config (dict): Configuration parameters for the experiment
            rir (np.ndarray): Room impulse response
            fs (int): Sampling frequency
            duration (float): Duration of the RIR in seconds
            experiment_id (str, optional): Unique ID for the experiment. If None, will be generated.
            skip_metrics (bool): If True, skip calculating metrics (useful for temporary objects)
        """
        self.config = config
        self.rir = rir
        self.fs = fs
        self.duration = duration
        self.timestamp = datetime.now().isoformat()
        
        # Generate a unique ID if not provided
        if experiment_id is None:
            # Create a hash of the configuration to use as ID
            id_config = self._prepare_config_for_id(config)
            config_str = json.dumps(self._make_serializable(id_config), sort_keys=True)
            self.experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:10]
        else:
            self.experiment_id = experiment_id
            
        # Calculate metrics if not skipped
        if not skip_metrics:
            self._calculate_metrics()
    
    def _prepare_config_for_id(self, config):
        """
        Prepare a configuration for ID generation.
        Excludes descriptive fields like 'info' that don't affect the experiment result.
        Focuses on parameters that actually impact the simulation.
        
        Args:
            config (dict): Original configuration dictionary
            
        Returns:
            dict: Configuration with only the fields that affect the experiment result
        """
        # Create a copy to avoid modifying the original
        id_config = {}  # Start with empty dict instead of copying to ensure consistent keys
        
        # Add simulation method
        if 'method' in config:
            id_config['method'] = config['method']
        
        # Add ISM-specific parameters
        if config.get('method') == 'ISM':
            if 'max_order' in config:
                id_config['max_order'] = config['max_order']
            if 'ray_tracing' in config:
                id_config['ray_tracing'] = config['ray_tracing']
            if 'use_rand_ism' in config:
                id_config['use_rand_ism'] = config['use_rand_ism']
                
        # Keep only room parameters with numerical values
        if 'room_parameters' in config:
            id_config['room_parameters'] = {}
            for key, value in config['room_parameters'].items():
                # Only include numeric values that affect simulation
                if isinstance(value, (int, float)):
                    id_config['room_parameters'][key] = value
        
        # Keep only relevant flags that affect the simulation (for SDN)
        if 'flags' in config and config.get('method') == 'SDN':
            id_config['flags'] = {}
            for key, value in config['flags'].items():
                if key in ['source_weighting', 'specular_source_injection', 
                          'scattering_matrix_update_coef', 'coef', 
                          'source_pressure_injection_coeff']:
                    id_config['flags'][key] = value
        
        # Add other essential parameters
        if 'fs' in config:
            id_config['fs'] = config['fs']
        if 'duration' in config:
            id_config['duration'] = config['duration']
            
        return id_config
    
    def _make_serializable(self, obj):
        """Convert a complex object to a JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            # Try to convert to a basic type
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
    
    def _calculate_metrics(self):
        """Calculate various acoustic metrics for the RIR."""
        self.metrics = {}
        
        # Skip metrics calculation if RIR is empty
        if len(self.rir) == 0:
            self.edc = np.array([])
            self.ned = np.array([])
            self.time_axis = np.array([])
            self.ned_time_axis = np.array([])
            return
        
        # Calculate RT60 if the RIR is long enough
        if self.duration > 0.7:
            self.metrics['rt60'] = pp.calculate_rt60_from_rir(self.rir, self.fs, plot=False)
        
        # Calculate EDC
        self.edc = an.compute_edc(self.rir, self.fs, label=self.get_label(), plot=False)
        
        # Calculate NED
        self.ned = ned.echoDensityProfile(self.rir, fs=self.fs)
        
        # Time axis for plotting
        self.time_axis = np.arange(len(self.rir)) / self.fs
        self.ned_time_axis = np.arange(len(self.ned)) / self.fs
    
    def get_label(self):
        """Generate a descriptive label for the experiment."""
        method = self.config.get('method', 'SDN')
        
        if 'label' in self.config and self.config['label']:
            label = f"{self.config['label']}"
        else:
            label = f""
            
        if 'info' in self.config and self.config['info']:
            label += f": {self.config['info']}"
            
        # Add method-specific details
        if method == 'ISM':
            if 'max_order' in self.config:
                label += f" order={self.config['max_order']}"
        
        return label
    
    def get_key_parameters(self):
        """Extract and return the key parameters that define this experiment."""
        params = {}
        
        # Add method
        params['method'] = self.config.get('method', 'SDN')
        
        # Add key flags if they exist (for SDN)
        if 'flags' in self.config and params['method'] == 'SDN':
            flags = self.config['flags']
            # Focus on commonly adjusted parameters
            key_params = ['source_weighting', 'specular_source_injection', 
                          'scattering_matrix_update_coef', 'coef', 'source_pressure_injection_coeff']
            
            for param in key_params:
                if param in flags:
                    params[param] = flags[param]
        
        # Add ISM-specific parameters
        if params['method'] == 'ISM':
            if 'max_order' in self.config:
                params['max_order'] = self.config['max_order']
            if 'ray_tracing' in self.config:
                params['ray_tracing'] = self.config['ray_tracing']
        
        # Add room parameters if they exist
        if 'room_parameters' in self.config:
            room = self.config['room_parameters']
            params['room_dimensions'] = f"{room.get('width', 0)}x{room.get('depth', 0)}x{room.get('height', 0)}"
            params['absorption'] = room.get('absorption', 0)
            params['src_pos'] = f"({room.get('source x', 0)}, {room.get('source y', 0)}, {room.get('source z', 0)})"
            params['mic_pos'] = f"({room.get('mic x', 0)}, {room.get('mic y', 0)}, {room.get('mic z', 0)})"
        
        return params
    
    def to_dict(self):
        """Convert the experiment to a dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'config': self._make_serializable(self.config),
            'timestamp': self.timestamp,
            'fs': self.fs,
            'duration': self.duration,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, data_dict, rir):
        """Create an SDNExperiment instance from a dictionary and RIR data."""
        experiment = cls(
            config=data_dict['config'],
            rir=rir,
            fs=data_dict['fs'],
            duration=data_dict['duration'],
            experiment_id=data_dict['experiment_id']
        )
        experiment.timestamp = data_dict['timestamp']
        experiment.metrics = data_dict.get('metrics', {})
        return experiment


class SDNExperimentManager:
    """Class to manage loading and accessing acoustic simulation experiments from storage."""
    
    def __init__(self, results_dir='results', is_batch_manager=False):
        """
        Initialize the experiment manager.
        
        Args:
            results_dir (str): Base directory to store experiment data
            is_batch_manager (bool): If True, this manager handles batch processing experiments
        """
        self.results_dir = results_dir
        self.is_batch_manager = is_batch_manager
        self.rooms = {}  # name -> Room
        self.ensure_dir_exists()
        self.load_experiments()
    
    def ensure_dir_exists(self):
        """Ensure the results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
        # Ensure the singulars directory exists if this is not a batch manager
        if not self.is_batch_manager:
            os.makedirs(os.path.join(self.results_dir, 'room_singulars'), exist_ok=True)
    
    def _get_room_dir(self, room_name):
        """Get the directory path for a room."""
        if not self.is_batch_manager:
            # For singular experiments, use the room_singulars folder
            return os.path.join(self.results_dir, 'room_singulars', room_name)
        else:
            # For batch experiments, use the normal structure
            return os.path.join(self.results_dir, 'rooms', room_name)
    
    def _get_source_dir(self, room_name, source_label):
        """Get the directory path for a source within a room."""
        room_dir = self._get_room_dir(room_name)
        return os.path.join(room_dir, source_label)  
    
    def load_experiments(self):
        """Load all experiments from the results directory."""
        self.rooms = {}
        
        if not self.is_batch_manager:
            # Load singular experiments from room_singulars folder
            singulars_dir = os.path.join(self.results_dir, 'room_singulars')
            if os.path.exists(singulars_dir):
                for room_name in os.listdir(singulars_dir):
                    room_path = os.path.join(singulars_dir, room_name)
                    if not os.path.isdir(room_path):
                        continue
                        
                    # Check for room info first (preferred method)
                    room_info_path = os.path.join(room_path, 'room_info.json')
                    if os.path.exists(room_info_path):
                        try:
                            with open(room_info_path, 'r') as f:
                                room_info = json.load(f)
                            
                            # Create room with saved display name
                            room_parameters = room_info.get('parameters', {})
                            room = Room(room_info.get('name', room_name), room_parameters)
                            if 'display_name' in room_info:
                                room.display_name = room_info['display_name']
                            
                            # Load experiments from this room
                            for experiment_file in os.listdir(room_path):
                                if experiment_file.endswith('.json') and experiment_file != 'room_info.json':
                                    config_path = os.path.join(room_path, experiment_file)
                                    rir_path = os.path.join(room_path, experiment_file.replace('.json', '.npy'))
                                    
                                    # Load experiment (singular case)
                                    if os.path.exists(config_path) and os.path.exists(rir_path):
                                        try:
                                            with open(config_path, 'r') as f:
                                                config_data = json.load(f)
                                            rir = np.load(rir_path)
                                            
                                            # Check if this is a nested config structure
                                            # (singular experiments store room_parameters inside a nested config key)
                                            if 'config' in config_data and 'room_parameters' in config_data['config']:
                                                experiment_config = config_data['config']
                                                experiment_id = config_data.get('experiment_id')
                                                fs = config_data.get('fs', 44100)
                                                duration = config_data.get('duration', 0.5)
                                            else:
                                                # Original format expected
                                                experiment_config = config_data
                                                experiment_id = config_data.get('experiment_id')
                                                fs = config_data.get('fs', 44100)
                                                duration = config_data.get('duration', 0.5)
                                            
                                            # Create experiment object
                                            experiment = SDNExperiment(
                                                config=experiment_config,
                                                rir=rir,
                                                fs=fs,
                                                duration=duration,
                                                experiment_id=experiment_id
                                            )
                                            
                                            # Add experiment to room
                                            room.add_experiment(experiment)
                                        except Exception as e:
                                            print(f"Error loading experiment {experiment_file}: {e}")
                            
                            self.rooms[room.name] = room
                        except Exception as e:
                            print(f"Error loading room {room_name}: {e}")
        else:
            # Load batch experiments from the structured directory
            rooms_dir = os.path.join(self.results_dir, 'rooms')
            if not os.path.exists(rooms_dir):
                return
            
            # Iterate through room directories
            for room_name in os.listdir(rooms_dir):
                room_path = os.path.join(rooms_dir, room_name)
                if not os.path.isdir(room_path):
                    continue
                    
                # Load room info
                room_info_path = os.path.join(room_path, 'room_info.json')
                if os.path.exists(room_info_path):
                    try:
                        with open(room_info_path, 'r') as f:
                            room_info = json.load(f)
                        
                        # Create room with saved parameters
                        room_parameters = room_info.get('parameters', {})
                        room = Room(room_info.get('name', room_name), room_parameters)
                        if 'display_name' in room_info:
                            room.display_name = room_info['display_name']
                        
                        # Load sources directly from room directory
                        for source_label in os.listdir(room_path):
                            source_path = os.path.join(room_path, source_label)
                            
                            # Skip non-directories and special files/directories
                            if not os.path.isdir(source_path) or source_label == 'room_info.json':
                                continue
                            
                            # For each method directory
                            for method in os.listdir(source_path):
                                method_path = os.path.join(source_path, method)
                                if not os.path.isdir(method_path):
                                    continue
                                
                                # For each parameter set directory
                                for param_set in os.listdir(method_path):
                                    param_path = os.path.join(method_path, param_set)
                                    if not os.path.isdir(param_path):
                                        continue
                                    
                                    # Load config and RIRs
                                    config_path = os.path.join(param_path, 'config.json')
                                    rirs_path = os.path.join(param_path, 'rirs.npy')
                                    
                                    # Load config and RIRs (batch case)
                                    if os.path.exists(config_path) and os.path.exists(rirs_path):
                                        try:
                                            with open(config_path, 'r') as f:
                                                config_data = json.load(f)
                                            rirs = np.load(rirs_path)
                                            
                                            # Check if this is a nested config structure
                                            if 'config' in config_data and 'room_parameters' in config_data['config']:
                                                actual_config = config_data['config']
                                                fs = config_data.get('fs', 44100)
                                                duration = config_data.get('duration', 0.5)
                                            else:
                                                actual_config = config_data
                                                fs = config_data.get('fs', 44100)
                                                duration = config_data.get('duration', 0.5)
                                            
                                            # Get receivers from the config
                                            receivers_data = actual_config.get('receivers', config_data.get('receivers', []))
                                            
                                            # Create experiment objects for each source-receiver pair
                                            for idx, receiver_info in enumerate(receivers_data):
                                                if idx >= len(rirs):
                                                    break
                                                    
                                                # Create a single experiment config
                                                receiver_config = actual_config.copy()
                                                # Remove the receivers array
                                                if 'receivers' in receiver_config:
                                                    del receiver_config['receivers']
                                                # Add the specific receiver info
                                                receiver_config['receiver'] = receiver_info
                                                
                                                # Use source info from config
                                                source_info = actual_config.get('source', config_data.get('source', {}))
                                                
                                                # Create experiment object
                                                experiment = SDNExperiment(
                                                    config=receiver_config,
                                                    rir=rirs[idx],
                                                    fs=fs,
                                                    duration=duration,
                                                    experiment_id=receiver_info.get('experiment_id')
                                                )
                                                
                                                # Add experiment to room
                                                room.add_experiment(experiment)
                                        except Exception as e:
                                            print("is this??")
                                            print(f"Error loading experiments from {param_path}: {e}")
                        
                        self.rooms[room.name] = room
                    except Exception as e:
                        print(f"Error loading room {room_name}: {e}")
    
    def get_experiment(self, experiment_id):
        """
        Get an experiment by ID.
        
        Args:
            experiment_id (str): The ID of the experiment
            
        Returns:
            SDNExperiment: The experiment object
        """
        for room in self.rooms.values():
            if experiment_id in room.experiments:
                return room.experiments[experiment_id]
        return None
    
    def get_experiments_by_label(self, label):
        """
        Get experiments by label.
        
        Args:
            label (str): The label to search for
            
        Returns:
            list: List of matching experiments
        """
        experiments = []
        for room in self.rooms.values():
            experiments.extend([exp for exp in room.experiments.values() if label in exp.get_label()])
        return experiments
    
    def get_all_experiments(self):
        """
        Get all experiments.
        
        Returns:
            list: List of all experiments
        """
        experiments = []
        for room in self.rooms.values():
            experiments.extend(list(room.experiments.values()))
        return experiments

def get_batch_manager():
    _batch_manager = SDNExperimentManager(
        results_dir='results',
        is_batch_manager=True
    )
    return _batch_manager

def get_singular_manager():

    _singular_manager = SDNExperimentManager(
        results_dir='results',
        is_batch_manager=False
    )
    return _singular_manager

# Load all experiments
# load_manager = SDNExperimentManager(results_dir='results')
# load_manager.load_experiments()

# Or use the singletons
# batch_manager = get_batch_manager()
# singular_manager = get_singular_manager()

# Access experiments
# experiments = batch_manager.get_all_experiments()
# experiment = batch_manager.get_experiment("some_experiment_id")

# For batch experiments
# from sdn_experiment_visualizer import SDNExperimentVisualizer

# Option 1: Pass the manager explicitly
# visualizer = SDNExperimentVisualizer(get_batch_manager())
# visualizer.show()

# Option 2: Let the visualizer select the appropriate manager
# visualizer = SDNExperimentVisualizer()
# visualizer.show(is_batch=True)  # For batch experiments
# visualizer.show(is_batch=False)  # For singular experiments