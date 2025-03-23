import os
import json
# import pickle
import numpy as np
import spatial_analysis as sa
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

# Set matplotlib backend to match main script
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5

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
        self.experiments[experiment.experiment_id] = experiment
        
        # Get source and mic positions
        room_params = experiment.config['room_parameters']
        source_pos = [
            room_params['source x'],
            room_params['source y'],
            room_params['source z']
        ]
        mic_pos = [
            room_params['mic x'],
            room_params['mic y'],
            room_params['mic z']
        ]
        
        # Add to position-based dictionary
        pos_key = self._get_position_key(source_pos, mic_pos)
        if pos_key not in self.experiments_by_position:
            self.experiments_by_position[pos_key] = []
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
        
        # For debugging
        # print(f"Generated label: {label}")
        
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
    """Class to manage multiple acoustic simulation experiments, store results, and provide visualization."""
    
    def __init__(self, results_dir='results'):
        """
        Initialize the experiment manager.
        
        Args:
            results_dir (str): Base directory to store experiment data
        """
        self.results_dir = results_dir
        self.rooms = {}  # name -> Room
        self.ensure_dir_exists()
        self.load_experiments()
    
    def ensure_dir_exists(self):
        """Ensure the results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _get_room_name(self, room_parameters):
        """Generate a unique room name based on parameters."""
        # Create a hash of room dimensions and absorption
        room_key = {k: room_parameters[k] for k in ['width', 'depth', 'height', 'absorption']}
        room_str = json.dumps(room_key, sort_keys=True)
        room_hash = hashlib.md5(room_str.encode()).hexdigest()[:6]
        return f"room_{room_hash}"
    
    def _get_room_dir(self, room_name):
        """Get the directory path for a room."""
        return os.path.join(self.results_dir, room_name)
    
    def load_experiments(self):
        """Load all experiments from the results directory."""
        self.rooms = {}
        
        # Check if the directory exists
        if not os.path.exists(self.results_dir):
            return
        
        # Iterate through room directories
        for room_dir in os.listdir(self.results_dir):
            room_path = os.path.join(self.results_dir, room_dir)
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
                    room = Room(room_info.get('name', room_dir), room_parameters)
                    if 'display_name' in room_info:
                        room.display_name = room_info['display_name']
                except Exception as e:
                    print(f"Error loading room info for {room_dir}: {e}")
                    # Fallback to room parameters if room info fails
                    room_params_path = os.path.join(room_path, 'room_parameters.json')
                    if os.path.exists(room_params_path):
                        with open(room_params_path, 'r') as f:
                            room_parameters = json.load(f)
                        room = Room(room_dir, room_parameters)
                    else:
                        continue  # Skip if no valid room data
            else:
                # Fallback to legacy room parameters
                room_params_path = os.path.join(room_path, 'room_parameters.json')
                if not os.path.exists(room_params_path):
                    continue
                    
                with open(room_params_path, 'r') as f:
                    room_parameters = json.load(f)
                
                # Create room object from legacy parameters
                room = Room(room_dir, room_parameters)
                
                # Create room_info.json from room_parameters.json for future use
                # and to standardize on a single format
                room_info_path = os.path.join(room_path, 'room_info.json')
                with open(room_info_path, 'w') as f:
                    json.dump(room.to_dict(), f, indent=2)
                    print(f"Upgraded {room_dir} to use room_info.json")
            
            # Load experiments for this room
            for filename in os.listdir(room_path):
                if filename.endswith('.json') and filename not in ['room_parameters.json', 'room_info.json']:
                    experiment_id = filename.split('.')[0]
                    metadata_path = os.path.join(room_path, filename)
                    rir_path = os.path.join(room_path, f"{experiment_id}.npy")
                    
                    try:
                        # Load metadata
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Load RIR if it exists
                        if os.path.exists(rir_path):
                            rir = np.load(rir_path)
                            
                            # Create experiment object
                            experiment = SDNExperiment.from_dict(metadata, rir)
                            room.add_experiment(experiment)
                    except Exception as e:
                        print(f"Error loading experiment {experiment_id}: {e}")
            
            self.rooms[room.name] = room
    
    def run_experiment(self, config, room_parameters, duration=0.5, fs=44100, 
                      force_rerun=False, room_name=None, method='SDN',
                      batch_processing=False, source_positions=None, receiver_positions=None):
        """
        Run an acoustic simulation experiment with the given configuration.
        
        Args:
            config (dict): Configuration for the experiment
            room_parameters (dict): Room parameters
            duration (float): Duration of the simulation in seconds
            fs (int): Sampling frequency
            force_rerun (bool): If True, rerun the experiment even if it exists
            room_name (str, optional): Explicit name for the room (e.g. 'room_aes')
            method (str): Simulation method ('SDN', 'ISM', 'TRE', etc.)
            batch_processing (bool): If True, run for multiple source-mic positions
            source_positions (list): List of source positions [(x,y,z,label), ...] 
            receiver_positions (list): List of receiver positions [(x,y,z), ...]
            
        Returns:
            SDNExperiment or list: Single experiment or list of experiment IDs
        """
        # Add method to config
        if 'method' not in config:
            config['method'] = method
        
        # Handle batch processing case
        if batch_processing:
            if source_positions is None or receiver_positions is None:
                raise ValueError("Both source_positions and receiver_positions must be provided for batch processing")
            
            experiment_ids = []
            total_combinations = len(source_positions) * len(receiver_positions)
            
            print(f"\nRunning batch processing: {len(source_positions)} sources × {len(receiver_positions)} receivers = {total_combinations} experiments")
            
            # Process each source-receiver combination
            for src_idx, src_pos in enumerate(source_positions):
                source_x, source_y, source_z, source_label = src_pos
                
                print(f"\nProcessing source {src_idx+1}/{len(source_positions)}: {source_label}")
                
                # Process each receiver for this source
                for rec_idx, rec_pos in enumerate(receiver_positions):
                    mic_x, mic_y = rec_pos
                    mic_z = room_parameters['mic z']
                    
                    # Create position identifier
                    pos_id = f"{source_label}_to_mic_{rec_idx+1}"
                    progress = ((src_idx * len(receiver_positions) + rec_idx + 1) / total_combinations) * 100
                    print(f"  [{progress:.1f}%] Position {src_idx * len(receiver_positions) + rec_idx + 1}/{total_combinations}: {pos_id}")
                    
                    # Create updated room parameters with current positions
                    current_params = room_parameters.copy()
                    current_params.update({
                        'source x': source_x,
                        'source y': source_y, 
                        'source z': source_z,
                        'mic x': mic_x,
                        'mic y': mic_y,
                        'mic z': mic_z
                    })
                    
                    # Add position ID to config
                    position_config = config.copy()
                    position_config['position_id'] = pos_id
                    
                    # Run single experiment with these positions
                    try:
                        experiment = self.run_experiment(
                            config=position_config,
                            room_parameters=current_params,
                            duration=duration,
                            fs=fs,
                            force_rerun=force_rerun,
                            room_name=room_name,
                            method=method
                        )
                        
                        if experiment:
                            experiment_ids.append(experiment.experiment_id)
                    except Exception as e:
                        print(f"    Error: {str(e)}")
            
            print(f"\nCompleted {len(experiment_ids)}/{total_combinations} experiments")
            return experiment_ids
        
        # Original single-position implementation (existing code)
        else:
            # Get or create room 
            if room_name is None:
                # Generate a hash-based name if not provided
                room_name = self._get_room_name(room_parameters)
            
            # Check if room exists with same parameters
            existing_room = None
            for existing_name, room in self.rooms.items():
                if room.matches_parameters(room_parameters):
                    existing_room = room
                    room_name = existing_name
                    break
                
            if existing_room is None:
                # Create new room with the specified name
                room = Room(room_name, room_parameters)
                self.rooms[room_name] = room
                
                # Save room info to disk
                room_dir = self._get_room_dir(room_name)
                os.makedirs(room_dir, exist_ok=True)
                
                # Save only room_info.json
                room_info_path = os.path.join(room_dir, 'room_info.json')
                with open(room_info_path, 'w') as f:
                    json.dump(room.to_dict(), f, indent=2)
            else:
                room = existing_room
            
            # Create a temporary config with all parameters for ID generation
            full_config = {
                **config,
                'room_parameters': room_parameters,
                'duration': duration,
                'fs': fs,
                'method': method
            }
            
            # Generate experiment ID
            temp_experiment = SDNExperiment(full_config, np.array([]), skip_metrics=True)
            experiment_id = temp_experiment.experiment_id
            
            # Check if experiment exists
            if experiment_id in room.experiments and not force_rerun:
                print(f"Experiment {experiment_id} already exists in room {room.display_name}. Using cached results.")
                return room.experiments[experiment_id]
            
            # Setup room geometry
            geom_room = geometry.Room(
                room_parameters['width'], 
                room_parameters['depth'], 
                room_parameters['height']
            )
            geom_room.set_microphone(
                room_parameters['mic x'], 
                room_parameters['mic y'], 
                room_parameters['mic z']
            )
            geom_room.set_source(
                room_parameters['source x'], 
                room_parameters['source y'], 
                room_parameters['source z'],
                signal="will be replaced", 
                Fs=fs
            )
            
            # Calculate reflection coefficient
            reflection = np.sqrt(1 - room_parameters['absorption'])
            geom_room.wallAttenuation = [reflection] * 6
            
            # Setup signal
            num_samples = int(fs * duration)
            impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
            geom_room.source.signal = impulse_dirac['signal']
            
            # Calculate RIR based on method
            if method == 'SDN':
                # Get flags from config
                flags = config.get('flags', {})
                
                # Create SDN instance with configured flags
                sdn = DelayNetwork(geom_room, Fs=fs, label=config.get('label', ''), **flags)
                
                # Calculate RIR
                rir = sdn.calculate_rir(duration)
                
            elif method == 'ISM':
                # Import pyroomacoustics
                import pyroomacoustics as pra
                
                # Get ISM parameters
                max_order = config.get('max_order', 12)
                ray_tracing = config.get('ray_tracing', False)
                use_rand_ism = config.get('use_rand_ism', False)
                
                # Setup source and mic locations
                source_loc = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
                mic_loc = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])
                room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])
                
                # Create pra room
                pra_room = pra.ShoeBox(
                    room_dim, 
                    fs=fs,
                    materials=pra.Material(room_parameters['absorption']),
                    max_order=max_order,
                    air_absorption=False, 
                    ray_tracing=ray_tracing,
                    use_rand_ism=use_rand_ism
                )
                pra_room.set_sound_speed(343)
                pra_room.add_source(source_loc).add_microphone(mic_loc)
                
                # Compute RIR
                pra_room.compute_rir()
                pra_rir = pra_room.rir[0][0]
                
                # Process the RIR
                global_delay = pra.constants.get("frac_delay_length") // 2
                pra_rir = pra_rir[global_delay:]  # Shift left by removing the initial delay
                pra_rir = np.pad(pra_rir, (0, global_delay))  # Pad with zeros at the end to maintain length
                rir = pra_rir[:num_samples]
                
            elif method == 'TRE':
                # Placeholder for Treble method
                # This would be implemented in the future
                raise NotImplementedError("Treble method not yet implemented")
                
            else:
                raise ValueError(f"Unknown simulation method: {method}")
            
            # Normalize RIR
            rir = rir / np.max(np.abs(rir))
            
            # Create experiment object
            experiment = SDNExperiment(
                config=full_config,
                rir=rir,
                fs=fs,
                duration=duration,
                experiment_id=experiment_id
            )
            
            # Save experiment
            self.save_experiment(experiment, room_name)
            
            # Add to room's experiments
            room.add_experiment(experiment)
            
            return experiment
    
    def save_experiment(self, experiment, room_name):
        """
        Save an experiment to disk.
        
        Args:
            experiment (SDNExperiment): The experiment to save
            room_name (str): Name of the room this experiment belongs to
        """
        room_dir = self._get_room_dir(room_name)
        os.makedirs(room_dir, exist_ok=True)
        
        # Save or update room info
        room = self.rooms.get(room_name)
        if room:
            room_info_path = os.path.join(room_dir, 'room_info.json')
            with open(room_info_path, 'w') as f:
                json.dump(room.to_dict(), f, indent=2)
        
        # Save experiment metadata
        metadata_path = os.path.join(room_dir, f"{experiment.experiment_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # Save RIR
        rir_path = os.path.join(room_dir, f"{experiment.experiment_id}.npy")
        np.save(rir_path, experiment.rir)
    
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
    
    def create_room_visualization(self, experiments, highlight_pos_idx=None):
        """
        Create a 2D top-view visualization of the room with source and receiver positions.
        
        Args:
            experiments (list): List of SDNExperiment objects or Room objects to visualize
            highlight_pos_idx (int, optional): Index of source-mic pair to highlight
            
        Returns:
            go.Figure: Plotly figure with room visualization
        """
        fig = go.Figure()
        
        # Keep track of room dimensions
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        # Get the current room and its experiments
        room = experiments[0] if isinstance(experiments[0], Room) else None
        if room is None:
            return fig
            
        # Get room dimensions
        width = room.parameters['width']
        depth = room.parameters['depth']
        height = room.parameters['height']
        
        # Update plot bounds
        min_x = min(min_x, 0)
        max_x = max(max_x, width)
        min_y = min(min_y, 0)
        max_y = max(max_y, depth)
        
        # Draw room outline
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=width, y1=depth,
            line=dict(color='black', width=2),
            fillcolor="rgba(240, 240, 240, 0.1)",
            layer="above"
        )
        
        # Get current source-mic pair
        current_pos = None
        if highlight_pos_idx is not None and room.source_mic_pairs:
            current_pos = room.source_mic_pairs[highlight_pos_idx % len(room.source_mic_pairs)]
            source_pos, mic_pos = current_pos
            
            # Add source marker (red)
            fig.add_trace(go.Scatter(
                x=[source_pos[0]], y=[source_pos[1]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Source'
            ))
            
            # Add microphone marker (green)
            fig.add_trace(go.Scatter(
                x=[mic_pos[0]], y=[mic_pos[1]],
                mode='markers',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Microphone'
            ))
        
        # Add some padding
        padding = 0.5
        x_range = [min_x - padding, max_x + padding]
        y_range = [min_y - padding, max_y + padding]
        
        # Update layout
        fig.update_layout(
            title=f"{room.display_name}: {room.dimensions_str}",
            xaxis=dict(
                title="Width (m)",
                range=x_range,
                constrain="domain",
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.2)',
                zerolinecolor='rgba(200, 200, 200, 0.2)'
            ),
            yaxis=dict(
                title="Depth (m)",
                range=y_range,
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor='rgba(200, 200, 200, 0.2)',
                zerolinecolor='rgba(200, 200, 200, 0.2)'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(t=30, b=0, l=0, r=0),  # Added top margin for title
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    def plot(self, room_name=None):
        """
        Launch an interactive dashboard to visualize experiments for a specific room.
        
        Args:
            room_name (str, optional): Name of the room to display. If None, displays the first room.
        """
        if not self.rooms:
            print("No rooms with experiments to plot.")
            return
        
        # Get the room to display
        if room_name is None:
            room_name = next(iter(self.rooms.keys()))
        elif room_name not in self.rooms:
            print(f"Room {room_name} not found.")
            return
        
        room = self.rooms[room_name]
        if not room.experiments:
            print(f"No experiments in room {room_name}.")
            return
        
        # Create a Dash app
        app = dash.Dash(__name__)
        server = app.server
        port = 8050

        # Add custom styles for dropdown options
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    /* Make dropdown option text light-colored */
                    .VirtualizedSelectOption {
                        color: #e0e0e0 !important;
                        background-color: #282c34 !important;
                    }
                    .VirtualizedSelectFocusedOption {
                        background-color: #1e2129 !important;
                    }
                    /* Style for dropdown input text and selected value */
                    .Select-value-label {
                        color: #e0e0e0 !important;
                    }
                    .Select-control {
                        background-color: #282c34 !important;
                        border-color: #404040 !important;
                    }
                    .Select-menu-outer {
                        background-color: #282c34 !important;
                        border-color: #404040 !important;
                    }
                    .Select-input > input {
                        color: #e0e0e0 !important;
                    }
                    /* Dropdown arrow color */
                    .Select-arrow {
                        border-color: #e0e0e0 transparent transparent !important;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

        # Get list of rooms for navigation
        room_names = list(self.rooms.keys())
        current_room_idx = room_names.index(room_name)
        
        # Dark theme colors
        dark_theme = {
            'background': '#2d3038',
            'paper_bg': '#282c34',
            'text': '#e0e0e0',
            'grid': 'rgba(255, 255, 255, 0.1)',
            'button_bg': '#404040',
            'button_text': '#ffffff',
            'header_bg': '#1e2129',
            'plot_bg': '#1e2129',
            'accent': '#61dafb'
        }
        
        # Create app layout
        app.layout = html.Div([
            # Room navigation header
            html.Div([
                html.Div([
                    html.H2(
                        id='room-header',
                        style={'margin': '0 20px', 'color': dark_theme['text']}
                    ),
                    html.H3(
                        id='rt-header',
                        style={'margin': '10px 20px', 'color': dark_theme['accent']}
                    )
                ], style={'display': 'inline-block', 'position': 'relative'}),
                html.Div([
                    html.Button('←', id='prev-room', style={
                        'fontSize': 24, 
                        'marginRight': '10px',
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 15px'
                    }),
                    html.Button('→', id='next-room', style={
                        'fontSize': 24,
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 15px'
                    }),
                ], style={'position': 'absolute', 'left': '50%', 'top': '%50', 'transform': 'translateX(-50%)'}),
                dcc.Store(id='current-room-idx', data=current_room_idx)
            ], style={'textAlign': 'center', 'margin': '20px', 'position': 'relative'}),
            
            # Main content with plots and room visualization
            html.Div([
                # Left side - plots and table (50% width instead of 75%)
                html.Div([
                    # Time range selector
                    html.Div([
                        html.H4("Time Range:", style={'display': 'inline-block', 'marginRight': '15px', 'color': dark_theme['text']}),
                        dcc.RadioItems(
                            id='time-range-selector',
                            options=[
                                {'label': 'Early Part (50ms)', 'value': 0.05},
                                {'label': 'First 0.5s', 'value': 0.5},
                                {'label': 'Whole RIR', 'value': 'full'}
                            ],
                            value='0.05',  # Default to early part view
                            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'cursor': 'pointer'},
                            style={'color': dark_theme['text'], 'fontSize': '14px'},
                            inputStyle={'marginRight': '5px'}
                        )
                    ], style={'margin': '10px 0 0 20px', 'display': 'flex', 'alignItems': 'center'}),
                    
                    # Plots
                    dcc.Tabs([
                        dcc.Tab(label="Room Impulse Responses", children=[
                            dcc.Graph(id='rir-plot', style={'height': '50vh'})
                        ],
                        style={
                            'backgroundColor': dark_theme['paper_bg'],
                            'color': dark_theme['text']
                        },
                        selected_style={
                            'backgroundColor': dark_theme['header_bg'],
                            'color': dark_theme['accent'],
                            'borderTop': f'2px solid {dark_theme["accent"]}'
                        }),
                        dcc.Tab(label="Energy Decay Curves", children=[
                            dcc.Graph(id='edc-plot', style={'height': '50vh'})
                        ],
                        style={
                            'backgroundColor': dark_theme['paper_bg'],
                            'color': dark_theme['text']
                        },
                        selected_style={
                            'backgroundColor': dark_theme['header_bg'],
                            'color': dark_theme['accent'],
                            'borderTop': f'2px solid {dark_theme["accent"]}'
                        }),
                        dcc.Tab(label="Normalized Echo Density", children=[
                            dcc.Graph(id='ned-plot', style={'height': '50vh'})
                        ],
                        style={
                            'backgroundColor': dark_theme['paper_bg'],
                            'color': dark_theme['text']
                        },
                        selected_style={
                            'backgroundColor': dark_theme['header_bg'],
                            'color': dark_theme['accent'],
                            'borderTop': f'2px solid {dark_theme["accent"]}'
                        })
                    ], style={'backgroundColor': dark_theme['paper_bg'], 'margin': '10px 0'}),
                    
                    # Experiment table for current position (now below the plots)
                    html.Div([
                        html.H3("Active Experiments", style={'color': dark_theme['text']}),
                        dash_table.DataTable(
                            id='experiment-table',
                            style_table={
                                'overflowX': 'auto',
                                'maxWidth': '100%'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'minWidth': '5px',     # Smaller minimum width
                                'width': '100px',      # Smaller default width
                                'maxWidth': '150px',   # Smaller maximum width
                                'padding': '5px',      # Reduced padding
                                'backgroundColor': dark_theme['paper_bg'],
                                'color': dark_theme['text'],
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis'
                            },
                            style_cell_conditional=[
                                # Generate conditional styling for each column from config
                                {'if': {'column_id': col_config['id']},
                                 'width': col_config['width']}
                                for col_config in self._get_column_config()
                                if 'width' in col_config
                            ],
                            style_header={
                                'backgroundColor': dark_theme['header_bg'],
                                'fontWeight': 'bold',
                                'color': dark_theme['text'],
                                'whiteSpace': 'normal',
                                'height': 'auto',
                                'textAlign': 'center'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgba(255, 255, 255, 0.05)'
                                }
                            ],
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi"
                        )
                    ], style={'margin': '20px'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Middle - room visualization (25% width)
                html.Div([
                    # Title
                    html.H3("Room Layout (Top View)", 
                           style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '65px', 'color': dark_theme['text']}),
                    
                    # Source and receiver dropdown selectors (NEW)
                    html.Div([
                        # Source selector
                        html.Div([
                            html.Label("Select source", 
                                     style={'fontSize': '14px', 'color': dark_theme['text'], 'display': 'block', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='source-selector',
                                options=[],  # Will be populated in callback
                                style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],  # Light text color for better readability
                                    'width': '100%'
                                },
                                # Add dropdown style for better readability of options
                                className='dropdown-light-text'
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'}),
                        
                        # Receiver selector
                        html.Div([
                            html.Label("Select receiver", 
                                     style={'fontSize': '14px', 'color': dark_theme['text'], 'display': 'block', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='receiver-selector',
                                options=[],  # Will be populated in callback
                                style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],  # Light text color for better readability
                                    'width': '100%'
                                },
                                # Add dropdown style for better readability of options
                                className='dropdown-light-text'
                            )
                        ], style={'width': '100%', 'marginBottom': '10px'})
                    ], style={'padding': '0px 10px'}),
                    
                    
                    # Store for current position index
                    dcc.Store(id='current-pos-idx', data=0),
                    
                    # Room visualization plot
                    dcc.Graph(
                        id='room-plot',
                        style={'height': '50vh', 'marginTop': '0px'}
                    ),
                    
                    # Position navigation buttons (at bottom)
                    html.Div([
                        html.Div(style={'textAlign': 'center', 'marginBottom': '5px', 'fontSize': '16px', 'color': dark_theme['text']}, children="Navigate Source-Mic Pairs"),
                        html.Button('←', id='prev-pos', style={
                            'fontSize': 20, 
                            'marginRight': '10px',
                            'backgroundColor': dark_theme['button_bg'],
                            'color': dark_theme['button_text'],
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '0px 15px'
                        }),
                        html.Button('→', id='next-pos', style={
                            'fontSize': 20,
                            'backgroundColor': dark_theme['button_bg'],
                            'color': dark_theme['button_text'],
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '0px 15px'
                        }),
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),
                    
                    # Position info text
                    html.Div(id='pos-info', 
                           style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '10px', 'fontSize': '14px', 'color': dark_theme['text']}),
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right side - empty space (25% width) for future use
                html.Div([
                    # Empty space for future extensions
                    html.H3("Future Plot Area", 
                           style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '65px', 'color': dark_theme['text'], 'opacity': '0.5'}),
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'alignItems': 'flex-start'})
        ], style={
            'backgroundColor': dark_theme['background'],
            'minHeight': '100vh',
            'fontFamily': 'Arial, sans-serif',
            'padding': '10px'
        })
        
        # Combine the navigation and dropdown callbacks into a single callback
        @app.callback(
            [Output('current-room-idx', 'data'),
             Output('current-pos-idx', 'data'),
             Output('pos-info', 'children'),
             Output('room-plot', 'figure'),
             Output('experiment-table', 'data'),
             Output('experiment-table', 'columns'),
             Output('room-header', 'children'),
             Output('rt-header', 'children'),
             Output('source-selector', 'options'),
             Output('source-selector', 'value'),
             Output('receiver-selector', 'options'),
             Output('receiver-selector', 'value')],
            [Input('prev-room', 'n_clicks'),
             Input('next-room', 'n_clicks'),
             Input('prev-pos', 'n_clicks'),
             Input('next-pos', 'n_clicks'),
             Input('source-selector', 'value'),
             Input('receiver-selector', 'value')],
            [State('current-room-idx', 'data'),
             State('current-pos-idx', 'data')]
        )
        def update_ui(prev_room_clicks, next_room_clicks, 
                    prev_pos_clicks, next_pos_clicks,
                    source_str, receiver_str,
                    current_room_idx, current_pos_idx):
            ctx = dash.callback_context
            
            # Initialize indices if they're None
            if current_room_idx is None:
                current_room_idx = 0
            if current_pos_idx is None:
                current_pos_idx = 0
            
            new_room_idx = current_room_idx
            new_pos_idx = current_pos_idx
            
            # Get the current room
            room = self.rooms[room_names[new_room_idx]]
            
            # Create a mapping of source-receiver pairs to position indices
            pos_indices = {}
            source_positions = {}
            receiver_positions = {}
            
            for idx, pos_key in enumerate(room.source_mic_pairs):
                source_pos, mic_pos = pos_key
                
                # Format source position string
                source_str_key = f"Source ({source_pos[0]:.1f}, {source_pos[1]:.1f})"
                source_positions[source_str_key] = source_pos
                
                # Format receiver position string
                if len(room.source_mic_pairs) <= 50:
                    receiver_str_key = f"Mic ({mic_pos[0]:.1f}, {mic_pos[1]:.1f})"
                else:
                    receiver_str_key = f"Mic {len(receiver_positions) + 1} ({mic_pos[0]:.1f}, {mic_pos[1]:.1f})"
                receiver_positions[receiver_str_key] = mic_pos
                
                # Store the mapping
                pos_indices[(source_str_key, receiver_str_key)] = idx
            
            # Handle button clicks or dropdown changes
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # Handle room navigation
                if trigger_id in ['prev-room', 'next-room']:
                    if trigger_id == 'prev-room':
                        new_room_idx = (current_room_idx - 1) % len(room_names)
                    else:  # next-room
                        new_room_idx = (current_room_idx + 1) % len(room_names)
                    # Reset position index when changing rooms
                    new_pos_idx = 0
                
                # Handle position navigation buttons
                elif trigger_id in ['prev-pos', 'next-pos']:
                    if room.source_mic_pairs:
                        if trigger_id == 'prev-pos':
                            new_pos_idx = (current_pos_idx - 1) % len(room.source_mic_pairs)
                        else:  # next-pos
                            new_pos_idx = (current_pos_idx + 1) % len(room.source_mic_pairs)
                
                # Handle dropdown selection changes
                elif trigger_id in ['source-selector', 'receiver-selector']:
                    # Only try to update if both dropdowns have values
                    if source_str and receiver_str:
                        pos_key = (source_str, receiver_str)
                        if pos_key in pos_indices:
                            new_pos_idx = pos_indices[pos_key]
            
            # Get updated room (in case room changed)
            room = self.rooms[room_names[new_room_idx]]
            
            # Get active data for the new position
            pos_info = room.get_position_info(new_pos_idx)
            active_experiments = room.get_experiments_for_position(new_pos_idx)
            table_data, columns = self._prepare_table_data(active_experiments)
            
            # Create room visualization with current position highlighted
            room_fig = self.create_room_visualization([room], highlight_pos_idx=new_pos_idx)
            room_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#1e2129",
                plot_bgcolor="#1e2129",
                font={"color": "#e0e0e0"}
            )
            
            # Create header information
            room_header = f"Room: {room.display_name}"
            rt_header = f"Dimensions: {room.dimensions_str}, abs={room.absorption_str}, {room.theoretical_rt_str}"
            
            # Re-create the sources and receivers options based on current room
            sources = []
            receivers = []
            pos_indices = {}
            
            for idx, pos_key in enumerate(room.source_mic_pairs):
                source_pos, mic_pos = pos_key
                
                # Format source position
                source_str_key = f"Source ({source_pos[0]:.1f}, {source_pos[1]:.1f})"
                source_opt = {'label': source_str_key, 'value': source_str_key}
                if source_opt not in sources:
                    sources.append(source_opt)
                
                # Format receiver position
                if len(room.source_mic_pairs) <= 50:
                    receiver_str_key = f"Mic ({mic_pos[0]:.1f}, {mic_pos[1]:.1f})"
                else:
                    # For large number of receivers, use numbered format
                    receiver_count = 0
                    for prev_idx in range(idx):
                        prev_src, prev_mic = room.source_mic_pairs[prev_idx]
                        if tuple(prev_mic) == tuple(mic_pos):
                            receiver_count += 1
                    receiver_str_key = f"Mic {receiver_count + 1} ({mic_pos[0]:.1f}, {mic_pos[1]:.1f})"
                
                receiver_opt = {'label': receiver_str_key, 'value': receiver_str_key}
                if receiver_opt not in receivers:
                    receivers.append(receiver_opt)
                
                # Store the position index mapping
                pos_indices[(source_str_key, receiver_str_key)] = idx
            
            # Store indices for future lookups
            app.server.pos_indices = pos_indices
            
            # Get the current position details for dropdown selection
            if room.source_mic_pairs:
                current_source_pos, current_mic_pos = room.source_mic_pairs[new_pos_idx]
                
                # Find the matching string representations
                current_source = f"Source ({current_source_pos[0]:.1f}, {current_source_pos[1]:.1f})"
                
                if len(room.source_mic_pairs) <= 50:
                    current_receiver = f"Mic ({current_mic_pos[0]:.1f}, {current_mic_pos[1]:.1f})"
                else:
                    # For large number of receivers, need to find the correct mic number
                    mic_count = 0
                    for idx, (src, mic) in enumerate(room.source_mic_pairs):
                        if tuple(mic) == tuple(current_mic_pos):
                            if idx < new_pos_idx:
                                mic_count += 1
                            else:
                                break
                    current_receiver = f"Mic {mic_count + 1} ({current_mic_pos[0]:.1f}, {current_mic_pos[1]:.1f})"
            else:
                current_source = None
                current_receiver = None
            
            return (
                new_room_idx,
                new_pos_idx,
                pos_info,
                room_fig,
                table_data,
                columns,
                room_header,
                rt_header,
                sources,
                current_source,
                receivers,
                current_receiver
            )

        # Add a callback to update all three plots simultaneously based on time range selection
        @app.callback(
            [Output('rir-plot', 'figure'),
             Output('edc-plot', 'figure'),
             Output('ned-plot', 'figure')],
            [Input('current-room-idx', 'data'),
             Input('current-pos-idx', 'data'),
             Input('time-range-selector', 'value')]
        )
        def update_all_plots(room_idx, pos_idx, time_range):
            if room_idx is None or pos_idx is None:
                # Return empty figures if data is not available
                return go.Figure(), go.Figure(), go.Figure()
            
            room = self.rooms[room_names[room_idx]]
            active_experiments = room.get_experiments_for_position(pos_idx)
            
            # Function to set the x-axis range based on selected time range
            def set_time_range(fig, max_time=None):
                if time_range != 'full':
                    fig.update_xaxes(range=[0, float(time_range)])
                elif max_time is not None:
                    fig.update_xaxes(range=[0, max_time])
                return fig
            
            # Create RIR plot
            rir_fig = go.Figure()
            max_time = 0
            
            # Number the experiments for this position
            for idx, exp in enumerate(active_experiments, 1):
                exp.display_id = idx  # Store the ID on the experiment
                method = exp.config.get('method', 'SDN')
                
                # Create enhanced legend name with ID and method
                legend_name = f"{idx}: {method} {exp.get_label()}"
                
                rir_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.rir,
                    mode='lines',
                    name=legend_name
                ))
                max_time = max(max_time, exp.time_axis[-1] if len(exp.time_axis) > 0 else 0)
            
            rir_fig.update_layout(
                title="Room Impulse Responses",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                template="plotly_dark",
                paper_bgcolor="#1e2129",
                plot_bgcolor="#1e2129",
                font={"color": "#e0e0e0"}
            )
            
            # Create EDC plot
            edc_fig = go.Figure()
            
            for exp in active_experiments:
                method = exp.config.get('method', 'SDN')
                # Use the same legend naming format for consistency
                legend_name = f"{exp.display_id}: {method} {exp.get_label()}"
                
                edc_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.edc,
                    mode='lines',
                    name=legend_name
                ))
            
            edc_fig.update_layout(
                title="Energy Decay Curves",
                xaxis_title="Time (s)",
                yaxis_title="Energy (dB)",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                template="plotly_dark",
                paper_bgcolor="#1e2129",
                plot_bgcolor="#1e2129",
                font={"color": "#e0e0e0"}
            )
            
            # Create NED plot
            ned_fig = go.Figure()
            
            for exp in active_experiments:
                method = exp.config.get('method', 'SDN')
                # Use the same legend naming format for consistency
                legend_name = f"{exp.display_id}: {method} {exp.get_label()}"
                
                ned_fig.add_trace(go.Scatter(
                    x=exp.ned_time_axis,
                    y=exp.ned,
                    mode='lines',
                    name=legend_name
                ))
            
            ned_fig.update_layout(
                title="Normalized Echo Density",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Echo Density",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                template="plotly_dark",
                paper_bgcolor="#1e2129",
                plot_bgcolor="#1e2129",
                font={"color": "#e0e0e0"}
            )
            
            # Apply time range to all plots
            set_time_range(rir_fig, max_time)
            set_time_range(edc_fig, max_time)
            set_time_range(ned_fig, max_time)
            
            return rir_fig, edc_fig, ned_fig
        
        # Open browser automatically
        def open_browser():
            webbrowser.open_new(f"http://127.0.0.1:{port}/")
        
        Timer(1, open_browser).start()
        
        # Print message to console
        server_address = f"http://127.0.0.1:{port}/"
        print("\n" + "=" * 70)
        print(f"Dash server is running at: {server_address}")
        print("If the browser doesn't open automatically, please copy and paste the URL above.")
        print("=" * 70)
        
        # Run the app
        app.run_server(debug=True, port=port)


    def run_ism_experiment(self, room_parameters, max_order=12, ray_tracing=False, use_rand_ism=False, duration=0.5, fs=44100, force_rerun=False, room_name=None, label="ISM"):
        """
        Run an Image Source Method (ISM) experiment.
        
        Args:
            room_parameters (dict): Room parameters
            max_order (int): Maximum reflection order
            ray_tracing (bool): Whether to use ray tracing
            use_rand_ism (bool): Whether to use randomized ISM
            duration (float): Duration of the simulation in seconds
            fs (int): Sampling frequency
            force_rerun (bool): If True, rerun the experiment even if it exists
            room_name (str, optional): Explicit name for the room
            label (str): Label for the experiment
            
        Returns:
            SDNExperiment: The experiment object
        """
        # Create config
        config = {
            'label': label,
            'info': "",
            'max_order': max_order,
            'ray_tracing': ray_tracing,
            'use_rand_ism': use_rand_ism,
        }
        
        # Run the experiment
        return self.run_experiment(
            config=config,
            room_parameters=room_parameters,
            duration=duration,
            fs=fs,
            force_rerun=force_rerun,
            room_name=room_name,
            method='ISM'
        )

    def _get_column_config(self):
        """
        Define the configuration for table columns.
        
        This single source of truth controls:
        1. Which columns are displayed
        2. The order of columns
        3. The display names of columns
        4. Any custom formatting for values
        
        Returns:
            list: List of column configurations in display order
        """
        return [
            # ID column (leftmost)
            {
                'id': '#',
                'display': True,
                'name': '#',
                'width': '30px',
            },
            # Core columns
            {
                'id': 'Method',
                'display': True,
                'name': 'Method',
                'width': '70px',
            },
            {
                'id': 'Label',
                'display': True,
                'name': 'Label',
                'width': '180px',
            },

            
            # SDN parameters
            {
                'id': 'sdn_source_weighting',
                'display': True,
                'name': 'SDN: source_weighting',
                'width': '100px',
            },
            {
                'id': 'sdn_specular_source_injection',
                'display': True,
                'name': 'SDN: specular_source_injection',
                'width': '120px',
            },
            {
                'id': 'sdn_scattering_matrix_update_coef',
                'display': True,
                'name': 'SDN: scattering_matrix_update_coef',
                'width': '120px',
            },
            {
                'id': 'sdn_coef',
                'display': True,
                'name': 'SDN: coef',
                'width': '100px',
            },
            {
                'id': 'sdn_source_pressure_injection_coeff',
                'display': True,
                'name': 'SDN: source_pressure_injection_coeff',
                'width': '120px',
            },
            {
                'id': 'sdn_specular_scattering',
                'display': True,
                'name': 'SDN: specular_scattering',
                'width': '100px',
            },
            
            # ISM parameters
            {
                'id': 'ism_max_order',
                'display': False,  # Hidden by default as it's in the label
                'name': 'ISM: max_order',
                'width': '100px',
            },
            {
                'id': 'ism_ray_tracing',
                'display': True,
                'name': 'ISM: ray_tracing',
                'width': '100px',
            },
            {
                'id': 'ism_use_rand_ism',
                'display': False,
                'name': 'ISM: use_rand_ism',
                'width': '100px',
            },

            # Metrics
            {
                'id': 'rt60',
                'display': True,
                'name': 'Metric: rt60',
                'format': lambda x: f"{x:.3f}" if x is not None else "",
                'width': '100px',
            },

            # duration
            {
                'id': 'Duration',
                'display': True,
                'name': 'Duration (s)',
                'width': '80px',
            },
        ]

    def _prepare_table_data(self, active_experiments):
        """
        Prepare data for the DataTable based on column configuration.
        
        Args:
            active_experiments (list): List of experiments to include in the table
            
        Returns:
            tuple: (table_data, columns) for the DataTable
        """
        # Get the column configuration
        column_config = {col['id']: col for col in self._get_column_config()}
        
        # Initialize table data
        table_data = []
        all_column_ids = set(['#'])  # Ensure ID column is always included
        
        # Process each experiment
        for idx, exp in enumerate(active_experiments, 1):
            method = exp.config.get('method', 'SDN')
            row = {}
            
            # Add row ID (1, 2, 3, ...)
            row['#'] = idx
            
            # Store the ID on the experiment object for reference in plots
            exp.display_id = idx
            
            # 1. Add core columns
            row['Method'] = method
            row['Label'] = exp.get_label()
            row['Duration'] = exp.duration
            
            # 2. Add metrics
            for metric, value in exp.metrics.items():
                # Use column_id directly as the metric name
                column_id = metric
                if column_id in column_config and 'format' in column_config[column_id]:
                    row[column_id] = column_config[column_id]['format'](value)
                else:
                    row[column_id] = f"{value:.3f}" if value is not None else ""
                all_column_ids.add(column_id)
            
            # 3. Add method-specific parameters
            if method == 'SDN' and 'flags' in exp.config:
                flags = exp.config['flags']
                for key, value in flags.items():
                    # Use standardized column_id format
                    column_id = f"sdn_{key}"
                    row[column_id] = str(value)
                    all_column_ids.add(column_id)
            
            elif method == 'ISM':
                ism_params = {
                    'max_order': 'ism_max_order', 
                    'ray_tracing': 'ism_ray_tracing', 
                    'use_rand_ism': 'ism_use_rand_ism'
                }
                for param, column_id in ism_params.items():
                    row[column_id] = str(exp.config.get(param, False))
                    all_column_ids.add(column_id)
            
            table_data.append(row)
        
        # Ensure all rows have entries for all columns used
        for row in table_data:
            for col_id in all_column_ids:
                if col_id not in row:
                    row[col_id] = ""
        
        # Build column definitions for Dash
        dash_columns = []
        
        # Loop through configured columns in their specified order
        for col_config in self._get_column_config():
            col_id = col_config['id']
            
            # Only include the column if:
            # 1. It's marked for display in the config, AND
            # 2. It actually appears in the data or is a core column
            if col_config['display'] and (col_id in all_column_ids or col_id in ['#', 'Method', 'Label', 'Duration']):
                dash_columns.append({
                    "name": col_config['name'],
                    "id": col_id
                })
        
        # Handle any dynamic columns that weren't in the config but are in the data
        # Sort these alphabetically after the configured columns
        for col_id in sorted(all_column_ids):
            if col_id not in column_config:
                dash_columns.append({
                    "name": col_id.replace('_', ' ').title(),
                    "id": col_id
                })
        
        return table_data, dash_columns

def run_main_experiments():
    """Run experiments from main.py configurations."""
    # Import main module to access configurations
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import main
    
    # Create experiment manager
    manager = SDNExperimentManager()
    
    # Determine which predefined room we're using
    room_name = None
    if hasattr(main, 'room_parameters'):
        # Check if it matches a known room
        params = main.room_parameters
        if params == getattr(main, 'room_waspaa', None):
            room_name = "room_waspaa"
        elif params == getattr(main, 'room_aes', None):
            room_name = "room_aes"
        elif params == getattr(main, 'room_journal', None):
            room_name = "room_journal"
    
    # Run ISM experiments if enabled
    if hasattr(main, 'PLOT_ISM_with_pra') and main.PLOT_ISM_with_pra:
        print("Running ISM experiment using pyroomacoustics")
        max_order = getattr(main, 'pra_order', 12)
        ray_tracing = getattr(main, 'ray_tracing_flag', False)
        use_rand_ism = getattr(main, 'use_rand_ism', False)
        manager.run_ism_experiment(
            room_parameters=main.room_parameters,
            max_order=max_order,
            ray_tracing=ray_tracing,
            use_rand_ism=use_rand_ism,
            duration=getattr(main, 'duration', 0.5),
            fs=getattr(main, 'Fs', 44100),
            room_name=room_name
        )
    
    # Run SDN experiments from main.py configurations
    if hasattr(main, 'sdn_tests') and hasattr(main, 'room_parameters'):
        for test_name, config in main.sdn_tests.items():
            if config.get('enabled', False):
                print(f"Running SDN experiment: {test_name}")
                # Ensure method is set to SDN
                config['method'] = 'SDN'
                manager.run_experiment(config, main.room_parameters, room_name=room_name)
    
    return manager

if __name__ == "__main__":
    # Method 1: Run experiments directly
    manager = SDNExperimentManager()
    experiments = manager.get_all_experiments()
    duration = 1

    # Example room parameters
    room_aes = {'width': 9, 'depth': 7, 'height': 4,
                'source x': 4.5, 'source y': 3.5, 'source z': 2,
                'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                'absorption': 0.2}

    room = room_aes

    
    # Run an SDN experiment
    # single source and receiver
    manager.run_experiment(
        config={
            'label': 'weighted psk',
            'info': '',
            'method': 'SDN',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 5,
            }
        },
        room_parameters=room_aes,
        duration=duration,
        fs=44100,
        room_name="room_aes"
    )

    # Method 2: Run experiments from main.py (recommended)
    # print("\nRunning experiments from main.py configuration...")
    # main_manager = run_main_experiments()
    # Optionally, visualize the main experiments
    # print("\nLaunching visualization of main.py experiments...")
    # main_manager.plot()

    # Generate source & receiver positions
    receiver_positions = sa.generate_receiver_grid(room['width'], room['depth'], 5)
    source_positions = sa.generate_source_positions(room)

    # Run the experiments in batch mode
    experiment_ids = manager.run_experiment(
    config={
        'label': 'weighted psk',
        'info': '',
        'method': 'SDN',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 5,
        }
    },
    room_parameters=room_aes,
    duration=0.5,  # Shorter duration for batch processing
    fs=44100,
    room_name="room_aes",
    batch_processing=True,  # Enable batch processing
    source_positions=source_positions,  # Provide source positions
    receiver_positions=receiver_positions  # Provide receiver positions
)

    print(f"Completed {len(experiment_ids)} experiments")

    # Launch visualization of direct experiments
    print("\nLaunching visualization of direct experiments...")
    manager.plot()