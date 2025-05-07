import os
import json
import numpy as np
from datetime import datetime
import hashlib
# import sys
# import spatial_analysis as sa
# from rir_calculators import rir_normalisation
# Import modules for core functionality
# import geometry
# import plot_room as pp
import EchoDensity as ned
import analysis as an

# Set matplotlib backend to match main script
# import matplotlib
# matplotlib.use('Qt5Agg')  # Set the backend to Qt5

# Note: Visualization functionality has been moved to sdn_experiment_visualizer.py
# For visualization, import and use the SDNExperimentVisualizer class

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
        # if experiment.experiment_id in self.experiments:
        #     # Just update the experiment if it already exists
        #     self.experiments[experiment.experiment_id] = experiment
        #     # Update in position-based dictionary if it exists there
        #     for pos_list in self.experiments_by_position.values():
        #         for i, exp in enumerate(pos_list):
        #             if exp.experiment_id == experiment.experiment_id:
        #                 pos_list[i] = experiment
        #     return
            
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
        rt60_sabine, rt60_eyring = an.calculate_rt60_theoretical(room_dim, self.parameters['absorption'])
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
    
    def __init__(self, config, rir, fs=44100, duration=None, experiment_id=None, skip_metrics=True):
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
                
        # Add HO-SDN specific parameters
        elif config.get('method') == 'HO-SDN':
            if 'order' in config:
                id_config['order'] = config['order']
            if 'source_signal' in config:
                id_config['source_signal'] = config['source_signal']
                
        # Add rimpy specific parameters
        elif config.get('method') == 'RIMPY':
            reflection_sign = config.get('reflection_sign')  # Default to positive
            id_config['reflection_sign'] = reflection_sign
                
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
            self.metrics['rt60'] = an.calculate_rt60_from_rir(self.rir, self.fs, plot=False)
        
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
        elif method == 'SDN':
            # Add SDN-specific flags that affect the simulation
            if 'flags' in self.config:
                flags = self.config['flags']
                if flags.get('source_weighting'):
                    label += f" {flags['source_weighting']}"
                if flags.get('specular_source_injection'):
                    label += " specular"
                if 'source_pressure_injection_coeff' in flags:
                    label += f" src constant coef={flags['source_pressure_injection_coeff']}"
                if 'scattering_matrix_update_coef' in flags:
                    label += f" scat={flags['scattering_matrix_update_coef']}"
        
        # Create label for legend with method included
        label_for_legend = f"{method}, {label}"
        
        # Return dictionary with both label versions
        return {
            "label": label,
            "label_for_legend": label_for_legend
        }
    
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
    """Class to manage multiple acoustic simulation experiments and store results."""
    
    def __init__(self, results_dir='results', is_batch_manager=False, dont_check_duplicates=False):
        """
        Initialize the experiment manager.
        
        Args:
            results_dir (str): Base directory to store experiment data. Can be customized
                             to separate different sets of experiments.
                              
            is_batch_manager (bool): If True, this manager handles batch processing experiments
                                   with multiple source/receiver positions.
            
            dont_check_duplicates (bool): If True, skip loading existing experiments.
                                        This can significantly speed up initialization
                                        when you don't need to check for duplicates.
        
        Directory Structure:
            When is_batch_manager=False (singular):
                {results_dir}/room_singulars/{project_name}/{experiment_id}.json
                {results_dir}/room_singulars/{project_name}/{experiment_id}.npy
                
            When is_batch_manager=True (batch):
                {results_dir}/rooms/{project_name}/{source_label}/{method}/{param_set}/config.json
                {results_dir}/rooms/{project_name}/{source_label}/{method}/{param_set}/rirs.npy
                
        Example:
            # Create a manager for singular experiments in custom directory
            singular_mgr = SDNExperimentManager(results_dir='custom_results', is_batch_manager=False)
            
            # Create a manager for batch experiments in default directory
            batch_mgr = SDNExperimentManager(results_dir='results', is_batch_manager=True)
        """
        self.results_dir = results_dir
        self.is_batch_manager = is_batch_manager
        self.projects = {}  # project_name -> Room (acoustic room object)
        self.ensure_dir_exists()
        
        # Only load experiments if not skipping duplicate checks
        if not dont_check_duplicates:
            # self.load_experiments()
            print("load_experiment() is removed. cant check the duplicates. retrieve from the previous commit if you want")
        else:
            print("Skipping experiment loading (dont_check_duplicates=True)")

    def _get_results_dir(self):
        """Get the base results directory."""
        return self.results_dir
        
    def ensure_dir_exists(self):
        """Ensure the results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
        # Ensure the singulars directory exists if this is not a batch manager
        if not self.is_batch_manager:
            os.makedirs(os.path.join(self.results_dir, 'room_singulars'), exist_ok=True)
    
    def _get_room_name(self, room_parameters):
        """Generate a unique room name based on parameters."""
        # Create a hash of room dimensions and absorption
        room_key = {k: room_parameters[k] for k in ['width', 'depth', 'height', 'absorption']}
        room_str = json.dumps(room_key, sort_keys=True)
        room_hash = hashlib.md5(room_str.encode()).hexdigest()[:6]
        return f"room_{room_hash}"
    
    def _get_room_dir(self, project_name):
        """
        Get the directory path for a room based on experiment type.
        
        Directory structure:
        - Singular experiments: {results_dir}/room_singulars/{room_name}/
          All files are stored directly in this directory.
          
        - Batch experiments: {results_dir}/rooms/{room_name}/
          Further organized by source/method/params structure:
          {results_dir}/rooms/{room_name}/{source_label}/{method}/{param_set}/
        
        Args:
            room_name (str): The name of the room
            
        Returns:
            str: The path to the room directory
        """
        if not self.is_batch_manager:
            # For singular experiments, use the room_singulars folder
            return os.path.join(self.results_dir, 'room_singulars', project_name)
        else:
            # For batch experiments, use the normal structure
            return os.path.join(self.results_dir, 'rooms', project_name)
    
    def _get_source_dir(self, project_name, source_label):
        """Get the directory path for a source within a room."""
        room_dir = self._get_room_dir(project_name)
        return os.path.join(room_dir, source_label)  
    
    def _get_simulation_dir(self, project_name, source_label, method, param_set):
        """Get the directory path for a simulation within a source."""
        source_dir = self._get_source_dir(project_name, source_label)
        return os.path.join(source_dir, method, param_set)
    
    def _get_source_label_from_pos(self, source_pos):
        """Generate a standardized label for a source position."""
        return f"source_{source_pos[0]}_{source_pos[1]}_{source_pos[2]}"
    
    def _get_mic_label_from_pos(self, mic_pos):
        """Generate a standardized label for a microphone position."""
        return f"mic_{mic_pos[0]}_{mic_pos[1]}_{mic_pos[2]}"
    
    def _generate_param_set_name(self, config):
        """Generate a standardized name for the parameter set based on the config."""
        method = config.get('method')
        param_set = ""
        
        if method == 'SDN':
            # Include key SDN parameters in the name
            flags = config.get('flags', {})
            #if flags.get('specular_source_injection', False):
                #param_set += "specular_"
            if 'source_weighting' in flags:
                param_set += f"sw{flags['source_weighting']}_"
            if 'scattering_matrix_update_coef' in flags:
                param_set += f"smu{flags['scattering_matrix_update_coef']}_"
            if 'source_pressure_injection_coeff' in flags:
                param_set += f"src{flags['source_pressure_injection_coeff']}_"
            
            # Remove trailing underscore
            param_set = param_set.rstrip('_')
            if not param_set:
                param_set = "original"
                
        elif method == 'ISM':
            # Include key ISM parameters in the name
            max_order = config.get('max_order', 12)
            param_set = f"order{max_order}"
            if config.get('ray_tracing', False):
                param_set += "_rt"
            if config.get('use_rand_ism', False):
                param_set += "_rand"
                
        elif method == 'TRE' or method == 'treble':
            # Include key Treble parameters in the name
            param_set = "hybrid"
            if 'max_order' in config:
                param_set += f"_order{config['max_order']}"
                
        elif method == 'HO-SDN':
            # Include key HO-SDN parameters in the name
            param_set = "ho-sdn"
            if 'order' in config:
                param_set += f"_order{config['order']}"
            if 'source_signal' in config and config['source_signal'] != 'dirac':
                param_set += f"_{config['source_signal']}"
                
        elif method == 'RIMPY':
            # Include key rimpy parameters in the name
            param_set = ""  # Don't include method name here, it will be added in save_experiment
            reflection_sign = config.get('reflection_sign')  # Default to positive
            if reflection_sign < 0:
                param_set += "negref"
            else:
                param_set += "posref"
                
        else:
            param_set = "unknown method"
            
        return param_set
    
    def create_experiment_config(self, method, label="", info="", **params):
        """
        LEGACY METHOD: Create an experiment configuration for run_with_config.
        
        It's recommended to create config dictionaries directly instead.
        See the docstring of run_experiment() for the expected config structure.
        
        Example of direct dictionary creation (preferred):
        ```python
        config = {
            'method': 'SDN',
            'label': 'weighted psk',
            'info': '',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 4
            }
        }
        ```
        
        Args:
            method (str): Simulation method ('SDN', 'ISM', 'HO-SDN', 'RIMPY', etc.)
            label (str): Label for the experiment
            info (str): Additional information about the experiment
            **params: Method-specific parameters:
                For 'SDN': specular_source_injection, source_weighting, etc.
                For 'ISM': max_order, ray_tracing, use_rand_ism
                For 'HO-SDN': order, source_signal
                For 'RIMPY': reflection_sign
                
        Returns:
            dict: Configuration dictionary ready for run_experiment
        """
        config = {
            'method': method,
            'label': label,
            'info': info
        }
        
        # Add method-specific parameters
        if method == 'SDN':
            # SDN params go into flags
            flags = {}
            for param, value in params.items():
                if param not in ['room_parameters', 'duration', 'fs']:
                    flags[param] = value
            config['flags'] = flags
                
        elif method == 'ISM':
            # ISM params go directly in config
            for param in ['max_order', 'ray_tracing', 'use_rand_ism']:
                if param in params:
                    config[param] = params[param]
                    
        elif method == 'HO-SDN':
            # HO-SDN params go directly in config
            for param in ['order', 'source_signal']:
                if param in params:
                    config[param] = params[param]
                    
        elif method == 'RIMPY':
            # RIMPY params go directly in config
            if 'reflection_sign' in params:
                config['reflection_sign'] = params['reflection_sign']
                
        return config

    def get_experiment(self, experiment_id):
        """
        Get an experiment by ID.
        
        Args:
            experiment_id (str): The ID of the experiment
            
        Returns:
            SDNExperiment: The experiment object
        """
        for project in self.projects.values():
            if experiment_id in project.experiments:
                return project.experiments[experiment_id]
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
        for project in self.projects.values():
            experiments.extend([
                exp for exp in project.experiments.values() 
                if label in exp.get_label()['label'] or label in exp.get_label()['label_for_legend']
            ])
        return experiments
    
    def get_all_experiments(self):
        """
        Get all experiments.
        
        Returns:
            list: List of all experiments
        """
        experiments = []
        for project in self.projects.values():
            experiments.extend(list(project.experiments.values()))
        return experiments

# Replace the two separate functions with a unified function
def get_experiment_manager(is_batch=False, results_dir='results', dont_check_duplicates=False):
    """
    Create an experiment manager for either batch or singular experiments.
    This unified function replaces the separate batch and singular manager functions.
    
    Args:
        is_batch (bool): If True, create a batch manager; otherwise a singular manager
        results_dir (str): Base directory to store experiment data
                         For batch: {results_dir}/rooms/
                         For singular: {results_dir}/room_singulars/
        dont_check_duplicates (bool): If True, skip loading existing experiments
                                    
    Returns:
        SDNExperimentManager: The experiment manager instance
    """
    manager = SDNExperimentManager(
        results_dir=results_dir,
        is_batch_manager=is_batch,
        dont_check_duplicates=dont_check_duplicates
    )
    return manager
