import os
import json
import numpy as np
from datetime import datetime
import hashlib
import sys
import spatial_analysis as sa

# Import modules for core functionality
import geometry
import plot_room as pp
import EchoDensity as ned
import analysis as an
from sdn_core import DelayNetwork

# Set matplotlib backend to match main script
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5

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
                {results_dir}/room_singulars/{room_name}/{experiment_id}.json
                {results_dir}/room_singulars/{room_name}/{experiment_id}.npy
                
            When is_batch_manager=True (batch):
                {results_dir}/rooms/{room_name}/{source_label}/{method}/{param_set}/config.json
                {results_dir}/rooms/{room_name}/{source_label}/{method}/{param_set}/rirs.npy
                
        Example:
            # Create a manager for singular experiments in custom directory
            singular_mgr = SDNExperimentManager(results_dir='custom_results', is_batch_manager=False)
            
            # Create a manager for batch experiments in default directory
            batch_mgr = SDNExperimentManager(results_dir='results', is_batch_manager=True)
        """
        self.results_dir = results_dir
        self.is_batch_manager = is_batch_manager
        self.rooms = {}  # name -> Room
        self.ensure_dir_exists()
        
        # Only load experiments if not skipping duplicate checks
        if not dont_check_duplicates:
            self.load_experiments()
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
        method = config.get('method', 'SDN')
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
                
        else:
            param_set = "unknown method"
            
        return param_set

    def load_experiments(self):
        """Load all experiments from the results directory."""
        self.rooms = {}
        
        if not self.is_batch_manager:
            # Singular case: Load from room_singulars folder
            singulars_dir = os.path.join(self.results_dir, 'room_singulars')
            if os.path.exists(singulars_dir):
                for project_name in os.listdir(singulars_dir):
                    room_path = os.path.join(singulars_dir, project_name)
                    if not os.path.isdir(room_path):
                        continue
                        
                    try:
                        # Load room info
                        room_info_path = os.path.join(room_path, 'room_info.json')
                        if os.path.exists(room_info_path):
                            with open(room_info_path, 'r') as f:
                                room_info = json.load(f)
                            
                            # Create room with saved parameters
                            room_parameters = room_info.get('parameters', {})
                            room = Room(room_info.get('name', project_name), room_parameters)
                            if 'display_name' in room_info:
                                room.display_name = room_info['display_name']
                            
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
                                            # Only add experiment if successfully loaded
                                            if experiment and hasattr(experiment, 'experiment_id'):
                                                room.add_experiment(experiment)
                                            else:
                                                print(f"Skipping experiment {experiment_id}: Invalid experiment object")
                                    except Exception as e:
                                        print(f"Error loading experiment {experiment_id}: {e}")
                                        continue  # Skip this experiment and continue with the next
                            
                            # Only add room if it has valid experiments
                            if room.experiments:
                                self.rooms[room.name] = room
                            
                    except Exception as e:
                        print(f"Error loading room {project_name}: {e}")
                        continue  # Skip this room and continue with the next
        else:
            # Load batch experiments from the structured directory
            rooms_dir = os.path.join(self.results_dir, 'rooms')
            if not os.path.exists(rooms_dir):
                return
            
            # Iterate through room directories
            for project_name in os.listdir(rooms_dir):
                room_path = os.path.join(rooms_dir, project_name)
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
                        room = Room(room_info.get('name', project_name), room_parameters)
                        if 'display_name' in room_info:
                            room.display_name = room_info['display_name']
                        
                        # Load sources directly from room directory
                        for source_label in os.listdir(room_path):
                            source_path = os.path.join(room_path, source_label)
                            
                            # Skip non-directories and special files/directories
                            if not os.path.isdir(source_path) or source_label == 'room_info.json':
                                continue
                            
                            # Method folders directly under source folder (no simulations level)
                            for method in os.listdir(source_path):
                                method_dir = os.path.join(source_path, method)
                                if not os.path.isdir(method_dir):
                                    continue
                                
                                for param_set in os.listdir(method_dir):
                                    param_dir = os.path.join(method_dir, param_set)
                                    if not os.path.isdir(param_dir):
                                        continue
                                    
                                    # Load configuration
                                    config_path = os.path.join(param_dir, 'config.json')
                                    if not os.path.exists(config_path):
                                        continue
                                        
                                    try:
                                        with open(config_path, 'r') as f:
                                            config = json.load(f)
                                        
                                        # Load RIRs
                                        rirs_path = os.path.join(param_dir, 'rirs.npy')
                                        if not os.path.exists(rirs_path):
                                            continue
                                            
                                        rirs = np.load(rirs_path)
                                        
                                        # Get receivers from config.json
                                        receivers_data = config.get('receivers', [])
                                        
                                        # Create experiment objects for each source-receiver pair
                                        for idx, receiver_info in enumerate(receivers_data):
                                            if idx >= len(rirs):
                                                break
                                                
                                            # Create a single experiment config
                                            receiver_config = config.copy()
                                            # Remove the receivers array
                                            if 'receivers' in receiver_config:
                                                del receiver_config['receivers']
                                            # Add the specific receiver info
                                            receiver_config['receiver'] = receiver_info
                                            
                                            # Use source info from config
                                            source_info = config.get('source', {})
                                            
                                            # Create experiment object
                                            experiment = SDNExperiment(
                                                config=receiver_config,
                                                rir=rirs[idx],
                                                fs=config.get('fs', 44100),
                                                duration=config.get('duration', 0.5),
                                                experiment_id=receiver_info.get('experiment_id')
                                            )
                                            
                                            # Add experiment to room
                                            if experiment and hasattr(experiment, 'experiment_id'):
                                                room.add_experiment(experiment)
                                            else:
                                                print(f"Skipping experiment in {param_dir}: Invalid experiment object")
                                    except Exception as e:
                                        print(f"Error loading simulation {param_set} for {source_label}: {e}")
                        
                        # Only add room if it has valid experiments
                        if room.experiments:
                            self.rooms[room.name] = room
                        
                    except Exception as e:
                        print(f"Error loading room {project_name}: {e}")
                        continue  # Skip this room and continue with the next
    
    def run_experiment(self, config, room_parameters, duration, fs=44100,
                      force_rerun=False, project_name=None, method='SDN',
                      batch_processing=False, source_positions=None, receiver_positions=None):
        """
        Run an acoustic simulation experiment with the given configuration.
        """
        if project_name is None:
            print("Generating project name based on method and parameters because project_name is None")
            project_name = self._generate_project_name(config, method)
            print("Generated project name:", project_name)
        
        # Add method to config if not present
        if 'method' not in config:
            config['method'] = method
        
        # Handle batch processing case
        if batch_processing:
            # Check if this manager is set up for batch processing
            if not self.is_batch_manager:
                # Create or get a batch manager
                batch_manager = get_batch_manager()
                # Delegate to batch manager
                return batch_manager.run_experiment(
                    config=config,
                    room_parameters=room_parameters,
                    duration=duration,
                    fs=fs,
                    force_rerun=force_rerun,
                    project_name=project_name,
                    method=method,
                    batch_processing=True,
                    source_positions=source_positions,
                    receiver_positions=receiver_positions
                )
            
            # Validate batch processing inputs
            if source_positions is None or receiver_positions is None:
                raise ValueError("Both source_positions and receiver_positions must be provided for batch processing")
            
            # Load and validate room info once at the start
            if 'aes' in project_name.lower():
                room_info_file = 'room_info_aes.json'
            elif 'journal' in project_name.lower():
                room_info_file = 'room_info_journal.json'
            elif 'waspaa' in project_name.lower():
                room_info_file = 'room_info_waspaa.json'
            else:
                raise ValueError(f"Project name '{project_name}' must contain 'aes', 'journal', or 'waspaa' to determine room type")

            # Load room info from unified location
            rooms_dir = os.path.join(self.results_dir, 'rooms')
            room_info_path = os.path.join(rooms_dir, room_info_file)
            if not os.path.exists(room_info_path):
                raise FileNotFoundError(f"Room info file not found: {room_info_path}")

            try:
                with open(room_info_path, 'r') as f:
                    room_info = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading room info from {room_info_path}: {e}")

            # Verify room parameters match
            for key in ['width', 'depth', 'height', 'absorption']:
                if abs(room_info['parameters'][key] - room_parameters[key]) > 1e-6:
                    raise ValueError(f"Room parameter '{key}' mismatch. Expected {room_info['parameters'][key]}, got {room_parameters[key]}")

            # Get or create room once
            if project_name not in self.rooms:
                room = Room(project_name, room_parameters)
                self.rooms[project_name] = room
            else:
                room = self.rooms[project_name]
            
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
                            project_name=project_name,
                            method=method,
                            batch_processing=False  # Important: prevent recursion
                        )
                        
                        experiment_ids.append(experiment.experiment_id)
                    except Exception as e:
                        print(f"    Error: {str(e)}")
            
            print(f"\nCompleted {len(experiment_ids)}/{total_combinations} experiments")
            return experiment_ids
        
        else:
            # Single experiment case
            print("single position run")
            # Determine room type from project name
            if 'aes' in project_name.lower():
                room_info_file = 'room_info_aes.json'
            elif 'journal' in project_name.lower():
                room_info_file = 'room_info_journal.json'
            elif 'waspaa' in project_name.lower():
                room_info_file = 'room_info_waspaa.json'
            else:
                raise ValueError(f"Project name '{project_name}' must contain 'aes', 'journal', or 'waspaa' to determine room type")

            # Load room info from unified location
            rooms_dir = os.path.join(self.results_dir, 'rooms')
            room_info_path = os.path.join(rooms_dir, room_info_file)
            if not os.path.exists(room_info_path):
                raise FileNotFoundError(f"Room info file not found: {room_info_path}")

            try:
                with open(room_info_path, 'r') as f:
                    room_info = json.load(f)
            except Exception as e:
                raise Exception(f"Error loading room info from {room_info_path}: {e}")

            # Verify room parameters match
            for key in ['width', 'depth', 'height', 'absorption']:
                if abs(room_info['parameters'][key] - room_parameters[key]) > 1e-6:
                    raise ValueError(f"Room parameter '{key}' mismatch. Expected {room_info['parameters'][key]}, got {room_parameters[key]}")

            # Get or create room
            if project_name not in self.rooms:
                room = Room(project_name, room_parameters)
                self.rooms[project_name] = room
            else:
                room = self.rooms[project_name]
            
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
            print(" num_samples: ", num_samples)
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
                if len(pra_rir) < num_samples:
                    # Pad with zeros to reach num_samples
                    rir = np.pad(pra_rir, (0, num_samples - len(pra_rir)))
                else:
                    # Truncate if longer
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
            self.save_experiment(experiment, project_name)
            
            # Add to room's experiments
            room.add_experiment(experiment)
            
            return experiment
    
    def save_experiment(self, experiment, project_name):
        """
        Save an experiment to disk.
        
        Args:
            experiment (SDNExperiment): The experiment to save
            project_name (str): Name of the project/experiment group (e.g. 'aes_abs20_comparison')
        """
        if not self.is_batch_manager:
            # Save using the old singular format
            room_dir = self._get_room_dir(project_name)
            os.makedirs(room_dir, exist_ok=True)
            
            # Generate descriptive filename based on experiment parameters
            method = experiment.config.get('method')
            filename = self._generate_param_set_name(experiment.config)
            
            # Add method prefix for singular case
            if method == 'SDN':
                filename = f"sdn_{filename}"
            else:
                filename = f"{method.lower()}"
                
            # Save metadata with descriptive filename
            metadata_path = os.path.join(room_dir, f"{filename}.json")
            with open(metadata_path, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)
            
            # Save RIR
            rir_path = os.path.join(room_dir, f"{filename}.npy")
            np.save(rir_path, experiment.rir)
        else:
            # Batch case: Save in structured directory
            source_label = experiment.config.get('position_id', '').split('_to_mic_')[0]
            method = experiment.config.get('method', 'SDN')
            param_set = self._generate_param_set_name(experiment.config)
            
            simulation_dir = self._get_simulation_dir(project_name, source_label, method, param_set)
            os.makedirs(simulation_dir, exist_ok=True)
            
            config_path = os.path.join(simulation_dir, 'config.json')
            rirs_path = os.path.join(simulation_dir, 'rirs.npy')
            
            # Load existing config and RIRs if they exist
            receivers = []
            rirs = np.array([])
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                    receivers = existing_config.get('receivers', [])
                if os.path.exists(rirs_path):
                    rirs = np.load(rirs_path)
            
            # Update config and save
            config_data = experiment.config.copy()
            config_data['receivers'] = receivers
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Save RIRs
            np.save(rirs_path, rirs)

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

    def run_ism_experiment(self, room_parameters, max_order=12, ray_tracing=False, use_rand_ism=False, duration=0.5, fs=44100, force_rerun=False, project_name=None, label="ISM", batch_processing=False, source_positions=None, receiver_positions=None):
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
            project_name (str): Name of the project/experiment group (e.g. 'aes_abs20_comparison')
            label (str): Label for the experiment
            batch_processing (bool): If True, run as batch processing with multiple source/receiver positions
            source_positions (list, optional): List of source positions for batch processing
            receiver_positions (list, optional): List of receiver positions for batch processing
            
        Returns:
            SDNExperiment or dict: The experiment object or dict of experiment objects for batch processing
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
            project_name=project_name,
            method='ISM',
            batch_processing=batch_processing,
            source_positions=source_positions,
            receiver_positions=receiver_positions
        )

# Singleton pattern for batch manager
# _batch_manager = None

def get_batch_manager(results_dir='results', dont_check_duplicates=False):
    """
    Get or create the batch experiment manager singleton.
    
    Args:
        results_dir (str): Custom directory to store batch experiment results
                          Default is 'results' with data in 'results/rooms/'
        dont_check_duplicates (bool): If True, skip loading existing experiments.
                                    This significantly speeds up initialization
                                    when you don't need to check for duplicates.
        
        Returns:
        SDNExperimentManager: The batch experiment manager instance
    """
    _batch_manager = SDNExperimentManager(
        results_dir=results_dir,
        is_batch_manager=True,
        dont_check_duplicates=dont_check_duplicates
    )
    return _batch_manager

def get_singular_manager(results_dir='results', dont_check_duplicates=False):
    """
    Get or create the singular experiment manager singleton.
        
        Args:
        results_dir (str): Custom directory to store singular experiment results
                          Default is 'results' with data in 'results/room_singulars/'
        dont_check_duplicates (bool): If True, skip loading existing experiments.
                                    This significantly speeds up initialization
                                    when you don't need to check for duplicates.
            
        Returns:
        SDNExperimentManager: The singular experiment manager instance  
    """
    global _singular_manager
    if _singular_manager is None or _singular_manager.results_dir != results_dir:
        _singular_manager = SDNExperimentManager(
            results_dir=results_dir,
            is_batch_manager=False,
            dont_check_duplicates=dont_check_duplicates
        )
    return _singular_manager

_singular_manager = None

if __name__ == "__main__":

    duration = 1

    # Use this as a parent directory for the results (optional)
    results_dir = 'results'  # Default: uses results/rooms/

    room_waspaa = {
        'width': 6, 'depth': 4, 'height': 7,
        'source x': 3.6, 'source y': 1.3, 'source z': 5.3,
        'mic x': 1.2, 'mic y': 2.4, 'mic z': 1.8,
        'absorption': 0.1,
    }

    room_aes = {'width': 9, 'depth': 7, 'height': 4,
                'source x': 4.5, 'source y': 3.5, 'source z': 2,
                'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                'absorption': 0.2,
                }

    room_journal = {'width': 3.2, 'depth': 4, 'height': 2.7,
                    'source x': 2, 'source y': 3., 'source z': 2,
                    'mic x': 1, 'mic y': 1, 'mic z': 1.5,
                    'absorption': 0.1,
                    }

    room = room_journal
    # room_name = "room_aes"
    project_name = "ZZZZZjournal_absorptioncoeffs"

    # Generate source & receiver positions
    # receiver_positions = sa.generate_receiver_grid(room['width'], room['depth'], 50)

    # receiver_positions = sa.generate_receiver_grid(room['width'] / 2, room['depth'] / 2, n_points=16, margin=0.5)  # room aes
    # source_positions = sa.generate_source_positions(room)

    run_single_experiments = True
    run_batch_experiments = False

    if run_batch_experiments:

        # Run batch experiments
        # Use custom results directory for batch experiments (optional)
        batch_manager = get_batch_manager(results_dir, dont_check_duplicates=True)  # Default: uses results/rooms/

        print(f"Batch experiments saved in: {batch_manager._get_room_dir(project_name)}")

        batch_manager.run_experiment(
            config={
                'label': 'weighted psk',
                'info': '',
                'method': 'SDN',
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 2,
                }
            },
            room_parameters=room,
            duration=duration,
            fs=44100,
            project_name=project_name,
            batch_processing=True,  # Enable batch processing
            source_positions=source_positions,  # Provide source positions
            receiver_positions=receiver_positions  # Provide receiver positions
        )

        # batch_manager.run_experiment(
        #     config={
        #         'label': 'weighted psk',
        #         'info': '',
        #         'method': 'SDN',
        #         'flags': {
        #             'specular_source_injection': True,
        #             'source_weighting': 4,
        #         }
        #     },
        #     room_parameters=room,
        #     duration=duration,
        #     fs=44100,
        #     project_name=project_name,
        #     batch_processing=True,  # Enable batch processing
        #     source_positions=source_positions,  # Provide source positions
        #     receiver_positions=receiver_positions  # Provide receiver positions
        # )

        # batch_manager.run_experiment(
        #     config={
        #         'label': 'weighted psk',
        #         'info': '',
        #         'method': 'SDN',
        #         'flags': {
        #             'specular_source_injection': True,
        #             'source_weighting': 5,
        #         }
        #     },
        #     room_parameters=room,
        #     duration=duration,
        #     fs=44100,
        #     project_name=project_name,
        #     batch_processing=True,  # Enable batch processing
        #     source_positions=source_positions,  # Provide source positions
        #     receiver_positions=receiver_positions  # Provide receiver positions
        # )

        # batch_manager.run_ism_experiment(
        #     room_parameters=room,
        #     duration=duration,  # Shorter duration for batch processing
        #     fs=44100,
        #     max_order=12,
        #     ray_tracing=False,
        #     project_name=project_name,
        #     batch_processing=True,  # Enable batch processing
        #     source_positions=source_positions,  # Provide source positions
        #     receiver_positions=receiver_positions,  # Provide receiver positions
        #     label = "pra"
        # )

    if run_single_experiments:
        # Method 1: Run experiments directly
        # Use custom results directory for singular experiments (optional)
        # single_manager = get_singular_manager("results_custom")  # Will use results_custom/room_singulars/
        single_manager = get_singular_manager(results_dir)  # Default: uses results/room_singulars/

        print(f"Singular experiments saved in: {single_manager._get_room_dir(project_name)}")

        # Run an SDN experiment - single source and receiver (uses singular manager)

        single_manager.run_experiment(
            config={
                'label': 'original',
                'info': '',
                'method': 'SDN',
                'flags': {
                    # 'specular_source_injection': True,
                    # 'source_weighting': 2,
                }
            },
            room_parameters=room,
            duration=duration,
            fs=44100,
            project_name=project_name
        )

        # single_manager.run_experiment(
        #     config={
        #         'label': 'weighted psk',
        #         'info': '',
        #         'method': 'SDN',
        #         'flags': {
        #             'specular_source_injection': True,
        #             'source_weighting': 3,
        #         }
        #     },
        #     room_parameters=room,
        #     duration=duration,
        #     fs=44100,
        #     project_name = project_name
        # )
        #
        # single_manager.run_experiment(
        #     config={
        #         'label': 'weighted psk',
        #         'info': '',
        #         'method': 'SDN',
        #         'flags': {
        #             'specular_source_injection': True,
        #             'source_weighting': 4,
        #         }
        #     },
        #     room_parameters=room,
        #     duration=duration,
        #     fs=44100,
        #     project_name=project_name
        # )
        #
        # single_manager.run_experiment(
        #     config={
        #         'label': 'weighted psk',
        #         'info': '',
        #         'method': 'SDN',
        #         'flags': {
        #             'specular_source_injection': True,
        #             'source_weighting': 5,
        #         }
        #     },
        #     room_parameters=room,
        #     duration=duration,
        #     fs=44100,
        #     project_name=project_name
        # )
        #
        # # Run an ISM experiment
        # single_manager.run_ism_experiment(
        #     room_parameters=room,
        #     max_order=12,
        #     ray_tracing=False,
        #     duration=duration,
        #     fs=44100,
        #     project_name= project_name,
        #     label="",
        # )

    #dump batch_manager to pickle
    # import pickle
    # with open('batch_manager_roomaes_4exp.pkl', 'wb') as f:
    #     pickle.dump(batch_manager.rooms, f)

