"""
Treble Data Processing Script

This script processes Treble simulation results and converts them into a format compatible with the SDN visualization system.

Input Directory Structure:
------------------------
./results/treble/single_experiments/           # Base directory for singular experiments
    ├── direct_result_folders/                 # Direct simulation result folders
    │   ├── simulation_info.json              # Simulation configuration
    │   └── *.h5                              # Simulation RIR data
    │
    └── grouped_experiment_sets/              # Groups of related experiments
        ├── experiment1/                      # Individual experiment folders
        │   ├── simulation_info.json
        │   └── *.h5
        └── experiment2/
            ├── simulation_info.json
            └── *.h5

OR

./results/treble/multi_experiments/           # Base directory for batch experiments
    └── project_name/                        # Project-specific folder
        ├── experiment1/                     # Individual experiment folders
        │   ├── simulation_info.json
        │   └── *.h5
        └── experiment2/
            ├── simulation_info.json
            └── *.h5

Output Directory Structure:
-------------------------
For Singular Case:
./results/room_singulars/                    # Output directory for singular experiments
    ├── room_info.json                       # Room configuration
    ├── config_experiment1.json              # Experiment configurations
    ├── rirs_experiment1.npy                 # Processed RIR data
    └── ...

For Batch Case:
./results/rooms/                             # Output directory for batch experiments
    └── room_name/                           # Room-specific folder
        ├── room_info.json                   # Room configuration
        └── source_label/                    # Source-specific folder
            └── method/                      # Method-specific folder (e.g., TREBLE)
                └── sim_set/               # Parameter set folder
                    ├── config.json          # Configuration
                    └── rirs.npy             # RIR data
"""

import os
import json
import numpy as np
from datetime import datetime
import hashlib

# Import modules from the main codebase
import EchoDensity as ned
import analysis as an
from sdn_experiment_manager_pythonANY import Room


class SDNExperiment:
    """Class to store and manage acoustic simulation experiment data and metadata."""
    
    def __init__(self, config, rir, fs=44100, duration=0.5, experiment_id=None, skip_metrics=False):
        # note: skip_metrics is not fully correct yet. due to table retrieval error. only use it if you dont need table error ui.
        """
        Initialize an acoustic simulation experiment.
        
        Args:
            config (dict): Configuration parameters for the experiment
            rir (np.ndarray): Room impulse response
            fs (int): Sampling frequency
            duration (float): Duration of the RIR in seconds
            experiment_id (str, optional): Unique ID for the experiment. If None, will be generated.
        """
        self.config = config
        self.rir = rir
        self.fs = fs
        self.duration = duration
        self.timestamp = datetime.now().isoformat()
        self._metrics_calculated = False
        self.metrics = {}
        self.skip_metrics = skip_metrics

        if not self.skip_metrics:
            self._calculate_metrics()
        
        # Generate a unique ID if not provided
        if experiment_id is None:
            # Create a hash of the configuration to use as ID
            id_config = self._prepare_config_for_id(config)
            config_str = json.dumps(self._make_serializable(id_config), sort_keys=True)
            self.experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:10]
        else:
            self.experiment_id = experiment_id


    
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
    
    def ensure_metrics_calculated(self):
        """Ensure metrics are calculated if they haven't been already."""
        if not hasattr(self, '_metrics_calculated') or not self._metrics_calculated:
            self._calculate_metrics()
        print("ensure ")
    
    def _calculate_metrics(self):
        """Calculate various acoustic metrics for the RIR."""
        self.metrics = {}
        
        # Skip metrics calculation if RIR is empty
        if len(self.rir) == 0:
            self.edc = np.array([])
            self.ned = np.array([])
            self.time_axis = np.array([])
            self.ned_time_axis = np.array([])
            self._metrics_calculated = True
            return
        
        # Calculate EDC once to avoid duplication - now getting both EDC curve and raw energy
        self.edc, time_edc, schroder_energy = an.compute_edc(self.rir, self.fs, label=self.get_label(), plot=False)
        
        # Calculate RT60 if the RIR is long enough - use the pre-calculated EDC and raw energy
        if self.duration > 0.7:
            # Pass the pre-calculated EDC and raw energy to the RT60 calculation function
            self.metrics['rt60'] = an.calculate_rt60_from_rir(
                self.rir, 
                self.fs, 
                plot=False, 
                pre_calculated_edc=self.edc,
                schroder_energy=schroder_energy
            )
        
        # Calculate NED - limit to first 0.5 seconds by default and downsample to 22050 Hz
        # This significantly reduces computation time for long RIRs
        self.ned = ned.echoDensityProfile(
            self.rir, 
            fs=self.fs, 
            max_duration=0.5,
        )
        
        # Time axis for plotting
        self.time_axis = np.arange(len(self.rir)) / self.fs
        self.ned_time_axis = np.arange(len(self.ned)) / self.fs
        self.edc_time_axis = time_edc
        self._metrics_calculated = True
    
    # Add property accessors that ensure metrics are calculated
    @property
    def rt60(self):
        """Get RT60 value, calculating metrics if needed."""
        if self.skip_metrics:
            return None
        else:
            self.ensure_metrics_calculated()
        return self.metrics.get('rt60', None)

    
    def get_label(self):
        """Generate a descriptive label for the experiment."""
        method = self.config.get('method')
        
        if 'label' in self.config and self.config['label']:
            label = f"{self.config['label']}"
        else:
            label = f""
            
        if 'info' in self.config and self.config['info']:
            label += f": {self.config['info']}"
            
        # Add method-specific details
        if method == 'ISM':
            if 'max_order' in self.config:
                label += f"PRA order={self.config['max_order']}"
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
        label_for_legend = f"{method}, {label}" # complete label
        # return label and label_for_legend as a single dictionary
        labels = {"label": label, "label_for_legend": label_for_legend}
        return labels
    
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


class ExperimentLoaderManager:
    """Class to manage loading and accessing acoustic simulation experiments from storage."""
    
    def __init__(self, results_dir='results', is_batch_manager=False, project_names=None, disable_unified_rooms=True, skip_metrics=False):
        #note: skip_metrics is not fully correct yet. due to table retrieval error. only use it if you dont need table error ui.
        """
        Initialize the experiment manager.
        
        Args:
            results_dir (str): Base directory to store experiment data
            is_batch_manager (bool): If True, this manager handles batch processing experiments
            project_names (list or str, optional): Specific project names to load. Can be:
                - None: Load all projects (default)
                - str: Load a single project
                - list: Load multiple specific projects
                For batch_manager=True, these are folder names in 'results/rooms/'
                For batch_manager=False, these are folder names in 'results/room_singulars/'
            disable_unified_rooms (bool): If True, only create individual rooms for singular mode 
                                         (saves memory by not duplicating experiments in unified rooms)
        """
        self.results_dir = results_dir
        self.is_batch_manager = is_batch_manager
        self.projects = {}  # name -> Room
        self.project_names = project_names
        self.disable_unified_rooms = disable_unified_rooms
        self.skip_metrics = skip_metrics
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
    
    # def _get_source_dir(self, room_name, source_label):
    #     """Get the directory path for a source within a room."""
    #     room_dir = self._get_room_dir(room_name)
    #     return os.path.join(room_dir, source_label)
    
    def get_display_name_from_folder(self, folder_name):
        """Create display name from folder name.
        e.g., 'aes_absorptioncoeffs' -> 'AES - absorptioncoeffs'"""
        parts = folder_name.split('_', 1)  # Split at first underscore
        if len(parts) == 2:
            prefix, rest = parts
            return f"{prefix.upper()} - {rest}"
        return folder_name.upper()

    @staticmethod
    def print_projects(projects_dict, projects_list, title="Available projects"):
        """Print available projects by category in a formatted way.
        
        Args:
            projects_dict (dict): Dictionary with categories as keys and project lists as values
            title (str): Title to display before the project listing
        """
        print(f"\n{title}:\n")
        for i, name in enumerate(projects_list):
            print(f"  {i}: {name}")

        print("\nby room category:")
        for category, projects in projects_dict.items():
            print(f"{category.upper()}:")
            for i, project in enumerate(projects):
                print(f"  {i}: {project}")

    @staticmethod
    def get_available_projects(results_dir='results', mode='batch'):
        """Get list of available project names in the results directory.
        
        Args:
            results_dir (str): Base directory to store experiment data
            mode (str): 'batch' for batch projects in 'rooms/' or 'singular' for singular projects in 'room_singulars/'
            
        Returns:
            dict: Dictionary with categories as keys and list of project names as values
                 Example: {
                     'aes': ['aes_absorption_experiment', 'aes_quartergrid'],
                     'journal': ['journal_experiment1'],
                     'waspaa': ['waspaa_test']
                 }
        """
        # Determine the correct directory based on mode
        if mode == 'singular':
            target_dir = os.path.join(results_dir, 'room_singulars')
        else:  # Default to batch mode
            target_dir = os.path.join(results_dir, 'rooms')
            
        if not os.path.exists(target_dir):
            return {}
        
        projects_grouped = {}
        projectsingular_projects = []
        for name in os.listdir(target_dir):
            path = os.path.join(target_dir, name)
            
            # Check if it's a directory and apply appropriate filters
            is_valid_dir = os.path.isdir(path)
            if mode == 'singular':
                # For singular mode, skip hidden folders and folders starting with underscore
                is_valid_dir = is_valid_dir and not name.startswith('.') and not name.startswith('_')

            if is_valid_dir:
                projectsingular_projects.append(name)
                # Determine category based on name prefix
                if 'aes' in name.lower():
                    category = 'aes'
                elif 'journal' in name.lower():
                    category = 'journal'
                elif 'waspaa' in name.lower():
                    category = 'waspaa'
                else:
                    category = 'other'
                
                if category not in projects_grouped:
                    projects_grouped[category] = []
                projects_grouped[category].append(name)
        
        # Sort project names within each category
        for category in projects_grouped:
            projects_grouped[category].sort()
        
        return projects_grouped, projectsingular_projects

    def load_experiments(self):
        """Load all experiments from the results directory."""
        self.projects = {}

        # Determine the directory path based on manager type
        base_dir = os.path.join(self.results_dir, 'room_singulars' if not self.is_batch_manager else 'rooms')
        if not os.path.exists(base_dir):
            print(f"No {base_dir} directory found")
            return

        # Create project_dirs list based on project_names
        if self.project_names is None:
            # Load all projects from the directory
            project_dirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d))]
            # For singular case, filter out hidden folders and folders starting with underscore
            if not self.is_batch_manager:
                project_dirs = [d for d in project_dirs if not d.startswith('.') and not d.startswith('_')]
            print(f"Loading all {len(project_dirs)} projects from {base_dir}")
        elif isinstance(self.project_names, str):
            # Load single project
            project_dirs = [self.project_names]
            print(f"Loading specified project: {self.project_names}")
        elif isinstance(self.project_names, list):
            # Load multiple specified projects
            project_dirs = self.project_names
            print(f"Loading {len(project_dirs)} specified projects")
        else:
            print("Warning: project_names should be None, a string, or a list")
            project_dirs = []

        # SINGULAR EXPERIMENTS
        if not self.is_batch_manager:
            # Singular case: Load from room_singulars with unified rooms
            unified_rooms = {} if not self.disable_unified_rooms else None  # Only create if not disabled
            
            # Process each experiment group (subfolder)
            for folder_name in project_dirs:
                folder_path = os.path.join(base_dir, folder_name)
                # Skip if folder doesn't exist
                if not os.path.isdir(folder_path):
                    print(f"Warning: Folder {folder_name} does not exist. Skipping...")
                    continue

                try:
                    # Determine which room_info to use based on folder name
                    if 'aes' in folder_name.lower():
                        room_info_file = 'room_info_aes.json'
                    elif 'journal' in folder_name.lower():
                        room_info_file = 'room_info_journal.json'
                    elif 'waspaa' in folder_name.lower():
                        room_info_file = 'room_info_waspaa.json'
                    else:
                        print(f"Warning: Could not determine room type for {folder_name}")
                        continue

                    # Load room info
                    room_info_path = os.path.join(base_dir, room_info_file)
                    if not os.path.exists(room_info_path):
                        print(f"Warning: Room info file {room_info_file} not found")
                        continue

                    with open(room_info_path, 'r') as f:
                        room_info = json.load(f)

                    # Create individual room with formatted display name
                    individual_room = Room(folder_name, room_info['parameters'])
                    individual_room.display_name = self.get_display_name_from_folder(folder_name)
                    self.projects[folder_name] = individual_room

                    # Get room type for unified room (if enabled)
                    if not self.disable_unified_rooms:
                        room_type = folder_name.split('_')[0].lower()

                        # Create unified room if it doesn't exist
                        if room_type not in unified_rooms:
                            unified_room = Room(room_info['name'], room_info['parameters'])
                            unified_room.display_name = room_info['display_name']
                            unified_rooms[room_type] = unified_room

                    # Load experiments using existing logic
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.json') and file_name != 'room_info.json':
                            config_path = os.path.join(folder_path, file_name)
                            
                            # Handle both naming patterns for RIR files
                            if file_name.startswith('config_'):
                                base_name = file_name[7:-5]  # Remove 'config_' and '.json'
                                rir_file = f'rirs_{base_name}.npy'
                            else:
                                rir_file = file_name[:-5] + '.npy'
                            
                            rir_path = os.path.join(folder_path, rir_file)
                            
                            # Load experiment if both files exist
                            if os.path.exists(config_path) and os.path.exists(rir_path):
                                try:
                                    with open(config_path, 'r') as f:
                                        config_data = json.load(f)
                                    
                                    # Load RIR data and ensure correct shape
                                    rir_data = np.load(rir_path)
                                    if rir_data.ndim > 1:
                                        # If RIR is 2D array, take first row
                                        rir = rir_data[0]
                                    else:
                                        # If RIR is already 1D array, use as is
                                        rir = rir_data
                                    
                                    # Handle both config formats
                                    if 'config' in config_data and 'room_parameters' in config_data['config']:
                                        experiment_config = config_data['config']
                                        fs = config_data.get('fs')
                                        duration = config_data.get('duration')
                                    else:
                                        experiment_config = config_data
                                        fs = config_data.get('fs')
                                        duration = config_data.get('duration')
                                    
                                    # Set experiment_id based on file naming
                                    if file_name.startswith('config_'):
                                        experiment_id = file_name[7:-5]
                                    else:
                                        experiment_id = config_data.get('experiment_id')
                                    
                                    # Create experiment object
                                    experiment = SDNExperiment(
                                        config=experiment_config,
                                        rir=rir,
                                        fs=fs,
                                        duration=duration,
                                        experiment_id=experiment_id,
                                        skip_metrics=self.skip_metrics
                                    )
                                    
                                    # Add experiment to individual room
                                    individual_room.add_experiment(experiment)
                                    
                                    # Also add to unified room if enabled
                                    if not self.disable_unified_rooms:
                                        unified_rooms[room_type].add_experiment(experiment)
                                    
                                    print(f"Successfully loaded experiment: {file_name} with ID: {experiment_id}")
                                except Exception as e:
                                    print(f"Error loading experiment {file_name}: {str(e)}")
                                    import traceback
                                    traceback.print_exc()

                    print(f"Loaded experiments from {folder_name} as room {individual_room.display_name}\n")
                except Exception as e:
                    print(f"Error processing folder {folder_name}: {str(e)}")

            # Add unified rooms to self.projects if not disabled
            if not self.disable_unified_rooms:
                for unified_room in unified_rooms.values():
                    self.projects[unified_room.name] = unified_room
                    print(f"Created unified room {unified_room.display_name} with {len(unified_room.experiments)} total experiments")

        # BATCH EXPERIMENTS
        else:
            # Batch case: Load from rooms directory without unification
            total_projects = len(project_dirs)
            print(f"\nLoading {total_projects} projects:")
            
            # Iterate through selected project directories
            for project_idx, project_name in enumerate(project_dirs, 1):
                project_path = os.path.join(base_dir, project_name)
                if not os.path.isdir(project_path):
                    print(f"Skipping non-directory or non-existent project: {project_name}")
                    continue
                
                print(f"\n[{project_idx}/{total_projects}] Loading project: {project_name}")
                
                # Load room info from unified location
                if 'aes' in project_name.lower():
                    print("room name: aes")
                    room_info_file = 'room_info_aes.json'
                elif 'journal' in project_name.lower():
                    print("room name: journal")
                    room_info_file = 'room_info_journal.json'
                elif 'waspaa' in project_name.lower():
                    print("room name: waspaa")
                    room_info_file = 'room_info_waspaa.json'
                else:
                    print(f"Warning: Could not determine room type for {project_name}")
                    continue

                # Load room info from unified location
                room_info_path = os.path.join(self.results_dir, 'rooms', room_info_file)
                try:
                    with open(room_info_path, 'r') as f:
                        room_info = json.load(f)
                    
                    # Create room with saved parameters
                    room_parameters = room_info.get('parameters', {})
                    room = Room(project_name, room_parameters)
                    room.display_name = self.get_display_name_from_folder(project_name)
                    
                    # Use os.walk to traverse the directory structure once
                    # This eliminates multiple os.listdir() calls
                    print(f"Scanning directory structure for {project_name}...")
                    
                    # Structure to store all paths
                    all_experiment_paths = []
                    
                    # Walk through the project directory to find all config.json and rirs.npy files
                    for root, dirs, files in os.walk(project_path):
                        if 'config.json' in files and 'rirs.npy' in files:
                            # Found a simulation set directory
                            rel_path = os.path.relpath(root, project_path)
                            parts = rel_path.split(os.sep)
                            
                            # For standard structure with 3 levels: source/method/param_set
                            if len(parts) >= 3:
                                source_label = parts[0]
                                method = parts[1]
                                param_set = parts[2]
                                
                                # Store this path
                                all_experiment_paths.append({
                                    'source_label': source_label,
                                    'method': method,
                                    'param_set': param_set,
                                    'config_path': os.path.join(root, 'config.json'),
                                    'rirs_path': os.path.join(root, 'rirs.npy')
                                })
                    
                    # Report how many experiment paths we found
                    total_experiments = len(all_experiment_paths)
                    print(f"Found {total_experiments} experiment configurations in {project_name}")
                    
                    # Process each experiment path
                    for exp_idx, exp_path in enumerate(all_experiment_paths, 1):
                        source_label = exp_path['source_label']
                        method = exp_path['method']
                        param_set = exp_path['param_set']
                        config_path = exp_path['config_path']
                        rirs_path = exp_path['rirs_path']
                        
                        print(f"  [{exp_idx}/{total_experiments}] Loading {source_label}/{method}/{param_set}")
                        
                        try:
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            rirs = np.load(rirs_path)
                            
                            # Get receivers from config.json
                            receivers_list_of_dicts = config_data.get('receivers', [])
                            total_receivers = len(receivers_list_of_dicts)

                            # Create experiment objects for each source-receiver pair
                            for rec_idx, receiver_single in enumerate(receivers_list_of_dicts, 1):
                                if rec_idx > len(rirs):
                                    break
                                    
                                # Create a single experiment config
                                receiver_config = config_data.copy()
                                # take full config, delete receivers list, put only current receiver
                                if 'receivers' in receiver_config:
                                    del receiver_config['receivers']
                                receiver_config['receiver'] = receiver_single
                                
                                # Create experiment object
                                experiment = SDNExperiment(
                                    config=receiver_config,
                                    rir=rirs[rec_idx-1],
                                    fs=config_data.get('fs'),
                                    duration=config_data.get('duration'),
                                    experiment_id=receiver_single.get('experiment_id'),
                                )
                                
                                # Add experiment to room
                                room.add_experiment(experiment)
                            
                            print(f"    Successfully loaded {total_receivers} receivers")
                            
                        except Exception as e:
                            print(f"    Error loading experiments from {param_set}: {e}")
                    
                    # Only add room if it has valid experiments
                    if room.experiments:
                        self.projects[room.name] = room
                        print(f"\n  Completed room {room.name}: {len(room.experiments)} total experiments loaded")
                    else:
                        print(f"\n  No valid experiments found in {project_name}")
                        
                except Exception as e:
                    print(f"Error loading project {project_name}: {e}")
            
            print(f"\nLoading complete! Total projects loaded: {len(self.projects)}")

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
            experiments.extend([exp for exp in project.experiments.values() if label in exp.get_label()])
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



# Example usage in main:
if __name__ == "__main__":

    results_dir = 'results'
    IS_SINGULAR = False  # Set to False for batch processing
    IS_BATCH = True  # Set to True for batch processing

    # Display available projects for both modes
    singular_category, singular_projects = ExperimentLoaderManager.get_available_projects(results_dir, mode='singular')
    ExperimentLoaderManager.print_projects(singular_category, singular_projects, "Available singular projects:")
    
    batch_category, batch_projects = ExperimentLoaderManager.get_available_projects(results_dir, mode='batch')
    ExperimentLoaderManager.print_projects(batch_category, batch_projects, "Available batch projects:")

    if IS_SINGULAR:
        # Example 1: Load all singular experiments
        singular_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=False, 
                                                  )
        
        # Example 2: Load a single specific singular experiment folder
        # singular_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=False, 
        #                                           project_names="waspaa_rr", 
        #                                           )
        
        # Example 3: Load multiple specific singular experiment folders
        # singular_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=False, 
        #                                           project_names=["aes_quartergrid", "journal_experiment1"],
        #                                           )

        from sdn_experiment_visualizer import ExperimentVisualizer
        single_visualizer = ExperimentVisualizer(singular_manager)
        single_visualizer.show(port=1991)

        import sdn_experiment_visualizer as sev
        import importlib
        importlib.reload(sev)
        single_visualizer = sev.ExperimentVisualizer(singular_manager)
        single_visualizer.show(port=1991)


    elif IS_BATCH: # Batch processing


        # Example 3: Load multiple specific batch projects
        #get list of categorized projects
        aes_projects = batch_category.get('aes', [])
        waspaa_projects = batch_category.get('waspaa', [])
        journal_projects = batch_category.get('journal', [])

        multiple_projects = False  # Set to True to select specific projects
        if multiple_projects:
            selected_projects = [batch_projects[0], batch_projects[1], batch_projects[3]]
            print("selected projects: ", selected_projects)
            batch_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=True, project_names=selected_projects)

        else:
            # Example 1: Load all batch projects
            # batch_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=True)

            # Example 2: Load a single specific batch project
            batch_manager = ExperimentLoaderManager(results_dir=results_dir, is_batch_manager=True,
                                                    project_names="aes_quartergrid_new")

        from sdn_experiment_visualizer import ExperimentVisualizer
        batch_visualizer = ExperimentVisualizer(batch_manager)
        # generate random 4 digit with python random
        import random
        port_no = random.randint(1000, 9999)
        batch_visualizer.show(port=port_no)
        #
        # # # # import analysis as an
        # import importlib
        # # importlib.reload(an)
        # import sdn_experiment_visualizer as sev
        # importlib.reload(sev)
        # import random
        # port_no = random.randint(1000, 9999)
        # batch_visualizer = sev.ExperimentVisualizer(batch_manager)
        # batch_visualizer.show(port=port_no)