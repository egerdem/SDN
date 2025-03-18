import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import hashlib
import importlib
import sys
from pathlib import Path
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
        self.experiments = {}  # experiment_id -> SDNExperiment
        self.experiments_by_position = {}  # (source_pos, mic_pos) -> list of experiments
        
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

class SDNExperiment:
    """Class to store and manage SDN experiment data and metadata."""
    
    def __init__(self, config, rir, fs=44100, duration=0.5, experiment_id=None, skip_metrics=False):
        """
        Initialize an SDN experiment.
        
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
            # Create a hash of the configuration to use as ID, but exclude 'info' field
            # as it's just a description and doesn't affect the experiment result
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
        
        # Remove purely descriptive fields
        if 'info' in config:
            # Don't include 'info' in id_config at all
            pass
        
        # Keep only room parameters with numerical values
        if 'room_parameters' in config:
            id_config['room_parameters'] = {}
            for key, value in config['room_parameters'].items():
                # Only include numeric values that affect simulation
                if isinstance(value, (int, float)):
                    id_config['room_parameters'][key] = value
        
        # Keep only relevant flags that affect the simulation
        if 'flags' in config:
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
        if 'label' in self.config and self.config['label']:
            label = self.config['label']
        else:
            label = f"SDN-{self.experiment_id[:6]}"
            
        if 'info' in self.config and self.config['info']:
            label += f": {self.config['info']}"
            
        return label
    
    def get_key_parameters(self):
        """Extract and return the key parameters that define this experiment."""
        params = {}
        
        # Add key flags if they exist
        if 'flags' in self.config:
            flags = self.config['flags']
            # Focus on commonly adjusted parameters
            key_params = ['source_weighting', 'specular_source_injection', 
                          'scattering_matrix_update_coef', 'coef', 'source_pressure_injection_coeff']
            
            for param in key_params:
                if param in flags:
                    params[param] = flags[param]
        
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
    """Class to manage multiple SDN experiments, store results, and provide visualization."""
    
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
                
            # Load room parameters
            room_params_path = os.path.join(room_path, 'room_parameters.json')
            if not os.path.exists(room_params_path):
                continue
                
            with open(room_params_path, 'r') as f:
                room_parameters = json.load(f)
            
            # Create room object
            room = Room(room_dir, room_parameters)
            
            # Load experiments for this room
            for filename in os.listdir(room_path):
                if filename.endswith('.json') and filename != 'room_parameters.json':
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
    
    def run_experiment(self, config, room_parameters, duration=0.5, fs=44100, force_rerun=False):
        """
        Run an SDN experiment with the given configuration.
        
        Args:
            config (dict): Configuration for the SDN experiment
            room_parameters (dict): Room parameters
            duration (float): Duration of the simulation in seconds
            fs (int): Sampling frequency
            force_rerun (bool): If True, rerun the experiment even if it exists
            
        Returns:
            SDNExperiment: The experiment object
        """
        # Get or create room
        room_name = self._get_room_name(room_parameters)
        if room_name not in self.rooms:
            room = Room(room_name, room_parameters)
            self.rooms[room_name] = room
            
            # Save room parameters
            room_dir = self._get_room_dir(room_name)
            os.makedirs(room_dir, exist_ok=True)
            with open(os.path.join(room_dir, 'room_parameters.json'), 'w') as f:
                json.dump(room_parameters, f, indent=2)
        else:
            room = self.rooms[room_name]
        
        # Create a temporary config with all parameters for ID generation
        full_config = {
            **config,
            'room_parameters': room_parameters,
            'duration': duration,
            'fs': fs
        }
        
        # Generate experiment ID
        temp_experiment = SDNExperiment(full_config, np.array([]), skip_metrics=True)
        experiment_id = temp_experiment.experiment_id
        
        # Check if experiment exists
        if experiment_id in room.experiments and not force_rerun:
            print(f"Experiment {experiment_id} already exists in room {room_name}. Using cached results.")
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
        
        # Create SDN instance with configured flags
        sdn = DelayNetwork(geom_room, Fs=fs, label=config.get('label', ''), **config.get('flags', {}))
        
        # Calculate RIR
        rir = sdn.calculate_rir(duration)
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
        
        # Save metadata
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
            
            # Add source marker
            fig.add_trace(go.Scatter(
                x=[source_pos[0]], y=[source_pos[1]],
                mode='markers',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Source'
            ))
            
            # Add microphone marker
            fig.add_trace(go.Scatter(
                x=[mic_pos[0]], y=[mic_pos[1]],
                mode='markers',
                marker=dict(
                    color='red',
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
            margin=dict(t=0, b=0, l=0, r=0),
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
        
        # Get list of rooms for navigation
        room_names = list(self.rooms.keys())
        current_room_idx = room_names.index(room_name)
        
        # Create app layout
        app.layout = html.Div([
            # Room navigation header
            html.Div([
                html.Div([
                    html.H2(
                        id='room-header',
                        style={'margin': '0 20px'}
                    ),
                    html.H3(
                        id='rt-header',
                        style={'margin': '10px 20px', 'color': '#666'}
                    )
                ], style={'display': 'inline-block', 'position': 'relative'}),
                html.Div([
                    html.Button('←', id='prev-room', style={'fontSize': 24, 'marginRight': '10px'}),
                    html.Button('→', id='next-room', style={'fontSize': 24}),
                ], style={'position': 'absolute', 'left': '50%', 'top': '%20', 'transform': 'translateX(-50%)'}),
                dcc.Store(id='current-room-idx', data=current_room_idx)
            ], style={'textAlign': 'center', 'margin': '20px', 'position': 'relative'}),
            
            # Main content with plots and room visualization
            html.Div([
                # Left side - plots (75% width)
                html.Div([
                    # Experiment table for current position
                    html.Div([
                        html.H3("Active Experiments"),
                        dash_table.DataTable(
                            id='experiment-table',
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'minWidth': '100px',  # Minimum column width
                                'width': '150px',     # Default column width
                                'maxWidth': '300px',  # Maximum column width
                                'padding': '10px'
                            },
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi"
                        )
                    ], style={'margin': '20px'}),
                    
                    dcc.Tabs([
                        dcc.Tab(label="Room Impulse Responses", children=[
                            dcc.Graph(id='rir-plot', style={'height': '60vh'})
                        ]),
                        dcc.Tab(label="Energy Decay Curves", children=[
                            dcc.Graph(id='edc-plot', style={'height': '60vh'})
                        ]),
                        dcc.Tab(label="Normalized Echo Density", children=[
                            dcc.Graph(id='ned-plot', style={'height': '60vh'})
                        ])
                    ])
                ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                # Right side - room visualization (25% width)
                html.Div([
                    # Title and source-mic navigation
                    html.Div([
                        html.H3("Room Layout (Top View)", 
                               style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '0px'}),
                        html.Div([
                            html.Button('←', id='prev-pos', style={'fontSize': 20, 'marginRight': '10px'}),
                            html.Button('→', id='next-pos', style={'fontSize': 20}),
                        ], style={'textAlign': 'center', 'marginBottom': '5px'}),
                        html.Div(id='pos-info', 
                               style={'textAlign': 'center', 'marginBottom': '5px', 'fontSize': '14px'}),
                        dcc.Store(id='current-pos-idx', data=0),
                    ], style={'marginBottom': '0px', 'padding': '0px'}),
                    dcc.Graph(
                        id='room-plot',
                        style={'height': '50vh', 'marginTop': '0px'}
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'top'})
            ])
        ])
        
        # Callback for room and position navigation
        @app.callback(
            [Output('current-room-idx', 'data'),
             Output('current-pos-idx', 'data'),
             Output('pos-info', 'children'),
             Output('room-plot', 'figure'),
             Output('experiment-table', 'data'),
             Output('experiment-table', 'columns'),
             Output('room-header', 'children'),
             Output('rt-header', 'children')],
            [Input('prev-room', 'n_clicks'),
             Input('next-room', 'n_clicks'),
             Input('prev-pos', 'n_clicks'),
             Input('next-pos', 'n_clicks')],
            [State('current-room-idx', 'data'),
             State('current-pos-idx', 'data')]
        )
        def update_navigation(prev_room_clicks, next_room_clicks, 
                            prev_pos_clicks, next_pos_clicks,
                            current_room_idx, current_pos_idx):
            ctx = dash.callback_context
            
            # Initialize indices if they're None
            if current_room_idx is None:
                current_room_idx = 0
            if current_pos_idx is None:
                current_pos_idx = 0
            
            new_room_idx = current_room_idx
            new_pos_idx = current_pos_idx
            
            # Handle button clicks if any
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                # Handle room navigation
                if button_id in ['prev-room', 'next-room']:
                    if button_id == 'prev-room':
                        new_room_idx = (current_room_idx - 1) % len(room_names)
                    else:  # next-room
                        new_room_idx = (current_room_idx + 1) % len(room_names)
                    # Reset position index when changing rooms
                    new_pos_idx = 0
                
                # Handle position navigation
                elif button_id in ['prev-pos', 'next-pos']:
                    room = self.rooms[room_names[current_room_idx]]
                    if room.source_mic_pairs:
                        if button_id == 'prev-pos':
                            new_pos_idx = (current_pos_idx - 1) % len(room.source_mic_pairs)
                        else:  # next-pos
                            new_pos_idx = (current_pos_idx + 1) % len(room.source_mic_pairs)
            
            # Get current room and position info
            room = self.rooms[room_names[new_room_idx]]
            pos_info = room.get_position_info(new_pos_idx)
            
            # Get active experiments for current position
            active_experiments = room.get_experiments_for_position(new_pos_idx)
            
            # Create table data for active experiments
            table_data = []
            for exp in active_experiments:
                row = {
                    'Label': exp.get_label(),
                    'Duration (s)': exp.duration
                }
                
                # Add metrics
                for metric, value in exp.metrics.items():
                    row[f"Metric: {metric}"] = f"{value:.3f}"
                
                # Add flags that affect the simulation
                if 'flags' in exp.config:
                    for key, value in exp.config['flags'].items():
                        if key in ['source_weighting', 'specular_source_injection']:
                            row[f"Flag: {key}"] = str(value)
                
                table_data.append(row)
            
            # Create columns for table
            columns = [{"name": col, "id": col} for col in (table_data[0].keys() if table_data else [])]
            
            # Create room visualization with current room and highlighted position
            room_fig = self.create_room_visualization([room], highlight_pos_idx=new_pos_idx)
            
            # Create room header text
            room_header = f"Room: {room.name}, dim: {room.dimensions_str}, abs={room.absorption_str}"
            rt_header = room.theoretical_rt_str
            
            return new_room_idx, new_pos_idx, pos_info, room_fig, table_data, columns, room_header, rt_header

        # Update plot callbacks to use new room methods
        @app.callback(
            Output('rir-plot', 'figure'),
            [Input('current-room-idx', 'data'),
             Input('current-pos-idx', 'data')]
        )
        def update_rir_plot(room_idx, pos_idx):
            fig = go.Figure()
            
            room = self.rooms[room_names[room_idx]]
            active_experiments = room.get_experiments_for_position(pos_idx)
            
            for exp in active_experiments:
                fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.rir,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            fig.update_layout(
                title="Room Impulse Responses",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig

        @app.callback(
            Output('edc-plot', 'figure'),
            [Input('current-room-idx', 'data'),
             Input('current-pos-idx', 'data')]
        )
        def update_edc_plot(room_idx, pos_idx):
            fig = go.Figure()
            
            room = self.rooms[room_names[room_idx]]
            active_experiments = room.get_experiments_for_position(pos_idx)
            
            for exp in active_experiments:
                fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.edc,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            fig.update_layout(
                title="Energy Decay Curves",
                xaxis_title="Time (s)",
                yaxis_title="Energy (dB)",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig

        @app.callback(
            Output('ned-plot', 'figure'),
            [Input('current-room-idx', 'data'),
             Input('current-pos-idx', 'data')]
        )
        def update_ned_plot(room_idx, pos_idx):
            fig = go.Figure()
            
            room = self.rooms[room_names[room_idx]]
            active_experiments = room.get_experiments_for_position(pos_idx)
            
            for exp in active_experiments:
                fig.add_trace(go.Scatter(
                    x=exp.ned_time_axis,
                    y=exp.ned,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            fig.update_layout(
                title="Normalized Echo Density",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Echo Density",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig
        
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

    def _identify_potential_duplicates(self, experiments):
        """
        Identify potential duplicate experiments (same parameters but different IDs).
        
        Args:
            experiments (list): List of SDNExperiment objects to check
            
        Returns:
            dict: Dictionary mapping parameter hash to list of experiment IDs
        """
        # Group experiments by parameter signature
        param_groups = {}
        
        for exp in experiments:
            # Create a simplified parameter signature for comparison
            params = {}
            
            # Add flags that significantly affect the algorithm
            if 'flags' in exp.config:
                flags = exp.config['flags']
                for key in ['source_weighting', 'specular_source_injection', 
                           'scattering_matrix_update_coef', 'coef']:
                    if key in flags:
                        params[key] = flags[key]
            
            # Add room parameters
            if 'room_parameters' in exp.config:
                room = exp.config['room_parameters']
                for key in ['width', 'depth', 'height', 'absorption', 
                           'source x', 'source y', 'source z',
                           'mic x', 'mic y', 'mic z']:
                    if key in room:
                        params[key] = room[key]
            
            # Create a string representation for this parameter set
            param_str = json.dumps(params, sort_keys=True)
            
            # Group by parameter string
            if param_str not in param_groups:
                param_groups[param_str] = []
            param_groups[param_str].append(exp.experiment_id)
        
        # Filter to only include parameter sets with multiple experiments
        return {k: v for k, v in param_groups.items() if len(v) > 1}


def run_main_experiments():
    """Run experiments from main.py configurations."""
    # Import main module to access configurations
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import main
    
    # Create experiment manager
    manager = SDNExperimentManager()
    
    # Run experiments from main.py configurations
    if hasattr(main, 'sdn_tests') and hasattr(main, 'room_parameters'):
        for test_name, config in main.sdn_tests.items():
            if config.get('enabled', False):
                print(f"Running experiment: {test_name}")
                manager.run_experiment(config, main.room_parameters)
    
    return manager

if __name__ == "__main__":

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

    # Create an experiment manager
    manager = SDNExperimentManager()
    duration = 2
    # Run an experiment with a specific configuration
    experiment = manager.run_experiment(
        config={
            'label': 'SDN',
            'info': 'specular node receives all the psk',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 5,
            }
        },
        room_parameters=room_aes,
        duration=duration,
        fs=44100
    )

    experiment = manager.run_experiment(
        config={
            'label': 'SDN',
            'info': 'specular node receives all the psk',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 4,
            }
        },
        room_parameters=room_aes,
        duration=duration,
        fs=44100
    )
    room_aes["source x"] = 2
    experiment = manager.run_experiment(
        config={
            'label': 'SDN',
            'info': '',
            'flags': {
                # 'specular_source_injection': True,
                # 'source_weighting': 4,
            }
        },
        room_parameters=room_aes,
        duration=duration,
        fs=44100
    )

    experiment = manager.run_experiment(
        config={
            'label': 'SDN',
            'info': '',
            'flags': {
                # 'specular_source_injection': True,
                # 'source_weighting': 4,
            }
        },
        room_parameters=room_journal,
        duration=duration,
        fs=44100
    )

    """    # Or run all experiments from main.py
    # import main

    # for test_name, config in main.sdn_tests.items():
    #     if config.get('enabled', False):
    #         manager.run_experiment(config, main.room_parameters)"""

    # Launch visualization
    manager.plot()