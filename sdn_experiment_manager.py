import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import dash
from dash import dcc, html, callback, Input, Output, dash_table
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
            # Create a hash of the configuration to use as ID
            config_str = json.dumps(self._make_serializable(config), sort_keys=True)
            self.experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:10]
        else:
            self.experiment_id = experiment_id
            
        # Calculate metrics if not skipped
        if not skip_metrics:
            self._calculate_metrics()
    
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
    
    def __init__(self, experiments_dir='experiments'):
        """
        Initialize the experiment manager.
        
        Args:
            experiments_dir (str): Directory to store experiment data
        """
        self.experiments_dir = experiments_dir
        self.experiments = {}
        self.ensure_dir_exists()
        self.load_experiments()
    
    def ensure_dir_exists(self):
        """Ensure the experiments directory exists."""
        os.makedirs(self.experiments_dir, exist_ok=True)
    
    def load_experiments(self):
        """Load all experiments from the experiments directory."""
        self.experiments = {}
        
        # Check if the directory exists
        if not os.path.exists(self.experiments_dir):
            return
        
        # Load each experiment
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.json'):
                experiment_id = filename.split('.')[0]
                metadata_path = os.path.join(self.experiments_dir, filename)
                rir_path = os.path.join(self.experiments_dir, f"{experiment_id}.npy")
                
                # Load metadata
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Load RIR if it exists
                    if os.path.exists(rir_path):
                        rir = np.load(rir_path)
                        
                        # Create experiment object
                        experiment = SDNExperiment.from_dict(metadata, rir)
                        self.experiments[experiment_id] = experiment
                except Exception as e:
                    print(f"Error loading experiment {experiment_id}: {e}")
    
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
        # Create a temporary config with all parameters for ID generation
        full_config = {
            **config,
            'room_parameters': room_parameters,
            'duration': duration,
            'fs': fs
        }
        
        # Generate experiment ID using a temporary experiment object with skip_metrics=True
        temp_experiment = SDNExperiment(full_config, np.array([]), skip_metrics=True)
        experiment_id = temp_experiment.experiment_id
        
        # Check if experiment exists
        if experiment_id in self.experiments and not force_rerun:
            print(f"Experiment {experiment_id} already exists. Using cached results.")
            return self.experiments[experiment_id]
        
        # Setup room
        room = geometry.Room(
            room_parameters['width'], 
            room_parameters['depth'], 
            room_parameters['height']
        )
        room.set_microphone(
            room_parameters['mic x'], 
            room_parameters['mic y'], 
            room_parameters['mic z']
        )
        room.set_source(
            room_parameters['source x'], 
            room_parameters['source y'], 
            room_parameters['source z'],
            signal="will be replaced", 
            Fs=fs
        )
        
        # Calculate reflection coefficient
        reflection = np.sqrt(1 - room_parameters['absorption'])
        room.wallAttenuation = [reflection] * 6
        
        # Setup signal
        num_samples = int(fs * duration)
        impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
        room.source.signal = impulse_dirac['signal']
        
        # Create SDN instance with configured flags
        sdn = DelayNetwork(room, Fs=fs, label=config.get('label', ''), **config.get('flags', {}))
        
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
        self.save_experiment(experiment)
        
        # Add to experiments dictionary
        self.experiments[experiment_id] = experiment
        
        return experiment
    
    def save_experiment(self, experiment):
        """
        Save an experiment to disk.
        
        Args:
            experiment (SDNExperiment): The experiment to save
        """
        # Ensure directory exists
        self.ensure_dir_exists()
        
        # Save metadata
        metadata_path = os.path.join(self.experiments_dir, f"{experiment.experiment_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # Save RIR
        rir_path = os.path.join(self.experiments_dir, f"{experiment.experiment_id}.npy")
        np.save(rir_path, experiment.rir)
    
    def get_experiment(self, experiment_id):
        """
        Get an experiment by ID.
        
        Args:
            experiment_id (str): The ID of the experiment
            
        Returns:
            SDNExperiment: The experiment object
        """
        return self.experiments.get(experiment_id)
    
    def get_experiments_by_label(self, label):
        """
        Get experiments by label.
        
        Args:
            label (str): The label to search for
            
        Returns:
            list: List of matching experiments
        """
        return [exp for exp in self.experiments.values() if label in exp.get_label()]
    
    def get_all_experiments(self):
        """
        Get all experiments.
        
        Returns:
            list: List of all experiments
        """
        return list(self.experiments.values())
    
    def plot(self, *experiment_ids, max_experiments=10):
        """
        Launch an interactive dashboard to visualize experiments.
        
        Args:
            *experiment_ids: IDs of experiments to plot. If none provided, plots all experiments.
            max_experiments (int): Maximum number of experiments to plot if no IDs are provided
        """
        # If no experiment IDs are provided, use all experiments (up to max_experiments)
        if not experiment_ids:
            experiments = list(self.experiments.values())[:max_experiments]
        else:
            experiments = [self.experiments[exp_id] for exp_id in experiment_ids if exp_id in self.experiments]
        
        if not experiments:
            print("No experiments to plot.")
            return
        
        # Create a Dash app
        app = dash.Dash(__name__)
        server = app.server
        port = 8050  # Default Dash port
        
        # Create experiment table data
        table_data = []
        for exp in experiments:
            row = {
                'ID': exp.experiment_id,
                'Label': exp.get_label(),
                'Timestamp': exp.timestamp,
                'Duration (s)': exp.duration
            }
            
            # Add metrics
            for metric, value in exp.metrics.items():
                row[metric] = f"{value:.3f}"
            
            # Add key configuration parameters
            if 'flags' in exp.config:
                for key, value in exp.config['flags'].items():
                    row[key] = str(value)
            
            table_data.append(row)
        
        # Create DataFrame for the table
        df = pd.DataFrame(table_data)
        
        # Create app layout
        app.layout = html.Div([
            html.H1("SDN Experiment Visualization", style={'textAlign': 'center'}),
            
            # Experiment table
            html.Div([
                html.H3("Experiment Parameters"),
                dash_table.DataTable(
                    id='experiment-table',
                    columns=[{"name": col, "id": col} for col in df.columns],
                    data=df.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
            ]),
            
            # Tabs for different visualizations
            dcc.Tabs([
                # RIR Tab
                dcc.Tab(label="Room Impulse Responses", children=[
                    dcc.Graph(id='rir-plot', style={'height': '70vh'})
                ]),
                
                # EDC Tab
                dcc.Tab(label="Energy Decay Curves", children=[
                    dcc.Graph(id='edc-plot', style={'height': '70vh'})
                ]),
                
                # NED Tab
                dcc.Tab(label="Normalized Echo Density", children=[
                    dcc.Graph(id='ned-plot', style={'height': '70vh'})
                ])
            ])
        ])
        
        # Callback for RIR plot
        @app.callback(
            Output('rir-plot', 'figure'),
            Input('experiment-table', 'derived_virtual_data'),
            Input('experiment-table', 'derived_virtual_selected_rows')
        )
        def update_rir_plot(rows, selected_rows):
            # Create figure
            fig = go.Figure()
            
            # Handle None values for rows
            if rows is None:
                rows = df.to_dict('records')
            
            # Add trace for each experiment
            for i, row in enumerate(rows):
                exp_id = row['ID']
                exp = self.experiments[exp_id]
                
                fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.rir,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            # Update layout
            fig.update_layout(
                title="Room Impulse Responses",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig
        
        # Callback for EDC plot
        @app.callback(
            Output('edc-plot', 'figure'),
            Input('experiment-table', 'derived_virtual_data'),
            Input('experiment-table', 'derived_virtual_selected_rows')
        )
        def update_edc_plot(rows, selected_rows):
            # Create figure
            fig = go.Figure()
            
            # Handle None values for rows
            if rows is None:
                rows = df.to_dict('records')
            
            # Add trace for each experiment
            for i, row in enumerate(rows):
                exp_id = row['ID']
                exp = self.experiments[exp_id]
                
                fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.edc,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            # Update layout
            fig.update_layout(
                title="Energy Decay Curves",
                xaxis_title="Time (s)",
                yaxis_title="Energy (dB)",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return fig
        
        # Callback for NED plot
        @app.callback(
            Output('ned-plot', 'figure'),
            Input('experiment-table', 'derived_virtual_data'),
            Input('experiment-table', 'derived_virtual_selected_rows')
        )
        def update_ned_plot(rows, selected_rows):
            # Create figure
            fig = go.Figure()
            
            # Handle None values for rows
            if rows is None:
                rows = df.to_dict('records')
            
            # Add trace for each experiment
            for i, row in enumerate(rows):
                exp_id = row['ID']
                exp = self.experiments[exp_id]
                
                fig.add_trace(go.Scatter(
                    x=exp.ned_time_axis,
                    y=exp.ned,
                    mode='lines',
                    name=exp.get_label()
                ))
            
            # Update layout
            fig.update_layout(
                title="Normalized Echo Density",
                xaxis_title="Time (s)",
                yaxis_title="Normalized Echo Density",
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
        
        # Print message to console with clear server address
        server_address = f"http://127.0.0.1:{port}/"
        print("\n" + "=" * 70)
        print(f"Dash server is running at: {server_address}")
        print("If the browser doesn't open automatically, please copy and paste the URL above into your browser.")
        print("=" * 70)
        
        # Run the app
        app.run_server(debug=True, port=port)


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
            'info': 'source weighting = 5: specular receives all psk',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 5,
            }
        },
        room_parameters=room_aes,
        duration=duration,
        fs=44100
    )

    """    # Or run all experiments from main.py
    import main

    for test_name, config in main.sdn_tests.items():
        if config.get('enabled', False):
            manager.run_experiment(config, main.room_parameters)"""

    # Launch visualization
    manager.plot()