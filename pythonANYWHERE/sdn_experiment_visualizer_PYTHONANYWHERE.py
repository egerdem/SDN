import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
import numpy as np
# import webbrowser
# from threading import Timer
from sdn_experiment_manager_pythonANY import Room
# import analysis as an
import io
import base64
import soundfile as sf
# import plot_room as pp
# import EchoDensity as ned
import plotly.express as px
from scipy.signal import convolve
import time
class ExperimentVisualizer:
    print("visualizer started")

    # Class-level constants for error metrics
    COMPARISON_TYPES = [
        {'label': 'Energy Decay Curve', 'value': 'edc'},
        {'label': 'Smoothed Energy', 'value': 'smoothed_energy'},
        {'label': 'Raw Energy', 'value': 'energy'}
    ]

    ERROR_METRICS = [
        {'label': 'RMSE', 'value': 'rmse'},
        {'label': 'MAE', 'value': 'mae'},
        {'label': 'Median', 'value': 'median'},
        {'label': 'Sum', 'value': 'sum'}
    ]

    """Class to visualize SDN experiment data using Dash."""

    def __init__(self, manager=None):
        """
        Initialize the visualizer with an experiment manager.

        Args:
            manager: An SDNExperimentManager instance to visualize
        """
        self.manager = manager
        # Dictionary to store audio data for each experiment
        self.audio_cache = {}
        # Dictionary to store bongo convolution audio data
        self.bongo_convolution_cache = {}
        # Dictionary to store calculated metrics for on-demand calculations
        self.metrics_cache = {}
        # Flag to control whether to use on-demand calculations
        self.use_on_demand_calculations = False
        # Bongo sound data
        self.bongo_data = None
        self.bongo_fs = None

        # Load trumpet sound
        try:
            self.trumpet_data, self.trumpet_fs = sf.read('trumpet_cr_Track34.wav')
            print("Trumpet sound loaded successfully")
        except Exception as e:
            print(f"Error loading trumpet sound: {str(e)}")
            self.trumpet_data = None
            self.trumpet_fs = None

        # Dictionary to store trumpet convolution audio data
        self.trumpet_convolution_cache = {}

        # Import analysis modules and store them as class attributes
        # This makes them accessible in callback functions via self
        import plot_room as pp
        import EchoDensity as ned
        import analysis as an
        self.pp = pp
        self.ned = ned
        self.an = an

        # Load bongo sound
        try:
            self.bongo_data, self.bongo_fs = sf.read('002_bongo_original_cr.wav')
            print("Bongo sound loaded successfully")
        except Exception as e:
            print(f"Error loading bongo sound: {str(e)}")
            self.bongo_data = None
            self.bongo_fs = None

    def calculate_metrics_for_experiment(self, experiment):
        # only called if use_on_demand_calculations is True
        """
        Calculate metrics for an experiment on demand, with caching.

        Args:
            experiment: The SDNExperiment object to calculate metrics for

        Returns:
            dict: Dictionary containing calculated metrics (edc, ned, etc.)
        """
        # Check if we already have metrics for this experiment
        if experiment.experiment_id in self.metrics_cache:
            return self.metrics_cache[experiment.experiment_id]

        # Create new metrics dictionary
        metrics = {}

        # Get RIR data from the experiment
        rir = experiment.rir
        fs = experiment.fs
        duration = experiment.duration

        # Skip metrics calculation if RIR is empty
        if len(rir) == 0:
            metrics['edc'] = np.array([])
            metrics['ned'] = np.array([])
            metrics['time_axis'] = np.array([])
            metrics['ned_time_axis'] = np.array([])
            metrics['edc_time_axis'] = np.array([])
            self.metrics_cache[experiment.experiment_id] = metrics
            return metrics

        # Calculate RT60 if the RIR is long enough
        if duration > 0.7:
            metrics['rt60'] = self.pp.calculate_rt60_from_rir(rir, fs, plot=False)

        # Calculate EDC
        metrics['edc'] = self.an.compute_edc(rir, fs, label=experiment.get_label()['label'], plot=False)

        # Calculate NED
        metrics['ned'] = self.ned.echoDensityProfile(rir, fs=fs)

        # Time axis for plotting
        metrics['time_axis'] = np.arange(len(rir)) / fs
        metrics['ned_time_axis'] = np.arange(len(metrics['ned'])) / fs
        metrics['edc_time_axis'] = np.arange(len(metrics['edc'])) / fs

        # Cache the results
        self.metrics_cache[experiment.experiment_id] = metrics
        return metrics

    def get_experiment_plot_data(self, experiment):
        """
        Get plot data for an experiment, either from cache or by calculating.

        Args:
            experiment: The SDNExperiment object

        Returns:
            dict: Data ready for plotting (edc, ned, time axes)
        """
        if self.use_on_demand_calculations:
            # Calculate or get from cache
            metrics = self.calculate_metrics_for_experiment(experiment)
            return {
                'edc': metrics['edc'],
                'ned': metrics['ned'],
                'time_axis': metrics['time_axis'],
                'ned_time_axis': metrics['ned_time_axis'],
                'edc_time_axis': metrics['edc_time_axis']
            }
        else:
            # Use pre-calculated values from the experiment
            return {
                'edc': experiment.edc,
                'ned': experiment.ned,
                'time_axis': experiment.time_axis,
                'ned_time_axis': experiment.ned_time_axis,
                'edc_time_axis': experiment.edc_time_axis
            }

    def create_room_visualization(self, experiments, highlight_pos_idx=None, show_error_contour=False,
                                  reference_id=None, comparison_type='edc', error_metric='rmse'):
        """
        Create a 2D top-view visualization of the room with source and receiver positions.

        Args:
            experiments (list): List of SDNExperiment objects or Room objects to visualize
            highlight_pos_idx (int, optional): Index of source-mic pair to highlight
            show_error_contour (bool): Whether to show error contour plot
            reference_id (str): ID of the reference experiment for error contour
            comparison_type (str): Type of comparison ('edc', 'smoothed_energy', 'energy')
            error_metric (str): Error metric to use ('rmse', 'mae', 'median', 'sum')

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

        # Draw all source-mic pairs with reduced opacity
        if room.source_mic_pairs:
            # Extract all unique source positions
            source_positions = []
            mic_positions = []

            for src_mic_pair in room.source_mic_pairs:
                source_pos, mic_pos = src_mic_pair

                # Add to unique positions lists if not already present
                if all(not np.array_equal(source_pos, pos) for pos in source_positions):
                    source_positions.append(source_pos)
                if all(not np.array_equal(mic_pos, pos) for pos in mic_positions):
                    mic_positions.append(mic_pos)

            # Add all sources with low opacity
            x_sources = [pos[0] for pos in source_positions]
            y_sources = [pos[1] for pos in source_positions]
            fig.add_trace(go.Scatter(
                x=x_sources, y=y_sources,
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2),
                    opacity=0.2
                ),
                customdata=[[i, 'source'] for i in range(len(source_positions))],
                hoverinfo='text',
                hovertext=[f"Source {i + 1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                           for i, pos in enumerate(source_positions)],
                name='All Sources',
                showlegend=False
            ))

            # Add all mics with low opacity
            x_mics = [pos[0] for pos in mic_positions]
            y_mics = [pos[1] for pos in mic_positions]
            fig.add_trace(go.Scatter(
                x=x_mics, y=y_mics,
                mode='markers',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2),
                    opacity=0.2
                ),
                customdata=[[i, 'receiver'] for i in range(len(mic_positions))],
                hoverinfo='text',
                hovertext=[f"Receiver {i + 1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.1f})"
                           for i, pos in enumerate(mic_positions)],
                name='All Microphones',
                showlegend=False
            ))

        # Get current source-mic pair and highlight it
        current_pos = None
        if highlight_pos_idx is not None and room.source_mic_pairs:
            current_pos = room.source_mic_pairs[highlight_pos_idx % len(room.source_mic_pairs)]
            source_pos, mic_pos = current_pos

            # Add highlighted source marker (red)
            fig.add_trace(go.Scatter(
                x=[source_pos[0]], y=[source_pos[1]],
                mode='markers',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Active Source'
            ))

            # Add highlighted microphone marker (green)
            fig.add_trace(go.Scatter(
                x=[mic_pos[0]], y=[mic_pos[1]],
                mode='markers',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Active Microphone'
            ))

        # Add some padding
        padding = 0.5
        x_range = [min_x - padding, max_x + padding]
        y_range = [min_y - padding, max_y + padding]

        # Update layout
        fig.update_layout(
            # title=f"{room.display_name}: {room.dimensions_str}", # commented out the room plot title
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
                yanchor="bottom",
                y=0.11,  # Move legend to the bottom
                xanchor="right",
                x=0.81,  # Move legend to the right
                bgcolor='rgba(40, 44, 52, 0.7)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1,
                font=dict(size=10)  # Decrease font size of legend
            ),
            margin=dict(t=30, b=0, l=0, r=0),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )

        # Add error contour if requested
        if show_error_contour and reference_id and len(experiments) >= 2:
            room = experiments[0]
            source_pos = room.source_mic_pairs[highlight_pos_idx][0] if highlight_pos_idx is not None else None

            # Get reference and comparison experiments
            ref_exp = None
            comp_exp = None
            for exp in room.experiments.values():
                if exp.experiment_id == reference_id:
                    ref_exp = exp
                elif exp.config.get('method') == 'SDN':  # Use SDN as comparison
                    comp_exp = exp

            if ref_exp and comp_exp and source_pos:
                contour = self.create_error_contour(
                    room, ref_exp, comp_exp,
                    comparison_type, error_metric  # Use the passed parameters
                )
                fig.add_trace(contour)

        return fig

    def create_app_for_deployment(self):
        """Create and configure the Dash app for deployment without running the server."""
        # Make sure we have a manager
        # Get the appropriate manager if not provided

        if not self.manager.projects:
            print("No rooms with experiments to visualize.")
            return

        # Create Dash app
        app = dash.Dash(__name__)
        server = app.server  # Uncomment this line to fix the NoneType error

        @app.server.after_request
        def add_header(response):
            response.headers["ngrok-skip-browser-warning"] = "true"
            return response

        # Add Flask route for bongo convolution
        @app.server.route('/convolve-bongo')
        def convolve_bongo():
            from flask import request, jsonify

            experiment_id = request.args.get('experiment_id')
            if not experiment_id:
                return jsonify({'error': 'No experiment ID provided'}), 400

            # Find the experiment
            found_experiment = None
            for room_name in self.manager.projects:
                room = self.manager.projects[room_name]
                for pos_idx in range(len(room.source_mic_pairs)):
                    experiments = room.get_experiments_for_position(pos_idx)
                    for exp in experiments:
                        if exp.experiment_id == experiment_id:
                            found_experiment = exp
                            break
                    if found_experiment:
                        break
                if found_experiment:
                    break

            if not found_experiment:
                return jsonify({'error': 'Experiment not found'}), 404

            # Get or generate the bongo convolution
            audio_data = self.generate_bongo_convolution_audio(
                found_experiment.rir,
                found_experiment.fs,
                experiment_id
            )

            if audio_data is None:
                return jsonify({'error': 'Failed to generate convolution'}), 500

            return jsonify({'audio_data': audio_data})

        # Add Flask route for trumpet convolution
        @app.server.route('/convolve-trumpet')
        def convolve_trumpet():
            from flask import request, jsonify

            experiment_id = request.args.get('experiment_id')
            if not experiment_id:
                return jsonify({'error': 'No experiment ID provided'}), 400

            # Find the experiment - reuse the same search logic
            found_experiment = None
            for room_name in self.manager.projects:
                room = self.manager.projects[room_name]
                for pos_idx in range(len(room.source_mic_pairs)):
                    experiments = room.get_experiments_for_position(pos_idx)
                    for exp in experiments:
                        if exp.experiment_id == experiment_id:
                            found_experiment = exp
                            break
                    if found_experiment:
                        break
                if found_experiment:
                    break

            if not found_experiment:
                return jsonify({'error': 'Experiment not found'}), 404

            # Get or generate the trumpet convolution
            audio_data = self.generate_trumpet_convolution_audio(
                found_experiment.rir,
                found_experiment.fs,
                experiment_id
            )

            if audio_data is None:
                return jsonify({'error': 'Failed to generate convolution'}), 500

            return jsonify({'audio_data': audio_data})

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

        # Get list of rooms and create room objects list
        room_names = list(self.manager.projects.keys())
        current_room_idx = 0

        # Create app layout (same as in SDNExperimentManager)
        app.layout = html.Div([
            # Room navigation header
            html.Div([
                # First row: Room selector and heading
                html.Div([
                    # Container with fixed width and positioning
                    html.Div([
                        # Dropdown for room selection
                        dcc.Dropdown(
                            id='room-selector',
                            options=[{'label': self.manager.projects[name].display_name, 'value': i}
                                     for i, name in enumerate(room_names)],
                            value=current_room_idx,
                            style={
                                'width': '240px',
                                'backgroundColor': dark_theme['paper_bg'],
                                'color': dark_theme['text']
                            }
                        ),
                    ], style={
                        'position': 'absolute',
                        'left': '40%',
                        'transform': 'translateX(-50%)',
                        'zIndex': 1
                    }),

                    # Room header with fixed position
                    html.H2(
                        id='room-header',
                        style={
                            'margin': '0',
                            'color': dark_theme['text'],
                            'position': 'absolute',
                            'left': 'calc(40% + 140px)',  # 25% (dropdown center) + half dropdown width + some spacing
                            'whiteSpace': 'nowrap',
                            'fontSize': '1.2em'  # Make header smaller
                        }
                    ),
                ], style={
                    'position': 'relative',
                    'height': '30px',  # Reduce height
                    'marginBottom': '5px'  # Reduce margin
                }),

                # Second row: Navigation buttons and RT info
                html.Div([
                    # Navigation buttons
                    html.Button('←', id='prev-room', style={
                        'fontSize': 18,  # Reduce button size
                        'marginRight': '10px',
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 10px',  # Reduce padding
                        'cursor': 'pointer',
                        'display': 'inline-block',
                        'verticalAlign': 'middle'
                    }),
                    html.Button('→', id='next-room', style={
                        'fontSize': 18,  # Reduce button size
                        'marginRight': '20px',
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 10px',  # Reduce padding
                        'cursor': 'pointer',
                        'display': 'inline-block',
                        'verticalAlign': 'middle'
                    }),
                    html.H3(
                        id='rt-header',
                        style={
                            'margin': '0',
                            'color': dark_theme['accent'],
                            'display': 'inline-block',
                            'verticalAlign': 'middle',
                            'fontSize': '0.9em'  # Make RT header smaller
                        }
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                }),
                dcc.Store(id='current-room-idx', data=current_room_idx)
            ], style={
                'textAlign': 'center',
                'margin': '2px',  # Reduce margin
                'marginBottom': '5px'  # Reduce bottom margin
            }),

            # Main content area
            html.Div([
                # Left side - plots and table
                html.Div([
                    # Time range selector and layout switch
                    html.Div([
                        html.H4("Time Range:",
                                style={'display': 'inline-block', 'marginRight': '15px', 'color': dark_theme['text'],
                                       'fontSize': '0.9em'}),  # Make text smaller
                        dcc.RadioItems(
                            id='time-range-selector',
                            options=[
                                {'label': 'Early Part (50ms)', 'value': 0.05},
                                {'label': 'First 0.5s', 'value': 0.5},
                                {'label': 'Whole RIR', 'value': 'full'}
                            ],
                            value=0.05,
                            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'cursor': 'pointer',
                                        'fontSize': '0.9em'},  # Make text smaller
                            style={'color': dark_theme['text']},
                            inputStyle={'marginRight': '5px'}
                        ),
                        # Layout switch moved to top right
                        html.Div([
                            dcc.RadioItems(
                                id='layout-switch',
                                options=[
                                    {'label': 'Laptop', 'value': 'laptop'},
                                    {'label': 'Monitor', 'value': 'monitor'}
                                ],
                                value='monitor',
                                labelStyle={'display': 'inline-block', 'marginRight': '10px', 'cursor': 'pointer',
                                            'fontSize': '0.9em'},  # Make text smaller
                                style={'color': dark_theme['text']},
                                inputStyle={'marginRight': '5px'}
                            )
                        ], style={'position': 'absolute', 'right': '20px', 'top': '5px'})  # Position in top right
                    ], style={'margin': '5px 0 0 20px', 'display': 'flex', 'alignItems': 'center',
                              'position': 'relative'}),  # Add relative positioning

                    # Plots container with conditional rendering
                    html.Div([
                        # Monitor layout (simultaneous plots)
                        html.Div([
                            # First row: EDC and NED plots side by side
                            html.Div([
                                # Left side - EDC plot
                                html.Div([
                                    dcc.Graph(id='edc-plot-monitor', style={'height': '40vh'})
                                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                                # Right side - NED plot
                                html.Div([
                                    dcc.Graph(id='ned-plot-monitor', style={'height': '40vh'})
                                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row'}),

                            # Second row: RIR plot full width
                            html.Div([
                                dcc.Graph(id='rir-plot-monitor', style={'height': '40vh'})
                            ], style={'width': '100%', 'marginTop': '10px'}),
                        ], id='monitor-layout', style={'display': 'none'}),

                        # Laptop layout (tabs)
                        html.Div([
                            dcc.Tabs([
                                dcc.Tab(label="RIR", children=[
                                    dcc.Graph(id='rir-plot-laptop', style={'height': '50vh'})
                                ], style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                }, selected_style={
                                    'backgroundColor': dark_theme['header_bg'],
                                    'color': dark_theme['accent'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                }),
                                dcc.Tab(label="EDC", children=[
                                    dcc.Graph(id='edc-plot-laptop', style={'height': '50vh'})
                                ], style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                }, selected_style={
                                    'backgroundColor': dark_theme['header_bg'],
                                    'color': dark_theme['accent'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                }),
                                dcc.Tab(label="NED", children=[
                                    dcc.Graph(id='ned-plot-laptop', style={'height': '50vh'})
                                ], style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                }, selected_style={
                                    'backgroundColor': dark_theme['header_bg'],
                                    'color': dark_theme['accent'],
                                    'height': '40px',
                                    'padding': '6px',
                                    'display': 'flex',
                                    'alignItems': 'center'
                                })
                            ], style={'width': '100%'})
                        ], id='laptop-layout', style={'display': 'none'}),

                        # Hidden audio components
                        html.Div(id='audio-components', style={'display': 'none'})
                    ], style={'width': '100%'}),

                    # Experiment table in a centered container
                    html.Div([
                        html.H3("Active Experiments",
                                style={'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '5px',
                                       'color': dark_theme['text']}),
                        # Add error metric controls
                        html.Div([
                            html.Div([
                                html.Label("Comparison Type:",
                                           style={'color': dark_theme['text'], 'marginRight': '10px'}),
                                dcc.Dropdown(
                                    id='comparison-type-selector',
                                    options=self.COMPARISON_TYPES,
                                    value='edc',
                                    style={
                                        'width': '200px',
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text']
                                    }
                                )
                            ], style={'marginRight': '20px', 'display': 'inline-block'}),
                            html.Div([
                                html.Label("Error Metric:", style={'color': dark_theme['text'], 'marginRight': '10px'}),
                                dcc.Dropdown(
                                    id='error-metric-selector',
                                    options=self.ERROR_METRICS,
                                    value='rmse',
                                    style={
                                        'width': '150px',
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text']
                                    }
                                )
                            ], style={'display': 'inline-block'}),
                            html.Div([
                                html.Label("Reference:", style={'color': dark_theme['text'], 'marginRight': '10px'}),
                                dcc.Dropdown(
                                    id='reference-selector',
                                    options=[],  # Will be populated in callback
                                    style={
                                        'width': '200px',
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text']
                                    }
                                )
                            ], style={'marginLeft': '20px', 'display': 'inline-block'})
                        ], style={'marginBottom': '5px'}),
                        # Create a flex container for tables
                        html.Div([
                            # Main experiment table
                            html.Div([
                                dash_table.DataTable(
                                    id='experiment-table',
                                    style_table={'marginTop': '28px'},  # Add margin to align with mean error table
                                    style_cell={
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text'],
                                        'textAlign': 'left',
                                        'padding': '2px 5px',
                                        'height': '24px',
                                        'minHeight': '24px',
                                        'maxHeight': '24px',
                                        'whiteSpace': 'nowrap',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'lineHeight': '24px'  # Match header row height
                                    },
                                    style_header={
                                        'backgroundColor': dark_theme['header_bg'],
                                        'fontWeight': 'bold',
                                        'height': '24px',
                                        'lineHeight': '24px'
                                    },
                                    style_cell_conditional=[
                                        {
                                            'if': {'column_id': 'play'},
                                            'width': '40px',
                                            'textAlign': 'center',
                                            'padding': '2px 0'
                                        },
                                        {
                                            'if': {'column_id': 'bongo'},
                                            'width': '40px',
                                            'textAlign': 'center',
                                            'padding': '2px 0'
                                        },
                                        {
                                            'if': {'column_id': 'trumpet'},
                                            'width': '40px',
                                            'textAlign': 'center',
                                            'padding': '2px 0'
                                        }
                                    ],
                                    style_data_conditional=[
                                        {
                                            'if': {'column_id': 'play'},
                                            'height': '24px'
                                        },
                                        {
                                            'if': {'column_id': 'bongo'},
                                            'height': '24px'
                                        },
                                        {
                                            'if': {'column_id': 'trumpet'},
                                            'height': '24px'
                                        }
                                    ],
                                    markdown_options={'html': True},
                                    columns=[
                                        {'name': 'Play', 'id': 'play', 'presentation': 'markdown'},
                                        {'name': '🥁', 'id': 'bongo', 'presentation': 'markdown'},
                                        {'name': '🎺', 'id': 'trumpet', 'presentation': 'markdown'},
                                        {'name': 'ID', 'id': 'id'},
                                        {'name': 'Method', 'id': 'method'},
                                        {'name': 'Label', 'id': 'label'},
                                        {'name': 'RT60', 'id': 'rt60'},
                                        {'name': 'Total Energy', 'id': 'total_energy'},
                                        {'name': f'{self.ERROR_METRICS[0]["label"]} [50ms]', 'id': 'error_50ms'},
                                        {'name': f'{self.ERROR_METRICS[0]["label"]} [500ms]', 'id': 'error_500ms'}
                                    ],
                                    data=[]  # Will be populated in callback
                                ),
                            ], style={'flex': '3'}),
                            # Mean error table
                            html.Div([
                                html.H4(f'Mean {self.ERROR_METRICS[0]["label"]} (all positions)',
                                        style={
                                            'textAlign': 'center',
                                            'margin': '0',
                                            'marginBottom': '4px',
                                            'color': dark_theme['text'],
                                            'height': '24px',
                                            'lineHeight': '24px'
                                        }),
                                dash_table.DataTable(
                                    id='mean-error-table',
                                    columns=[
                                        {'name': f'{self.ERROR_METRICS[0]["label"]} [50ms]', 'id': 'error_50ms'},
                                        {'name': f'{self.ERROR_METRICS[0]["label"]} [500ms]', 'id': 'error_500ms'},
                                        {'name': 'Mean Total Energy', 'id': 'total_energy'}
                                    ],
                                    data=[],
                                    style_table={},
                                    style_header={
                                        'backgroundColor': dark_theme['header_bg'],
                                        'color': dark_theme['text'],
                                        'fontWeight': 'bold',
                                        'height': '24px',
                                        'lineHeight': '24px'
                                    },
                                    style_cell={
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text'],
                                        'textAlign': 'center',
                                        'padding': '2px 5px',
                                        'height': '24px',
                                        'minHeight': '24px',
                                        'maxHeight': '24px',
                                        'whiteSpace': 'nowrap',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                        'lineHeight': '24px'  # Match header row height
                                    }
                                )
                            ], style={'flex': '1', 'marginLeft': '10px', 'display': 'flex', 'flexDirection': 'column',
                                      'justifyContent': 'flex-start'})
                        ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '30px'})
                    ], style={'width': '71.4%', 'margin': '0 auto', 'marginLeft': 'auto', 'marginRight': '0'})
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right side - room visualization
                html.Div([
                    html.H3(
                        # "Room Layout (Top View)",
                        style={'textAlign': 'left', 'marginBottom': '5px', 'marginTop': '60px',
                               'color': dark_theme['text']}),
                    dcc.Store(id='current-pos-idx', data=0),
                    dcc.Graph(
                        id='room-plot',
                        style={
                            'height': '40vh',
                            'marginTop': '0px',
                            'backgroundColor': dark_theme['paper_bg'],
                            'width': '100%'
                        },
                        config={'displayModeBar': True}
                    ),

                    # Source and receiver dropdown selectors
                    html.Div([
                        # Source selector
                        html.Div([
                            dcc.Dropdown(
                                id='source-selector',
                                options=[],  # Will be populated in callback
                                style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],
                                    'width': '100%'
                                },
                                className='dropdown-light-text'
                            )
                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),

                        # Receiver selector
                        html.Div([
                            dcc.Dropdown(
                                id='receiver-selector',
                                options=[],  # Will be populated in callback
                                style={
                                    'backgroundColor': dark_theme['paper_bg'],
                                    'color': dark_theme['text'],
                                    'width': '100%'
                                },
                                className='dropdown-light-text'
                            )
                        ], style={'width': '48%', 'display': 'inline-block'})
                    ], style={'padding': '10px 0', 'marginBottom': '0px', 'width': '100%'}),

                    # Navigation buttons
                    html.Div([
                        html.Button('←', id='prev-pos', style={
                            'fontSize': 16,  # Further reduced size
                            'marginRight': '10px',
                            'backgroundColor': dark_theme['button_bg'],
                            'color': dark_theme['button_text'],
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '0px 8px',  # Reduced padding
                            'height': '25px',  # Reduced height
                            'width': '25px'  # Reduced width
                        }),
                        html.Button('→', id='next-pos', style={
                            'fontSize': 16,  # Further reduced size
                            'backgroundColor': dark_theme['button_bg'],
                            'color': dark_theme['button_text'],
                            'border': 'none',
                            'borderRadius': '4px',
                            'padding': '0px 8px',  # Reduced padding
                            'height': '25px',  # Reduced height
                            'width': '25px'  # Reduced width
                        })
                    ], style={'textAlign': 'center', 'marginTop': '2px', 'width': '100%'})
                ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
            ], style={'display': 'flex', 'alignItems': 'flex-start'})
        ], style={
            'backgroundColor': dark_theme['background'],
            'minHeight': '100vh',
            'fontFamily': 'Arial, sans-serif',
            'padding': '10px'
        })

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
                    /* Button hover effects */
                    button {
                        transition: all 0.2s ease-in-out !important;
                    }
                    button:hover {
                        background-color: #505050 !important;
                        transform: scale(1.05) !important;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
                    }
                    button:active {
                        transform: scale(0.95) !important;
                    }
                    /* Play button styles */
                    .play-button {
                        background-color: #404040;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        width: 20px;
                        height: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                        padding: 0;
                        line-height: 1;
                        min-height: 20px;
                    }
                    .play-button:hover {
                        background-color: #505050;
                    }
                    .bongo-button {
                        background-color: #604020;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        width: 20px;
                        height: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                        padding: 0;
                        line-height: 1;
                        min-height: 20px;
                    }
                    .bongo-button:hover {
                        background-color: #705030;
                    }
                    .trumpet-button {
                        background-color: #404080;
                        color: #e0e0e0;
                        border: none;
                        border-radius: 4px;
                        width: 20px;
                        height: 20px;
                        cursor: pointer;
                        font-size: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto;
                        padding: 0;
                        line-height: 1;
                        min-height: 20px;
                    }
                    .trumpet-button:hover {
                        background-color: #505090;
                    }
                    /* Table styles */
                    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                        background-color: #282c34 !important;
                        color: #e0e0e0 !important;
                        height: 24px !important;
                        min-height: 24px !important;
                        max-height: 24px !important;
                        line-height: 24px !important;
                        white-space: nowrap !important;
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                    }
                    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                        background-color: #1e2129 !important;
                        color: #e0e0e0 !important;
                        height: 24px !important;
                        line-height: 24px !important;
                    }
                    /* Remove any potential scrollbars in cells */
                    .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td > div {
                        overflow: visible !important;
                    }
                    /* Ensure consistent cell sizing */
                    .dash-table-container .dash-cell-value {
                        overflow: hidden !important;
                        text-overflow: ellipsis !important;
                        padding: 2px 5px !important;
                    }

                    /* Additional table styles to remove fixed height and scrollbars */
                    #experiment-table .dash-spreadsheet, 
                    #mean-error-table .dash-spreadsheet {
                        max-height: none !important;
                    }

                    #experiment-table .dash-spreadsheet-container,
                    #mean-error-table .dash-spreadsheet-container {
                        max-height: none !important;
                        overflow: visible !important;
                    }

                    #experiment-table .dash-spreadsheet-inner,
                    #mean-error-table .dash-spreadsheet-inner {
                        max-height: none !important;
                        overflow: visible !important;
                    }
                </style>
                <script>
                    // Function to handle play button clicks
                    document.addEventListener('DOMContentLoaded', function() {
                        let animationFrameId = null;
                        let lastUpdateTime = 0;
                        const UPDATE_INTERVAL = 16; // 60fps
                        let currentPlot = null;
                        let currentTraceIndex = -1;

                        // Function to find RIR plot and time bar trace
                        function findRIRPlotAndTimeBar() {
                            const plots = document.querySelectorAll('.js-plotly-plot');
                            for (const plot of plots) {
                                const traces = plot.data;
                                const timeBarIndex = traces.findIndex(trace => trace.name === 'Time Bar');
                                if (timeBarIndex !== -1) {
                                    return { plot, timeBarIndex };
                                }
                            }
                            return null;
                        }

                        // Function to update time bar
                        function updateTimeBar(audioElement) {
                            if (!audioElement || !currentPlot) return;

                            const currentTime = performance.now();
                            if (currentTime - lastUpdateTime >= UPDATE_INTERVAL) {
                                // Update only the time bar trace using restyle
                                Plotly.restyle(currentPlot, {
                                    x: [[audioElement.currentTime, audioElement.currentTime]]
                                }, [currentTraceIndex]);

                                lastUpdateTime = currentTime;
                            }

                            if (!audioElement.paused) {
                                animationFrameId = requestAnimationFrame(() => updateTimeBar(audioElement));
                            }
                        }

                        // Use event delegation for dynamically added buttons
                        document.body.addEventListener('click', function(e) {
                            if (e.target && e.target.id) {
                                if (e.target.id.startsWith('play-button-')) {
                                // Extract experiment ID from button ID
                                const experimentId = e.target.id.replace('play-button-', '');
                                playAudio(experimentId, e.target);
                                } else if (e.target.id.startsWith('bongo-button-')) {
                                    // Extract experiment ID from button ID
                                    const experimentId = e.target.id.replace('bongo-button-', '');
                                    playBongoAudio(experimentId, e.target);
                                } else if (e.target.id.startsWith('trumpet-button-')) {
                                    // Extract experiment ID from button ID
                                    const experimentId = e.target.id.replace('trumpet-button-', '');
                                    playTrumpetAudio(experimentId, e.target);
                                }
                            }
                        });

                        // Function to play audio
                        function playAudio(experimentId, buttonElement) {
                            const audioElement = document.getElementById('audio-' + experimentId);

                            if (audioElement) {
                                // Check if the audio is already playing and handle pause
                                if (!audioElement.paused) {
                                    audioElement.pause();
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                    return;
                                }

                                // Stop all other audio elements first
                                document.querySelectorAll('audio').forEach(audio => {
                                    if (audio.id !== 'audio-' + experimentId) {
                                        audio.pause();
                                        audio.currentTime = 0;
                                    }
                                });

                                // Cancel any existing animation frame
                                if (animationFrameId) {
                                    cancelAnimationFrame(animationFrameId);
                                }

                                // Find the current RIR plot and time bar trace
                                const plotInfo = findRIRPlotAndTimeBar();
                                if (plotInfo) {
                                    currentPlot = plotInfo.plot;
                                    currentTraceIndex = plotInfo.timeBarIndex;
                                }

                                // Play the selected audio
                                audioElement.currentTime = 0;
                                audioElement.play();

                                // Update button text
                                if (buttonElement) {
                                    buttonElement.textContent = '⏸';

                                    // Add event listener for when audio ends
                                    audioElement.onended = function() {
                                        buttonElement.textContent = '▶';
                                        if (animationFrameId) {
                                            cancelAnimationFrame(animationFrameId);
                                        }
                                        // Reset time bar to start
                                        if (currentPlot) {
                                            Plotly.restyle(currentPlot, {
                                                x: [[0, 0]]
                                            }, [currentTraceIndex]);
                                        }
                                    };
                                }

                                // Start the animation loop for smooth time bar updates
                                lastUpdateTime = performance.now();
                                animationFrameId = requestAnimationFrame(() => updateTimeBar(audioElement));

                                // Add pause handler
                                audioElement.onpause = function() {
                                    if (animationFrameId) {
                                        cancelAnimationFrame(animationFrameId);
                                    }
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                };
                            }
                        }

                        // Function to play bongo audio with convolution
                        function playBongoAudio(experimentId, buttonElement) {
                            // Check if we already have the convolved audio
                            const audioElement = document.getElementById('bongo-audio-' + experimentId);

                            if (audioElement && audioElement.src) {
                                // If audio already exists, check if playing and handle pause
                                if (!audioElement.paused) {
                                    audioElement.pause();
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                    return;
                                }

                                // Stop all other audio elements first
                                document.querySelectorAll('audio').forEach(audio => {
                                    if (audio.id !== 'bongo-audio-' + experimentId) {
                                        audio.pause();
                                        audio.currentTime = 0;
                                    }
                                });

                                // Cancel any existing animation frame
                                if (animationFrameId) {
                                    cancelAnimationFrame(animationFrameId);
                                }

                                // Find the current RIR plot and time bar trace
                                const plotInfo = findRIRPlotAndTimeBar();
                                if (plotInfo) {
                                    currentPlot = plotInfo.plot;
                                    currentTraceIndex = plotInfo.timeBarIndex;
                                }

                                // Play the selected audio
                                audioElement.currentTime = 0;
                                audioElement.play();

                                // Update button text
                                if (buttonElement) {
                                    buttonElement.textContent = '⏸';
                                    buttonElement.disabled = false;

                                    // Add event listener for when audio ends
                                    audioElement.onended = function() {
                                        buttonElement.textContent = '▶';
                                        if (animationFrameId) {
                                            cancelAnimationFrame(animationFrameId);
                                        }
                                        // Reset time bar to start
                                        if (currentPlot) {
                                            Plotly.restyle(currentPlot, {
                                                x: [[0, 0]]
                                            }, [currentTraceIndex]);
                                        }
                                    };
                                }

                                // Start the animation loop for smooth time bar updates
                                lastUpdateTime = performance.now();
                                animationFrameId = requestAnimationFrame(() => updateTimeBar(audioElement));

                                // Add pause handler
                                audioElement.onpause = function() {
                                    if (animationFrameId) {
                                        cancelAnimationFrame(animationFrameId);
                                    }
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                };
                            } else {
                                // First, update the button to show a loading state
                                if (buttonElement) {
                                    buttonElement.textContent = '🕒'; // Clock emoji
                                    buttonElement.disabled = true;
                                }

                                // We need to request the convolution from the server
                                // Use fetch to call our convolution endpoint
                                fetch('/convolve-bongo?experiment_id=' + experimentId)
                                    .then(response => {
                                        if (!response.ok) {
                                            throw new Error('Network response was not ok');
                                        }
                                        return response.json();
                                    })
                                    .then(data => {
                                        // Create or update the audio element
                                        let audioElement = document.getElementById('bongo-audio-' + experimentId);
                                        if (!audioElement) {
                                            audioElement = document.createElement('audio');
                                            audioElement.id = 'bongo-audio-' + experimentId;
                                            audioElement.controls = true;
                                            audioElement.style.display = 'none';
                                            document.getElementById('audio-components').appendChild(audioElement);
                                        }

                                        // Set the src and play the audio
                                        audioElement.src = 'data:audio/wav;base64,' + data.audio_data;

                                        // Re-enable the button and play
                                        if (buttonElement) {
                                            buttonElement.textContent = '▶';
                                            buttonElement.disabled = false;

                                            // Call the play function now that we have the audio
                                            playBongoAudio(experimentId, buttonElement);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching convolved audio:', error);
                                        if (buttonElement) {
                                            buttonElement.textContent = '❌'; // Error indicator
                                            setTimeout(() => {
                                                buttonElement.textContent = '▶';
                                                buttonElement.disabled = false;
                                            }, 2000);
                                        }
                                    });
                            }
                        }

                        // Function to play trumpet audio with convolution
                        function playTrumpetAudio(experimentId, buttonElement) {
                            // Check if we already have the convolved audio
                            const audioElement = document.getElementById('trumpet-audio-' + experimentId);

                            if (audioElement && audioElement.src) {
                                // If audio already exists, check if playing and handle pause
                                if (!audioElement.paused) {
                                    audioElement.pause();
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                    return;
                                }

                                // Stop all other audio elements first
                                document.querySelectorAll('audio').forEach(audio => {
                                    if (audio.id !== 'trumpet-audio-' + experimentId) {
                                        audio.pause();
                                        audio.currentTime = 0;
                                    }
                                });

                                // Cancel any existing animation frame
                                if (animationFrameId) {
                                    cancelAnimationFrame(animationFrameId);
                                }

                                // Find the current RIR plot and time bar trace
                                const plotInfo = findRIRPlotAndTimeBar();
                                if (plotInfo) {
                                    currentPlot = plotInfo.plot;
                                    currentTraceIndex = plotInfo.timeBarIndex;
                                }

                                // Play the selected audio
                                audioElement.currentTime = 0;
                                audioElement.play();

                                // Update button text
                                if (buttonElement) {
                                    buttonElement.textContent = '⏸';
                                    buttonElement.disabled = false;

                                    // Add event listener for when audio ends
                                    audioElement.onended = function() {
                                        buttonElement.textContent = '▶';
                                        if (animationFrameId) {
                                            cancelAnimationFrame(animationFrameId);
                                        }
                                        // Reset time bar to start
                                        if (currentPlot) {
                                            Plotly.restyle(currentPlot, {
                                                x: [[0, 0]]
                                            }, [currentTraceIndex]);
                                        }
                                    };
                                }

                                // Start the animation loop for smooth time bar updates
                                lastUpdateTime = performance.now();
                                animationFrameId = requestAnimationFrame(() => updateTimeBar(audioElement));

                                // Add pause handler
                                audioElement.onpause = function() {
                                    if (animationFrameId) {
                                        cancelAnimationFrame(animationFrameId);
                                    }
                                    if (buttonElement) {
                                        buttonElement.textContent = '▶';
                                    }
                                };
                            } else {
                                // First, update the button to show a loading state
                                if (buttonElement) {
                                    buttonElement.textContent = '🕒'; // Clock emoji
                                    buttonElement.disabled = true;
                                }

                                // We need to request the convolution from the server
                                // Use fetch to call our convolution endpoint
                                fetch('/convolve-trumpet?experiment_id=' + experimentId)
                                    .then(response => {
                                        if (!response.ok) {
                                            throw new Error('Network response was not ok');
                                        }
                                        return response.json();
                                    })
                                    .then(data => {
                                        // Create or update the audio element
                                        let audioElement = document.getElementById('trumpet-audio-' + experimentId);
                                        if (!audioElement) {
                                            audioElement = document.createElement('audio');
                                            audioElement.id = 'trumpet-audio-' + experimentId;
                                            audioElement.controls = true;
                                            audioElement.style.display = 'none';
                                            document.getElementById('audio-components').appendChild(audioElement);
                                        }

                                        // Set the src and play the audio
                                        audioElement.src = 'data:audio/wav;base64,' + data.audio_data;

                                        // Re-enable the button and play
                                        if (buttonElement) {
                                            buttonElement.textContent = '▶';
                                            buttonElement.disabled = false;

                                            // Call the play function now that we have the audio
                                            playTrumpetAudio(experimentId, buttonElement);
                                        }
                                    })
                                    .catch(error => {
                                        console.error('Error fetching convolved audio:', error);
                                        if (buttonElement) {
                                            buttonElement.textContent = '❌'; // Error indicator
                                            setTimeout(() => {
                                                buttonElement.textContent = '▶';
                                                buttonElement.disabled = false;
                                            }, 2000);
                                        }
                                    });
                            }
                        }

                        // Check for hash changes to trigger audio playback
                        window.addEventListener('hashchange', function() {
                            const hash = window.location.hash;
                            if (hash.startsWith('#play-')) {
                                const experimentId = hash.replace('#play-', '');
                                playAudio(experimentId);
                            } else if (hash.startsWith('#bongo-')) {
                                const experimentId = hash.replace('#bongo-', '');
                                playBongoAudio(experimentId);
                            } else if (hash.startsWith('#trumpet-')) {
                                const experimentId = hash.replace('#trumpet-', '');
                                playTrumpetAudio(experimentId);
                            }
                        });

                        // Check initial hash on page load
                        if (window.location.hash.startsWith('#play-')) {
                            const experimentId = window.location.hash.replace('#play-', '');
                            playAudio(experimentId);
                        } else if (window.location.hash.startsWith('#bongo-')) {
                            const experimentId = window.location.hash.replace('#bongo-', '');
                            playBongoAudio(experimentId);
                        } else if (window.location.hash.startsWith('#trumpet-')) {
                            const experimentId = window.location.hash.replace('#trumpet-', '');
                            playTrumpetAudio(experimentId);
                        }
                    });
                </script>
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

        # Add callback for layout switching
        @app.callback(
            [Output('monitor-layout', 'style'),
             Output('laptop-layout', 'style')],
            [Input('layout-switch', 'value')]
        )
        def update_layout(layout_value):
            if layout_value == 'monitor':
                return {'display': 'block'}, {'display': 'none'}
            else:  # laptop
                return {'display': 'none'}, {'display': 'block'}

        # Update the main callback to handle both layouts
        @app.callback(
            [Output('current-room-idx', 'data'),
             Output('current-pos-idx', 'data'),
             Output('room-plot', 'figure'),
             Output('room-header', 'children'),
             Output('rt-header', 'children'),
             Output('source-selector', 'options'),
             Output('receiver-selector', 'options'),
             Output('source-selector', 'value'),
             Output('receiver-selector', 'value'),
             Output('room-selector', 'value'),
             Output('experiment-table', 'data'),
             Output('experiment-table', 'columns'),
             Output('reference-selector', 'options'),
             Output('reference-selector', 'value'),
             # Monitor layout outputs
             Output('rir-plot-monitor', 'figure'),
             Output('edc-plot-monitor', 'figure'),
             Output('ned-plot-monitor', 'figure'),
             # Laptop layout outputs
             Output('rir-plot-laptop', 'figure'),
             Output('edc-plot-laptop', 'figure'),
             Output('ned-plot-laptop', 'figure'),
             Output('mean-error-table', 'data'),
             Output('audio-components', 'children')],
            [Input('prev-room', 'n_clicks'),
             Input('next-room', 'n_clicks'),
             Input('prev-pos', 'n_clicks'),
             Input('next-pos', 'n_clicks'),
             Input('source-selector', 'value'),
             Input('receiver-selector', 'value'),
             Input('room-selector', 'value'),
             Input('room-plot', 'clickData'),
             Input('reference-selector', 'value'),
             Input('comparison-type-selector', 'value'),
             Input('error-metric-selector', 'value'),
             Input('time-range-selector', 'value'),
             Input('layout-switch', 'value')],
            [State('current-room-idx', 'data'),
             State('current-pos-idx', 'data')]
        )
        def update_all(prev_room, next_room, prev_pos, next_pos,
                       source_value, receiver_value, room_selector_value,
                       click_data, reference_id,
                       comparison_type, error_metric, time_range,
                       layout_switch,
                       room_idx, pos_idx):
            ctx = dash.callback_context
            if not ctx.triggered:
                button_id = 'no-click'
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle room navigation
            if button_id == 'prev-room':
                room_idx = (room_idx - 1) % len(room_names)
                pos_idx = 0
            elif button_id == 'next-room':
                room_idx = (room_idx + 1) % len(room_names)
                pos_idx = 0
            elif button_id == 'room-selector':
                room_idx = room_selector_value
                pos_idx = 0
            elif button_id == 'prev-pos':
                room = self.manager.projects[room_names[room_idx]]
                pos_idx = (pos_idx - 1) % len(room.source_mic_pairs)
            elif button_id == 'next-pos':
                room = self.manager.projects[room_names[room_idx]]
                pos_idx = (pos_idx + 1) % len(room.source_mic_pairs)

            room = self.manager.projects[room_names[room_idx]]
            experiments = room.get_experiments_for_position(pos_idx)

            # Get unique sources and receivers for dropdown menus
            source_positions = {}
            receiver_positions = {}

            for idx, pos_key in enumerate(room.source_mic_pairs):
                source_pos, mic_pos = pos_key
                source_key = f"({source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f})"
                receiver_key = f"({mic_pos[0]:.1f}, {mic_pos[1]:.1f}, {mic_pos[2]:.1f})"

                if source_key not in source_positions:
                    source_positions[source_key] = idx

                if receiver_key not in receiver_positions:
                    receiver_positions[receiver_key] = idx

            # Create dropdown options with explicit labels
            source_options = [{'label': f'Source: {key}', 'value': key} for key in source_positions.keys()]
            receiver_options = [{'label': f'Receiver: {key}', 'value': key} for key in receiver_positions.keys()]

            # Handle clicks on the room plot
            if button_id == 'room-plot' and click_data is not None:
                # Extract the customdata to identify what was clicked
                if 'customdata' in click_data['points'][0]:
                    point_idx = click_data['points'][0]['customdata'][0]
                    point_type = click_data['points'][0]['customdata'][1]

                    # Get list of position keys
                    source_keys = list(source_positions.keys())
                    receiver_keys = list(receiver_positions.keys())

                    # Determine which unique positions we need
                    if point_type == 'source' and point_idx < len(source_keys):
                        source_value = source_keys[point_idx]
                        for i, (s_pos, r_pos) in enumerate(room.source_mic_pairs):
                            s_key = f"({s_pos[0]:.1f}, {s_pos[1]:.1f}, {s_pos[2]:.1f})"
                            r_key = f"({r_pos[0]:.1f}, {r_pos[1]:.1f}, {r_pos[2]:.1f})"
                            if s_key == source_value and (not receiver_value or r_key == receiver_value):
                                pos_idx = i
                                break

                    elif point_type == 'receiver' and point_idx < len(receiver_keys):
                        receiver_value = receiver_keys[point_idx]
                        for i, (s_pos, r_pos) in enumerate(room.source_mic_pairs):
                            s_key = f"({s_pos[0]:.1f}, {s_pos[1]:.1f}, {s_pos[2]:.1f})"
                            r_key = f"({r_pos[0]:.1f}, {r_pos[1]:.1f}, {r_pos[2]:.1f})"
                            if r_key == receiver_value and (not source_value or s_key == source_value):
                                pos_idx = i
                                break

            # Handle dropdown selection
            if button_id == 'source-selector' and source_value:
                for i, (s_pos, r_pos) in enumerate(room.source_mic_pairs):
                    s_key = f"({s_pos[0]:.1f}, {s_pos[1]:.1f}, {s_pos[2]:.1f})"
                    r_key = f"({r_pos[0]:.1f}, {r_pos[1]:.1f}, {r_pos[2]:.1f})"
                    if s_key == source_value and (not receiver_value or r_key == receiver_value):
                        pos_idx = i
                        break

            elif button_id == 'receiver-selector' and receiver_value:
                for i, (s_pos, r_pos) in enumerate(room.source_mic_pairs):
                    s_key = f"({s_pos[0]:.1f}, {s_pos[1]:.1f}, {s_pos[2]:.1f})"
                    r_key = f"({r_pos[0]:.1f}, {r_pos[1]:.1f}, {r_pos[2]:.1f})"
                    if r_key == receiver_value and (not source_value or s_key == source_value):
                        pos_idx = i
                        break

            # Get current position info
            if not room.source_mic_pairs:
                current_source = None
                current_receiver = None
            else:
                current_pos = room.source_mic_pairs[pos_idx % len(room.source_mic_pairs)]
                source_pos, mic_pos = current_pos
                current_source = f"({source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f})"
                current_receiver = f"({mic_pos[0]:.1f}, {mic_pos[1]:.1f}, {mic_pos[2]:.1f})"

            # Sort experiments by label for consistent ordering
            sorted_experiments = sorted(experiments, key=lambda exp: exp.get_label()['label_for_legend'])

            # Create dropdown options from experiment labels
            dropdown_options = [{'label': f"{idx}: {exp.get_label()['label_for_legend']}",
                                 'value': exp.get_label()['label_for_legend']}
                                for idx, exp in enumerate(sorted_experiments, 1)]

            # Prepare table columns
            columns = [
                {'name': 'Play', 'id': 'play', 'presentation': 'markdown'},
                {'name': '🥁', 'id': 'bongo', 'presentation': 'markdown'},
                {'name': '🎺', 'id': 'trumpet', 'presentation': 'markdown'},
                {'name': 'ID', 'id': 'id'},
                {'name': 'Method', 'id': 'method'},
                {'name': 'Label', 'id': 'label'},
                {'name': 'RT60', 'id': 'rt60'},
                {'name': 'Total Energy', 'id': 'total_energy'},
                {'name': f'{self.ERROR_METRICS[0]["label"]} [50ms]', 'id': 'error_50ms'},
                {'name': f'{self.ERROR_METRICS[0]["label"]} [500ms]', 'id': 'error_500ms'}
            ]

            # Prepare table data
            table_data = []
            mean_error_data = []

            if reference_id:  # reference_id is now the label_for_legend
                # Get reference experiment for current position
                try:
                    ref_exp = next(
                        exp for exp in sorted_experiments if exp.get_label()['label_for_legend'] == reference_id)

                    # MAIN TABLE
                    # Prepare table data and calculate errors
                    for idx, exp in enumerate(sorted_experiments, 1):
                        label_dict = exp.get_label()
                        experiment_id = exp.experiment_id
                        row = {
                            'play': f'<button id="play-button-{experiment_id}" class="play-button">▶</button>',
                            'bongo': f'<button id="bongo-button-{experiment_id}" class="bongo-button">▶</button>',
                            'trumpet': f'<button id="trumpet-button-{experiment_id}" class="trumpet-button">▶</button>',
                            'id': idx,
                            'method': exp.config.get('method', 'Unknown'),
                            'label': label_dict['label'],
                            'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A',
                            'total_energy': 'N/A',
                            'error_50ms': 'N/A',
                            'error_500ms': 'N/A'
                        }

                        # Slice to appropriate time ranges
                        samples_50ms = int(0.05 * ref_exp.fs)
                        samples_500ms = int(0.5 * ref_exp.fs)

                        # Add total energy to the row (sum of raw energy)
                        _, exp_energy, _ = self.an.calculate_err(exp.rir, Fs=exp.fs)
                        total_energy = np.sum(exp_energy)
                        row['total_energy'] = f"{total_energy:.6f}"

                        # Get signals based on comparison type
                        if comparison_type == 'edc':
                            # Calculate full EDC first
                            sig1_full = ref_exp.edc
                            sig2_full = exp.edc

                        elif comparison_type == 'smoothed_energy':
                            # Calculate full smoothed energy once
                            sig1_full = self.an.calculate_smoothed_energy(ref_exp.rir, window_length=30, Fs=ref_exp.fs)
                            sig2_full = self.an.calculate_smoothed_energy(exp.rir, window_length=30, Fs=exp.fs)

                        else:  # raw energy
                            # Calculate full energy once
                            sig1_full, _, _ = self.an.calculate_err(ref_exp.rir, Fs=ref_exp.fs)
                            # sig2_full, _, _ = self.an.calculate_err(exp.rir, Fs=exp.fs)
                            sig2_full = exp_energy

                        sig1_50ms = sig1_full[:samples_50ms]
                        sig2_50ms = sig2_full[:samples_50ms]
                        sig1_500ms = sig1_full[:samples_500ms]
                        sig2_500ms = sig2_full[:samples_500ms]

                        # Calculate errors for current position
                        error_50ms = self.an.compute_RMS(sig1_50ms, sig2_50ms, Fs=ref_exp.fs, method=error_metric)
                        error_500ms = self.an.compute_RMS(sig1_500ms, sig2_500ms, Fs=ref_exp.fs, method=error_metric)
                        row['error_50ms'] = f"{error_50ms:.6f}"
                        row['error_500ms'] = f"{error_500ms:.6f}"
                        table_data.append(row)

                        # Calculate mean errors across all receivers for current source
                        current_src = tuple(room.source_mic_pairs[pos_idx][0])
                        receiver_indices = [i for i, (src, _) in enumerate(room.source_mic_pairs)
                                            if tuple(src) == current_src]

                        all_errors_50ms = []
                        all_errors_500ms = []
                        all_total_energies = []
                        for rec_idx in receiver_indices:
                            try:
                                pos_exps = room.get_experiments_for_position(rec_idx)
                                pos_exp_by_label = {exp.get_label()['label_for_legend']: exp for exp in pos_exps}

                                # Check if reference and current experiment exist for this position
                                if reference_id not in pos_exp_by_label or exp.get_label()[
                                    'label_for_legend'] not in pos_exp_by_label:
                                    continue

                                # Get reference and current experiment for this position
                                pos_ref = pos_exp_by_label[reference_id]  # Use label_for_legend directly
                                pos_exp = pos_exp_by_label[exp.get_label()['label_for_legend']]

                                # Calculate total energy for this position
                                _, pos_exp_energy, _ = self.an.calculate_err(pos_exp.rir, Fs=pos_exp.fs)
                                pos_total_energy = np.sum(pos_exp_energy)
                                all_total_energies.append(pos_total_energy)

                                # Slice to appropriate time ranges
                                samples_50ms = int(0.05 * pos_ref.fs)
                                samples_500ms = int(0.5 * pos_ref.fs)

                                # Get signals and calculate errors
                                if comparison_type == 'edc':
                                    # Calculate full EDC first
                                    sig1_full = pos_ref.edc
                                    sig2_full = pos_exp.edc

                                elif comparison_type == 'smoothed_energy':
                                    # Calculate full smoothed energy once
                                    sig1_full = self.an.calculate_smoothed_energy(pos_ref.rir, window_length=30,
                                                                                  Fs=pos_ref.fs)
                                    sig2_full = self.an.calculate_smoothed_energy(pos_exp.rir, window_length=30,
                                                                                  Fs=pos_exp.fs)

                                else:  # raw energy
                                    # Calculate full energy once
                                    sig1_full, _, _ = self.an.calculate_err(pos_ref.rir, Fs=pos_ref.fs)
                                    # sig2_full, _, _ = self.an.calculate_err(pos_exp.rir, Fs=pos_exp.fs)
                                    sig2_full = pos_exp_energy

                                sig1_50ms = sig1_full[:samples_50ms]
                                sig2_50ms = sig2_full[:samples_50ms]
                                sig1_500ms = sig1_full[:samples_500ms]
                                sig2_500ms = sig2_full[:samples_500ms]

                                error_50ms = self.an.compute_RMS(sig1_50ms, sig2_50ms, Fs=pos_ref.fs,
                                                                 method=error_metric)
                                error_500ms = self.an.compute_RMS(sig1_500ms, sig2_500ms, Fs=pos_ref.fs,
                                                                  method=error_metric)

                                all_errors_50ms.append(error_50ms)
                                all_errors_500ms.append(error_500ms)
                            except Exception as e:
                                # Skip this position if any error occurs
                                print(f"Warning: Error processing position {rec_idx}: {str(e)}")
                                continue

                        # Add mean errors to mean error table (only if we have data)
                        if all_errors_50ms:
                            mean_error_data.append({
                                'error_50ms': f"{np.mean(all_errors_50ms):.6f}",
                                'error_500ms': f"{np.mean(all_errors_500ms):.6f}",
                                'total_energy': f"{np.mean(all_total_energies):.6f}"
                            })
                        else:
                            # Add N/A if no data available
                            mean_error_data.append({
                                'error_50ms': 'N/A',
                                'error_500ms': 'N/A',
                                'total_energy': 'N/A'
                            })
                except StopIteration:
                    # Reference experiment not found in this position
                    print(f"Warning: Reference experiment '{reference_id}' not found at position {pos_idx}")
                    # Just populate table with experiment info, no error calculations
                    for idx, exp in enumerate(sorted_experiments, 1):
                        label_dict = exp.get_label()
                        table_data.append({
                            'play': f'<button id="play-button-{exp.experiment_id}" class="play-button">▶</button>',
                            'bongo': f'<button id="bongo-button-{exp.experiment_id}" class="bongo-button">▶</button>',
                            'trumpet': f'<button id="trumpet-button-{exp.experiment_id}" class="trumpet-button">▶</button>',
                            'id': idx,
                            'method': exp.config.get('method', 'Unknown'),
                            'label': label_dict['label'],
                            'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A',
                            'total_energy': 'N/A',
                            'error_50ms': 'N/A',
                            'error_500ms': 'N/A'
                        })
            else:
                # Just populate table with experiment info, no error calculations
                for idx, exp in enumerate(sorted_experiments, 1):
                    label_dict = exp.get_label()
                    table_data.append({
                        'play': f'<button id="play-button-{exp.experiment_id}" class="play-button">▶</button>',
                        'bongo': f'<button id="bongo-button-{exp.experiment_id}" class="bongo-button">▶</button>',
                        'trumpet': f'<button id="trumpet-button-{exp.experiment_id}" class="trumpet-button">▶</button>',
                        'id': idx,
                        'method': exp.config.get('method', 'Unknown'),
                        'label': label_dict['label'],
                        'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A',
                        'total_energy': 'N/A',
                        'error_50ms': 'N/A',
                        'error_500ms': 'N/A'
                    })

            # Create room visualization without error contour
            room_plot = self.create_room_visualization([room], highlight_pos_idx=pos_idx)

            # Update room plot colors to match original implementation
            room_plot.update_layout(
                plot_bgcolor=dark_theme['plot_bg'],
                paper_bgcolor=dark_theme['paper_bg'],
                font={'color': dark_theme['text']},
                title={'font': {'color': dark_theme['text']}},
                xaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                yaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']}
            )

            # For room outline, update to match dark theme
            for shape in room_plot.layout.shapes:
                shape.line.color = 'rgba(255, 255, 255, 0.5)'
                shape.fillcolor = 'rgba(50, 50, 50, 0.1)'

            room_header = f"Room: {room.display_name}"
            rt_header = f"Dimensions: {room.dimensions_str}, abs={room.absorption_str}, {room.theoretical_rt_str}"

            # Create plots for both layouts
            rir_fig_monitor = go.Figure()
            edc_fig_monitor = go.Figure()
            ned_fig_monitor = go.Figure()

            rir_fig_laptop = go.Figure()
            edc_fig_laptop = go.Figure()
            ned_fig_laptop = go.Figure()

            # Create audio components for each experiment
            audio_components = []

            # Define consistent colors for each experiment
            colors = px.colors.qualitative.Plotly  # Use a standard color palette

            for idx, exp in enumerate(sorted_experiments, 1):
                label_dict = exp.get_label()
                experiment_id = exp.experiment_id
                legend_name = f"{idx}: {label_dict['label_for_legend']}"
                color = colors[idx % len(colors)]  # Cycle through colors

                # Generate audio data for this experiment
                audio_data = self.generate_audio_data(exp.rir, exp.fs, experiment_id)

                # Create hidden audio component
                audio_components.append(
                    html.Audio(
                        id=f'audio-{experiment_id}',
                        src=f'data:audio/wav;base64,{audio_data}',
                        controls=True,
                        style={'display': 'none'}
                    )
                )

                # Create placeholders for bongo audio (will be populated when button is clicked)
                audio_components.append(
                    html.Audio(
                        id=f'bongo-audio-{experiment_id}',
                        controls=True,
                        style={'display': 'none'}
                    )
                )

                # Create placeholders for trumpet audio (will be populated when button is clicked)
                audio_components.append(
                    html.Audio(
                        id=f'trumpet-audio-{experiment_id}',
                        controls=True,
                        style={'display': 'none'}
                    )
                )

                """ MONITOR """
                # Get data for plotting
                plot_data = self.get_experiment_plot_data(exp)

                # Add traces to monitor layout plots - use consistent naming and coloring
                rir_fig_monitor.add_trace(go.Scatter(
                    x=plot_data['time_axis'],
                    y=exp.rir,
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color),
                    customdata=[experiment_id]
                ))

                # Add time bar trace (initially at x=0)
                rir_fig_monitor.add_trace(go.Scatter(
                    x=[0, 0],
                    y=[-1, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Time Bar',
                    showlegend=False,
                    hoverinfo='skip'
                ))

                edc_fig_monitor.add_trace(go.Scatter(
                    x=plot_data['edc_time_axis'],
                    y=plot_data['edc'],
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color)
                ))

                ned_fig_monitor.add_trace(go.Scatter(
                    x=plot_data['ned_time_axis'],
                    y=plot_data['ned'],  # Use calculated NED from plot_data
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color)
                ))

                """ LAPTOP """
                # Add traces to laptop layout plots - use consistent naming and coloring
                rir_fig_laptop.add_trace(go.Scatter(
                    x=plot_data['time_axis'],
                    y=exp.rir,
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color),
                    customdata=[experiment_id]
                ))

                # Add time bar trace (initially at x=0)
                rir_fig_laptop.add_trace(go.Scatter(
                    x=[0, 0],
                    y=[-1, 1],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Time Bar',
                    showlegend=False,
                    hoverinfo='skip'
                ))

                edc_fig_laptop.add_trace(go.Scatter(
                    x=plot_data['time_axis'],
                    y=plot_data['edc'],
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color)
                ))

                ned_fig_laptop.add_trace(go.Scatter(
                    x=plot_data['ned_time_axis'],
                    y=plot_data['ned'],
                    name=legend_name,
                    mode='lines',
                    line=dict(color=color)
                ))

            # Find monitor plot layouts and add uirevision
            # Update plot layouts for monitor view
            for fig, title in [(rir_fig_monitor, "Room Impulse Response"),
                               (edc_fig_monitor, "Energy Decay Curve"),
                               (ned_fig_monitor, "Normalized Echo Density")]:
                fig.update_layout(
                    title=title,
                    plot_bgcolor=dark_theme['plot_bg'],
                    paper_bgcolor=dark_theme['paper_bg'],
                    font={'color': dark_theme['text']},
                    xaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    yaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    margin=dict(t=30, b=20, l=50, r=20),
                    uirevision='constant'  # This preserves zoom level when data changes
                )

                if time_range != 'full':
                    fig.update_xaxes(range=[0, float(time_range)])

                # Set y-axis ranges
                if fig == rir_fig_monitor:
                    # fig.update_yaxes(range=[-0.5, 1.0])
                    a = 4
                elif fig == edc_fig_monitor and (time_range == 0.05 or time_range == '0.05'):
                    fig.update_yaxes(range=[-10, 2])

                # Set y-axis ranges
                # if fig == ned_fig_monitor and (time_range == 0.05 or time_range == '0.05'):
                #    fig.update_xaxes(range=[0, 0.2])

            # Update plot layouts for laptop view
            for fig, title in [(rir_fig_laptop, "Room Impulse Response"),
                               (edc_fig_laptop, "Energy Decay Curve"),
                               (ned_fig_laptop, "Normalized Echo Density")]:
                fig.update_layout(
                    title=title,
                    plot_bgcolor=dark_theme['plot_bg'],
                    paper_bgcolor=dark_theme['paper_bg'],
                    font={'color': dark_theme['text']},
                    xaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    yaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    margin=dict(t=30, b=20, l=50, r=20),
                    uirevision='constant'  # This preserves zoom level when data changes
                )

                if time_range != 'full':
                    fig.update_xaxes(range=[0, float(time_range)])

                # Set y-axis ranges
                if fig == rir_fig_laptop:
                    # fig.update_yaxes(range=[-0.5, 1.0])
                    a = 4
                elif fig == edc_fig_laptop and (time_range == 0.05 or time_range == '0.05'):
                    fig.update_yaxes(range=[-10, 2])

            # Add unified legend for all plots
            legend_config = dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(40, 44, 52, 0.7)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            )

            # EDC specific legend config
            edc_legend_config = dict(
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(40, 44, 52, 0.7)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            )

            # NED specific legend config
            ned_legend_config = dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(40, 44, 52, 0.7)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            )

            rir_fig_monitor.update_layout(legend=legend_config)
            edc_fig_monitor.update_layout(legend=edc_legend_config)
            ned_fig_monitor.update_layout(legend=ned_legend_config)

            rir_fig_laptop.update_layout(legend=legend_config)
            edc_fig_laptop.update_layout(legend=edc_legend_config)
            ned_fig_laptop.update_layout(legend=ned_legend_config)

            return (room_idx, pos_idx, room_plot, room_header, rt_header,
                    source_options, receiver_options, current_source, current_receiver,
                    room_idx, table_data, columns, dropdown_options,
                    reference_id,
                    rir_fig_monitor, edc_fig_monitor, ned_fig_monitor,
                    rir_fig_laptop, edc_fig_laptop, ned_fig_laptop,
                    mean_error_data,
                    audio_components)

        # Add callback for legend clicks
        @app.callback(
            Output('dummy-output', 'children'),
            [Input('rir-plot-monitor', 'clickData'),
             Input('rir-plot-laptop', 'clickData')],
            [State('layout-switch', 'value')]
        )
        def handle_legend_click(click_data_monitor, click_data_laptop, layout_value):
            # Determine which click data to use based on the current layout
            click_data = click_data_monitor if layout_value == 'monitor' else click_data_laptop

            if click_data and 'points' in click_data and click_data['points']:
                point = click_data['points'][0]
                if 'customdata' in point and point['customdata']:
                    experiment_id = point['customdata'][0]
                    # Trigger JavaScript to play the audio
                    return dcc.Location(id='dummy-output', pathname=f'#play-{experiment_id}', refresh=False)
            return dash.no_update

        # Add a dummy div for the callback output
        app.layout.children.append(html.Div(id='dummy-output', style={'display': 'none'}))

        # All the layout and callback code from the show() method should be copied here
        # Dark theme colors, layout, callbacks, etc.
        # But don't call app.run_server()

        # The rest of the code is identical to the show() method, except without the call to app.run_server()
        # Dark theme, room list, app layout, callbacks

        # Instead, we just return the app for WSGI to handle
        return app

    def generate_bongo_convolution_audio(self, rir, fs, experiment_id):
        """
        Generate audio data by convolving RIR with bongo sound.

        Args:
            rir (np.ndarray): Room impulse response
            fs (int): Sampling frequency
            experiment_id (str): Experiment ID for caching

        Returns:
            str: Base64 encoded audio data or None if convolution failed
        """
        if self.bongo_data is None or self.bongo_fs is None:
            return None

        # Check if we already have this convolution in cache
        cache_key = f"bongo_{experiment_id}"
        if cache_key in self.bongo_convolution_cache:
            return self.bongo_convolution_cache[cache_key]

        print(f"Generating convolved audio for experiment {experiment_id}...")
        start_time = time.time()

        try:
            # Resample if needed
            if fs != self.bongo_fs:
                # Simple resampling by interpolation (not ideal but simple)
                # For production, use proper resampling library like librosa
                rir_resampled = np.interp(
                    np.linspace(0, len(rir) - 1, int(len(rir) * self.bongo_fs / fs)),
                    np.arange(len(rir)),
                    rir
                )
                rir = rir_resampled
                fs = self.bongo_fs

            # Normalize RIR
            rir_normalized = rir / np.max(np.abs(rir)) * 0.8

            # Perform convolution
            convolved = convolve(self.bongo_data, rir_normalized)

            # Normalize result
            max_val = np.max(np.abs(convolved))
            if max_val > 0:  # Avoid division by zero
                convolved = convolved / max_val * 0.8

            # Create an in-memory buffer
            buffer = io.BytesIO()

            # Write the audio data to the buffer
            sf.write(buffer, convolved, fs, format='WAV')

            # Get the buffer content and encode as base64
            buffer.seek(0)
            audio_data = base64.b64encode(buffer.read()).decode('utf-8')

            # Cache the result
            self.bongo_convolution_cache[cache_key] = audio_data

            elapsed = time.time() - start_time
            print(f"Convolution completed in {elapsed:.2f} seconds")

            return audio_data
        except Exception as e:
            print(f"Error in bongo convolution for {experiment_id}: {str(e)}")
            return None

    def generate_audio_data(self, rir, fs, experiment_id):
        """
        Generate audio data from RIR and return as base64 encoded string.

        Args:
            rir (np.ndarray): Room impulse response
            fs (int): Sampling frequency
            experiment_id (str): Experiment ID for caching

        Returns:
            str: Base64 encoded audio data
        """
        # Check if we already have this audio in cache
        if experiment_id in self.audio_cache:
            return self.audio_cache[experiment_id]

        # Normalize and scale the RIR
        rir_normalized = rir / np.max(np.abs(rir)) * 0.8

        # Create an in-memory buffer
        buffer = io.BytesIO()

        # Write the audio data to the buffer
        sf.write(buffer, rir_normalized, fs, format='WAV')

        # Get the buffer content and encode as base64
        buffer.seek(0)
        audio_data = base64.b64encode(buffer.read()).decode('utf-8')

        # Cache the result
        self.audio_cache[experiment_id] = audio_data

        return audio_data

    @property
    def rt60(self):
        """Get RT60 value, calculating metrics if needed."""
        # Keep the method, but it doesn't do anything in this class
        # since metrics are calculated on a per-experiment basis
        return None

    def generate_trumpet_convolution_audio(self, rir, fs, experiment_id):
        """
        Generate audio data by convolving RIR with trumpet sound.

        Args:
            rir (np.ndarray): Room impulse response
            fs (int): Sampling frequency
            experiment_id (str): Experiment ID for caching

        Returns:
            str: Base64 encoded audio data or None if convolution failed
        """
        if self.trumpet_data is None or self.trumpet_fs is None:
            return None

        # Check if we already have this convolution in cache
        cache_key = f"trumpet_{experiment_id}"
        if cache_key in self.trumpet_convolution_cache:
            return self.trumpet_convolution_cache[cache_key]

        print(f"Generating trumpet convolved audio for experiment {experiment_id}...")
        start_time = time.time()

        try:
            # Resample if needed
            if fs != self.trumpet_fs:
                # Simple resampling by interpolation (not ideal but simple)
                # For production, use proper resampling library like librosa
                rir_resampled = np.interp(
                    np.linspace(0, len(rir) - 1, int(len(rir) * self.trumpet_fs / fs)),
                    np.arange(len(rir)),
                    rir
                )
                rir = rir_resampled
                fs = self.trumpet_fs

            # Normalize RIR
            rir_normalized = rir / np.max(np.abs(rir)) * 0.8

            # Perform convolution
            convolved = convolve(self.trumpet_data, rir_normalized)

            # Normalize result
            max_val = np.max(np.abs(convolved))
            if max_val > 0:  # Avoid division by zero
                convolved = convolved / max_val * 0.8

            # Create an in-memory buffer
            buffer = io.BytesIO()

            # Write the audio data to the buffer
            sf.write(buffer, convolved, fs, format='WAV')

            # Get the buffer content and encode as base64
            buffer.seek(0)
            audio_data = base64.b64encode(buffer.read()).decode('utf-8')

            # Cache the result
            self.trumpet_convolution_cache[cache_key] = audio_data

            elapsed = time.time() - start_time
            print(f"Trumpet convolution completed in {elapsed:.2f} seconds")

            return audio_data
        except Exception as e:
            print(f"Error in trumpet convolution for {experiment_id}: {str(e)}")
            return None