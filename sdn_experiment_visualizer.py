import os
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
import numpy as np
import webbrowser
from threading import Timer
from sdn_experiment_manager import Room
import spatial_analysis
import analysis as an
from scipy.interpolate import griddata

class SDNExperimentVisualizer:
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
        
    def create_error_contour(self, room, reference_exp, comparison_exp, comparison_type, error_metric):
        """Create a contour plot of error metrics using existing receiver positions."""
        # Get all receiver positions from the room's source_mic_pairs
        receiver_positions = []
        receiver_errors = []
        
        # Extract unique receiver positions and calculate errors
        seen_positions = set()
        for source_pos, mic_pos in room.source_mic_pairs:
            # Convert position to tuple for hashable type
            pos_tuple = tuple(mic_pos)
            if pos_tuple not in seen_positions:
                seen_positions.add(pos_tuple)
                receiver_positions.append(mic_pos)
                
                # Get RIRs for this position
                ref_rir = reference_exp.rir
                comp_rir = comparison_exp.rir
                
                # Get signals based on comparison type
                if comparison_type == 'edc':
                    sig1_50ms = reference_exp.edc
                    sig2_50ms = comparison_exp.edc
                    
                elif comparison_type == 'smoothed_energy':
                    _, sig1_50ms = an.calculate_smoothed_energy(ref_rir, window_length=30, range=50, Fs=reference_exp.fs)
                    _, sig2_50ms = an.calculate_smoothed_energy(comp_rir, window_length=30, range=50, Fs=comparison_exp.fs)
                    _, sig1_500ms = an.calculate_smoothed_energy(ref_rir, window_length=30, range=500, Fs=reference_exp.fs)
                    _, sig2_500ms = an.calculate_smoothed_energy(comp_rir, window_length=30, range=500, Fs=comparison_exp.fs)
                else:  # raw energy
                    sig1_50ms, _ = an.calculate_smoothed_energy(ref_rir, window_length=30, range=50, Fs=reference_exp.fs)
                    sig2_50ms, _ = an.calculate_smoothed_energy(comp_rir, window_length=30, range=50, Fs=comparison_exp.fs)
                    sig1_500ms, _ = an.calculate_smoothed_energy(ref_rir, window_length=30, range=500, Fs=reference_exp.fs)
                    sig2_500ms, _ = an.calculate_smoothed_energy(comp_rir, window_length=30, range=500, Fs=comparison_exp.fs)
                
                # Calculate error
                error_50ms = an.compute_RMS(sig1_50ms, sig2_50ms, range=50, Fs=reference_exp.fs, method=error_metric)
                error_500ms = an.compute_RMS(sig1_500ms, sig2_500ms, range=500, Fs=reference_exp.fs, method=error_metric)
                receiver_errors.append(error_50ms)
                receiver_errors.append(error_500ms)

        
        # Convert positions to arrays for plotting
        positions = np.array(receiver_positions)
        x = positions[:, 0]
        y = positions[:, 1]
        z = np.array(receiver_errors)
        
        # Create contour plot
        contour = go.Contour(
            x=x, y=y, z=z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f'{error_metric.upper()} Error')
        )
        
        return contour

    def create_room_visualization(self, experiments, highlight_pos_idx=None, show_error_contour=False, reference_id=None, comparison_type='edc', error_metric='rmse'):
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
                hovertext=[f"Source {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" 
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
                hovertext=[f"Receiver {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" 
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

    def show(self, is_batch=True, port=9050):
        """Launch the visualization dashboard."""
        # Get the appropriate manager if not provided
        if self.manager is None:
            self.manager = get_batch_manager() if is_batch else get_singular_manager()

        if not self.manager.projects:
            print("No rooms with experiments to visualize.")
            return

        # Create Dash app
        app = dash.Dash(__name__)
        server = app.server

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
                            'whiteSpace': 'nowrap'
                        }
                    ),
                ], style={
                    'position': 'relative',
                    'height': '40px',  # Fixed height for the container
                    'marginBottom': '10px'
                }),

                # Second row: Navigation buttons and RT info
                html.Div([
                    # Navigation buttons
                    html.Button('←', id='prev-room', style={
                        'fontSize': 24, 
                        'marginRight': '10px',
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 15px',
                        'cursor': 'pointer',
                        'display': 'inline-block',
                        'verticalAlign': 'middle'
                    }),
                    html.Button('→', id='next-room', style={
                        'fontSize': 24,
                        'marginRight': '20px',
                        'backgroundColor': dark_theme['button_bg'],
                        'color': dark_theme['button_text'],
                        'border': 'none',
                        'borderRadius': '4px',
                        'padding': '0px 15px',
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
                            'verticalAlign': 'middle'
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
                'margin': '5px',
                'marginBottom': '10px'
            }),

            # Main content area
            html.Div([
                # Left side - plots and table
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
                            value=0.05,
                            labelStyle={'display': 'inline-block', 'marginRight': '20px', 'cursor': 'pointer'},
                            style={'color': dark_theme['text'], 'fontSize': '14px'},
                            inputStyle={'marginRight': '5px'}
                        )
                    ], style={'margin': '10px 0 0 20px', 'display': 'flex', 'alignItems': 'center'}),

                    # Plots
                    html.Div([
                    dcc.Tabs([
                        dcc.Tab(label="Room Impulse Responses", children=[
                            dcc.Graph(id='rir-plot', style={'height': '50vh'})
                        ],
                            style={
                                'backgroundColor': dark_theme['paper_bg'],
                                'color': dark_theme['text'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            },
                            selected_style={
                                'backgroundColor': dark_theme['header_bg'],
                                'color': dark_theme['accent'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            }),
                        dcc.Tab(label="Energy Decay Curves", children=[
                            dcc.Graph(id='edc-plot', style={'height': '50vh'})
                        ],
                            style={
                                'backgroundColor': dark_theme['paper_bg'],
                                'color': dark_theme['text'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            },
                            selected_style={
                                'backgroundColor': dark_theme['header_bg'],
                                'color': dark_theme['accent'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            }),
                        dcc.Tab(label="Normalized Echo Density", children=[
                            dcc.Graph(id='ned-plot', style={'height': '50vh'})
                        ],
                            style={
                                'backgroundColor': dark_theme['paper_bg'],
                                'color': dark_theme['text'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            },
                            selected_style={
                                'backgroundColor': dark_theme['header_bg'],
                                'color': dark_theme['accent'],
                                'height': '40px',
                                'padding': '6px',
                                'display': 'flex',
                                'alignItems': 'center'
                            })
                        ], style={'backgroundColor': dark_theme['paper_bg'], 'margin': '0px 0'})
                    ], style={'width': '100%'}),  # Full width for plots container

                    # Experiment table in a centered container
                    html.Div([
                        html.H3("Active Experiments", 
                               style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px', 'color': dark_theme['text']}),
                        # Add error metric controls
                        html.Div([
                            html.Div([
                                html.Label("Comparison Type:", style={'color': dark_theme['text'], 'marginRight': '10px'}),
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
                        ], style={'marginBottom': '10px'}),
                        # Create a flex container for tables
                        html.Div([
                            # Main experiment table
                            html.Div([
                                dash_table.DataTable(
                                    id='experiment-table',
                                    style_table={'height': '25vh', 'overflowY': 'auto'},
                                    style_cell={
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text'],
                                        'textAlign': 'left',
                                        'padding': '5px'
                                    },
                                    style_header={
                                        'backgroundColor': dark_theme['header_bg'],
                                        'fontWeight': 'bold'
                                    },
                                    columns=[
                                        {'name': 'ID', 'id': 'id'},
                                        {'name': 'Method', 'id': 'method'},
                                        {'name': 'Label', 'id': 'label'},
                                        {'name': 'RT60', 'id': 'rt60'},
                                        {'name': f'Error (50ms) ({self.ERROR_METRICS[0]["label"]})', 'id': 'error_50ms'},
                                        {'name': f'Error (500ms) ({self.ERROR_METRICS[0]["label"]})', 'id': 'error_500ms'}
                                    ],
                                    data=[]  # Will be populated in callback
                                ),
                            ], style={'flex': '3'}),
                            # Mean error table
                            html.Div([
                                html.H4("Mean Error (all positions)", style={'textAlign': 'center', 'margin': '0', 'marginBottom': '4px', 'color': dark_theme['text']}),
                                dash_table.DataTable(
                                    id='mean-error-table',
                                    columns=[
                                        {'name': 'Error (50ms)', 'id': 'error_50ms'},
                                        {'name': 'Error (500ms)', 'id': 'error_500ms'}
                                    ],
                                    data=[],
                                    style_table={'height': '25vh', 'overflowY': 'auto'},
                                    style_header={
                                        'backgroundColor': dark_theme['header_bg'],
                                        'color': dark_theme['text'],
                                        'fontWeight': 'bold'
                                    },
                                    style_cell={
                                        'backgroundColor': dark_theme['paper_bg'],
                                        'color': dark_theme['text'],
                                        'textAlign': 'center',
                                        'padding': '5px'
                                    }
                                )
                            ], style={'flex': '1', 'marginLeft': '10px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'flex-end'})
                        ], style={'display': 'flex', 'alignItems': 'flex-end'})
                    ], style={'width': '71.4%', 'margin': '0 auto'})
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right side - room visualization
                html.Div([
                    html.H3("Room Layout (Top View)", 
                           style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '65px', 'color': dark_theme['text']}),
                    dcc.Store(id='current-pos-idx', data=0),
                    dcc.Graph(
                        id='room-plot',
                        style={
                            'height': '50vh',
                            'marginTop': '0px',
                            'backgroundColor': dark_theme['paper_bg']
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
                    ], style={'padding': '10px 20px', 'marginBottom': '0px'}),
                    
                    # Navigation buttons
                    html.Div([
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
                        })
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),

                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
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

        # Combined callback for room/position navigation, error metrics, and plots
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
             Output('rir-plot', 'figure'),
             Output('edc-plot', 'figure'),
             Output('ned-plot', 'figure'),
             Output('mean-error-table', 'data')],
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
             Input('time-range-selector', 'value')],
            [State('current-room-idx', 'data'),
             State('current-pos-idx', 'data')]
        )
        def update_all(prev_room, next_room, prev_pos, next_pos, 
                      source_value, receiver_value, room_selector_value, 
                      click_data, reference_id,
                      comparison_type, error_metric, time_range,
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
                {'name': 'ID', 'id': 'id'},
                {'name': 'Method', 'id': 'method'},
                {'name': 'Label', 'id': 'label'},
                {'name': 'RT60', 'id': 'rt60'},
                {'name': f'Error (50ms) ({error_metric.upper()})', 'id': 'error_50ms'},
                {'name': f'Error (500ms) ({error_metric.upper()})', 'id': 'error_500ms'}
            ]

            # Prepare table data
            table_data = []
            mean_error_data = []
            
            if reference_id:  # reference_id is now the label_for_legend
                # Get reference experiment for current position
                ref_exp = next(exp for exp in sorted_experiments if exp.get_label()['label_for_legend'] == reference_id)
                
                # Prepare table data and calculate errors
                for idx, exp in enumerate(sorted_experiments, 1):
                    label_dict = exp.get_label()
                    row = {
                        'id': idx,
                        'method': exp.config.get('method', 'Unknown'),
                        'label': label_dict['label'],
                        'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A',
                        'error_50ms': 'N/A',
                        'error_500ms': 'N/A'
                    }
                    
                    # Get signals based on comparison type
                    if comparison_type == 'edc':
                        sig1_50ms = ref_exp.edc
                        sig2_50ms = exp.edc
                        sig1_500ms = ref_exp.edc
                        sig2_500ms = exp.edc
                    elif comparison_type == 'smoothed_energy':
                        _, sig1_50ms = an.calculate_smoothed_energy(ref_exp.rir, window_length=30, range=50, Fs=ref_exp.fs)
                        _, sig2_50ms = an.calculate_smoothed_energy(exp.rir, window_length=30, range=50, Fs=exp.fs)
                        _, sig1_500ms = an.calculate_smoothed_energy(ref_exp.rir, window_length=30, range=500, Fs=ref_exp.fs)
                        _, sig2_500ms = an.calculate_smoothed_energy(exp.rir, window_length=30, range=500, Fs=exp.fs)
                    else:  # raw energy
                        sig1_50ms, _ = an.calculate_smoothed_energy(ref_exp.rir, window_length=30, range=50, Fs=ref_exp.fs)
                        sig2_50ms, _ = an.calculate_smoothed_energy(exp.rir, window_length=30, range=50, Fs=exp.fs)
                        sig1_500ms, _ = an.calculate_smoothed_energy(ref_exp.rir, window_length=30, range=500, Fs=ref_exp.fs)
                        sig2_500ms, _ = an.calculate_smoothed_energy(exp.rir, window_length=30, range=500, Fs=exp.fs)
                    
                    # Calculate errors for current position
                    error_50ms = an.compute_RMS(sig1_50ms, sig2_50ms, range=50, Fs=ref_exp.fs, method=error_metric)
                    error_500ms = an.compute_RMS(sig1_500ms, sig2_500ms, range=500, Fs=ref_exp.fs, method=error_metric)
                    row['error_50ms'] = f"{error_50ms:.6f}"
                    row['error_500ms'] = f"{error_500ms:.6f}"
                    table_data.append(row)
                    
                    # Calculate mean errors across all receivers for current source
                    current_src = tuple(room.source_mic_pairs[pos_idx][0])
                    receiver_indices = [i for i, (src, _) in enumerate(room.source_mic_pairs) 
                                     if tuple(src) == current_src]
                    
                    all_errors_50ms = []
                    all_errors_500ms = []
                    for rec_idx in receiver_indices:
                        pos_exps = room.get_experiments_for_position(rec_idx)
                        pos_exp_by_label = {exp.get_label()['label_for_legend']: exp for exp in pos_exps}
                        
                        # Get reference and current experiment for this position
                        pos_ref = pos_exp_by_label[reference_id]  # Use label_for_legend directly
                        pos_exp = pos_exp_by_label[exp.get_label()['label_for_legend']]
                        
                        # Get signals and calculate errors
                        if comparison_type == 'edc':
                            sig1 = pos_ref.edc
                            sig2 = pos_exp.edc
                        elif comparison_type == 'smoothed_energy':
                            _, sig1 = an.calculate_smoothed_energy(pos_ref.rir, window_length=30, range=50, Fs=pos_ref.fs)
                            _, sig2 = an.calculate_smoothed_energy(pos_exp.rir, window_length=30, range=50, Fs=pos_exp.fs)
                        else:  # raw energy
                            sig1, _ = an.calculate_smoothed_energy(pos_ref.rir, window_length=30, range=50, Fs=pos_ref.fs)
                            sig2, _ = an.calculate_smoothed_energy(pos_exp.rir, window_length=30, range=50, Fs=pos_exp.fs)
                        
                        error_50ms = an.compute_RMS(sig1, sig2, range=50, Fs=pos_ref.fs, method=error_metric)
                        error_500ms = an.compute_RMS(sig1, sig2, range=500, Fs=pos_ref.fs, method=error_metric)
                        
                        all_errors_50ms.append(error_50ms)
                        all_errors_500ms.append(error_500ms)
                    
                    # Add mean errors to mean error table
                    mean_error_data.append({
                        'error_50ms': f"{np.mean(all_errors_50ms):.6f}",
                        'error_500ms': f"{np.mean(all_errors_500ms):.6f}"
                    })
            else:
                # Just populate table with experiment info, no error calculations
                for idx, exp in enumerate(sorted_experiments, 1):
                    label_dict = exp.get_label()
                    table_data.append({
                        'id': idx,
                        'method': exp.config.get('method', 'Unknown'),
                        'label': label_dict['label'],
                        'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A',
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

            # Create plots
            rir_fig = go.Figure()
            edc_fig = go.Figure()
            ned_fig = go.Figure()

            for idx, exp in enumerate(sorted_experiments, 1):
                label_dict = exp.get_label()
                
                # Add traces to plots
                rir_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.rir,
                    name=f"{idx}: {label_dict['label_for_legend']}",
                    mode='lines'
                ))

                edc_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.edc,
                    name=f"{idx}: {label_dict['label_for_legend']}",
                    mode='lines'
                ))

                ned_fig.add_trace(go.Scatter(
                    x=exp.ned_time_axis,
                    y=exp.ned,
                    name=f"{idx}: {label_dict['label_for_legend']}",
                    mode='lines'
                ))
            
            # Update plot layouts
            for fig, title in [(rir_fig, "Room Impulse Response"),
                             (edc_fig, "Energy Decay Curve"),
                             (ned_fig, "Normalized Echo Density")]:
                fig.update_layout(
                    title=title,
                    plot_bgcolor=dark_theme['plot_bg'],
                    paper_bgcolor=dark_theme['paper_bg'],
                    font={'color': dark_theme['text']},
                    xaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    yaxis={'gridcolor': dark_theme['grid'], 'zerolinecolor': dark_theme['grid']},
                    margin=dict(t=30, b=20, l=50, r=20)
                )
                
                if time_range != 'full':
                    fig.update_xaxes(range=[0, float(time_range)])

                # Set y-axis ranges
                if fig == rir_fig:
                    fig.update_yaxes(range=[-0.5, 1.0])
                elif fig == edc_fig and (time_range == 0.05 or time_range == '0.05'):
                    fig.update_yaxes(range=[-10, 2])

            return (room_idx, pos_idx, room_plot, room_header, rt_header, 
                   source_options, receiver_options, current_source, current_receiver,
                   room_idx, table_data, columns, dropdown_options,
                   reference_id,
                   rir_fig, edc_fig, ned_fig,
                   mean_error_data)

        # Open browser automatically
        def open_browser():
            webbrowser.open_new(f"http://127.0.0.1:{port}/")

        Timer(1, open_browser).start()
        server_address = f"http://127.0.0.1:{port}/"
        print("\n" + "=" * 70)
        print(f"Dash server is running at: {server_address}")
        print("If the browser doesn't open automatically, please copy and paste the URL above.")
        print("=" * 70)
        # Run the app
        app.run_server(debug=True, port=port)

# Import needed for Room class access
# from sdn_manager_load_sims import Room
# from sdn_manager_load_sims import get_batch_manager, get_singular_manager

if __name__ == "__main__":

    from sdn_manager_load_sims import get_batch_manager, get_singular_manager

    # Create a visualizer using the singular manager
    print("single manager")
    single_manager = get_singular_manager()

    singular_visualizer = SDNExperimentVisualizer(single_manager)
    singular_visualizer.show(port=9052)

    # Create a visualizer using the batch manager
    # print("batch")
    # batch_manager = get_batch_manager()
    # batch_visualizer = SDNExperimentVisualizer(batch_manager)
    # batch_visualizer.show(port=9062)

    # import sdn_experiment_visualizer as sev
    # import importlib
    # importlib.reload(sev)
    # batch_visualizer = sev.SDNExperimentVisualizer(batch_manager)
    # batch_visualizer.show(port=9062)