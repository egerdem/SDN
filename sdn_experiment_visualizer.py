import os
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.graph_objects as go
import numpy as np
import webbrowser
from threading import Timer
from sdn_experiment_manager import Room

class SDNExperimentVisualizer:
    print("visualizer started")

    """Class to visualize SDN experiment data using Dash."""
    
    def __init__(self, manager=None):
        """
        Initialize the visualizer with an experiment manager.
        
        Args:
            manager: An SDNExperimentManager instance to visualize
        """
        self.manager = manager
        
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
                # Add customdata to identify points when clicked
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
                # Add customdata to identify points when clicked
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
        
        return fig

    def show(self, is_batch=True, port=9050):
        """Launch the visualization dashboard."""
        # Get the appropriate manager if not provided
        if self.manager is None:
            self.manager = get_batch_manager() if is_batch else get_singular_manager()

        if not self.manager.rooms:
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

        # Get list of rooms
        room_names = list(self.manager.rooms.keys())
        current_room_idx = 0

        # Create app layout (same as in SDNExperimentManager)
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

            # Main content area
            html.Div([
                # Left side - plots
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
                    dcc.Tabs([
                        dcc.Tab(label="Room Impulse Responses", children=[
                            dcc.Graph(id='rir-plot', style={'height': '50vh'})
                        ],
                        style={'backgroundColor': dark_theme['paper_bg'], 'color': dark_theme['text']},
                        selected_style={'backgroundColor': dark_theme['header_bg'], 'color': dark_theme['accent']}),
                        dcc.Tab(label="Energy Decay Curves", children=[
                            dcc.Graph(id='edc-plot', style={'height': '50vh'})
                        ],
                        style={'backgroundColor': dark_theme['paper_bg'], 'color': dark_theme['text']},
                        selected_style={'backgroundColor': dark_theme['header_bg'], 'color': dark_theme['accent']}),
                        dcc.Tab(label="Normalized Echo Density", children=[
                            dcc.Graph(id='ned-plot', style={'height': '50vh'})
                        ],
                        style={'backgroundColor': dark_theme['paper_bg'], 'color': dark_theme['text']},
                        selected_style={'backgroundColor': dark_theme['header_bg'], 'color': dark_theme['accent']})
                    ], style={'backgroundColor': dark_theme['paper_bg'], 'margin': '10px 0'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Middle - room visualization
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

                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right side - experiment table
                html.Div([
                    html.H3("Active Experiments", 
                           style={'textAlign': 'center', 'marginBottom': '5px', 'marginTop': '65px', 'color': dark_theme['text']}),
                    dash_table.DataTable(
                        id='experiment-table',
                        style_table={'height': '60vh', 'overflowY': 'auto'},
                        style_cell={
                            'backgroundColor': dark_theme['paper_bg'],
                            'color': dark_theme['text'],
                            'textAlign': 'left',
                            'padding': '5px'
                        },
                        style_header={
                            'backgroundColor': dark_theme['header_bg'],
                            'fontWeight': 'bold'
                        }
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
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

        # Combined callback for room/position navigation and dropdown selection
        @app.callback(
            [Output('current-room-idx', 'data'),
             Output('current-pos-idx', 'data'),
             Output('room-plot', 'figure'),
             Output('room-header', 'children'),
             Output('rt-header', 'children'),
             Output('source-selector', 'options'),
             Output('receiver-selector', 'options'),
             Output('source-selector', 'value'),
             Output('receiver-selector', 'value')],
            [Input('prev-room', 'n_clicks'),
             Input('next-room', 'n_clicks'),
             Input('prev-pos', 'n_clicks'),
             Input('next-pos', 'n_clicks'),
             Input('source-selector', 'value'),
             Input('receiver-selector', 'value'),
             Input('room-plot', 'clickData')],
            [State('current-room-idx', 'data'),
             State('current-pos-idx', 'data')]
        )
        def update_room_and_position(prev_room, next_room, prev_pos, next_pos, 
                                   source_value, receiver_value, click_data, room_idx, pos_idx):
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
            elif button_id == 'prev-pos':
                room = self.manager.rooms[room_names[room_idx]]
                pos_idx = (pos_idx - 1) % len(room.source_mic_pairs)
            elif button_id == 'next-pos':
                room = self.manager.rooms[room_names[room_idx]]
                pos_idx = (pos_idx + 1) % len(room.source_mic_pairs)

            room = self.manager.rooms[room_names[room_idx]]
            
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
                        # Get the source position string key
                        source_value = source_keys[point_idx]
                        
                        # Find position with this source and current receiver
                        for i, (s_pos, r_pos) in enumerate(room.source_mic_pairs):
                            s_key = f"({s_pos[0]:.1f}, {s_pos[1]:.1f}, {s_pos[2]:.1f})"
                            r_key = f"({r_pos[0]:.1f}, {r_pos[1]:.1f}, {r_pos[2]:.1f})"
                            if s_key == source_value and (not receiver_value or r_key == receiver_value):
                                pos_idx = i
                                break
                    
                    elif point_type == 'receiver' and point_idx < len(receiver_keys):
                        # Get the receiver position string key
                        receiver_value = receiver_keys[point_idx]
                        
                        # Find position with this receiver and current source
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
                
                # Set current dropdown values
                current_source = f"({source_pos[0]:.1f}, {source_pos[1]:.1f}, {source_pos[2]:.1f})"
                current_receiver = f"({mic_pos[0]:.1f}, {mic_pos[1]:.1f}, {mic_pos[2]:.1f})"
            
            # Create room visualization
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

            return (room_idx, pos_idx, room_plot, room_header, rt_header, 
                   source_options, receiver_options, current_source, current_receiver)

        @app.callback(
            [Output('rir-plot', 'figure'),
             Output('edc-plot', 'figure'),
             Output('ned-plot', 'figure'),
             Output('experiment-table', 'data'),
             Output('experiment-table', 'columns')],
            [Input('current-room-idx', 'data'),
             Input('current-pos-idx', 'data'),
             Input('time-range-selector', 'value')]
        )
        def update_plots_and_table(room_idx, pos_idx, time_range):
            room = self.manager.rooms[room_names[room_idx]]
            experiments = room.get_experiments_for_position(pos_idx)

            # Create plots
            rir_fig = go.Figure()
            edc_fig = go.Figure()
            ned_fig = go.Figure()

            # Prepare table data
            table_data = []
            columns = [
                {'name': 'ID', 'id': 'id'},
                {'name': 'Method', 'id': 'method'},
                {'name': 'Label', 'id': 'label'},
                {'name': 'RT60', 'id': 'rt60'}
            ]
            
            sorted_experiments = sorted(experiments, key=lambda exp: exp.get_label()['label_for_legend'])
            for idx, exp in enumerate(sorted_experiments, 1):
                # Add traces to plots
                label_dict= exp.get_label()
                rir_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.rir,

                    name=f"{idx}: {label_dict['label_for_legend']}"
                ))

                edc_fig.add_trace(go.Scatter(
                    x=exp.time_axis,
                    y=exp.edc,
                    name=f"{idx}: {label_dict['label_for_legend']}"
                ))

                ned_fig.add_trace(go.Scatter(
                    x=exp.ned_time_axis,
                    y=exp.ned,
                    name=f"{idx}: {label_dict['label_for_legend']}"
                ))

                # Add table row
                table_data.append({
                    'id': idx,
                    'method': exp.config.get('method', 'Unknown'),
                    'label': label_dict['label'],
                    'rt60': f"{exp.metrics.get('rt60', 'N/A'):.2f}" if 'rt60' in exp.metrics else 'N/A'
                })

            # Update plot layouts
            for fig, title in [(rir_fig, "Room Impulse Responses"),
                             (edc_fig, "Energy Decay Curves"),
                             (ned_fig, "Normalized Echo Density")]:
                fig.update_layout(
                    title=title,
                    template="plotly_dark",
                    paper_bgcolor="#1e2129",
                    plot_bgcolor="#1e2129",
                    font={"color": "#e0e0e0"}
                )
                if time_range != 'full':
                    fig.update_xaxes(range=[0, float(time_range)])

                # Set y-axis range for RIR plot when early part (50ms) is selected
                if time_range == 0.05 or time_range == '0.05':
                    edc_fig.update_yaxes(range=[-10, 2])

            return rir_fig, edc_fig, ned_fig, table_data, columns

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
    # Create a visualizer using the batch manager
    print("d")
    batch_manager = get_batch_manager()
    # single_manager = get_singular_manager()

    print("r")

    # print("r")
    batch_visualizer = SDNExperimentVisualizer(batch_manager)
    batch_visualizer.show(port=9062)
    
    # Create a visualizer using the singular manager
    # singular_visualizer = SDNExperimentVisualizer(get_singular_manager())
    # singular_visualizer.show(port=9051)

    import sdn_experiment_visualizer as sev
    import importlib
    importlib.reload(sev)
    batch_visualizer = sev.SDNExperimentVisualizer(batch_manager)
    batch_visualizer.show(port=9062)