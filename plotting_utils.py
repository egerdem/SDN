import numpy as np
import os
from typing import Dict

# --- Room Definitions ---
# Central place to define room parameters for consistency.
ROOM_AES = { 'display_name': 'AES Room' }
ROOM_JOURNAL = { 'display_name': 'Journal Room' }
ROOM_WASPAA = { 'display_name': 'WASPAA Room' }

# --- Plotting Style and Configuration ---
# This dictionary holds all the specific parameters for plotting,
# allowing for easy adjustments for each room without changing the script logic.
PLOT_CONFIG = {
    'AES Room': {
        'edc': {
            'main_xlim_focused': (0.005, 0.07),
            'main_ylim_focused': (-8, 0.2),
            'main_xlim_full': (0, 1.1),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0.012, 0.03),
            'inset_ylim': (-5, 0.4)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.05, 0.51, 0.2, 0.45],  # x, y, width, height
            'inset_xlim': (0.025, 0.052),
            'inset_ylim': (0.0, 0.6)
        }
    },
    'Journal Room': {
        'edc': {
            'main_xlim_focused': (0.00, 0.07),
            'main_ylim_focused': (-8, 0.2),
            'main_xlim_full': (0, 1.1),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0, 1.2),
            'inset_ylim': (-62, 5)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.50, 0.07, 0.2, 0.45], # x, y, width, height
            'inset_xlim': (0.0, 0.025),
            'inset_ylim': (0, 0.3)
        }
    },
    'WASPAA Room': {
        'edc': {
            'main_xlim_focused': (0.01, 0.07),
            'main_ylim_focused': (-5, 0.2),
            'main_xlim_full': (0, 1.8),
            'main_ylim_full': (-62, 5),
            'inset_rect': [0.07, 0.07, 0.4, 0.3],
            'inset_xlim': (0.012, 0.03),
            'inset_ylim': (-5, 0.4)
        },
        'ned': {
            'main_xlim': (0, 0.2),
            'main_ylim': (0, 1.3),
            'inset_rect': [0.50, 0.07, 0.2, 0.45], # x, y, width, height
            'inset_xlim': (0.015, 0.035),
            'inset_ylim': (0, 0.25)
        }
    },
}

def get_display_name(method_key: str, method_configs: dict, name_map: dict) -> str:
    """
    Generate a publication-ready display name for a given method.
    This provides a single place to control legend labels.
    """
    config = method_configs.get(method_key, {})
    info = config.get('info', method_key) # Default to key if no info
    return name_map.get(method_key, info)

def get_linestyle(method_key: str) -> str:
    """
    Assign a specific linestyle based on the method type for the plot.
    This provides a single place to control line styles.
    """
    if method_key.startswith('HO-SDN'):
        return ':'  # Dotted line for all HO-SDN methods
    elif method_key == 'ISM':
        return '--' # Dashed line for ISM
    elif method_key.startswith('SDN-'):
        return '-'  # Solid line for the user's SW-SDN approach
    else:
        return '-.' # A default style for any other method

def load_data(data_path: str) -> dict:
    """Load the pre-calculated simulation data from an .npz file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. "
                              f"Please run generate_paper_data.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    
    # Extract data into a more usable dictionary format
    results = {
        'edcs': {},
        'neds': {},
        'rirs': {},
        'receiver_positions': data['receiver_positions'],
        'source_pos': data['source_pos'],
        'Fs': int(data['Fs']),
        'duration': float(data['duration']),
        'method_configs': data['method_configs'][0],
        'room_params': data['room_params'][0]
    }
    
    for key in data.keys():
        if key.startswith('rirs_'):
            method = key.replace('rirs_', '')
            results['rirs'][method] = data[key]
        elif key.startswith('edcs_'):
            method = key.replace('edcs_', '')
            results['edcs'][method] = data[key]
        elif key.startswith('neds_'):
            method = key.replace('neds_', '')
            results['neds'][method] = data[key]
            
    return results 