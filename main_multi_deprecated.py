"""
main_multi.py - Multi-Receiver Spatial Analysis Mode

Same as main.py but for multi-receiver grid experiments.
Enables fast iteration during development without needing to save/load .npz files.

Usage:
    1. Configure methods (ISM, SDN, HO-SDN) by enabling flags below
    2. Set USE_GRID = True and adjust GRID_N_POINTS
    3. Run: python main_multi.py
    4. View comprehensive metrics table + interactive plot for first receiver
"""

import numpy as np
import geometry
import matplotlib.pyplot as plt

from analysis import plot_room as pp
from analysis import analysis as an
from analysis.spatial_analysis import generate_receiver_grid_old

from rir_calculators import calculate_pra_rir, calculate_rimpy_rir, calculate_sdn_rir, calculate_sdn_rir_fast
from rir_calculators import calculate_ho_sdn_rir, rir_normalisation

# =============================================================================
# MULTI-RECEIVER GRID CONFIGURATION
# =============================================================================
USE_GRID = True  # If False, behaves like single-receiver main.py
GRID_N_POINTS = 16  # Number of receiver positions (4x4 grid)
GRID_MARGIN = 0.5  # Margin from walls in meters

SHOW_SPATIAL_SUMMARY_TABLE = True  # Show comprehensive metrics table
SHOW_INTERACTIVE_PLOT = False  # Show unified interactive plot for first receiver

# =============================================================================
# METHOD FLAGS (Same as main.py)
# =============================================================================

""" Method flags """
# Single test flags
wall1, wall2, wall3, wall4, wall5, wall6 = False, False, False, False, False, False
t1111, t111, t11 = False, False, False
t1,t2,t3,t4,t5 = False, False, False, False, False
fast1, fast2 = False, False
testx = False

# SDN Tests
RUN_SDN_Test0 = False
RUN_SDN_Test1 = False  # c=1 original
RUN_SDN_Test2 = False
RUN_SDN_Test3 = True
RUN_SDN_Test4 = False
RUN_SDN_Test5 = False
RUN_SDN_Test6 = False
RUN_SDN_Test7 = False

# HO-SDN Tests
RUN_MY_HO_SDN_n1 = False
RUN_MY_HO_SDN_n2 = False
RUN_MY_HO_SDN_n3 = False
RUN_MY_HO_SDN_n2_swc5 = False
RUN_MY_HO_SDN_n2_swc3 = False
RUN_MY_HO_SDN_n3_swc3 = False

# ISM Methods
PLOT_ISM_with_pra = False
PLOT_ISM_with_pra_rand10 = False
PLOT_ISM_rimPy_pos = False
PLOT_ISM_rimPy_pos_rand10 = False
PLOT_ISM_rimPy_neg = False
PLOT_ISM_rimPy_neg_rand10 = False  # Reference method

# =============================================================================
# ROOM SETUP (Same as main.py)
# =============================================================================

room_aes = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

room_aes_rx00 = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 0.5, 'mic y': 0.5, 'mic z': 1.5,
    'absorption': 0.2,
}

room_waspaa = {
    'display_name': 'WASPAA Room',
    'width': 6, 'depth': 7, 'height': 4,
    'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
    'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
    'absorption': 0.1,
}

# Choose the room
room_parameters = room_aes

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

duration = 1  # seconds
Fs = 44100
num_samples = int(Fs * duration)

# Assign source signals
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)

# Setup room geometry
room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal=impulse_dirac['signal'], Fs=Fs)

room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# =============================================================================
# METHOD CONFIGURATIONS (Same as main.py)
# =============================================================================

# ISM Methods
ism_methods = {
    'ISM-pra': {
        'enabled': PLOT_ISM_with_pra,
        'info': 'pra 100',
        'function': calculate_pra_rir,
        'params': {'max_order': 100, 'use_rand_ism': False}
    },
    'ISM-pra-rand10': {
        'enabled': PLOT_ISM_with_pra_rand10,
        'info': 'pra 100 + 10cm randomness',
        'function': calculate_pra_rir,
        'params': {'max_order': 100, 'use_rand_ism': True, 'max_rand_disp': 0.1}
    },
    'rimPy-pos': {
        'enabled': PLOT_ISM_rimPy_pos,
        'info': 'Positive Reflection',
        'function': calculate_rimpy_rir,
        'params': {'reflection_sign': 1, 'tw_fractional_delay_length': 0}
    },
    'rimPy-pos-rand10': {
        'enabled': PLOT_ISM_rimPy_pos_rand10,
        'info': 'Positive Reflection + 10cm randomness',
        'function': calculate_rimpy_rir,
        'params': {'reflection_sign': 1, 'tw_fractional_delay_length': 0, 'randDist': 0.1}
    },
    'rimPy-neg': {
        'enabled': PLOT_ISM_rimPy_neg,
        'info': 'Negative Reflection',
        'function': calculate_rimpy_rir,
        'params': {'reflection_sign': -1, 'tw_fractional_delay_length': 0}
    },
    'rimPy-neg-rand10': {
        'enabled': PLOT_ISM_rimPy_neg_rand10,
        'info': 'Negative Reflection + 10cm Randomness',
        'function': calculate_rimpy_rir,
        'params': {'reflection_sign': -1, 'tw_fractional_delay_length': 0, 'randDist': 0.1}
    }
}

# SDN Tests (abbreviated - add more as needed from main.py)
sdn_tests = {
    'fast1': {
        'enabled': fast1, 'use_fast_method': True,
        'info': "fast [4.35,6.05,2.85,4.02,1.68,2.17]",
        'flags': {
            'specular_source_injection': True,
            'injection_c_vector':[4.35,6.05,2.85,4.02,1.68,2.17]
        }, 
        'label': "SDN"
    },
    'fast2': {
        'enabled': fast2, 'use_fast_method': True,
        'info': "optimized c-vector",
        'flags': {
            'specular_source_injection': True,
            'injection_c_vector':[4.79,6.20,1.00,7.00,1.00,2.09]
        }, 
        'label': "SDN"
    },
    'Test2.998':  {
        'enabled': testx,
        'info': "c 2.998",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2.998,
        }, 
        'label': "SDN"
    },
    'Test0': {
        'enabled': RUN_SDN_Test0,
        'info': "c0",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 0,
        },
        'label': "SDN"
    },
    'Test1': {
        'enabled': RUN_SDN_Test1,
        'info': "c1 original",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 1,
        },
        'label': "SDN"
    },
    'Test2': {
        'enabled': RUN_SDN_Test2,
        'info': "c2",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2,
        },
        'label': "SDN"
    },
    'Test3': {
        'enabled': RUN_SDN_Test3,
        'info': "c3",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 3,
        },
        'label': "SDN"
    },
    'Test4': {
        'enabled': RUN_SDN_Test4,
        'info': "c4",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 4,
        },
        'label': "SDN"
    },
    'Test5': {
        'enabled': RUN_SDN_Test5,
        'info': "c5",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 5,
        },
        'label': "SDN Test 5"
    },
    'Test6': {
        'enabled': RUN_SDN_Test6,
        'info': "c6",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 6,
        },
        'label': "SDN Test 6"
    },
    'Test7': {
        'enabled': RUN_SDN_Test7,
        'info': "c7",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 7,
        },
        'label': "SDN Test 7"
    },
}

# HO-SDN Tests
ho_sdn_tests = {
    'TestHO_N1': {
        'enabled': RUN_MY_HO_SDN_n1,
        'info': "HO-SDN order 1",
        'order': 1,
        'source_signal': 'dirac',
        'label': "HO-SDN N=1"
    },
    'TestHO_N2': {
        'enabled': RUN_MY_HO_SDN_n2,
        'info': "HO-SDN order 2",
        'order': 2,
        'source_signal': 'dirac',
        'label': "HO-SDN N=2"
    },
    'TestHO_N3': {
        'enabled': RUN_MY_HO_SDN_n3,
        'info': "HO-SDN order 3",
        'order': 3,
        'source_signal': 'dirac',
        'label': "HO-SDN N=3"
    },
    'TestHO_N2_swc5': {
        'enabled': RUN_MY_HO_SDN_n2_swc5,
        'info': "sw-c5-ho-N2",
        'order': 2,
        'source_weighting': 5,
        'source_signal': 'dirac',
        'label': "SW-c5-HO-N2"
    },
    'TestHO_N2_swc3': {
        'enabled': RUN_MY_HO_SDN_n2_swc3,
        'info': "sw-c3-ho-N2",
        'order': 2,
        'source_weighting': 3,
        'source_signal': 'dirac',
        'label': "SW-c3-HO-N2"
    },
    'TestHO_N3_swc3': {
        'enabled': RUN_MY_HO_SDN_n3_swc3,
        'info': "sw-c3-ho-N3",
        'order': 3,
        'source_weighting': 3,
        'source_signal': 'dirac',
        'label': "SW-c3-HO-N3"
    },
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("MULTI-RECEIVER SPATIAL ANALYSIS MODE (main_multi.py)")
    print("="*80)
    
    # Generate receiver positions
    if USE_GRID:
        # Use quarter-grid pattern like in generate_paper_data.py
        receiver_positions_2d = generate_receiver_grid_old(
            room_parameters['width'] / 2,
            room_parameters['depth'] / 2,
            margin=GRID_MARGIN,
            n_points=GRID_N_POINTS
        )
        # Add z-coordinate
        receiver_positions = [(x, y, room_parameters['mic z']) for x, y in receiver_positions_2d]
        print(f"\nRunning in MULTI-POSITION (QUARTER-GRID) mode for room '{room_parameters['display_name']}'")
    else:
        receiver_positions = [(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])]
        print(f"\nRunning in SINGLE-POSITION mode for room '{room_parameters['display_name']}'")
    
    num_receivers = len(receiver_positions)
    print(f"Generated {num_receivers} receiver positions")
    print(f"Grid margin: {GRID_MARGIN}m")
    print(f"First receiver: ({receiver_positions[0][0]:.2f}m, {receiver_positions[0][1]:.2f}m, {receiver_positions[0][2]:.2f}m)")
    
    # Dictionary to store RIRs: {method_label: [rir_receiver0, rir_receiver1, ...]}
    all_rirs_by_method = {}
    
    # Collect enabled methods
    enabled_ism_methods = [(name, config) for name, config in ism_methods.items() if config['enabled']]
    enabled_sdn_methods = [(name, config) for name, config in sdn_tests.items() if config['enabled']]
    enabled_ho_methods = [(name, config) for name, config in ho_sdn_tests.items() if config['enabled']]
    
    total_methods = len(enabled_ism_methods) + len(enabled_sdn_methods) + len(enabled_ho_methods)
    print(f"\nCalculating RIRs for {total_methods} methods across {num_receivers} receivers...")
    print(f"Total RIR calculations: {total_methods * num_receivers}")
    
    # Loop through all receiver positions
    for rx_idx, (rx, ry, rz) in enumerate(receiver_positions):
        print(f"\n--- Receiver {rx_idx+1}/{num_receivers}: ({rx:.2f}m, {ry:.2f}m, {rz:.2f}m) ---")
        
        # Update microphone position
        room.set_microphone(rx, ry, rz)
        room_parameters_temp = room_parameters.copy()
        room_parameters_temp.update({'mic x': rx, 'mic y': ry, 'mic z': rz})
        
        # Calculate ISM methods for this receiver
        for method_name, config in enabled_ism_methods:
            rir, label = config['function'](
                room_parameters_temp,
                duration,
                Fs,
                **config['params']
            )
            if label not in all_rirs_by_method:
                all_rirs_by_method[label] = []
            all_rirs_by_method[label].append(rir)
        
        # Calculate SDN methods for this receiver
        for test_name, config in enabled_sdn_methods:
            if config.get('use_fast_method', False):
                sdn, rir, label, is_default = calculate_sdn_rir_fast(
                    room_parameters_temp, test_name, room, duration, Fs, config
                )
            else:
                sdn, rir, label, is_default = calculate_sdn_rir(
                    room_parameters_temp, test_name, room, duration, Fs, config
                )
            if label not in all_rirs_by_method:
                all_rirs_by_method[label] = []
            all_rirs_by_method[label].append(rir)
        
        # Calculate HO-SDN methods for this receiver
        for test_name, config in enabled_ho_methods:
            order = config.get('order', 2)
            source_signal_type = config.get('source_signal', 'dirac')
            rir, label = calculate_ho_sdn_rir(
                room_parameters_temp, Fs, duration, source_signal_type, order=order
            )
            if label not in all_rirs_by_method:
                all_rirs_by_method[label] = []
            all_rirs_by_method[label].append(rir)
    
    print("\n" + "="*80)
    print("RIR CALCULATIONS COMPLETE!")
    print("="*80)
    
    # =========================================================================
    # SPATIAL ANALYSIS: Calculate Metrics
    # =========================================================================
    
    if SHOW_SPATIAL_SUMMARY_TABLE:
        print("\n" + "="*80)
        print("SPATIAL ANALYSIS SUMMARY")
        print("="*80)
        
        # Determine reference method (first ISM method or first method overall)
        reference_label = list(all_rirs_by_method.keys())[0]
        if enabled_ism_methods:
            # Use first enabled ISM method as reference
            rir_ref, ref_label = enabled_ism_methods[0][1]['function'](
                room_parameters, duration, Fs, **enabled_ism_methods[0][1]['params']
            )
            reference_label = ref_label
        
        print(f"Reference Method: {reference_label}\n")
        
        # Calculate metrics for each method
        method_metrics = {}
        
        for method_label, rirs_list in all_rirs_by_method.items():
            metrics = {
                'edc_rmse_50ms': [],
                'smoothed_rir_50ms': [],
                'smoothed_rir_full': [],
                'energy': [],
                'rt60': []
            }
            
            ref_rirs = all_rirs_by_method[reference_label]
            
            for i in range(num_receivers):
                ref_rir = ref_rirs[i]
                test_rir = rirs_list[i]
                
                # EDC RMSE (50ms)
                edc_ref, _, _ = an.compute_edc(ref_rir, Fs, plot=False)
                edc_test, _, _ = an.compute_edc(test_rir, Fs, plot=False)
                err_samples = int(50 / 1000 * Fs)
                edc_rmse = an.compute_RMS(
                    edc_ref[:err_samples], edc_test[:err_samples],
                    range=50, Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True
                )
                metrics['edc_rmse_50ms'].append(edc_rmse if method_label != reference_label else 0.0)
                
                # Smoothed RIR (50ms)
                smooth_ref = an.calculate_smoothed_energy(ref_rir, window_length=30, Fs=Fs)
                smooth_test = an.calculate_smoothed_energy(test_rir, window_length=30, Fs=Fs)
                smooth_rmse_50ms = an.compute_RMS(
                    smooth_ref[:err_samples], smooth_test[:err_samples],
                    range=50, Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True
                )
                metrics['smoothed_rir_50ms'].append(smooth_rmse_50ms if method_label != reference_label else 0.0)
                
                # Smoothed RIR (full)
                smooth_rmse_full = an.compute_RMS(
                    smooth_ref, smooth_test,
                    range=None, Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True
                )
                metrics['smoothed_rir_full'].append(smooth_rmse_full if method_label != reference_label else 0.0)
                
                # Energy
                _, energy, _ = an.calculate_err(test_rir, Fs=Fs)
                metrics['energy'].append(np.sum(energy))
                
                # RT60 (if duration is sufficient)
                if duration > 0.7:
                    rt60 = an.calculate_rt60_from_rir(test_rir, Fs, plot=False)
                    metrics['rt60'].append(rt60 if rt60 is not None else 0.0)
            
            # Store averaged metrics
            method_metrics[method_label] = {
                'edc_rmse': np.mean(metrics['edc_rmse_50ms']),
                'smooth_50ms': np.mean(metrics['smoothed_rir_50ms']),
                'smooth_full': np.mean(metrics['smoothed_rir_full']),
                'energy': np.mean(metrics['energy']),
                'rt60': np.mean(metrics['rt60']) if metrics['rt60'] else None
            }
        
        # Print summary table
        print(f"{'Method':<30} {'EDC-50ms':>10} {'Smooth-50ms':>12} {'Smooth-Full':>12} {'Energy':>10} {'RT60':>8}")
        print("-" * 92)
        
        for method_label, metrics in method_metrics.items():
            rt60_str = f"{metrics['rt60']:.3f}" if metrics['rt60'] is not None else "N/A"
            print(f"{method_label:<30} {metrics['edc_rmse']:>10.4f} {metrics['smooth_50ms']:>12.4f} "
                  f"{metrics['smooth_full']:>12.4f} {metrics['energy']:>10.2f} {rt60_str:>8}")
        
        print("-" * 92)
        print(f"\nAnalyzed {num_receivers} receivers\n")
    
    # =========================================================================
    # INTERACTIVE PLOT: Show First Receiver
    # =========================================================================
    
    if SHOW_INTERACTIVE_PLOT:
        print("Creating interactive plot for first receiver...")
        
        # Extract RIRs for first receiver
        first_receiver_rirs = {}
        for method_label, rirs_list in all_rirs_by_method.items():
            first_receiver_rirs[method_label] = rirs_list[0]
        
        # Reverse order for display (newest first)
        reversed_rirs = dict(reversed(list(first_receiver_rirs.items())))
        
        # Update room parameters for first receiver
        plot_room_params = room_parameters.copy()
        plot_room_params['mic x'] = receiver_positions[0][0]
        plot_room_params['mic y'] = receiver_positions[0][1]
        plot_room_params['mic z'] = receiver_positions[0][2]
        
        pp.create_unified_interactive_plot(
            reversed_rirs, Fs, plot_room_params,
            reflection_times=None
        )
        plt.show(block=True)  # block=True to keep window open
        print(f"Interactive plot displayed for first receiver at ({receiver_positions[0][0]:.2f}m, {receiver_positions[0][1]:.2f}m)")
    
    print("\n" + "="*80)
    print("MULTI-RECEIVER ANALYSIS COMPLETE")
    print("="*80)
