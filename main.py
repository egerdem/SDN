import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
import path_tracker
from sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
import frequency as ff
import EchoDensity as ned  # Import the EchoDensity module
import analysis as an
import pickle  # Added for loading pickled Treble RI
from rir_calculators import calculate_pra_rir, calculate_rimpy_rir,calculate_sdn_rir
from rir_calculators import calculate_ho_sdn_rir, rir_normalisation

""" Method flags """
PLOT_SDN_BASE = False
RUN_SDN_Test0 = True
RUN_SDN_Test1 = True
RUN_SDN_Test2 = True
RUN_SDN_Test3 = False
RUN_SDN_Test4 = True
RUN_SDN_Test5 = True
RUN_SDN_Test6 = True

RUN_HO_N2 = False
RUN_HO_N2g = False
RUN_HO_N3 = False
RUN_HO_N3g = False

PLOT_TREBLE = False

PLOT_ISM_with_pra = True
PLOT_ISM_rimPy_pos = False  # rimPy ISM with positive reflection
PLOT_ISM_rimPy_neg = False  # rimPy ISM with negative reflection
pra_order = 100

PICKLE_LOAD_RIRS = False  # Load RIRs from pickle file
file_name = "rir_data_rim_ism_ho23_ho23gauss_sdn_c3c5_c3c5gauss_aes.pkl"

""" Visualization flags """
PLOT_ROOM = False        # 3D Room Visualisation
PLOT_ISM_PATHS = False   # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False    # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)
PLOT_REFLECTION_LINES = True     # Plot vertical lines at reflection arrival times
SAVE_audio = False

""" Analysis flags """
PLOT_EDC = False
PLOT_NED = False         # Plot Normalized Echo Density
PLOT_lsd = False         # Plot LSD
PLOT_FREQ = False # Frequency response plot
UNIFIED_PLOTS = True    # Flag to switch between unified and separated plots
normalize_to_first_impulse = True  # Set this to True if you want to normalize to first impulse

Print_RIR_comparison_metrics = True
interactive_rirs = True
pulse_analysis = "upto_4"
plot_smoothed_rirs = False

""" Room Setup """

# ho-waspaa paper room
room_waspaa = {
        'display_name': 'WASPAA Room',
        'width': 6, 'depth': 7, 'height': 4,
        # 'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
        'source x': 3.6, 'source y': 6, 'source z': 1.3,
        # 'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
        'mic x': 1.833333, 'mic y': 3, 'mic z': 2.4,
        'absorption': 0.1,
    }

room_aes = {
        'display_name': 'AES Room',
        'width': 9, 'depth': 7, 'height': 4,
        'source x': 4.5, 'source y': 3.5, 'source z': 2,
        'mic x': 2, 'mic y': 2, 'mic z': 1.5,
        'absorption': 0.2,
        # 'air': {'humidity': 50,
        #        'temperature': 20,
        #        'pressure': 100},
    }

room_journal = {
        'display_name': 'Journal Room',
        'width': 3.2, 'depth': 4, 'height': 2.7,
        'source x': 2, 'source y': 3., 'source z': 2,
        'mic x': 1, 'mic y': 1, 'mic z': 1.5,
        'absorption': 0.1,
    }

# room_journal_random = {
#         'display_name': 'Journal Room',
#         'width': 3.2, 'depth': 4, 'height': 2.7,
#         'source x': 1.5, 'source y': 2.4, 'source z': 2,
#         'mic x': 1.3, 'mic y': 3, 'mic z': 1.4,
#         'absorption': 0.1,
#     }

room_parameters = room_aes  # Choose the room
# room_parameters = room_waspaa  # Choose the room
# room_parameters = room_journal

# Parameters
duration = 1. # seconds
duration_in_ms = 1000 * duration  # Convert to milliseconds
Fs = 44100
num_samples = int(Fs * duration)
rirs = {}
default_rirs = set()  # Track which RIRs should be black
rirs_analysis = {}
# Assign source signals
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)

#print room name and duration of the experiment
print(f"\n=== {room_parameters['display_name']} ===")
print(f"Duration: {duration} seconds")

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal = "will be replaced", Fs = Fs)

room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

# Setup signal
room.source.signal = impulse_dirac['signal']
# room.source.signal = impulse_gaussian['signal']

# Calculate reflection coefficient - will be overwritten for each method
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# SDN Test Configurations
sdn_tests = {

    'TestOptimizer': {
            'enabled': False,
            # 'absorption': 0.2,
            'info': "[9.    5.879 9.    9.    9.    9.   ]",
            'flags': {
                'specular_source_injection': True,
                'injection_c_vector': [9.,    5.879, 9.,    9.,    9.,    9.,   ],
            },
            'label': "SDN"
        },

    'TestX': {
                'enabled': False,
                # 'absorption': 0.2,
                'info': "[4, 4, 1, 1, 1, 1]",
                'flags': {
                    'specular_source_injection': True,
                    'injection_c_vector': [4, 4, 1, 1, 1, 1],
                },
                'label': "SDN"
            },

    'TestY': {
            'enabled': False,
            # 'absorption': 0.2,
            'info': "[ 7.  7.  7.  7.  7. -5.]",
            'flags': {
                'specular_source_injection': True,
                'injection_c_vector': [ 7.,  7.,  7.,  7.,  7., -5.],
            # "ignore_wall_absorption" : True,
            # "ignore_src_node_atten" : True,
            # "ignore_node_mic_atten" : True,
            },
            'label': "SDN"
                },

    'TestZ': {
            'enabled': False,
            # 'absorption': 0.2,
            'info': "[7, 7, -3, -3, -3, 5]",
            'flags': {
                'specular_source_injection': True,
                # 'injection_c_vector': [5, 1, 1, 1, 1, 1],
                'injection_c_vector': [7, 7, -3, -3, -3, 5],
            },
            'label': "SDN"
                },
    'TestT': {
                'enabled': False,
                # 'absorption': 0.2,
                'info': "[8, 1, 1, 1, 1, 1]",
                'flags': {
                    'specular_source_injection': True,
                    'injection_c_vector': [8, 1, 1, 1, 1, 1],

                },
                'label': "SDN"
            },
'Test0': {
            'enabled': RUN_SDN_Test0,
            # 'absorption': 0.2,
            # 'info': "[5,4,3,2,1,1]",
            'info': "c-3",
            'flags': {
                'specular_source_injection': True,
                # 'injection_c_vector': [5, 4, 3, 2, 1, 1],
                'source_weighting': -3,
            # 'scattering_matrix_update_coef' : 0.05
                "ignore_wall_absorption" : True,
                "ignore_src_node_atten" : True,
                "ignore_node_mic_atten" : True,
            },
            'label': "SDN"
        },

    'Test1': {
        'enabled': RUN_SDN_Test1,
        # 'absorption': 0.2,
        'info': "c1 orjinal",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 1,
            # 'specular_scattering': True,
        # 'scattering_matrix_update_coef' : 0.05
            "ignore_wall_absorption" : True,
            "ignore_src_node_atten" : True,
            "ignore_node_mic_atten" : True,
        },
        'label': "SDN"
    },
    'Test2': {
        'enabled': RUN_SDN_Test2,
        # 'absorption': 0.2,
        'info': "c-2",
        'flags': {
            "ignore_wall_absorption" : True,
            "ignore_src_node_atten" : True,
            "ignore_node_mic_atten" : True,
            'specular_source_injection': True,
            'source_weighting': -2,
            'source_pressure_injection_coeff': 1.,
            'coef': 1/5,
            # 'print_parameter_summary': True,
            # 'scattering_matrix_update_coef' : -0.02
            # 'specular_scattering': True,
        },
        'label': "SDN"
    },
    'Test3': {  # New test configuration
        'enabled': RUN_SDN_Test3,
        # 'absorption': 0.3,
        # 'info': "specular + specular -0.02",
        'info': "c-1",
        'flags': {
        # "ignore_wall_absorption" : True,
        # "ignore_src_node_atten" : True,
        # "ignore_node_mic_atten" : True,
        'specular_source_injection': True,
        'source_weighting': -1,
        # "source_pressure_injection_coeff": 0.01,
        # "coef": -0.01
        # 'scattering_matrix_update_coef' : -0.02
        },
        'label': "SDN"
    },

    'Test4': {  # New test configuration
            'enabled': RUN_SDN_Test4,
            # 'absorption': 0.3,
            'info': "c2 eq",
            'flags': {
                "ignore_wall_absorption" : True,
                "ignore_src_node_atten" : True,
                "ignore_node_mic_atten" : True,
                'specular_source_injection': True,
                'source_weighting':2,
                'coef': 1/5,
                'source_pressure_injection_coeff': 1,
            },
            'label': "SDN"
        },

    'Test5': {  # New test configuration
                'enabled': RUN_SDN_Test5,
                # 'absorption': 0.2,
                'info': "c3",
                # 'source_signal': 'gaussian',  # New parameter to specify the source signal type
                'flags': {
                "ignore_wall_absorption": True,
                "ignore_src_node_atten": True,
                "ignore_node_mic_atten": True,
                'specular_source_injection': True,
                'source_weighting': 3,
                },
                'label': "SDN Test 5"
            },
    'Test6': {  # New test configuration
                    'enabled': RUN_SDN_Test6,
                    # 'absorption': 0.3,
                    'info': "c5",
                    'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 5,
                    "ignore_wall_absorption": True,
                    "ignore_src_node_atten": True,
                    "ignore_node_mic_atten": True,
                    'coef': 1,
                    'source_pressure_injection_coeff': 0.2,
                    },
                    'label': "SDN Test 6"
        }
}

# HO-SDN Test Configurations
ho_sdn_tests = {
    'N2': {
        'enabled': RUN_HO_N2,
        'info': "Dirac",
        'source_signal': 'dirac',
        'order': 2,
        'label': "HO-SDN N2"
    },

    'N2g': {
            'enabled': RUN_HO_N2g,
            'info': "Gaussian",
            'source_signal': 'gaussian',
            'order': 2,
            'label': "HO-SDN N2"
        },

    'N3': {
        'enabled': RUN_HO_N3,
        'info': "Dirac",
        'source_signal': 'dirac',
        'order': 3,
        'label': "HO-SDN N3"
    },

    'N3g': {
            'enabled': RUN_HO_N3g,
            'info': "Gaussian",
            'source_signal': 'gaussian',
            'order': 3,
            'label': "HO-SDN N3"
        }
}

# Function to run SDN tests with given configuration, new implementation (SDN-Ege)
def run_sdn_test(test_name, config):
    # Store original signal
    original_signal = room.source.signal
    
    # Check if a specific source signal is requested
    if 'source_signal' in config:
        if config['source_signal'] == 'gaussian':
            room.source.signal = impulse_gaussian['signal']
            print(f"Using Gaussian impulse for {test_name}")
        elif config['source_signal'] == 'dirac':
            room.source.signal = impulse_dirac['signal']
            print(f"Using Dirac impulse for {test_name}")
    
    sdn, rir, label, is_default = calculate_sdn_rir(room_parameters, test_name, room, duration, Fs, config)

    if 'source_signal' in config:
            room.source.signal = original_signal

    rirs[label] = rir

    # Track if this is a default configuration
    if is_default:
        default_rirs.add(label)
    
    return sdn

# Function to run HO-SDN tests with given configuration
def run_ho_sdn_test(test_name, config):
    # Store original signal
    original_signal = room.source.signal
    
    # Check if a specific source signal is requested
    if 'source_signal' in config:
        if config['source_signal'] == 'gaussian':
            room.source.signal = impulse_gaussian['signal']
            print(f"Using Gaussian impulse for {test_name}")
        elif config['source_signal'] == 'dirac':
            room.source.signal = impulse_dirac['signal']
            print(f"Using Dirac impulse for {test_name}")
    
    # Get order from config
    order = config.get('order')  # Default to order 2 if not specified
    
    # Calculate HO-SDN RIR
    rir, label = calculate_ho_sdn_rir(room_parameters, Fs, duration, config['source_signal'], order=order)
    
    # Restore original signal
    if 'source_signal' in config:
        room.source.signal = original_signal
    
    # Add info to label if provided
    if 'info' in config and config['info']:
        label = f"{label}: {config['info']}"
    
    rirs[label] = rir
    
    return rir, label

def get_method_pairs():
    """Get pairs of methods with their original labels and configurations.
    Returns a list of dictionaries with clean method names and their configurations."""
    pairs = []

    # Get actual labels from rirs dictionary
    method_labels = list(rirs.keys())

    # Generate pairs with configurations
    for i in range(len(method_labels)):
        for j in range(i + 1, len(method_labels)):
            label1 = method_labels[i]
            label2 = method_labels[j]

            # Get base labels (before the colon)
            base1 = label1.split(': ')[0] if ': ' in label1 else label1
            base2 = label2.split(': ')[0] if ': ' in label2 else label2

            # Extract configuration info if available (after the colon)
            info1 = label1.split(': ')[1] if ': ' in label1 else ''
            info2 = label2.split(': ')[1] if ': ' in label2 else ''

            # Get binary flags from the label // new part
            flags1 = {}
            flags2 = {}
            
            # Extract flags from the label
            if 'Test' in base1:
                test_name = base1.split('Test')[1].split()[0]  # Get test number
                if test_name in sdn_tests:
                    flags1 = sdn_tests[test_name]['flags']
            
            if 'Test' in base2:
                test_name = base2.split('Test')[1].split()[0]  # Get test number
                if test_name in sdn_tests:
                    flags2 = sdn_tests[test_name]['flags']

            # Combine configurations and flags
            combined_info = ""
            if info1 and info2:
                combined_info = f"{info1} vs {info2}"
            elif info1:
                combined_info = info1
            elif info2:
                combined_info = info2

            # Add flag differences to combined_info // new part
            flag_diffs = []
            all_flags = set(flags1.keys()) | set(flags2.keys())
            for flag in all_flags:
                if flag not in flags1 or flag not in flags2 or flags1[flag] != flags2[flag]:
                    flag_diffs.append(f"{flag}={flags1.get(flag, 'False')} vs {flags2.get(flag, 'False')}")
            
            if flag_diffs:
                if combined_info:
                    combined_info += " | "
                combined_info += " | ".join(flag_diffs)

            pairs.append({
                'pair': f"{base1} vs {base2}",
                'label1': label1,
                'label2': label2,
                'info': combined_info
            })

    return pairs


if __name__ == '__main__':

    if not PICKLE_LOAD_RIRS:
        """ Image Source Methods and Treble """
        if PLOT_ISM_with_pra:
            pra_rir, label = calculate_pra_rir(room_parameters, duration, Fs, pra_order)
            rirs[label] = pra_rir

        # Calculate rimPy ISM RIR
        # Run rimPy with positive reflection coefficient
        if PLOT_ISM_rimPy_pos:
            rimpy_rir_pos, label = calculate_rimpy_rir(room_parameters, duration, Fs, reflection_sign=1)
            rirs[label] = rimpy_rir_pos

        # Run rimPy with negative reflection coefficient
        if PLOT_ISM_rimPy_neg:
            rimpy_rir_neg, label = calculate_rimpy_rir(room_parameters, duration, Fs, reflection_sign=-1)
            rirs[label] = rimpy_rir_neg

        if duration > 0.7:
            # Theoretical RT60 if room dimensions and absorption are provided
            if room_dim is not None and room_parameters['absorption'] is not None:
                rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, room_parameters['absorption'])
                print(f"\nTheoretical RT60 values of the room with a= {room_parameters['absorption']}:")
                print(f"Sabine: {rt60_sabine:.3f} s")
                print(f"Eyring: {rt60_eyring:.3f} s")

        # Calculate SDN-Base RIR
        if PLOT_SDN_BASE:
            from sdn_base import calculate_sdn_base_rir
            sdn_base_rir = calculate_sdn_base_rir(room_parameters, duration, Fs)
            label = 'SDN-Base (original)'
            rirs[label] = sdn_base_rir

        # Run SDN tests
        for test_name, config in sdn_tests.items():
            if config['enabled']:
                sdn = run_sdn_test(test_name, config)

                # Additional analysis: path logging
                if sdn and sdn.enable_path_logging:
                    print(f"\n=== {test_name} Path Analysis ===")
                    print("\n=== First Arriving Paths ===")
                    complete_paths = sdn.path_logger.get_complete_paths_sorted()
                    print("lennnn", len(complete_paths))
                    for path_key, packet in complete_paths[:10]:
                        print(f"{path_key}: arrives at n={packet.birth_sample + packet.delay}, value={packet.value:.6f}")

                    # Create rir_with_paths dictionary with pressure and arrival time
                    rir_with_paths = {}
                    for path_key, packet in complete_paths:
                        rir_with_paths[path_key] = {
                            "pressure": packet.value,
                            "arrival": packet.birth_sample + packet.delay
                        }

                    # Create a path-based RIR from complete_paths
                    if complete_paths:
                        # Find the maximum arrival time to determine the length of the RIR
                        max_arrival = max(packet.birth_sample + packet.delay for _, packet in complete_paths)

                        # Create an empty RIR array
                        path_based_rir = np.zeros(max_arrival + 1)

                        # Fill in the RIR with pressure values at their arrival times
                        for _, packet in complete_paths:
                            arrival_time = packet.birth_sample + packet.delay
                            path_based_rir[arrival_time] += packet.value

                        path_based_rir = path_based_rir / np.max(np.abs(path_based_rir))

                        # Add the path-based RIR to the rirs dictionary
                        rirs[f"SDN {test_name} (Path-based-PROOF)"] = path_based_rir

                        # Also add it to default_rirs to highlight it
                        default_rirs.add(f"{test_name} (Path-based)")

                sdn.get_path_summary()

        # Run HO-SDN tests
        for test_name, config in ho_sdn_tests.items():
            if config['enabled']:
                rir, label = run_ho_sdn_test(test_name, config)

        # Normalize all RIRs
        
        rirs = rir_normalisation(rirs, room, Fs, normalize_to_first_impulse)

    else: # rirs are loaded from pickle:
        with open(file_name, 'rb') as f:
            rirs = pickle.load(f)

    # Calculate reflection arrival times if needed
    reflection_times = None
    if PLOT_REFLECTION_LINES:
        path_tracker = path_tracker.PathTracker()
        sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        sdn_calc.set_path_tracker(path_tracker)
        ism_calc.set_path_tracker(path_tracker)

        # Compare paths and analyze invalid ISM paths (without printing)
        PathCalculator.compare_paths(sdn_calc, ism_calc, max_order=3, print_comparison=False)
        ism_calc.analyze_paths(max_order=3, print_invalid=False)

        # Get arrival times for each order
        arrival_times = path_tracker.get_latest_arrival_time_by_order('ISM')
        reflection_times = {
            'first_order': arrival_times.get(1),
            'second_order': arrival_times.get(2),
            'third_order': arrival_times.get(3)
        }

        if pulse_analysis == "upto_4":
            # Get the arrival time for 3rd order reflections
            arrival_time_3rd = arrival_times.get(4)  # in seconds
            if arrival_time_3rd is not None:
                sample_idx_3rd = int(arrival_time_3rd * Fs)  # convert to samples
                
                # Count nonzero samples for SDN and ISM up to 3rd order arrival
                threshold = 0  # threshold for considering a sample nonzero
                
                # For SDN original
                sdn_rir = rirs['SDN-Original:  ']
                sdn_nonzero = np.sum(np.abs(sdn_rir[:sample_idx_3rd]) > threshold)
                print(f"\nSDN nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {sdn_nonzero}")
                
                # For ISM rimPy negative
                
                ism_rir = rirs['ISM-rimpy-negREF']
                ism_nonzero = np.sum(np.abs(ism_rir[:sample_idx_3rd]) > threshold)
                print(f"ISM-rimpy-neg nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {ism_nonzero}")
                
                # Print valid ISM paths count
                path_tracker.print_valid_paths_count(4)

    if interactive_rirs:    
        if UNIFIED_PLOTS:
            pp.create_unified_interactive_plot(rirs, Fs, default_rirs, room_parameters, reflection_times=reflection_times)
            plt.show(block=False)
        else:
            pp.create_interactive_rir_plot(rirs)
            plt.show(block=False)

    # Calculate RT60 values for all RIRs
    if duration > 0.7:

        rt60_values = {}
        for label, rir in rirs.items():
            # rt60_values[label] = pp.calculate_rt60_from_rir(rir, Fs, plot=False)
            rt60_values[label] = an.calculate_rt60_from_rir(rir, Fs, plot=False)

        print("\nReverberation Time Analysis:")
        print("-" * 50)
        print("\nMeasured RT60 values:")
        for label, rt60 in rt60_values.items():
            print(f"{label}: {rt60:.3f} s")


    if PLOT_EDC:
        if not UNIFIED_PLOTS:
            pp.create_interactive_edc_plot(rirs, Fs, default_rirs)

    if PLOT_NED and not UNIFIED_PLOTS:
        plt.figure(figsize=(12, 6))
        print("old NED plot")
        # Plot both normalized and raw echo densities
        for label, rir in rirs.items():
            # Normalized Echo Density
            echo_density = ned.echoDensityProfile(rir, fs=Fs)
            # Create time array in seconds
            time_axis = np.arange(len(echo_density)) / Fs
            plt.plot(time_axis, echo_density, label=label, alpha=0.7)

        # Configure plots
        plt.title('Normalized Echo Density')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Echo Density')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    if PLOT_lsd:
    # Get all unique pairs of RIR labels
        rir_labels = list(rirs.keys())
        i = 0
        for j in range(i + 1, len(rir_labels)):
            label1 = rir_labels[i]
            label2 = rir_labels[j]
            rir1 = rirs[label1]
            rir2 = rirs[label2]
            lsd_values = an.compute_LSD(rir1, rir2, Fs)
            print(f"{label1} vs {label2}: {lsd_values:.2f} dB LSD difference")
            an.plot_spectral_comparison(rir1, rir2, Fs=Fs, label1=label1, label2=label2)

    if PLOT_FREQ:
        # Calculate and plot frequency responses for all RIRs
        for rir_label, rir in rirs.items():
            # Add RT60 to the label
            plot_label = f"{rir_label} (RT60: {rt60_values[rir_label]:.3f}s)"
            freq, magnitude = ff.calculate_frequency_response(rir, Fs)
            ff.plot_frequency_response(freq, magnitude, label=f'FRF {plot_label}')


    # Process method pairs once
    method_pairs = get_method_pairs()

    if Print_RIR_comparison_metrics:
        # Dictionary to store all RIR analyses
        for rir_label, rir in rirs.items():
            
            smoothed = an.calculate_smoothed_energy(rir, window_length=30, range=50, Fs=Fs)
            early_energy, energy, ERR = an.calculate_err(rir, early_range=50, Fs=Fs)
            c50 = an.compute_clarity_c50(rir, Fs=Fs)
            c80 = an.compute_clarity_c80(rir, Fs=Fs)

            rirs_analysis[rir_label] = {
            "smoothed_energy": smoothed,
            "early_energy": early_energy,
            "energy": energy,
            "ERR": ERR,
            "c50": c50,
            "c80": c80
            }

            print("\nEnergy Total {} = {:.3f}".format(rir_label, sum(energy)))
            print("Energy 50ms {} = {:.3f}".format(rir_label, sum(early_energy)))
            print("ERR: Energy50ms/EnergyTotal {} = {:.3f}".format(rir_label, ERR))
            print("C50 {} = {:.3f}".format(rir_label, c50))
            print("C80 {} = {:.3f}".format(rir_label, c80))

        # Dictionary to store comparison results for both energy and smoothed signals
        comparison_results = {
            'early_energy': [],
            'smoothed_energy': []
        }

        # Process energy signal comparisons
        print("\n" + "="*130)
        print(" "*50 + "Energy (squared RIRs) Signal Comparison Results")
        print("="*130)
        print(f"{'Method Pair':<40} {'Rmse':>12} {'MAE':>12} {'Median':>12} {'Info':>20}")
        print("-"*130)

        for pair_info in method_pairs:
            l1, l2 = pair_info['label1'], pair_info['label2']

            # Compare energy signals
            energy1 = rirs_analysis[l1]['early_energy']
            energy2 = rirs_analysis[l2]['early_energy']

            # Store energy comparison results
            energy_result = {
                'pair': pair_info['pair'],
                'info': pair_info['info'],
                'rmse': an.compute_RMS(energy1, energy2, method="rmse"),
                'mae': an.compute_RMS(energy1, energy2, method="mae"),
                'median': an.compute_RMS(energy1, energy2, method="median")
            }
            comparison_results['early_energy'].append(energy_result)

            # Print energy comparison results
            print(f"{energy_result['pair']:<40} {energy_result['rmse']:12.6f} {energy_result['mae']:12.6f} "
                  f"{energy_result['median']:12.6f}  {energy_result['info']:>20}")

        print("="*130)

        # Process smoothed signal comparisons
        print("\n" + "="*130)
        print(" "*50 + "Smoothed (hann windowed squared RIRs) Signal Comparison Results")
        print("="*130)
        print(f"{'Method Pair':<40} {'Rmse':>12} {'MAE':>12} {'Median':>12} {'Info':>20}")
        print("-"*130)

        for pair_info in method_pairs:
            l1, l2 = pair_info['label1'], pair_info['label2']

            # Compare smoothed signals
            smoothed1 = rirs_analysis[l1]['smoothed_energy']
            smoothed2 = rirs_analysis[l2]['smoothed_energy']

            # Store smoothed comparison results
            smoothed_result = {
                'pair': pair_info['pair'],
                'info': pair_info['info'],
                'rmse': an.compute_RMS(smoothed1, smoothed2, method="rmse"),
                'mae': an.compute_RMS(smoothed1, smoothed2, method="mae"),
                'median': an.compute_RMS(smoothed1, smoothed2, method="median")
            }
            comparison_results['smoothed_energy'].append(smoothed_result)

            # Print smoothed comparison results
            print(f"{smoothed_result['pair']:<40} {smoothed_result['rmse']:12.6f} {smoothed_result['mae']:12.6f} "
                  f"{smoothed_result['median']:12.6f}  {smoothed_result['info']:>20}")

        print("="*130)
        print("\n")

        # Compare all distinct pairs of EDCs
        print("\nEDC Comparison (First 50ms RMSE Differences):")
        print("-" * 50)

        # Get all unique pairs of RIR labels
        i = 0

        for pair_info in get_method_pairs():
            label1, label2 = pair_info['label1'], pair_info['label2']

            # Calculate EDCs for both RIRs without plotting
            edc1, _,_ = an.compute_edc(rirs[label1], Fs, label1, plot=False)
            edc2, _,_ = an.compute_edc(rirs[label2], Fs, label2, plot=False)

            # Compare EDCs
            rms_diff = an.compute_RMS(edc1, edc2, range=50, Fs=Fs, method="mae")  # range=50ms
            print(f"{label1} vs {label2}: {rms_diff:.2f} dB RMSE difference")

    if plot_smoothed_rirs:
        # Iterate through each method in rirs_analysis
        for rir_label, signals in rirs_analysis.items():
            plt.figure(figsize=(12, 6))

            # Plot Energy Signal
            plt.plot(signals["energy"], label=f'Energy Signal - {rir_label}', color='blue')

            # Plot Smoothed Signal
            plt.plot(signals["smoothed_energy"], label=f'Smoothed Energy Signal - {rir_label}', color='orange')

            # Add titles and labels
            plt.title(f'Energy and Smoothed Energy Signals for {rir_label}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            # Show the plot
            plt.show()

# Only Path Length Analysis, No RIR Calculation 
if ISM_SDN_PATH_DIFF_TABLE:
    
    if not PLOT_REFLECTION_LINES:
        path_tracker = path_tracker.PathTracker()
        sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        sdn_calc.set_path_tracker(path_tracker)
        ism_calc.set_path_tracker(path_tracker)

    PathCalculator.compare_paths(sdn_calc, ism_calc, max_order=3, print_comparison=True)
    # analyze_paths() returns a list of invalid paths (only for ISM calculator)
    invalid_paths = ism_calc.analyze_paths(max_order=3, print_invalid=True)

    # Visualize example ISM paths
    example_paths = [
        # ['s', 'east', 'west', 'm'],
        # ['s', 'west', 'm'],
        # ['s', 'ceiling', 'm'],
        # ['s', 'm'],
        # ['s', 'west', 'east', 'north', 'm']
    ]

    for path in example_paths:
        pp.plot_ism_path(room, ism_calc, path)
        plt.show()

if PLOT_ROOM:
    if not ISM_SDN_PATH_DIFF_TABLE:
        pp.plot_room(room)


if pulse_analysis == "all":

    print("\n=== RIR Pulse Analysis ===")
    from scipy.signal import find_peaks
    for rir_label, rir in rirs.items():
        print(f"\n{rir_label}:")

        if "ISM-pra" in rir_label:  # For PRA method, use peak detection
            # Find peaks that are at least 1% of the maximum amplitude
            threshold = 0.01 * np.max(np.abs(rir))
            peaks, _ = find_peaks(np.abs(rir), height=threshold)

            print(f"  Total significant peaks: {len(peaks)}")
            print(f"  First peak at: {peaks[0]/Fs*1000:.2f} ms")
            print(f"  Last peak at: {peaks[-1]/Fs*1000:.2f} ms")
            print(f"  Time span: {(peaks[-1] - peaks[0])/Fs*1000:.2f} ms")

        else:  # For other methods, use non-zero analysis
            # Count total nonzero pulses (using small threshold to account for floating point)
            threshold = 1e-10  # Adjust this threshold based on your needs
            nonzero_indices = np.where(np.abs(rir) > threshold)[0]
            nonzero_count = len(nonzero_indices)

            # Calculate percentage of nonzero samples
            percentage = (nonzero_count / len(rir)) * 100

            # Find first and last nonzero indices
            first_pulse = nonzero_indices[0] if nonzero_count > 0 else 0
            last_pulse = nonzero_indices[-1] if nonzero_count > 0 else 0

            print(f"  Total (nonzero) pulses: {nonzero_count}")
            print(f"  Percentage of nonzero samples: {percentage:.2f}%")
            print(f"  First pulse at: {first_pulse/Fs*1000:.2f} ms")
            print(f"  Last pulse at: {last_pulse/Fs*1000:.2f} ms")
            print(f"  Time span: {(last_pulse - first_pulse)/Fs*1000:.2f} ms")


"""import pickle
# # Save the RIRs to a file

try:
    with open(file_name, 'rb') as f:
        existing_rirs = pickle.load(f)
    # If the file exists, write a message
    print("File already exists. Not overwriting. Please delete the file to save new data.")

# dump the file otherwise
except FileNotFoundError:
    # If the file doesn't exist, write the new data
    print("File not found. Saving new data.")
    # Save the RIRs to the file
    with open(file_name, 'wb') as f:
        # Save the RIRs to the file
        pickle.dump(rirs, f)

"""

if SAVE_audio:
    # Save the RIRs as wav
    import soundfile as sf
    for rir_label, rir_audio in rirs.items():
        sf.write(f"{rir_label}.wav", rir_audio * 0.8, Fs)
