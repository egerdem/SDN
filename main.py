import numpy as np
import geometry
import matplotlib.pyplot as plt
import pickle 

from analysis import plot_room as pp
from analysis import path_tracker
from analysis.sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
from analysis import frequency as ff
from analysis import EchoDensity as ned
from analysis import analysis as an

from rir_calculators import calculate_pra_rir, calculate_rimpy_rir, calculate_sdn_rir, calculate_sdn_rir_fast
from rir_calculators import calculate_ho_sdn_rir, rir_normalisation

import experiment_configs as exp_config

# --- Method Configurations ---
# Import from experiment_configs
ism_methods = exp_config.ism_methods
sdn_tests = exp_config.sdn_tests
ho_sdn_tests = exp_config.ho_sdn_tests

""" Method flags """
PLOT_SDN_BASE = False
RUN_SDN_Test1b = False
RUN_SDN_Test1c = False
PLOT_TREBLE = False

PICKLE_LOAD_RIRS = False # Load RIRs from pickle file
# file_name = "rirs_c_cSN_cSPMAT_rimneg_ho3ho2.pkl"
# file_name = "rirs_c_cSN_cSPMAT_rimneg_ho3ho2.pkl"
# file_name = "rirs_c_cSN.pkl"
file_name = "pra_n2swc3_n2swc5_n3_swc3_hoN2_hoN3_c5_c1.pkl"  #

""" Visualization flags """
PLOT_ROOM = False  # 3D Room Visualisation
PLOT_ISM_PATHS = False  # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False  # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)
PLOT_REFLECTION_LINES = True  # Plot vertical lines at reflection arrival times
SAVE_audio = False

""" Analysis flags """
PLOT_EDC = False
PLOT_NED = False  # Plot Normalized Echo Density
PLOT_lsd = False  # Plot LSD
PLOT_FREQ = False  # Frequency response plot
UNIFIED_PLOTS = True  # Flag to switch between unified and separated plots
normalize_to_first_impulse = True  # Set this to True if you want to normalize to first impulse

Print_RIR_comparison_metrics = True
interactive_rirs = True  # Set to True to enable interactive RIR comparison
pulse_analysis = "upto_4"
plot_smoothed_rirs = False


# --- Room Setup ---
# Use active room from experiment_configs
room_parameters = exp_config.active_room

# Parameters
duration = 1  # seconds
duration_in_ms = 1000 * duration  # Convert to milliseconds

Fs = 44100
num_samples = int(Fs * duration)
rirs = {}
default_rirs = set()  # Track which RIRs should be black
rirs_analysis = {}
# Assign source signals
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)

# print room name and duration of the experiment
print(f"\n=== {room_parameters['display_name']} ===")
print(f"Duration: {duration} seconds")

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal="will be replaced", Fs=Fs)

room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

# Setup signal
room.source.signal = impulse_dirac['signal']
# room.source.signal = impulse_gaussian['signal']

# Calculate reflection coefficient - will be overwritten for each method
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6


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

    if config.get('use_fast_method', False):
         print(f"--- Running {test_name} with FAST method (Analytic Reconstruction) ---")
         sdn, rir, label, is_default = calculate_sdn_rir_fast(room_parameters, test_name, room, duration, Fs, config)
    else:
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
        # Loop through ISM/rimPy methods
        for method_name, config in ism_methods.items():
            if config['enabled']:
                print(f"--- Running {method_name}: {config.get('info', '')} ---")
                # Call the respective function with its parameters
                rir, label = config['function'](
                    room_parameters,
                    duration,
                    Fs,
                    **config['params']
                )
                rirs[label] = rir

        if duration > 0.7:
            # Theoretical RT60 if room dimensions and absorption are provided
            if room_dim is not None and room_parameters['absorption'] is not None:
                rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, room_parameters['absorption'])
                print(f"\nTheoretical RT60 values of the room with a= {room_parameters['absorption']}:")
                print(f"Sabine: {rt60_sabine:.3f} s")
                print(f"Eyring: {rt60_eyring:.3f} s")

        # Calculate SDN-Base RIR
        if PLOT_SDN_BASE:
            from archive.sdn_base import calculate_sdn_base_rir

            sdn_base_rir = calculate_sdn_base_rir(room_parameters, duration, Fs)
            label = 'SDN-Base (original)'
            rirs[label] = sdn_base_rir

        # Run SDN tests
        for test_name, config in sdn_tests.items():
            if config['enabled']:
                sdn = run_sdn_test(test_name, config)

        # Run HO-SDN tests
        for test_name, config in ho_sdn_tests.items():
            if config['enabled']:
                rir, label = run_ho_sdn_test(test_name, config)

        # Normalize all RIRs

        rirs = rir_normalisation(rirs, room, Fs, normalize_to_first_impulse)

    else:  # rirs are loaded from pickle:
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
                sdn_rir = rirs.get('SDN-Original:  ')
                if sdn_rir is not None:
                     sdn_nonzero = np.sum(np.abs(sdn_rir[:sample_idx_3rd]) > threshold)
                     print(f"\nSDN nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {sdn_nonzero}")

                # For ISM rimPy negative

                ism_rir = rirs.get('ISM-rimpy-negREF')
                if ism_rir is not None:
                    ism_nonzero = np.sum(np.abs(ism_rir[:sample_idx_3rd]) > threshold)
                    print(f"ISM-rimpy-neg nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {ism_nonzero}")

                # Print valid ISM paths count
                path_tracker.print_valid_paths_count(4)

    if interactive_rirs:
        reversed_rirs = dict(reversed(list(rirs.items())))
        if UNIFIED_PLOTS:
            pp.create_unified_interactive_plot(reversed_rirs, Fs, room_parameters,
                                               reflection_times=reflection_times)
            plt.show(block=False)
        else:
            import importlib
            from analysis import plot_room as pp
            importlib.reload(pp)
            pp.create_interactive_rir_plot(rirs, Fs)
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
            pp.create_interactive_edc_plot(rirs, Fs)

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
        # Compute all RIR metrics in batch (moved to analysis.py)
        rirs_analysis = an.compute_rir_metrics_batch(rirs, Fs)
        
        # Print individual RIR metrics
        an.print_rir_metrics(rirs_analysis)
        
        # Store comparison results for both energy and smoothed signals
        comparison_results = {
            'early_energy': [],
            'smoothed_energy': []
        }
        
        # Compare energy signals between all pairs
        energy_comparisons = an.compare_rir_pairs(rirs_analysis, method_pairs, comparison_type='early_energy')
        comparison_results['early_energy'] = energy_comparisons
        an.print_comparison_results(energy_comparisons, "Energy (squared RIRs) Signal Comparison Results")
        
        # Compare smoothed signals between all pairs
        smoothed_comparisons = an.compare_rir_pairs(rirs_analysis, method_pairs, comparison_type='smoothed_energy')
        comparison_results['smoothed_energy'] = smoothed_comparisons
        an.print_comparison_results(smoothed_comparisons, "Smoothed (hann windowed squared RIRs) Signal Comparison Results")
        
        print("\n")
        
        # Compare EDCs between all pairs
        edc_comparisons = an.compare_edc_pairs(rirs, get_method_pairs(), Fs)
        an.print_edc_comparisons(edc_comparisons)

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
    # Pulse analysis moved to analysis.py for better organization
    pulse_results = an.analyze_rir_pulses(rirs, Fs, print_results=True)

# import pickle
# # # Save the RIRs to a file
#
# try:
#     with open(file_name, 'rb') as f:
#         existing_rirs = pickle.load(f)
#     # If the file exists, write a message
#     print("File already exists. Not overwriting. Please delete the file to save new data.")
#
# # dump the file otherwise
# except FileNotFoundError:
#     # If the file doesn't exist, write the new data
#     print("File not found. Saving new data.")
#     # Save the RIRs to the file
#     with open(file_name, 'wb') as f:
#         # Save the RIRs to the file
#         pickle.dump(rirs, f)


if SAVE_audio:
    # Save the RIRs as wav
    import soundfile as sf

    for rir_label, rir_audio in rirs.items():
        sf.write(f"{rir_label}.wav", rir_audio * 0.8, Fs)
