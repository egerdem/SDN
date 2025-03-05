import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork
from ISM_manual import calculate_ism_rir
import pyroomacoustics as pra
import path_tracker
import random
from sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
import frequency as ff
import EchoDensity as ned  # Import the EchoDensity module
import analysis as an
import dsp

""" Method flags """
PLOT_SDN_BASE = False
PLOT_SDN_Test1 = True
PLOT_SDN_Test2 = True
PLOT_SDN_Test3 = True
PLOT_SDN_Test4 = False
PLOT_SDN_Test5 = False
PLOT_SDN_Test6 = False

PLOT_ISM = False # manual ISM
PLOT_ISM_with_pra = False
PLOT_ISM_TEST = False
pra_order = 12

""" Visualization flags """
PLOT_ROOM = False        # 3D Room Visualisation
PLOT_ISM_PATHS = False   # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False    # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)

""" Analysis flags """
PLOT_EDC = False
PLOT_FREQ = False        # Frequency response plot
PLOT_NED = False         # Plot Normalized Echo Density
PLOT_lsd = True         # Plot LSD

Print_RIR_comparison_metrics = True
interactive_rirs = True
pulse_analysis = False
plot_smoothed_rirs = False

""" Source Signals """

# source function moved to geometry.py as Source.generate_signal()

duration = 0.3  # seconds
Fs = 44100
num_samples = int(Fs * duration)
rirs = {}
default_rirs = set()  # Track which RIRs should be black
rirs_analysis = {}
# Assign signals to methods
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)
# ISM_signal = geometry.Source.generate_signal('dirac', num_samples) # only for manual ISM

""" Room Setup """

# ho-waspaa paper room
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

room_parameters = room_aes

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal = "will be replaced", Fs = Fs)

room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

# Calculate reflection coefficient - will be overwritten for each method


room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])

room.wallAttenuation = [room_parameters['reflection']] * 6

# geometry.rmse(room)

# Note: If ignore_wall_absorption is True in DelayNetwork initialization,
# the wall attenuation will be automatically overridden to 1.0 in the DelayNetwork class

if PLOT_ISM_with_pra:
    # Override absorption for ISM
    ray_tracing_flag = False
    # Setup room for ISM with PRA package
    source_loc = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
    mic_loc = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])

    # if room_parameters == room_journal:

    pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                   materials=pra.Material(room_parameters['absorption']),
                                max_order=pra_order,
                                air_absorption=False, ray_tracing=ray_tracing_flag, use_rand_ism=False)
    pra_room.set_sound_speed(343)
    pra_room.add_source(source_loc).add_microphone(mic_loc)

    pra_room.compute_rir()
    pra_rir = pra_room.rir[0][0]
    pra_rir =  pra_rir / np.max(np.abs(pra_rir))

    global_delay = pra.constants.get("frac_delay_length") // 2
    pra_rir = pra_rir[global_delay:]  # Shift left by removing the initial delay
    pra_rir = np.pad(pra_rir, (0, global_delay))  # Pad with zeros at the end to maintain length
    pra_rir = pra_rir[:num_samples]
    label = 'ISM'
    rirs[label] = pra_rir

# Calculate ISM test RIR if needed
if PLOT_ISM_TEST:
    from ISM import ISM

    # Setup parameters for ISM test
    xs = np.array([room_parameters['source x'],
                   room_parameters['source y'],
                   room_parameters['source z']])
    xr = np.array([[room_parameters['mic x'],
                    room_parameters['mic y'],
                    room_parameters['mic z']]])
    xr = np.transpose(xr)
    L = np.array([room_parameters['width'],
                  room_parameters['depth'],
                  room_parameters['height']])
    N = np.array([0, 0, 0])
    beta = room_parameters['reflection']  # Use reflection coefficient directly
    Tw = 11
    Fc = 0.9
    Rd = 0.08
    Nt = round(Fs / 2)
    c = 343

    # Calculate RIR using ISM test method
    B = ISM(xr, xs, L, beta, N, Nt, Rd, [], Tw, Fc, Fs, c)
    ism_test_rir = B[0].flatten()  # Flatten the 2D array to 1D
    ism_test_rir = ism_test_rir / np.max(np.abs(ism_test_rir))
    ism_test_rir = ism_test_rir[:num_samples]
    rirs['ISM_test'] = ism_test_rir

# Calculate manual ISM RIR if needed
if PLOT_ISM:
    ISM_signal = impulse_dirac  # or impulse_dirac (only change this line, not the subsequent)
    room.source.signal = ISM_signal['signal']

    ism_rir, fs = calculate_ism_rir(room, max_order = 6 ,duration=duration)
    ism_rir = ism_rir / np.max(np.abs(ism_rir))
    ism_manual_rir = ism_rir[:num_samples]
    rirs['ISM_manual'] = ism_manual_rir
    
# SDN Test Configurations
sdn_tests = {
    'Test1': {
        'enabled': PLOT_SDN_Test1,
        # 'absorption': 0.2,
        # 'info': "should be same with original SDN",
        'flags': {
            # 'more_absorption': True,
        },
        'label': "Original SDN"
    },
    'Test2': {
        'enabled': PLOT_SDN_Test2,
        # 'absorption': 0.2,
        'info': "path",
        'flags': {
            # "ignore_wall_absorption" : True,
            # "ignore_src_node_atten" : True,
            # "ignore_node_mic_atten" : True,
            # 'specular_source_injection': True,
            # 'source_weighting': 3,
            # "coef": 1,
            "enable_path_logging": True,
            # 'more_absorption': True,
        },
        'label': "SDN"
    },
    'Test3': {  # New test configuration
        'enabled': PLOT_SDN_Test3,
        # 'absorption': 0.3,
        'info': "c=4",
        'flags': {
        'specular_source_injection': True,
            'source_weighting': 3,
            "enable_path_logging": True,
        },
        'label': "SDN"
    },

    'Test4': {  # New test configuration
            'enabled': PLOT_SDN_Test4,
            # 'absorption': 0.3,
            'info': "c=8",
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 8,
            },
            'label': "SDN"
        },

    'Test5': {  # New test configuration
                'enabled': PLOT_SDN_Test5,
                # 'absorption': 0.3,
                'info': "c=5",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 5,
                },
                'label': "SDN Test 5"
            },
    'Test6': {  # New test configuration
                    'enabled': PLOT_SDN_Test6,
                    # 'absorption': 0.3,
                    'info': "source weigth 1",
                    'flags': {
                        'specular_source_injection': True,
                        'source_weighting': 1,
                    },
                    'label': "SDN Test 6"
                }
}

# Function to run SDN tests with given configuration, new implementation (SDN-Ege)
def run_sdn_test(test_name, config):
    if not config['enabled']:
        return
    print("Running SDN test:", test_name)
    if 'absorption' in config:
        # Override absorption
        room_parameters['absorption'] = config['absorption']
        room_parameters['reflection'] = np.sqrt(1 - config['absorption'])
        room.wallAttenuation = [room_parameters['reflection']] * 6
    
    # Setup signal
    room.source.signal = impulse_dirac['signal']
    
    # Create SDN instance with configured flags
    sdn = DelayNetwork(room, Fs=Fs, label=config['label'], **config['flags'])
    
    # Calculate RIR
    rir = sdn.calculate_rir(duration)
    rir = rir / np.max(np.abs(rir))
    
    # Check if this is a default configuration (no flags)
    is_default = len(config.get('flags', {})) == 0
    
    # Store result with optional info
    if is_default:
        label = 'SDN-Original'
    else:
        label = f'SDN-{test_name}'
        
    if 'info' in config:
        label += f': {config["info"]}'
    
    rirs[label] = rir
    
    # Track if this is a default configuration
    if is_default:
        default_rirs.add(label)
    
    return sdn

# Calculate SDN-Base RIR
if PLOT_SDN_BASE:
    from sdn_base import calculate_sdn_base_rir
    # Override absorption for SDN-Base
    sdn_base_rir = calculate_sdn_base_rir(room_parameters, duration, Fs)
    label = 'SDN-Base (original)'

    rirs[label] = sdn_base_rir

# Run SDN tests
for test_name, config in sdn_tests.items():
    sdn = run_sdn_test(test_name, config)


    # Additional analysis if needed (e.g., path logging)
    if sdn and sdn.enable_path_logging:
        print(f"\n=== {test_name} Path Analysis ===")
        print("\n=== First Arriving Paths ===")
        complete_paths = sdn.path_logger.get_complete_paths_sorted()
        for path_key, packet in complete_paths[:10]:
            print(f"{path_key}: arrives at n={packet.birth_sample + packet.delay}, value={packet.value:.6f}")

if duration > 0:

    # Calculate RT60 values for all RIRs
    rt60_values = {}
    for label, rir in rirs.items():
        rt60_values[label] = pp.calculate_rt60_from_rir(rir, Fs, plot=False)

    print("\nReverberation Time Analysis:")
    print("-" * 50)
    print("\nMeasured RT60 values:")
    for label, rt60 in rt60_values.items():
        print(f"{label}: {rt60:.3f} s")

    # Theoretical RT60 if room dimensions and absorption are provided
    if room_dim is not None and room_parameters['absorption'] is not None:
        rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, 0.2)
        print(f"\nTheoretical RT60 values of the room with a=0.2:")
        print(f"Sabine: {rt60_sabine:.3f} s")
        print(f"Eyring: {rt60_eyring:.3f} s")

if PLOT_EDC:

    pp.create_interactive_edc_plot(rirs, Fs, default_rirs)

    # commenting out the below "static" plot since I have an interactive plot now
    """ 
    # Plot energy decay curves (matching SDN_timu implementation)
    plt.figure(figsize=(12, 6))
    for label, rir in rirs.items():
        if duration > 1.5:
            plot_label = f"{label} (RT60: {rt60_values[label]:.3f}s)"
        else:
            plot_label = label

        color = 'black' if label in default_rirs or label == "SDN-Base (original)" else None

        # Use black for default RIRs, default color cycle otherwise
        # if label in default_rirs:
        #     an.compute_edc(rir, Fs, label=plot_label, color='black')
        # else:
        an.compute_edc(rir, Fs, label=plot_label, color=color)
        

    plt.title('Energy Decay Curves')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.legend()
    plt.grid(True)
    plt.ylim(-65, 5)
    plt.show(block=False)
    """

    # Compare all distinct pairs of EDCs
    print("\nEDC Comparison (First 50ms RMSE Differences):")
    print("-" * 50)
    
    # Get all unique pairs of RIR labels
    rir_labels = list(rirs.keys())
    i = 0
    for j in range(i + 1, len(rir_labels)):
        label1 = rir_labels[i]
        label2 = rir_labels[j]

        # Calculate EDCs for both RIRs without plotting
        edc1 = an.compute_edc(rirs[label1], Fs, label1, plot=False)
        edc2 = an.compute_edc(rirs[label2], Fs, label2, plot=False)

        # Compare EDCs
        rms_diff = an.compute_RMS(edc1, edc2, range=50, Fs=Fs, method="mae") # range=50ms
        # sum_of_raw_diff = an.compute_RMS(edc1, edc2, range=50, Fs=Fs, method="sum_of_raw_diff")
        print(f"{label1} vs {label2}: {rms_diff:.2f} dB RMSE difference")
        # print(f"{label1} vs {label2}: {sum_of_raw_diff:.2f} dB sum of raw differences")

if PLOT_NED:
    plt.figure(figsize=(12, 6))

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

if interactive_rirs:
    pp.create_interactive_rir_plot(rirs)
    plt.show(block=False)


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
            
            # Combine configurations
            combined_info = ""
            if info1 and info2:
                combined_info = f"{info1} vs {info2}"
            elif info1:
                combined_info = info1
            elif info2:
                combined_info = info2
                
            pairs.append({
                'pair': f"{base1} vs {base2}",
                'label1': label1,
                'label2': label2,
                'info': combined_info
            })
    
    return pairs

# Process method pairs once
method_pairs = get_method_pairs()

if Print_RIR_comparison_metrics:
    # Dictionary to store all RIR analyses
    for rir_label, rir in rirs.items():
        en, sm = an.calculate_smoothed_energy(rir, window_length=30, range=50, Fs=Fs)
        rirs_analysis[rir_label] = {
            "energy": en,
            "smoothed_energy": sm
        }

    # Dictionary to store comparison results for both energy and smoothed signals
    comparison_results = {
        'energy': [],
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
        energy1 = rirs_analysis[l1]['energy']
        energy2 = rirs_analysis[l2]['energy']
        
        # Store energy comparison results
        energy_result = {
            'pair': pair_info['pair'],
            'info': pair_info['info'],
            'rmse': an.compute_RMS(energy1, energy2, method="rmse"),
            'mae': an.compute_RMS(energy1, energy2, method="mae"),
            'median': an.compute_RMS(energy1, energy2, method="median")
        }
        comparison_results['energy'].append(energy_result)
        
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


"""

# dsp.calculate_edf(rirs['SDN-Test2'], sr=Fs, plot=True)
# dsp.calculate_rt(rirs['SDN-Base (original)'], sr=Fs, plot=False)
# dsp.calculate_drr(rirs['SDN-Test2'])


rir = rirs['SDN-Base (original)']

# 1) Check global max amplitude (absolute value)
global_max = np.max(np.abs(rir))
print("Global max amplitude:", global_max)

# 2) Check if there's any portion that is exactly zero
num_zeros = np.sum(rir == 0.0)
print(f"There are {num_zeros} samples that are exactly zero.")

# 3) Check total energy
total_energy = np.sum(rir**2)
print("Total energy:", total_energy)


# Filter into octave bands (if that's what your code is doing)
filtered = dsp.filter_octaveband(rir, sr=16000)

# Now, each column is an octave-band filtered signal
# Check max amplitude in each band
for band_idx in range(filtered.shape[1]):
    band_max = np.max(np.abs(filtered[:, band_idx]))
    print(f"Band {band_idx} max amplitude: {band_max}")

edc_db = dsp.calculate_edf(rir, sr=Fs)
print("EDC range:", np.min(edc_db), np.max(edc_db))
"""

"""
# Only Path Length Analysis, No RIR Calculation 
# Create shared path tracker and calculate paths
if ISM_SDN_PATH_DIFF_TABLE:
    path_tracker = path_tracker.PathTracker()
    sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
    ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
    sdn_calc.set_path_tracker(path_tracker)
    ism_calc.set_path_tracker(path_tracker)

    # Compare paths and analyze invalid ISM paths
    PathCalculator.compare_paths(sdn_calc, ism_calc,
                                 max_order=3)  # compare_paths() prints the comparison table but doesn't return anything

    # analyze_paths() returns a list of invalid paths (only for ISM calculator)
    # Each path is a list of node labels ['s', 'wall1', 'wall2', ..., 'm']
    invalid_paths = ism_calc.analyze_paths(max_order=3)

    # Visualize example ISM paths
    example_paths = [
        ['s', 'east', 'west', 'm'],
        ['s', 'west', 'm'],
        ['s', 'west', 'east', 'north', 'm']
    ]

    for path in example_paths:
        pp.plot_ism_path(room, ism_calc, path)
        plt.show()

if PLOT_ROOM:
    pp.plot_room(room)
"""

if pulse_analysis:
    print("\n=== RIR Pulse Analysis ===")
    for rir_label, rir in rirs.items():
        # Count total nonzero pulses (using small threshold to account for floating point)
        threshold = 1e-10  # Adjust this threshold based on your needs
        nonzero_count = np.sum(np.abs(rir) > threshold)

        # Calculate percentage of nonzero samples
        percentage = (nonzero_count / len(rir)) * 100

        # Find first and last nonzero indices
        nonzero_indices = np.where(np.abs(rir) > threshold)[0]
        first_pulse = nonzero_indices[0] if len(nonzero_indices) > 0 else 0
        last_pulse = nonzero_indices[-1] if len(nonzero_indices) > 0 else 0

        # Calculate time span of pulses
        time_span_ms = (last_pulse - first_pulse) / Fs * 1000  # in milliseconds

        print(f"\n{rir_label}:")
        print(f"  Total pulses: {nonzero_count}")
        print(f"  Percentage of nonzero samples: {percentage:.2f}%")
        print(f"  First pulse at: {first_pulse/Fs*1000:.2f} ms")
        print(f"  Last pulse at: {last_pulse/Fs*1000:.2f} ms")
        print(f"  Time span: {time_span_ms:.2f} ms")


# import numpy as np
# import matplotlib.pyplot as plt

# # Example RIR data (replace this with your actual RIR data)
# non_zero_rir = rir[rir != 0]
# # Compute the histogram
# hist, bin_edges = np.histogram(non_zero_rir, bins=30)
# # Print the histogram counts and bin edges
# print("Histogram counts:", hist)
# print("Bin edges:", bin_edges)
# plt.figure()
# # Plotting the histogram
# plt.hist(non_zero_rir, bins=30, color='blue', alpha=0.7)
# plt.title('Histogram of RIR')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid()
# plt.show()