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
PLOT_SDN_BASE = True
PLOT_SDN_Test1 = True # Reference SDN implementation
PLOT_SDN_Test2 = True # Ege's SDN implementation
PLOT_ISM = False # Manual ISM implementation
PLOT_ISM_with_pra = True # ISM implementation with PRA package
PLOT_ISM_TEST = False # New ISM test implementation
pra_order = 12 # Maximum reflection order for PRA

""" Visualization flags """
PLOT_ROOM = False        # 3D Room Visualisation
PLOT_ISM_PATHS = False   # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False    # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)

""" Analysis flags """
PLOT_EDC = True
PLOT_FREQ = False        # Frequency response plot
PLOT_NED = True         # Plot Normalized Echo Density
interactive_rirs = True
pulse_analysis = True

""" Source Signals """

# source function moved to geometry.py as Source.generate_signal()

duration = 2  # seconds
Fs = 44100
num_samples = int(Fs * duration)
rirs = {}

# Assign signals to methods
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)
# ISM_signal = geometry.Source.generate_signal('dirac', num_samples) # only for manual ISM

""" Room Setup """
# room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   # 'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   # 'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   # 'absorption': 0.2,
                   # }

room_parameters = {'width': 6, 'depth': 4, 'height': 7,
                   'source x': 3.6, 'source y': 1.3, 'source z': 5.3,
                   'mic x': 1.2, 'mic y': 2.4, 'mic z': 1.8,
                   'absorption': 0.1,
                   }

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal = "will be replaced", Fs = Fs)

room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

# Calculate reflection coefficient
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# geometry.test_specular_matrices(room)

# Note: If ignore_wall_absorption is True in DelayNetwork initialization,
# the wall attenuation will be automatically overridden to 1.0 in the DelayNetwork class

""" Only Path Length Analysis, No RIR Calculation """
# Create shared path tracker and calculate paths
if ISM_SDN_PATH_DIFF_TABLE:
    path_tracker = path_tracker.PathTracker()
    sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
    ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
    sdn_calc.set_path_tracker(path_tracker)
    ism_calc.set_path_tracker(path_tracker)

    # Compare paths and analyze invalid ISM paths
    PathCalculator.compare_paths(sdn_calc, ism_calc, max_order=3) # compare_paths() prints the comparison table but doesn't return anything
    
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

# Calculate SDN-Base RIR
if PLOT_SDN_BASE:
    from sdn_base import calculate_sdn_base_rir
    sdn_base_rir = calculate_sdn_base_rir(room_parameters, duration, Fs)
    rirs['SDN-Base (original)'] = sdn_base_rir

# Calculate Reference SDN RIR
if PLOT_SDN_Test1:
    SDN_Test1_signal = impulse_dirac # or impulse_dirac (only change this line, not the subsequent)
    room.source.signal = SDN_Test1_signal['signal']

    reference_sdn = DelayNetwork(room, Fs=Fs,
                                 specular_source_injection=True,
                                 source_weighting=3,
                                 label="SDN Test 1")
    reference_rir = reference_sdn.calculate_rir(duration)
    reference_rir = reference_rir / np.max(np.abs(reference_rir))
    rirs['SDN-Test1'] = reference_rir

# Calculate Ege's SDN RIR
if PLOT_SDN_Test2:
    SDN_Test2_signal = impulse_dirac  # or impulse_dirac (only change this line, not the subsequent)
    room.source.signal = SDN_Test2_signal['signal']

    sdn = DelayNetwork(room, Fs=Fs,
                    coef = 1,
                    source_pressure_injection_coeff= 0.2,
                    specular_source_injection = True,
                    source_weighting = 6,
                    label="SDN Test 2")
    sdn_rir = sdn.calculate_rir(duration)
    sdn_rir = sdn_rir / np.max(np.abs(sdn_rir))
    rirs['SDN-Test2'] = sdn_rir

    # sdn.plot_wall_incoming_sums()

    # Analyze paths if path logging is enabled
    if sdn.enable_path_logging:
        print("\n=== Path Analysis ===")
        print("\n=== First Arriving Paths ===")
        complete_paths = sdn.path_logger.get_complete_paths_sorted()
        for path_key, packet in complete_paths[:10]:  # Show first 10 paths
            print(f"{path_key}: arrives at n={packet.birth_sample + packet.delay}, value={packet.value:.6f}")

# Calculate manual ISM RIR if needed
if PLOT_ISM:
    ISM_signal = impulse_dirac  # or impulse_dirac (only change this line, not the subsequent)
    room.source.signal = ISM_signal['signal']

    ism_rir, fs = calculate_ism_rir(room, max_order= pra_order ,duration=duration)
    ism_rir = ism_rir / np.max(np.abs(ism_rir))
    rirs['ISM_manual'] = ism_rir

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
    Nt = round(Fs/2)
    c = 343
    
    # Calculate RIR using ISM test method
    B = ISM(xr, xs, L, beta, N, Nt, Rd, [], Tw, Fc, Fs, c)
    ism_test_rir = B[0].flatten()  # Flatten the 2D array to 1D
    ism_test_rir = ism_test_rir / np.max(np.abs(ism_test_rir))
    rirs['ISM_test'] = ism_test_rir

# Calculate ISM PRA RIR if needed
if PLOT_ISM_with_pra:
    ray_tracing_flag = False
    # Setup room for ISM with PRA package

    source_loc = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
    mic_loc = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])

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
    rirs['ISM'] = pra_rir

# Analyze pulse counts in RIRs
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

# Create list of enabled flags for SDN
enabled_flags = []
# if PLOT_SDN_Test2 and sdn.use_identity_scattering:
#     enabled_flags.append("Identity Scattering")
# if PLOT_SDN_Test2 and sdn.ignore_wall_absorption:
#     enabled_flags.append("No Wall Absorption")
# if PLOT_SDN_Test2 and sdn.ignore_src_node_atten:
#     enabled_flags.append("No Src-Node Atten")
# if PLOT_SDN_Test2 and sdn.ignore_node_mic_atten:
#     enabled_flags.append("No Node-Mic Atten")
if PLOT_SDN_Test1:
    enabled_flags.append(f"SDN-Test1: {SDN_Test1_signal['label']} impulse")
if PLOT_SDN_Test2:
    enabled_flags.append(f"SDN-Test2: {SDN_Test2_signal['label']} impulse")
if PLOT_ISM_with_pra:
    enabled_flags.append(f"ISM Ray Tracing: {ray_tracing_flag}")
    enabled_flags.append(f"ISM max order: {pra_order}")

# Plot Normalized Echo Density if enabled
if PLOT_NED:
    plt.figure(figsize=(12, 6))

    # Plot both normalized and raw echo densities
    for rir_label, rir in rirs.items():
        # Normalized Echo Density
        echo_density = ned.echoDensityProfile(rir, fs=Fs)
        # Create time array in seconds
        time_axis = np.arange(len(echo_density)) / Fs
        plt.plot(time_axis, echo_density, label=rir_label, alpha=0.7)

    # Configure plots
    plt.title('Normalized Echo Density')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Echo Density')
    plt.grid(True)
    plt.legend()
    # plt.xscale('log')
    # plt.xlim(left=0)  # Start from 0 seconds

    # Add flags to top-right corner
    if enabled_flags:
        flag_text = '\n'.join(enabled_flags)
        plt.text(0.05, 0.5, flag_text,
                transform=plt.gca().transAxes,
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()  # Adjust subplot parameters for better layout
    plt.show(block=False)  # Non-blocking

if PLOT_FREQ:
    # Calculate and plot frequency responses for all RIRs
    for rir_label, rir in rirs.items():
        freq, magnitude = ff.calculate_frequency_response(rir, Fs)
        ff.plot_frequency_response(freq, magnitude, label=f'FRF {rir_label}')

if PLOT_EDC:
    # Plot energy decay curves (matching SDN_timu implementation)
    plt.figure(figsize=(12, 6))
    for label, rir in rirs.items():
        an.compute_edc(rir, Fs, label=label)

    plt.title('Energy Decay Curves')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim(-60, 5)
    plt.show(block=False)  # Non-blocking


# Create interactive plot
if interactive_rirs:
    pp.create_interactive_rir_plot(enabled_flags, rirs)
    plt.show(block=False)  # Non-blocking

PLOT_RT = False
print("\nReverberation Time Analysis:")
print("-" * 50)

# Theoretical RT60 if room dimensions and absorption are provided
if room_dim is not None and room_parameters['absorption'] is not None:
    rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, room_parameters['absorption'])
    print(f"\nTheoretical RT60 values of the room:")
    print(f"Sabine: {rt60_sabine:.3f} s")
    print(f"Eyring: {rt60_eyring:.3f} s")

# Calculate RT60 from RIRs
print("\nMeasured RT60 values:")
for rir_label, rir in rirs.items():
    rt60 = pp.calculate_rt60_from_rir(rir, Fs, plot=PLOT_RT)
    print(f"{rir_label}: {rt60:.3f} s")


"""

# dsp.calculate_edf(rirs['SDN-Test2'], sr=Fs, plot=True)
dsp.calculate_rt(rirs['SDN-Base (original)'], sr=Fs, plot=False)
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