import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork
from ISM_manual import calculate_ism_rir
import pyroomacoustics as pra
import path_tracker
import random
from sdn_path_calculator import SDNCalculator, ISMCalculator
import frequency as ff
import EchoDensity as ned  # Import the EchoDensity module

""" Method flags """
# Reference SDN implementation
PLOT_SDN_REF = True
# Ege's SDN implementation
PLOT_SDN_EGE = True
# Manual ISM implementation
PLOT_ISM = False
# ISM implementation with PRA package
PLOT_ISM_with_pra = False
pra_order = 10 # Maximum reflection order for PRA

""" Visualization flags """
PLOT_ROOM = False        # 3D Room Visualisation
PLOT_ISM_PATHS = False   # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False    # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)

""" Analysis flags """
PLOT_EDC = False
PLOT_FREQ = False        # Frequency response plot
PLOT_NED = False         # Plot Normalized Echo Density
interactive_rirs = False

""" Source Signals """

def source(label):
    # be careful, num_samples and Fs is hardcoded, global variables
    if label == 'dirac':
        # Generate Dirac Impulse
        signal = np.array([1.0] + [0.0] * (num_samples - 1))
    elif label == 'gaussian':
        # Generate Gaussian pulse
        signal = ff.gaussian_impulse(Fs, num_gaussian_samples = 10, std_dev= 5 ,plot=False)
    else:
        raise ValueError('Invalid source label')

    return {'signal': signal,
            'label': label}

duration = 0.2  # seconds
Fs = 44100
num_samples = int(Fs * duration)
rirs = {}

# Assign signals to methods
SDN_REF_signal = source('dirac')
SDN_EGE_signal = source('dirac')
# ISM_signal = source('dirac') # only for manual ISM

""" Room Setup """
room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal = "will be replaced", Fs = Fs)

# Calculate reflection coefficient
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

""" Only Path Length Analysis, No RIR Calculation """
# Create shared path tracker and calculate paths
if ISM_SDN_PATH_DIFF_TABLE:
    path_tracker = path_tracker.PathTracker()
    sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
    ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
    sdn_calc.set_path_tracker(path_tracker)
    ism_calc.set_path_tracker(path_tracker)

    # Calculate paths up to order 3
    sdn_calc.calculate_paths_up_to_order(3)
    ism_calc.calculate_paths_up_to_order(3)

    # Print path comparison
    print("\n=== Path Comparison between ISM and SDN ===")
    path_tracker.print_path_comparison()

    # Get and print invalid ISM paths
    print("\n=== Invalid ISM Paths ===")
    invalid_paths = []
    for order in range(4):  # Up to order 3
        paths = path_tracker.get_paths_by_order(order, 'ISM')
        invalid_paths.extend([p.nodes for p in paths if not p.is_valid])

    if invalid_paths:
        print(f"Found {len(invalid_paths)} invalid paths:")
        for path in invalid_paths[:10]:  # Print first 10 invalid paths
            print(f"  {' -> '.join(path)}")
        if len(invalid_paths) > 10:
            print(f"  ... and {len(invalid_paths) - 10} more")
    else:
        print("No invalid paths found")

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

# Calculate Reference SDN RIR
if PLOT_SDN_REF:
    room.source.signal = SDN_REF_signal['signal']
    reference_sdn = DelayNetwork(room, Fs=Fs,
                                use_identity_scattering=False,
                                ignore_wall_absorption=False,
                                ignore_src_node_atten=False,
                                ignore_node_mic_atten=False,
                                enable_path_logging=False,
                                plot_wall_incoming_sums=False, label="Reference SDN")
    reference_rir = reference_sdn.calculate_rir(duration)
    first_nonzero_index = np.nonzero(reference_rir)[0][0]
    reference_rir = reference_rir / np.max(np.abs(reference_rir))
    rirs['SDN-Ref'] = reference_rir

# Calculate Ege's SDN RIR
if PLOT_SDN_EGE:
    room.source.signal = SDN_EGE_signal['signal']

    sdn = DelayNetwork(room, Fs=Fs,
                      use_identity_scattering=False,
                      ignore_wall_absorption=False,
                      ignore_src_node_atten=False,
                      ignore_node_mic_atten=False,
                      enable_path_logging=False,
                    plot_wall_incoming_sums=False, label="SDN-Ege")
    sdn_rir = sdn.calculate_rir(duration)
    sdn_rir = sdn_rir / np.max(np.abs(sdn_rir))
    rirs['SDN-Ege'] = sdn_rir

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
    room.source.signal = ISM_signal['ISM_signal']
    ism_rir, fs = calculate_ism_rir(room, duration=duration)
    ism_rir = ism_rir / np.max(np.abs(ism_rir))
    rirs['ISM_manual'] = ism_rir

# Calculate ISM PRA RIR if needed
if PLOT_ISM_with_pra:
    ray_tracing_flag = False
    # Setup room for ISM with PRA package

    room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])
    source_loc = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
    mic_loc = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])


    pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                   materials=pra.Material(room_parameters['absorption']),
                                   max_order=pra_order, air_absorption=False, ray_tracing=ray_tracing_flag, use_rand_ism=False)
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

# Create list of enabled flags for SDN
enabled_flags = []
if PLOT_SDN_EGE and sdn.use_identity_scattering:
    enabled_flags.append("Identity Scattering")
if PLOT_SDN_EGE and sdn.ignore_wall_absorption:
    enabled_flags.append("No Wall Absorption")
if PLOT_SDN_EGE and sdn.ignore_src_node_atten:
    enabled_flags.append("No Src-Node Atten")
if PLOT_SDN_EGE and sdn.ignore_node_mic_atten:
    enabled_flags.append("No Node-Mic Atten")
if PLOT_SDN_EGE:
    enabled_flags.append(f"SDN-Ege: {SDN_EGE_signal['label']} impulse")
if PLOT_SDN_REF:
    enabled_flags.append(f"SDN-REF: {SDN_REF_signal['label']} impulse")
if PLOT_ISM_with_pra:
    enabled_flags.append(f"ISM Ray Tracing: {ray_tracing_flag}")
    enabled_flags.append(f"ISM max order: {pra_order}")

plt.figure(figsize=(12, 6))
for rir_label, rir in rirs.items():
    plt.plot(rir, label=rir_label, alpha=0.7)
plt.title('Room Impulse Response Comparison')
plt.xlabel('Time (sample)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Add flags to top-right corner
if enabled_flags:
    flag_text = '\n'.join(enabled_flags)
    plt.text(0.98, 0.98, flag_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.show()

# Plot Normalized Echo Density if enabled
if PLOT_NED:
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Normalized Echo Density
    if PLOT_SDN_REF:
        echo_density = ned.echoDensityProfile(reference_rir, fs=Fs)
        ax1.plot(echo_density, label='SDN-Ref', alpha=1)
    if PLOT_SDN_EGE:
        echo_density = ned.echoDensityProfile(sdn_rir, fs=Fs)
        ax1.plot(echo_density, label='SDN-Test', alpha=0.7)
    if PLOT_ISM:
        echo_density = ned.echoDensityProfile(ism_rir, fs=Fs)
        ax1.plot(echo_density, label='ISM', alpha=0.7)
    if PLOT_ISM_with_pra:
        echo_density = ned.echoDensityProfile(pra_rir, fs=Fs)
        ax1.plot(echo_density, label='ISM with PRA', alpha=0.7)

    ax1.set_title('Normalized Echo Density')
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Normalized Echo Density')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_xlim(left=100)

    # Plot Raw (Non-normalized) Echo Density
    if PLOT_SDN_REF:
        echo_density = ned.echoDensityProfileRaw(reference_rir, fs=Fs)
        ax2.plot(echo_density, label='SDN-ref', alpha=1)
    if PLOT_SDN_EGE:
        echo_density = ned.echoDensityProfileRaw(sdn_rir, fs=Fs)
        ax2.plot(echo_density, label='SDN-Ege', alpha=0.7)
    if PLOT_ISM:
        echo_density = ned.echoDensityProfileRaw(ism_rir, fs=Fs)
        ax2.plot(echo_density, label='ISM', alpha=0.7)
    if PLOT_ISM_with_pra:
        echo_density = ned.echoDensityProfileRaw(pra_rir, fs=Fs)
        ax2.plot(echo_density, label='ISM with PRA', alpha=0.7)

    ax2.set_title('Raw Echo Density (Non-normalized)')
    ax2.set_xlabel('Time (samples)')
    ax2.set_ylabel('Raw Echo Density Count')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_xlim(left=100)

    # Add flags to top-right corner
    if enabled_flags:
        flag_text = '\n'.join(enabled_flags)
        # Add text to the first subplot (ax1)
        ax1.text(0.05, 0.5, flag_text,
                 transform=ax1.transAxes,
                 verticalalignment='center',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        # Add text to the second subplot (ax2)
        ax2.text(0.05, 0.5, flag_text,
                 transform=ax2.transAxes,
                 verticalalignment='center',
                 horizontalalignment='left',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()  # Adjust subplot parameters for better layout
    plt.show()

if PLOT_FREQ:

    if PLOT_SDN_REF:
        freq, magnitude = ff.calculate_frequency_response(reference_rir, Fs)
        ff.plot_frequency_response(freq, magnitude, label='FRF TEST SDN')

    if PLOT_SDN_EGE:
        freq, magnitude = ff.calculate_frequency_response(sdn_rir, Fs)
        ff.plot_frequency_response(freq, magnitude, label='FRF Reference SDN')

    if PLOT_ISM:
        freq, magnitude = ff.calculate_frequency_response(ism_rir, Fs)
        ff.plot_frequency_response(freq, magnitude, label='FRF ISM')

if PLOT_EDC:
    # Plot energy decay curves
    plt.figure(figsize=(12, 6))
    #edcc
    for rir, label in [(sdn_rir, 'SDN-ege'), (pra_rir, 'ISM-pra'), (reference_rir, 'SDN-ref')]:
        pEnergy = (np.cumsum(rir[::-1] ** 2) / np.sum(rir[::-1]))[::-1]
        pEdB = 10.0 * np.log10(pEnergy / np.max(pEnergy))
        plt.plot(np.arange(len(pEdB)) / Fs, pEdB, label=label, alpha=0.7)

    plt.title('Energy Decay Curves')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim(-60, 5)
    plt.show()

# Create interactive plot
if interactive_rirs:
    pp.create_interactive_rir_plot(enabled_flags, rirs)



    