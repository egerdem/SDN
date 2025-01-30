import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork
from ISM_manual import calculate_ism_rir
import path_tracker
import random
from sdn_path_calculator import SDNCalculator, ISMCalculator
import frequency as ff

# Plotting flags
PLOT_SDN_REF = True      # Plot reference SDN implementation
PLOT_SDN_EGE = True      # Plot Ege's SDN implementation
PLOT_ISM = True         # Plot ISM implementation
PLOT_ISM_PATHS = False   # Visualize example ISM paths
PLOT_ROOM = False        # 3D Room Visualisation
PLOT_EDC = False
PLOT_FREQ = True        # Frequency response plot
# Path analysis flag
ISM_SDN_PATH_DIFF_TABLE = False    # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)

#source signal
duration = 0.2  # seconds
Fs = 44100
num_samples = int(Fs * duration)

# Generate Gaussian pulse
source_signal_gauss = ff.gaussian_impulse(Fs, num_gaussian_samples = 10, std_dev= 5 ,plot=False)

# Generate Dirac Impulse
source_signal_impulse = np.zeros(num_samples)
source_signal_impulse[0] = 1.0


# Define the Room
room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }

# Setup room
room = geometry.Room(room_parameters['width'],
                     room_parameters['depth'],
                     room_parameters['height'])

room.set_microphone(room_parameters['mic x'],
                    room_parameters['mic y'],
                    room_parameters['mic z'])

room.set_source(room_parameters['source x'],
                room_parameters['source y'],
                room_parameters['source z'],
                signal = "")

# Calculate reflection coefficient
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

if PLOT_ROOM:
    pp.plot_room(room)

# Calculate Reference SDN RIR if needed
if PLOT_SDN_REF:
    room.source.signal = source_signal_impulse
    reference_sdn = DelayNetwork(room,
                                use_identity_scattering=False,
                                ignore_wall_absorption=False,
                                ignore_src_node_atten=False,
                                ignore_node_mic_atten=False,
                                enable_path_logging=False)
    reference_rir = reference_sdn.calculate_rir(duration)
    reference_rir = reference_rir / np.max(np.abs(reference_rir))

# Calculate Ege's SDN RIR if needed
if PLOT_SDN_EGE:
    # room.source.signal = source_signal_gauss
    room.source.signal = source_signal_impulse

    sdn = DelayNetwork(room,
                      use_identity_scattering=False,
                      ignore_wall_absorption=False,
                      ignore_src_node_atten=False,
                      ignore_node_mic_atten=False,
                      enable_path_logging=True)
    sdn_rir = sdn.calculate_rir(duration)
    sdn_rir = sdn_rir / np.max(np.abs(sdn_rir))
    sdn.plot_wall_incoming_sums()

    # Analyze paths if path logging is enabled
    if sdn.enable_path_logging:
        print("\n=== Path Analysis ===")
        print("\n=== First Arriving Paths ===")
        complete_paths = sdn.path_logger.get_complete_paths_sorted()
        for path_key, packet in complete_paths[:10]:  # Show first 10 paths
            print(f"{path_key}: arrives at n={packet.birth_sample + packet.delay}, value={packet.value:.6f}")

# Calculate ISM RIR if needed
if PLOT_ISM:
    ism_rir, fs = calculate_ism_rir(room_parameters, duration=duration)
    ism_rir = ism_rir / np.max(np.abs(ism_rir))

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

# Plot RIRs if any implementation is enabled
if any([PLOT_SDN_REF, PLOT_SDN_EGE, PLOT_ISM]):
    plt.figure(figsize=(12, 6))
    
    if PLOT_SDN_REF:
        plt.plot(reference_rir, label='Reference SDN', alpha=1)
    if PLOT_SDN_EGE:
        plt.plot(sdn_rir, label='Ege-SDN', alpha=0.7)
    if PLOT_ISM:
        plt.plot(ism_rir, label='ISM', alpha=0.7)

    plt.title('Room Impulse Response Comparison')
    plt.xlabel('Time (s)')
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
    for rir, label in [(sdn_rir, 'SDN'), (ism_rir, 'ISM')]:
        pEnergy = (np.cumsum(rir[::-1] ** 2) / np.sum(rir[::-1]))[::-1]
        pEdB = 10.0 * np.log10(pEnergy / np.max(pEnergy))
        plt.plot(np.arange(len(pEdB)) / fs, pEdB, label=label, alpha=0.7)

    plt.title('Energy Decay Curves')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.grid(True)
    plt.legend()
    plt.ylim(-60, 5)
    plt.show()

# Create shared path tracker and calculate paths if needed
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

    