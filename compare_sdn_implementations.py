import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork
import sys
import os
import time
from sdn_path_calculator import ISMCalculator
import path_tracker
from ISM_core import ISMNetwork
import plot_room as pp
import frequency as ff
import EchoDensity as ned  # Import EchoDensity module
import pyroomacoustics as pra  # Import pyroomacoustics
import analysis as an

# Add SDN_algo3 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SDN_algo3'))
import Signal as sig
import Source as src
import Microphone as mic
import Simulation as sim
import Geometry as geom
from SDN_timu import Room as TimuRoom, Source as TimuSource, Microphone as TimuMicrophone, SoundFileRW

def plot_rir_comparison(rirs, labels=None, PLOT_EDC=False, PLOT_NED=True, Fs=44100, duration=None, room_dim=None, absorption=None):
    """Plot multiple RIRs for comparison and calculate RT60 values.

    Args:
        rirs: List of RIR arrays to compare
        labels: List of labels for each RIR
        Fs: Sampling frequency (Hz)
        duration: Duration to plot (seconds). If None, plots entire RIRs
        room_dim: Room dimensions [width, depth, height] for theoretical calculations
        absorption: Average absorption coefficient for theoretical calculations
    """
    if labels is None:
        labels = [f'RIR {i + 1}' for i in range(len(rirs))]

    if duration:
        samples = int(duration * Fs)
        rirs = [rir[:samples] if len(rir) > samples else rir for rir in rirs]

    # Create time axis
    times = [np.arange(len(rir)) / Fs for rir in rirs]

    # Plot RIRs
    plt.figure(figsize=(12, 6))
    c = 0.7
    for t, rir, label in zip(times, rirs, labels):
        if label == "SDN-Timu":
            c = 0.5
        plt.plot(t, rir, label=label, alpha=c)

    plt.title('Room Impulse Response Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    if PLOT_FREQ:
        plt.figure(figsize=(10, 6))  # Create a single figure for all plots
        for rir, label in zip(rirs, labels):
            freq, magnitude = ff.calculate_frequency_response(rir, Fs)
            # Filter frequencies within the range 50 to 300 Hz
            mask = (freq >= 50) & (freq <= 300)
            plt.plot(freq[mask], magnitude[mask], label=label)  # Plot each RIR's frequency response
        
        plt.title('Frequency Response of RIRs (50-300 Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.legend()  # Add a legend to distinguish between different RIRs
        plt.show()

    if PLOT_EDC:
        # Plot energy decay curves (matching SDN_timu implementation)
        plt.figure(figsize=(12, 6))
        for rir, label in zip(rirs, labels):
            # edc_timu = an.EDC_timu(rir, Fs, label)
            # edc = an.EDC(rir)
            # edc = an.EDC_dp(rir)
            # plt.plot(np.array(range(len(edc))) / Fs,  edc, label=label)
            an.compute_edc(rir, Fs, label=label)

        plt.title('Energy Decay Curves')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (dB)')
        plt.grid(True)
        plt.legend()
        plt.ylim(-60, 5)

    # Plot Normalized Echo Density if enabled
    if PLOT_NED:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Normalized Echo Density
        for rir, label in zip(rirs, labels):
            echo_density = ned.echoDensityProfile(rir, fs=Fs)
            ax1.plot(echo_density, label=label, alpha=0.7)

        ax1.set_title('Normalized Echo Density')
        ax1.set_xlabel('Time (samples)')
        ax1.set_ylabel('Normalized Echo Density')
        ax1.grid(True)
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_xlim(left=100)

        # Plot Raw (Non-normalized) Echo Density
        for rir, label in zip(rirs, labels):
            echo_density_raw = ned.echoDensityProfileRaw(rir, fs=Fs)
            ax2.plot(echo_density_raw, label=label, alpha=0.7)

        ax2.set_title('Raw Echo Density (Non-normalized)')
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Raw Echo Density Count')
        ax2.grid(True)
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_xlim(left=100)

        plt.tight_layout()  # Adjust subplot parameters for better layout

    # Calculate and print RT60 values
    print("\nReverberation Time Analysis:")
    print("-" * 50)

    # Theoretical RT60 if room dimensions and absorption are provided
    if room_dim is not None and absorption is not None:
        rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, absorption)
        print(f"\nTheoretical RT60 values of the room:")
        print(f"Sabine: {rt60_sabine:.3f} s")
        print(f"Eyring: {rt60_eyring:.3f} s")

    # Calculate RT60 from RIRs
    print("\nMeasured RT60 values:")
    for rir, label in zip(rirs, labels):
        rt60 = pp.calculate_rt60_from_rir(rir, Fs)
        print(f"{label}: {rt60:.3f} s")


def run_comparison(room_parameters, duration=0.05, max_order=5):
    """Run and compare SDN implementations.
    
    Args:
        room_parameters: Dictionary containing room setup parameters
        duration: Duration of simulation in seconds
    """
    timing_results = {}
    rirs = {}
    Fs = 44100
    # SDN-EGE CALCULATIONS ****************************************************
    if CALC_SDN_EGE:

        # Generate Dirac Impulse
        source_signal_impulse = np.zeros(Fs)
        source_signal_impulse[0] = 1.0

        # Calculate reflection coefficient
        room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])

        # Setup room for our implementation
        room = geometry.Room(room_parameters['width'],
                           room_parameters['depth'],
                           room_parameters['height'])

        room.set_microphone(room_parameters['mic x'],
                          room_parameters['mic y'],
                          room_parameters['mic z'])

        room.set_source(room_parameters['source x'],
                      room_parameters['source y'],
                      room_parameters['source z'],
                      signal= source_signal_impulse)

        room.wallAttenuation = [room_parameters['reflection']] * 6

        if PRINT_TIMING:
            start_time = time.time()

        # Initialize and run our SDN implementation
        sdn = DelayNetwork(room, use_identity_scattering=False,
                           ignore_wall_absorption=False,
                           ignore_src_node_atten=False,
                           ignore_node_mic_atten=False,
                           enable_path_logging=False)
        our_rir = sdn.calculate_rir(duration)
        our_rir = our_rir / np.max(np.abs(our_rir))
        rirs['SDN-Ege'] = our_rir

        if PRINT_TIMING:
            timing_results['SDN-Ege'] = time.time() - start_time

    # SDN-BASE CALCULATIONS ****************************************************
    if CALC_SDN_BASE:
        # Setup and run reference implementation

        length = int(duration * Fs)
        data = np.zeros(length, dtype=float)
        data[0] = 1.0
        signal = sig.Signal(Fs, data)

        # Setup reference room
        ref_room = geom.Room()
        ref_room.shape = geom.Cuboid(room_parameters['width'],
                                    room_parameters['depth'],
                                    room_parameters['height'])

        nWalls = ref_room.shape.nWalls
        ref_room.wallAttenuation = [room_parameters['reflection'] for i in range(nWalls)]

        # Simple filter coefficients
        b = np.array([1.0, 0])
        a = np.array([1.0, 0])
        filtOrder = 1
        ref_room.wallFilters = [[geom.WallFilter(filtOrder, b, a)
                               for j in range(nWalls-1)] for i in range(nWalls)]

        # Setup source and mic
        srcPosition = geom.Point(4.5, 3.5, 2)
        # create source at specified position, with input signal
        source = src.Source(srcPosition, signal)

        micPosition = geom.Point(2, 2, 1.5)
        microphone = mic.Microphone(micPosition)

        # Run reference simulation
        frameSize = 8
        nSamples = length

        if PRINT_TIMING:
            start_time = time.time()

        simulate = sim.Simulation(ref_room, source, microphone, frameSize, nSamples)
        ref_rir = simulate.run()
        ref_rir = ref_rir / np.max(np.abs(ref_rir))
        rirs['SDN-Base'] = ref_rir

        if PRINT_TIMING:
            timing_results['SDN-Base'] = time.time() - start_time

    # SDN-TIMU CALCULATIONS ****************************************************
    if CALC_SDN_TIMU:
        timu_mic = TimuMicrophone(np.array([room_parameters['mic x'], 
                                          room_parameters['mic y'], 
                                          room_parameters['mic z']]))
        timu_src = TimuSource(np.array([room_parameters['source x'], 
                                      room_parameters['source y'], 
                                      room_parameters['source z']]))
        timu_room = TimuRoom(room_parameters['width'], 
                           room_parameters['depth'], 
                           room_parameters['height'],
                           room_parameters['absorption'],
                           timu_mic,
                           timu_src)

        if PRINT_TIMING:
            start_time = time.time()

        # Find images and nodes
        timu_room.find_images()
        timu_room.find_sdn_nodes()
        timu_room.create_delay_lines()
        timu_room.find_distances()

        # Setup input signal
        read_write = SoundFileRW(0, 0)
        read_write.read_sound_file("", timu_src)  # Empty filename for impulse

        # Run simulation
        upto = int(duration * Fs)
        for i in range(0, upto):
            timu_room.TickFunction()

        timu_rir = np.array(timu_mic.output)
        timu_rir = timu_rir / np.max(np.abs(timu_rir))
        rirs['SDN-Timu'] = timu_rir

        if PRINT_TIMING:
            timing_results['SDN-Timu'] = time.time() - start_time

    # ISM CALCULATIONS ********************************************************
    if CALC_ISM:
        if PRINT_TIMING:
            start_time = time.time()

        # Create path tracker
        path_tracker_obj = path_tracker.PathTracker()
        path_tracker_obj.add_path(['src', 'mic'],
                                room.source.srcPos.getDistance(room.micPos),
                                'ISM',
                                is_valid=True,
                                segment_distances=[room.micPos.getDistance(room.source.srcPos)])

        # Initialize ISM calculator with path tracker
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc.set_path_tracker(path_tracker_obj)

        # Calculate paths up to max_order
        ism_calc.calculate_paths_up_to_order(max_order)

        # Initialize ISM network
        ism_network = ISMNetwork(room)

        # Add direct path (order 0)
        direct_path = path_tracker_obj.get_paths_by_order(0, 'ISM')[0]
        ism_network.add_reflection_path('path_0_0', direct_path.distance, 0)

        # Add reflection paths
        for order in range(1, max_order + 1):
            paths = path_tracker_obj.get_paths_by_order(order, 'ISM')
            for i, path in enumerate(paths):
                if path.is_valid:  # Only add valid paths
                    path_key = f'path_{order}_{i}'
                    ism_network.add_reflection_path(path_key, path.distance, order)

        # Calculate RIR
        ism_rir = ism_network.calculate_rir(duration)
        ism_rir = ism_rir / np.max(np.abs(ism_rir))
        rirs['ISM'] = ism_rir

        if PRINT_TIMING:
            timing_results['ISM'] = time.time() - start_time

    # ISM WITH PRA CALCULATIONS ****************************************************
    if CALC_ISM_PRA:
        if PRINT_TIMING:
            start_time = time.time()

        # Setup room for ISM with PRA package
        room_dim = np.array([room_parameters['width'],
                            room_parameters['depth'],
                            room_parameters['height']])

        source_loc = np.array([room_parameters['source x'],
                              room_parameters['source y'],
                              room_parameters['source z']])

        mic_loc = np.array([room_parameters['mic x'],
                           room_parameters['mic y'],
                           room_parameters['mic z']])

        pra_room = pra.ShoeBox(room_dim, fs=Fs,
                              materials=pra.Material(room_parameters['absorption']),
                              max_order=max_order,
                              air_absorption=False, ray_tracing=False, use_rand_ism=False)

        pra_room.set_sound_speed(343)
        pra_room.add_source(source_loc).add_microphone(mic_loc)

        pra_room.compute_rir()
        pra_rir = pra_room.rir[0][0]
        pra_rir = pra_rir / np.max(np.abs(pra_rir))

        # Align PRA RIR by removing global delay
        global_delay = pra.constants.get("frac_delay_length") // 2
        pra_rir = pra_rir[global_delay:]  # Shift left by removing the initial delay
        pra_rir = np.pad(pra_rir, (0, global_delay))  # Pad with zeros at the end to maintain length
        pra_rir = pra_rir[:int(duration * Fs)]  # Trim to match duration

        rirs['ISM-PRA'] = pra_rir

        if PRINT_TIMING:
            timing_results['ISM-PRA'] = time.time() - start_time

    if PRINT_TIMING and timing_results:
        print("\nTiming Results:")
        print("-" * 40)
        for impl, duration in timing_results.items():
            print(f"{impl:10s}: {duration:.4f} seconds")
        print("-" * 40)

    return rirs


if __name__ == "__main__":
    # Define the Room
    room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                      'source x': 4.5, 'source y': 3.5, 'source z': 2,
                      'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                      'absorption': 0.2,
                      }

    # Implementation flags
    CALC_SDN_EGE = True  # Calculate Ege's SDN implementation
    CALC_SDN_BASE = True  # Calculate base SDN implementation
    CALC_SDN_TIMU = False  # Calculate Timu's SDN implementation
    CALC_ISM = False  # Calculate ISM implementation
    CALC_ISM_PRA = False  # Calculate ISM with PRA implementation
    PRINT_TIMING = True  # Print timing information
    PLOT_EDC = True  # Plot energy decay curves
    PLOT_FREQ = False  # Plot frequency response
    PLOT_NED = False  # Plot normalized echo density
    # Run comparison
    duration = 0.5
    rirs = run_comparison(room_parameters, duration=duration, max_order=6)

    # Plot RIR comparison
    plot_rir_comparison(
        rirs=list(rirs.values()),
        labels=list(rirs.keys()),
        PLOT_EDC=PLOT_EDC,
        PLOT_NED=PLOT_NED,
        Fs=44100,
        duration=duration,
        room_dim=[room_parameters['width'], 
                 room_parameters['depth'], 
                 room_parameters['height']],
        absorption=room_parameters['absorption']
    )
    plt.show() 