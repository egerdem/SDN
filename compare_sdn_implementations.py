import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork
import sys
import os
import time  # Add time import

# Add SDN_algo3 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SDN_algo3'))
import Signal as sig
import Source as src
import Microphone as mic
import Simulation as sim
import Geometry as geom
from SDN_timu import Room as TimuRoom, Source as TimuSource, Microphone as TimuMicrophone, SoundFileRW

def run_comparison(room_parameters, duration=0.05, time_comparison=False):
    """Run and compare SDN implementations.
    
    Args:
        room_parameters: Dictionary containing room setup parameters
        duration: Duration of simulation in seconds
        time_comparison: If True, print timing information for each implementation
    """
    timing_results = {}

    # SDN-EGE CALCULATIONS ****************************************************

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
                   signal=np.array([1]))

    room.wallAttenuation = [room_parameters['reflection']] * 6

    if time_comparison:
        start_time = time.time()

    # Initialize and run our SDN implementation
    sdn = DelayNetwork(room, use_identity_scattering=False,
                        ignore_wall_absorption=False,
                        ignore_src_node_atten=False,
                        ignore_node_mic_atten = False)
    our_rir = sdn.calculate_rir(duration)
    our_rir = our_rir / np.max(np.abs(our_rir))

    if time_comparison:
        timing_results['SDN-Ege'] = time.time() - start_time

    # SDN-BASE CALCULATIONS ****************************************************

    # Setup and run reference implementation
    Fs = 44100
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
    srcPosition = geometry.Point(room_parameters['source x'],
                               room_parameters['source y'],
                               room_parameters['source z'])
    source = src.Source(srcPosition, signal)

    micPosition = geometry.Point(room_parameters['mic x'],
                               room_parameters['mic y'],
                               room_parameters['mic z'])
    microphone = mic.Microphone(micPosition)

    # Run reference simulation
    frameSize = 8
    nSamples = length

    if time_comparison:
        start_time = time.time()

    simulate = sim.Simulation(ref_room, source, microphone, frameSize, nSamples)
    ref_rir = simulate.run()
    ref_rir = ref_rir / np.max(np.abs(ref_rir))

    if time_comparison:
        timing_results['SDN-Base'] = time.time() - start_time

    # SDN-TIMU CALCULATIONS ****************************************************

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

    if time_comparison:
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

    if time_comparison:
        timing_results['SDN-Timu'] = time.time() - start_time
        print("\nTiming Results:")
        print("-" * 40)
        for impl, duration in timing_results.items():
            print(f"{impl:10s}: {duration:.4f} seconds")
        print("-" * 40)

    return our_rir, ref_rir, timu_rir, room

if __name__ == "__main__":
    # Define the Room
    room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                       'source x': 4.5, 'source y': 3.5, 'source z': 2,
                       'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                       'absorption': 0.2,
                       }

    # Run comparison with timing enabled
    duration = 0.05
    our_rir, ref_rir, timu_rir, room = run_comparison(room_parameters, duration = duration, time_comparison=True)

    # Visualize room setup
    # pp.plot_room(room)

    # Plot RIR comparison
    pp.plot_rir_comparison(
        [our_rir, ref_rir, timu_rir],
        labels=['SDN-Ege', 'SDN-Base', 'SDN-Timu'],
        fs=44100,
        duration=duration,
        room_dim=[room_parameters['width'], 
                 room_parameters['depth'], 
                 room_parameters['height']],
        absorption=room_parameters['absorption']
    )
    plt.show() 