import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SDN_algo3'))
import Signal as sig
import Source as src
import Microphone as mic
import Simulation as sim
import Geometry as geom

def calculate_sdn_base_rir(room_parameters, duration, Fs):
    """Calculate RIR using SDN-Base implementation.
    
    Args:
        room_parameters (dict): Dictionary containing room setup parameters
        duration (float): Duration of simulation in seconds
        Fs (int): Sampling frequency
        
    Returns:
        numpy.ndarray: Normalized room impulse response
    """
    # Setup input signal
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
    reflection = np.sqrt(1 - room_parameters['absorption'])
    ref_room.wallAttenuation = [reflection for i in range(nWalls)]

    # Simple filter coefficients
    b = np.array([1.0, 0])
    a = np.array([1.0, 0])
    filtOrder = 1
    ref_room.wallFilters = [[geom.WallFilter(filtOrder, b, a)
                           for j in range(nWalls-1)] for i in range(nWalls)]

    # Setup source and mic
    srcPosition = geom.Point(room_parameters['source x'],
                           room_parameters['source y'],
                           room_parameters['source z'])
    source = src.Source(srcPosition, signal)

    micPosition = geom.Point(room_parameters['mic x'],
                           room_parameters['mic y'],
                           room_parameters['mic z'])
    microphone = mic.Microphone(micPosition)

    # Run simulation
    frameSize = 8
    nSamples = length
    simulate = sim.Simulation(ref_room, source, microphone, frameSize, nSamples)
    rir = simulate.run()
    rir = rir / np.max(np.abs(rir))
    
    return rir 