import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_path_calculator import ISMCalculator
import path_tracker
from ISM_core import ISMNetwork

def calculate_ism_rir(room, duration, max_order=6):
    """Calculate RIR using ISM method.
    
    Args:
        max_order: Maximum reflection order to consider
        
    Returns:
        rir: Normalized room impulse response
        fs: Sampling frequency
    """


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
    for order in range(1, max_order+1):
        paths = path_tracker_obj.get_paths_by_order(order, 'ISM')
        for i, path in enumerate(paths):
            if path.is_valid:  # Only add valid paths
                path_key = f'path_{order}_{i}'
                ism_network.add_reflection_path(path_key, path.distance, order)

    # Calculate RIR
    rir = ism_network.calculate_rir(duration)

    # Normalize RIR
    # rir = rir / np.max(np.abs(rir))
    
    return rir, room.source.Fs

if __name__ == '__main__':
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
                    signal=np.array([1]))

    # Calculate reflection coefficient
    room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
    room.wallAttenuation = [room_parameters['reflection']] * 6

    # Calculate RIR
    rir, fs = calculate_ism_rir(room)

    # Plot RIR
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(rir))/fs, rir)
    plt.title('Room Impulse Response (ISM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

