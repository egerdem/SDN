import numpy as np
import geometry
import plot_room as pp
from sdn_path_calculator import SDNCalculator, ISMCalculator
import matplotlib.pyplot as plt
import path_tracker
import random

# Define the Room
room_parameters = {'width': 9, 'depth': 7, 'height': 4,
                   'source x': 4.5, 'source y': 3.5, 'source z': 2,
                   'mic x': 2, 'mic y': 2, 'mic z': 1.5,
                   'absorption': 0.2,
                   }

room = geometry.Room(room_parameters['width'],
                     room_parameters['depth'],
                     room_parameters['height'])

room.set_microphone(room_parameters['mic x'],
                    room_parameters['mic y'],
                    room_parameters['mic z'])

room.set_source(room_parameters['source x'],
                room_parameters['source y'],
                room_parameters['source z'],
                signal = np.array([1]))

room_parameters['reflection'] = 1 * np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

pp.plot_room(room)



# Calculate paths with ray tracing
# sdn_calc = SDNCalculator(
#     room.walls,
#     room.source.srcPos,
#     room.micPos,
#     use_ray_tracing=True,
#     n_rays=10000,
#     rt_max_order=10
# )

# Calculate paths up to order 10 (ISM for 0-2, RT for 3-10)
# sdn_calc.calculate_paths_up_to_order(10)

# Create shared path tracker
path_tracker = path_tracker.PathTracker()

# Initialize calculators with shared tracker
sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
sdn_calc.set_path_tracker(path_tracker)
ism_calc.set_path_tracker(path_tracker)

# Calculate paths up to order 3
sdn_calc.calculate_paths_up_to_order(3)
ism_calc.calculate_paths_up_to_order(3)

# Print path comparison using shared tracker
path_tracker.print_path_comparison()

# Get all invalid ISM paths
# invalid_paths = []
# for order in range(4):  # Up to order 3
#     paths = path_tracker.get_paths_by_order(order, 'ISM')
#     invalid_paths.extend([p.nodes for p in paths if not p.is_valid])
#
# # Select 10 random paths (or all if less than 10 exist)
# num_examples = min(10, len(invalid_paths))
# example_paths = random.sample(invalid_paths, num_examples)
#
# print("\nVisualizing random invalid paths:")
# for path in example_paths:
#     print(f"\nExamining invalid path: {' → '.join(path)}")
#     pp.plot_ism_path(room, ism_calc, path)
#     plt.show()

# Visualize some example paths
example_paths = [
    ['s', 'east', 'west', 'm'],
    ['s', 'west', 'm'],
    ['s', 'west', 'east', 'north', 'm']
]

for path in example_paths:
    pp.plot_ism_path(room, ism_calc, path)
    plt.show()