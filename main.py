import numpy as np
import geometry
import plot_room as pp
import matplotlib.pyplot as plt
from sdn_core import DelayNetwork

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
                signal = np.array([1]))

# Calculate reflection coefficient
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# Visualize the room setup
# pp.plot_room(room)

# Initialize SDN for reference (with all physics enabled)
reference_sdn = DelayNetwork(room,
                           use_identity_scattering=False,
                           ignore_wall_absorption=False,
                           ignore_src_node_atten=False,
                           ignore_node_mic_atten=False,
                             enable_path_logging=False)
#
# Calculate reference RIR
duration = 0.05  # seconds

reference_rir = reference_sdn.calculate_rir(duration)
reference_rir = reference_rir / np.max(np.abs(reference_rir))

# Initialize SDN with test flags
sdn = DelayNetwork(room,
                   use_identity_scattering=False,
                   ignore_wall_absorption=False,
                   ignore_src_node_atten=False,
                   ignore_node_mic_atten=False,
                   enable_path_logging=True)

# Calculate test RIR
rir = sdn.calculate_rir(duration)
rir = rir / np.max(np.abs(rir))

# Analyze paths
# sdn.get_path_summary()  # Print all paths
# sdn.get_path_summary('src->ceiling->mic')  # Print specific path
# sdn.get_paths_through_node('ceiling')  # Get all paths through ceiling
# sdn.get_active_paths_at_sample(305)  # Get paths active at sample 305

# Create list of enabled flags
enabled_flags = []
if sdn.use_identity_scattering:
    enabled_flags.append("Identity Scattering")
if sdn.ignore_wall_absorption:
    enabled_flags.append("No Wall Absorption")
if sdn.ignore_src_node_atten:
    enabled_flags.append("No Src-Node Atten")
if sdn.ignore_node_mic_atten:
    enabled_flags.append("No Node-Mic Atten")

# Plot RIR
plt.figure(figsize=(12, 6))


# Plot test RIR
plt.plot(rir, label='Test RIR')

# Plot reference RIR with reduced opacity
plt.plot(reference_rir, color='orange', alpha=0.6, label='Reference RIR')

plt.title('Room Impulse Response (SDN)')
plt.xlabel('Sample')
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

# The code below is the order-based implementation that we'll revisit later
# after implementing and validating the traditional sample-based SDN.
# 
# # Calculate paths up to order 10 (ISM for 0-2, RT for 3-10)
# # sdn_calc.calculate_paths_up_to_order(10)
# 
# # Create shared path tracker
# path_tracker = path_tracker.PathTracker()
# 
# # Initialize calculators with shared tracker
# sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
# ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
# sdn_calc.set_path_tracker(path_tracker)
# ism_calc.set_path_tracker(path_tracker)
# 
# # Calculate paths up to order 3
# sdn_calc.calculate_paths_up_to_order(3)
# ism_calc.calculate_paths_up_to_order(3)
# 
# # Print path comparison using shared tracker
# path_tracker.print_path_comparison()
# 
# # Get all invalid ISM paths
# # invalid_paths = []
# # for order in range(4):  # Up to order 3
# #     paths = path_tracker.get_paths_by_order(order, 'ISM')
# #     invalid_paths.extend([p.nodes for p in paths if not p.is_valid])
# #
# # # Select 10 random paths (or all if less than 10 exist)
# # num_examples = min(10, len(invalid_paths))
# # example_paths = random.sample(invalid_paths, num_examples)
# #

# Visualize some example ISM paths
# example_paths = [
#     ['s', 'east', 'west', 'm'],
#     ['s', 'west', 'm'],
#     ['s', 'west', 'east', 'north', 'm']
# ]
#
# for path in example_paths:
#     pp.plot_ism_path(room, ism_calc, path)
#     plt.show()