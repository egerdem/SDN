import path_tracker
import numpy as np
import geometry
from sdn_path_calculator import ISMCalculator
import os

# --- Parameters ---
Fs = 44100 # Sample rate, needed for path_tracker time conversion

# --- Room Definitions ---
room_waspaa = {
    'display_name': 'WASPAA Room',
    'width': 6, 'depth': 7, 'height': 4,
    'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
    'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
    'absorption': 0.1,
}
room_aes = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}
room_journal = {
    'display_name': 'Journal Room',
    'width': 3.2, 'depth': 4, 'height': 2.7,
    'source x': 2, 'source y': 3., 'source z': 2,
    'mic x': 1.6, 'mic y': 2, 'mic z': 1.8,
    'absorption': 0.1,
}
    
# List of rooms to process
rooms_to_process = [room_journal, room_aes, room_waspaa]

all_room_data = {}

print("Calculating reflection times for all rooms...")

for room_params in rooms_to_process:
    display_name = room_params['display_name']
    print(f"Processing: {display_name}")

    # --- Setup Room Geometry ---
    room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
    
    source_pos = (room_params['source x'], room_params['source y'], room_params['source z'])
    mic_pos = (room_params['mic x'], room_params['mic y'], room_params['mic z'])
    
    # Set source and mic positions (signal is not needed for path calculation)
    room.set_source(source_pos[0], source_pos[1], source_pos[2], signal=None, Fs=Fs)
    room.set_microphone(mic_pos[0], mic_pos[1], mic_pos[2])

    # --- Calculate ISM Path Times ---
    tracker = path_tracker.PathTracker()
    ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
    ism_calc.set_path_tracker(tracker)
    
    # Analyze paths to populate the tracker
    ism_calc.analyze_paths(max_order=3, print_invalid=False)

    # Get latest arrival times for each order from the tracker
    arrival_times = tracker.get_latest_arrival_time_by_order('ISM')
    
    reflection_times = {
        'first_order': arrival_times.get(1),
        'second_order': arrival_times.get(2),
        'third_order': arrival_times.get(3)
    }

    print(f"  First order reflection time: {reflection_times.get('first_order', 0):.4f}s")
    print(f"  Second order reflection time: {reflection_times.get('second_order', 0):.4f}s")
    print(f"  Third order reflection time: {reflection_times.get('third_order', 0):.4f}s")
    
    # --- Store Data ---
    all_room_data[display_name] = {
        'params': room_params,
        'reflection_times': reflection_times
    }

# --- Save to File ---
output_dir = "results/paper_data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "reflection_times.npz")

# Save the dictionary. Using savez allows us to load it easily.
# The dictionary will be saved as an object array under the key 'all_room_data'.
np.savez(output_path, all_room_data=all_room_data)

print(f"\nSuccessfully saved reflection times for all rooms to:\n{output_path}")

