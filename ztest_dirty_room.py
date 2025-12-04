
import geometry
import numpy as np

def test_dirty_room():
    print("--- Testing Dirty Room Hypothesis ---")
    
    # 1. Create Room
    room = geometry.Room(9, 7, 4)
    room.set_source(4.5, 6.0, 2.0)
    
    # 2. Set Mic to Pos A
    pos_A = (0.5, 0.5, 1.5)
    room.set_microphone(*pos_A)
    print(f"Set Mic to A: {pos_A}")
    
    # Capture node positions for A
    nodes_A = {}
    for wall_id, wall in room.walls.items():
        nodes_A[wall_id] = (wall.node_positions.x, wall.node_positions.y, wall.node_positions.z)
        print(f"  Wall {wall_id} node: {nodes_A[wall_id]}")
        
    # 3. Set Mic to Pos B
    pos_B = (4.0, 3.0, 1.5)
    room.set_microphone(*pos_B)
    print(f"\nSet Mic to B: {pos_B}")
    
    # Capture node positions for B
    nodes_B = {}
    match = True
    for wall_id, wall in room.walls.items():
        nodes_B[wall_id] = (wall.node_positions.x, wall.node_positions.y, wall.node_positions.z)
        print(f"  Wall {wall_id} node: {nodes_B[wall_id]}")
        
        if nodes_A[wall_id] != nodes_B[wall_id]:
            match = False
            
    if match:
        print("\n[FAIL] Node positions DID NOT CHANGE! The room is dirty/stale.")
    else:
        print("\n[PASS] Node positions changed.")

if __name__ == "__main__":
    test_dirty_room()
