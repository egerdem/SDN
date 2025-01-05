from collections import deque
import numpy as np
from typing import Dict, List, Optional
from geometry import Room, Point

class DelayNetwork:
    """Core SDN implementation focusing on sample-based processing with accessible delay lines."""
    
    def __init__(self, room: Room, Fs: int = 44100, c: float = 343.0):
        self.room = room
        self.Fs = Fs
        self.c = c
        
        # Network parameters
        self.num_nodes = len(room.walls)  # Number of scattering nodes (6 for cuboid)
        self.scattering_matrix = self._create_scattering_matrix()
        
        # Initialize delay lines with public access using descriptive keys
        self.source_to_nodes = {}    # Format: "src_to_{wall_id}"
        self.node_to_mic = {}        # Format: "{wall_id}_to_mic"
        self.node_to_node = {}       # Format: "{wall1_id}_to_{wall2_id}"
        self._setup_delay_lines()
        
        # Initialize attenuation factors with matching keys
        self.source_to_node_gains = {}  # g_sk, Format: "src_to_{wall_id}"
        self.node_to_mic_gains = {}     # g_km, Format: "{wall_id}_to_mic"
        self._calculate_attenuations()
        
        # State variables
        self.current_sample = 0
        self.node_pressures = {wall_id: 0.0 for wall_id in room.walls}
        
        # For storing outgoing wave variables at each node
        self.outgoing_waves = {wall_id: {} for wall_id in room.walls}
        for wall1_id in room.walls:
            for wall2_id in room.walls:
                if wall1_id != wall2_id:
                    self.outgoing_waves[wall1_id][wall2_id] = 0.0
    
    def _create_scattering_matrix(self) -> np.ndarray:
        """Create the scattering matrix S = (2/(N-1))1_{(N-1)×(N-1)} - I
        This matrix is used for each node's scattering operation.
        Size is (N-1)×(N-1) as each node has N-1 connections (excluding self)."""
        N = self.num_nodes
        size = N - 1  # Matrix size for each node's operation
        return (2/size) * np.ones((size, size)) - np.eye(size)
    
    def _setup_delay_lines(self):
        """Initialize all delay lines in the network."""
        # Source to nodes
        for wall_id, wall in self.room.walls.items():
            key = f"src_to_{wall_id}"
            dist = wall.node_positions.getDistance(self.room.source.srcPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.source_to_nodes[key] = deque([0.0] * delay_samples, maxlen=delay_samples)
        
        # Nodes to microphone
        for wall_id, wall in self.room.walls.items():
            key = f"{wall_id}_to_mic"
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.node_to_mic[key] = deque([0.0] * delay_samples, maxlen=delay_samples)
        
        # Between nodes
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    # key = f"{wall1_id}_to_{wall2_id}"
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = int(np.floor((self.Fs * dist) / self.c))
                    self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
    
    def _calculate_attenuations(self):
        """Calculate attenuation factors for all connections."""
        G = self.c / self.Fs  # unit distance
        
        for wall_id, wall in self.room.walls.items():
            node_pos = wall.node_positions
            src_dist = node_pos.getDistance(self.room.source.srcPos)
            mic_dist = node_pos.getDistance(self.room.micPos)
            
            # Source to node: g_sk = G/||x_S - x_k||
            src_key = f"src_to_{wall_id}"
            self.source_to_node_gains[src_key] = G / src_dist
            
            # Node to mic: g_km = 1/(1 + ||x_k - x_M||/||x_S - x_k||)
            mic_key = f"{wall_id}_to_mic"
            self.node_to_mic_gains[mic_key] = 1.0 / (1.0 + mic_dist/src_dist)
    
    def process_sample(self, input_sample: float = 0.0) -> float:
        """Process one sample through the network and return the output sample.
        
        The processing follows these steps:
        1. Distribute source input to nodes (with proper wave variable conversion)
        2. Process each node (scattering operation)
        3. Update node-to-node connections
        4. Calculate microphone output
        """
        output_sample = 0.0
        
        # Step 1: Distribute input to nodes
        for wall_id in self.room.walls:
            src_key = f"src_to_{wall_id}"
            # Calculate source pressure contribution with attenuation
            source_pressure = input_sample * self.source_to_node_gains[src_key]
            self.source_to_nodes[src_key].append(source_pressure)
        
        # Step 2: Process each node
        for wall_id in self.room.walls:
            # Get source contribution
            src_key = f"src_to_{wall_id}"
            source_pressure = self.source_to_nodes[src_key][0]
            
            # Collect incoming wave variables from other nodes (excluding self)
            incoming_waves = []
            other_nodes = []
            for other_id in self.room.walls:
                if other_id != wall_id:
                    other_nodes.append(other_id)
                    # Read from delay line and add half of source pressure
                    p = self.node_to_node[other_id][wall_id][0]
                    p_tilde = p + 0.5 * source_pressure  # Eq. (7) from paper
                    incoming_waves.append(p_tilde)
            
            # Apply scattering matrix to get outgoing waves
            outgoing_waves = np.dot(self.scattering_matrix, incoming_waves)
            
            # Store outgoing waves for each connection
            for idx, other_id in enumerate(other_nodes):
                self.outgoing_waves[wall_id][other_id] = outgoing_waves[idx]
            
            # Calculate node pressure: p_k(n) = p_Sk(n) + 2/(N-1) * Σ p_ki^+(n)
            node_pressure = source_pressure + (2/(self.num_nodes-1)) * sum(incoming_waves)
            self.node_pressures[wall_id] = node_pressure
            
            # Send to microphone (using outgoing waves)
            mic_key = f"{wall_id}_to_mic"
            mic_pressure = (2/(self.num_nodes-1)) * sum(outgoing_waves) * self.node_to_mic_gains[mic_key]
            self.node_to_mic[mic_key].append(mic_pressure)
            output_sample += self.node_to_mic[mic_key][0]
        
        # Step 3: Update node-to-node connections using stored outgoing waves
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Get outgoing wave and apply wall attenuation
                    outgoing_wave = self.outgoing_waves[wall_id][other_id]
                    attenuated_wave = outgoing_wave * self.room.wallAttenuation[0]
                    self.node_to_node[wall_id][other_id].append(attenuated_wave)
        
        self.current_sample += 1
        return output_sample
    
    def calculate_rir(self, duration: float) -> np.ndarray:
        """Calculate room impulse response for given duration."""
        num_samples = int(self.Fs * duration)
        rir = np.zeros(num_samples)
        
        # Input signal is unit impulse
        rir[0] = self.process_sample(1.0)
        
        # Process remaining samples
        for n in range(1, num_samples):
            rir[n] = self.process_sample(0.0)
        
        return rir 