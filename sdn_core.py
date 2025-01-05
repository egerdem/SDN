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
        
        # Initialize delay lines with public access
        self.source_to_nodes = {}    # Source to each node
        self.node_to_mic = {}        # Each node to microphone
        self.node_to_node = {}       # Between nodes (2D dictionary)
        self._setup_delay_lines()
        
        # Initialize attenuation factors
        self.source_to_node_gains = {}  # g_sk
        self.node_to_mic_gains = {}     # g_km
        self._calculate_attenuations()
        
        # State variables
        self.current_sample = 0
        self.node_pressures = {wall_id: 0.0 for wall_id in room.walls}
    
    def _create_scattering_matrix(self) -> np.ndarray:
        """Create the scattering matrix S = (2/(N-1))1_{(N-1)×(N-1)} - I"""
        N = self.num_nodes
        return (2/(N-1)) * np.ones((N, N)) - np.eye(N)
    
    def _setup_delay_lines(self):
        """Initialize all delay lines in the network."""
        # Source to nodes
        for wall_id, wall in self.room.walls.items():
            dist = wall.node_positions.getDistance(self.room.source.srcPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.source_to_nodes[wall_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
        
        # Nodes to microphone
        for wall_id, wall in self.room.walls.items():
            dist = wall.node_positions.getDistance(self.room.micPos)
            delay_samples = int(np.floor((self.Fs * dist) / self.c))
            self.node_to_mic[wall_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
        
        # Between nodes
        for wall1_id, wall1 in self.room.walls.items():
            self.node_to_node[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    dist = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = int(np.floor((self.Fs * dist) / self.c))
                    self.node_to_node[wall1_id][wall2_id] = deque([0.0] * delay_samples, maxlen=delay_samples)
    
    def _calculate_attenuations(self):
        """Calculate attenuation factors for all connections."""
        G = self.c / self.Fs  # Conversion factor
        
        for wall_id, wall in self.room.walls.items():
            node_pos = wall.node_positions
            src_dist = node_pos.getDistance(self.room.source.srcPos)
            mic_dist = node_pos.getDistance(self.room.micPos)
            
            # Source to node: g_sk = G/||x_S - x_k||
            self.source_to_node_gains[wall_id] = G / src_dist
            
            # Node to mic: g_km = 1/(1 + ||x_k - x_M||/||x_S - x_k||)
            self.node_to_mic_gains[wall_id] = 1.0 / (1.0 + mic_dist/src_dist)
    
    def process_sample(self, input_sample: float = 0.0) -> float:
        """Process one sample through the network and return the output sample."""
        output_sample = 0.0
        
        # Step 1: Distribute input to nodes
        for wall_id in self.room.walls:
            # Input contribution with attenuation
            input_contribution = input_sample * self.source_to_node_gains[wall_id] * 0.5
            self.source_to_nodes[wall_id].append(input_contribution)
        
        # Step 2: Process each node
        for i, (wall_id, wall) in enumerate(self.room.walls.items()):
            # Collect incoming pressures from other nodes
            incoming_pressures = []
            for other_id in self.room.walls:
                if other_id != wall_id:
                    p = self.node_to_node[other_id][wall_id][0]  # Read from delay line
                    incoming_pressures.append(p)
            
            # Apply scattering matrix to incoming pressures
            scattered = np.dot(self.scattering_matrix[i], incoming_pressures)
            
            # Add source contribution
            node_pressure = self.source_to_nodes[wall_id][0] + sum(scattered)
            
            # Store node pressure
            self.node_pressures[wall_id] = node_pressure
            
            # Send to microphone
            self.node_to_mic[wall_id].append(node_pressure * self.node_to_mic_gains[wall_id])
            output_sample += self.node_to_mic[wall_id][0]
        
        # Step 3: Update node-to-node connections
        for wall_id in self.room.walls:
            for other_id in self.room.walls:
                if wall_id != other_id:
                    # Apply wall reflection coefficient
                    reflected = self.node_pressures[wall_id] * self.room.wallAttenuation[0]
                    self.node_to_node[wall_id][other_id].append(reflected)
        
        self.current_sample += 1
        return output_sample
    
    def calculate_rir(self, duration: float) -> np.ndarray:
        """Calculate room impulse response for given duration."""
        num_samples = int(self.Fs * duration)
        rir = np.zeros(num_samples)
        
        # Process impulse input
        rir[0] = self.process_sample(1.0)  # Impulse at t=0
        
        # Process remaining samples
        for n in range(1, num_samples):
            rir[n] = self.process_sample(0.0)
        
        return rir 