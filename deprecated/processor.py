from collections import deque
import numpy as np

class DelayLine:
    def __init__(self, max_delay: int):
        self.buffer = deque([0] * max_delay, maxlen=max_delay)
    
    def write(self, sample):
        self.buffer.append(sample)
    
    def read(self):
        return self.buffer[0]

class SDNProcessor:
    def __init__(self, room, Fs: int = 44100, c: float = 343.0, duration: float = 1.0):
        self.room = room
        self.Fs = Fs
        self.c = c
        self.num_samples = int(Fs * duration)
        self.G = self.c / self.Fs
        
        # Initialize nodes (one per wall)
        self.N = len(room.walls)  # Number of nodes (6 for cuboid)
        
        # Calculate scattering matrix
        self.S = (2/(self.N-1)) * np.ones((self.N-1, self.N-1)) - np.eye(self.N-1)
        
        # Initialize delay lines between nodes
        self.delay_lines = {}
        self.setup_delay_lines()
        
        # Calculate attenuations
        self.g_sk = {}  # source to node attenuations
        self.g_km = {}  # node to mic attenuations
        self.calculate_attenuations()
        
        # Output buffer
        self.rir = np.zeros(self.num_samples)

    def setup_delay_lines(self):
        """Setup delay lines between all nodes."""
        for wall1_id, wall1 in self.room.walls.items():
            self.delay_lines[wall1_id] = {}
            for wall2_id, wall2 in self.room.walls.items():
                if wall1_id != wall2_id:
                    # Calculate delay in samples
                    distance = wall1.node_positions.getDistance(wall2.node_positions)
                    delay_samples = int(np.floor((self.Fs * distance) / self.c))
                    self.delay_lines[wall1_id][wall2_id] = DelayLine(delay_samples)

    def calculate_attenuations(self):
        """Calculate source-node and node-mic attenuations."""
        for wall_id, wall in self.room.walls.items():
            node_pos = wall.node_positions
            src_dist = node_pos.getDistance(self.room.source.srcPos)
            mic_dist = node_pos.getDistance(self.room.micPos)
            
            # Source to node attenuation
            self.g_sk[wall_id] = 1.0 / src_dist
            
            # Node to mic attenuation
            self.g_km[wall_id] = 1.0 * self.G / (1 + mic_dist/src_dist)

    def process_sample(self, n: int):
        """Process a single time step."""
        # Input signal (impulse at n=0)
        input_signal = 1.0 if n == 0 else 0.0
        
        # For each node
        node_pressures = {}
        for wall_id in self.room.walls:
            # 1. Get source contribution
            p_source = input_signal * self.g_sk[wall_id]
            
            # 2. Collect incoming pressures from other nodes
            incoming = []
            for other_id in self.room.walls:
                if other_id != wall_id:
                    p = self.delay_lines[other_id][wall_id].read()
                    incoming.append(p)
            
            # 3. Apply scattering matrix
            scattered = np.dot(self.S, incoming)
            
            # 4. Total pressure at node
            node_pressures[wall_id] = p_source + sum(scattered)
            
            # 5. Contribution to microphone
            self.rir[n] += node_pressures[wall_id] * self.g_km[wall_id]
        
        # Update delay lines with new pressures
        for wall_id, pressure in node_pressures.items():
            for other_id in self.room.walls:
                if other_id != wall_id:
                    self.delay_lines[wall_id][other_id].write(pressure)

    def calculate_rir(self):
        """Calculate complete room impulse response."""
        for n in range(self.num_samples):
            self.process_sample(n)
        return self.rir

class SDNOrderProcessor:
    """SDN processor that operates based on reflection orders."""
    def __init__(self, room, max_order: int, Fs: int = 44100, c: float = 343.0):
        self.room = room
        self.Fs = Fs
        self.c = c
        self.max_order = max_order
        
        # Calculate maximum possible delay based on room dimensions
        max_distance = np.sqrt(room.x**2 + room.y**2 + room.z**2) * (max_order + 1)
        self.max_samples = int(np.ceil((Fs * max_distance) / c))
        
        # Initialize RIR buffer
        self.rir = np.zeros(self.max_samples)
        
        # Use natural wall ordering from room.walls
        self.wall_order = list(room.walls.keys())
        
        # Calculate attenuations
        self.g_sk = {}  # source to node attenuations
        self.g_km = {}  # node to mic attenuations
        self.calculate_attenuations()

    def calculate_attenuations(self):
        """Calculate source-node and node-mic attenuations."""
        for wall_id, wall in self.room.walls.items():
            node_pos = wall.node_positions
            src_dist = node_pos.getDistance(self.room.source.srcPos)
            mic_dist = node_pos.getDistance(self.room.micPos)
            
            # Source to node attenuation
            self.g_sk[wall_id] = 1.0 / src_dist
            
            # Node to mic attenuation
            self.g_km[wall_id] = 1.0 / (1 + mic_dist/src_dist)

    def process_path(self, path: 'Path'):
        """Process a single path and add its contribution to RIR."""
        if not path.segment_distances:
            return
            
        # Calculate delay in samples for each segment
        delays = [int(np.floor((self.Fs * d) / self.c)) 
                 for d in path.segment_distances]
        total_delay = sum(delays)
        
        if path.order == 0:  # Direct path
            attenuation = 1.0
        else:
            # First wall hit
            first_wall = path.nodes[1]
            SK_attenuation = 0.5 * self.g_sk[first_wall]  # Source to first node with 1/2 factor
            
            # Initialize attenuation
            attenuation = SK_attenuation



            if path.order >= 2:
                # Apply varying weights for each middle segment
                for i in range(1, len(path.nodes) - 2):
                    current_node = path.nodes[i]
                    next_node = path.nodes[i + 1]
                    
                    # Use a random weight between -1 and 1
                    weight = np.random.uniform(-0.4, 1)
                    
                    # Apply wall reflection coefficient
                    wall_reflection = 0.89
                    attenuation *= wall_reflection * weight
            
            # Final node to mic attenuation
            last_wall = path.nodes[-2]  # Last wall before mic
            attenuation *= self.g_km[last_wall]
        
        # Add contribution to RIR
        if total_delay < self.max_samples:
            self.rir[total_delay] += attenuation

    def calculate_rir(self, path_tracker):
        """Calculate room impulse response using path information."""
        # Process paths order by order
        for order in range(self.max_order + 1):
            paths = path_tracker.get_paths_by_order(order, 'SDN')
            for path in paths:
                self.process_path(path)
        
        return self.rir

