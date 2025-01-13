"""Path logging mechanism for SDN implementation analysis."""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class PathSegment:
    """Represents a segment of a path through the SDN network."""
    from_node: str  # 'src' or wall_id
    to_node: str    # wall_id or 'mic'
    sample_idx: int
    pressure: float
    attenuation: float
    delay: int
    origin: str = "direct"  # 'direct' for source->node, or the wall_id where energy came from

@dataclass
class PathSnapshot:
    """Captures the state of a complete path at a specific sample."""
    sample_idx: int
    segments: List[PathSegment]
    total_pressure: float
    total_delay: int  # Will be calculated from segments
    
    def __post_init__(self):
        """Calculate total delay as sum of segment delays."""
        self.total_delay = sum(seg.delay for seg in self.segments)
    
    def __str__(self):
        """Create string representation showing complete path in chronological order."""
        # Start with source
        components = ['src']
        
        # Track the path chronologically
        for i, seg in enumerate(self.segments):
            if seg.from_node == 'incoming':
                # Add the origin node in parentheses to show where the wave came from
                components.append(f"({seg.origin})")
            elif seg.from_node != 'src':  # Skip src as it's already added
                components.append(seg.from_node)
                
        # Add final destination if not already added
        if self.segments[-1].to_node != components[-1]:
            components.append(self.segments[-1].to_node)
            
        path_str = " -> ".join(components)
        return f"[n={self.sample_idx}] {path_str}: pressure={self.total_pressure:.6f}, delay={self.total_delay}"

class PathLogger:
    """Logs and analyzes pressure propagation paths in SDN."""
    
    def __init__(self, threshold: float = 1e-10):
        self.threshold = threshold  # Minimum pressure to log
        self.paths: Dict[str, List[PathSnapshot]] = {}
        self.current_sample = 0
        
    def log_direct_path(self, sample_idx: int, pressure: float, delay: int, attenuation: float):
        """Log direct sound path (source to mic)."""
        segment = PathSegment('src', 'mic', sample_idx, pressure, attenuation, delay, origin="direct")
        path_key = 'src->mic'
        snapshot = PathSnapshot(sample_idx, [segment], pressure, delay)
        
        if abs(pressure) > self.threshold:
            if path_key not in self.paths:
                self.paths[path_key] = []
            self.paths[path_key].append(snapshot)
    
    def log_source_to_node(self, sample_idx: int, node_id: str, pressure: float, 
                          delay: int, attenuation: float):
        """Log source to node path."""
        segment = PathSegment('src', node_id, sample_idx, pressure, attenuation, delay, origin="direct")
        path_key = f'src->{node_id}'
        snapshot = PathSnapshot(sample_idx, [segment], pressure, delay)
        
        if abs(pressure) > self.threshold:
            if path_key not in self.paths:
                self.paths[path_key] = []
            self.paths[path_key].append(snapshot)
    
    def log_node_to_mic(self, sample_idx: int, node_id: str, pressure: float, 
                        delay: int, attenuation: float, source_pressure: float = None):
        """Log node to microphone path, including source contribution if available."""
        segments = []
        total_pressure = pressure
        
        # Add source to node segment if source pressure is provided
        if source_pressure is not None:
            src_segment = PathSegment('src', node_id, sample_idx, source_pressure, 
                                    attenuation, delay, origin="direct")
            segments.append(src_segment)
            
        # Add node to mic segment with origin from previous segment if it exists
        origin = "direct" if not segments else segments[-1].origin
        mic_segment = PathSegment(node_id, 'mic', sample_idx, pressure, attenuation, delay, origin=origin)
        segments.append(mic_segment)
        
        path_key = f'src->{node_id}->mic' if source_pressure is not None else f'{node_id}->mic'
        snapshot = PathSnapshot(sample_idx, segments, total_pressure, delay)
        
        if abs(total_pressure) > self.threshold:
            if path_key not in self.paths:
                self.paths[path_key] = []
            self.paths[path_key].append(snapshot)
    
    def log_node_to_node(self, sample_idx: int, from_node: str, to_node: str, 
                         pressure: float, delay: int, attenuation: float,
                         source_pressure: Optional[float] = None,
                         incoming_pressure: Optional[float] = None):
        """Log node to node path, including source and incoming pressures if available."""
        segments = []
        
        # Build path key and segments based on available information
        path_components = []
        
        if source_pressure is not None:
            src_segment = PathSegment('src', from_node, sample_idx, source_pressure, 
                                    attenuation, delay, origin="direct")
            segments.append(src_segment)
            path_components.append('src')
            
        if incoming_pressure is not None:
            # This represents pressure arriving from other nodes
            in_segment = PathSegment('incoming', from_node, sample_idx, incoming_pressure, 
                                   1.0, 0, origin=from_node)  # Mark origin as the sending node
            segments.append(in_segment)
            
        # Add the current node-to-node segment with origin from previous segment
        origin = "direct" if not segments else segments[-1].origin
        node_segment = PathSegment(from_node, to_node, sample_idx, pressure, 
                                 attenuation, delay, origin=origin)
        segments.append(node_segment)
        
        # Build path key showing origins for non-direct paths
        if origin != "direct":
            path_components.append(f"{from_node}({origin})")
        else:
            path_components.append(from_node)
        path_components.append(to_node)
        path_key = '->'.join(path_components)
        
        snapshot = PathSnapshot(sample_idx, segments, pressure, delay)
        
        if abs(pressure) > self.threshold:
            if path_key not in self.paths:
                self.paths[path_key] = []
            self.paths[path_key].append(snapshot)
    
    def get_path_history(self, path_key: str) -> List[PathSnapshot]:
        """Get history of a specific path."""
        return self.paths.get(path_key, [])
    
    def get_paths_through_node(self, node_id: str) -> Dict[str, List[PathSnapshot]]:
        """Get all paths that pass through a specific node."""
        return {k: v for k, v in self.paths.items() if node_id in k}
    
    def get_active_paths_at_sample(self, sample_idx: int) -> Dict[str, PathSnapshot]:
        """Get all paths active at a specific sample index."""
        active_paths = {}
        for path_key, snapshots in self.paths.items():
            matching_snapshots = [s for s in snapshots if s.sample_idx == sample_idx]
            if matching_snapshots:
                active_paths[path_key] = matching_snapshots[-1]  # Get most recent snapshot
        return active_paths
    
    def print_path_summary(self, path_key: Optional[str] = None):
        """Print summary of paths, optionally filtered by path_key."""
        paths_to_print = {path_key: self.paths[path_key]} if path_key else self.paths
        
        for key, snapshots in paths_to_print.items():
            print(f"\nPath: {key}")
            print("-" * 50)
            for snapshot in snapshots:
                print(snapshot)
            print("-" * 50) 