# SDN Manager Bug Fixes Summary

## 1. Source-Microphone Position Dictionary Bug

### Original Issue
The `experiments_by_position` dictionary in the `Room` class was incorrectly handling source-microphone position pairs, leading to:
- Only 4 positions being tracked instead of the expected 16
- Duplicate experiments being added for the same position
- Inconsistent position tracking between batch and singular cases

### Root Causes
1. **Position Extraction Inconsistency**:
   - Batch experiments stored positions in `receiver['position']` and `source['position']`
   - Singular experiments stored positions in `room_parameters` (`source x`, `source y`, etc.)
   - The `add_experiment` method only checked `room_parameters`, missing batch format positions

2. **Duplicate Handling**:
   - No proper check for existing experiments at a position
   - No update mechanism for existing experiments
   - Missing position key standardization between formats

### Solution
1. **Unified Position Extraction**:
```python
# Check both formats for positions
receiver_info = experiment.config.get('receiver', {})
if receiver_info and 'position' in receiver_info:
    mic_pos = receiver_info['position']
else:
    mic_pos = [
        room_params.get('mic x', 0),
        room_params.get('mic y', 0),
        room_params.get('mic z', 0)
    ]
```

2. **Proper Duplicate Handling**:
```python
if experiment.experiment_id in self.experiments:
    # Update in both dictionaries
    self.experiments[experiment.experiment_id] = experiment
    for pos_list in self.experiments_by_position.values():
        for i, exp in enumerate(pos_list):
            if exp.experiment_id == experiment.experiment_id:
                pos_list[i] = experiment
```

## 2. SDN Folder Saving Disappearance

### Original Issue
The SDN experiment folders were not being saved after fixing the position tracking bug, indicating a regression in the saving functionality.

### Root Cause
The separation of singular and batch cases in `load_experiments` affected the saving logic:
- When we fixed the position tracking, we inadvertently changed how experiments were being saved
- The batch/singular separation in loading needed corresponding changes in saving
- The `save_experiment` method wasn't properly handling both cases

### Solution
1. **Consistent Directory Structure**:
```python
def _get_room_dir(self, room_name):
    """Get the directory path for a room."""
    if not self.is_batch_manager:
        return os.path.join(self.results_dir, 'room_singulars', room_name)
    else:
        return os.path.join(self.results_dir, 'rooms', room_name)
```

2. **Separate Save Logic**:
```python
def save_experiment(self, experiment, room_name):
    if not self.is_batch_manager:
        # Save in flat structure for singular experiments
        room_dir = self._get_room_dir(room_name)
        os.makedirs(room_dir, exist_ok=True)
        # Save metadata and RIR...
    else:
        # Save in structured directories for batch experiments
        # Create source/method/param directories...
```

## Key Learnings

1. **Data Structure Consistency**:
   - Maintain consistent position tracking across different experiment types
   - Use standardized keys and formats for position storage
   - Handle both batch and singular cases uniformly

2. **File System Organization**:
   - Keep clear separation between batch and singular experiment storage
   - Maintain consistent directory structure
   - Ensure saving logic matches loading logic

3. **Experiment Management**:
   - Properly handle experiment updates and duplicates
   - Maintain data integrity across different storage formats
   - Keep position tracking synchronized with experiment storage

## Impact on Load-Only Implementation

The load-only version (`sdn_manager_load_sims.py`) needs similar fixes for:
1. Position tracking consistency
2. Duplicate experiment handling
3. Proper directory structure handling

However, it can omit the saving-related fixes since it's focused only on loading functionality. 