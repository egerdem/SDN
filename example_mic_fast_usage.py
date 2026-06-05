"""
Example demonstrating mic-side fast mode in calculate_sdn_rir_fast()

This shows how to use the new use_fast_mic_method flag for efficient computation
when optimizing microphone-side parameters (mic_weighting or mic_weighting_vector).

Follows the same pattern as existing use_fast_method for source-side optimization.

IMPORTANT: 
- Fast method ONLY works with ONE-PARAMETER FAMILY patterns: [c, cn, cn, cn, cn]
- Use mic_weighting_vector (6-element) for fast mode, NOT collection_vector (arbitrary 5-element)
- collection_vector is for arbitrary patterns and is NOT compatible with fast method
"""

# Example 1: Mic-side SCALAR mode (optimizing mic_weighting)
config_mic_scalar = {
    'label': 'MicWeightingTest',
    'cache_label': 'example_mic_scalar',
    'use_fast_mic_method': True,  # Enable mic-side fast mode (NEW)
    'flags': {
        'specular_mic_pickup': True,  # Required for mic weighting
        'mic_weighting': 2.5,  # The parameter being optimized (can be any value)
        
        # Source-side parameters (if any) should be fixed
        'specular_source_injection': True,
        'source_weighting': 1.0,  # Fixed value
    }
}

# Example 2: Mic-side VECTOR mode (optimizing mic_weighting_vector)
config_mic_vector = {
    'label': 'MicWeightingVectorTest',
    'cache_label': 'example_mic_vector',
    'use_fast_mic_method': True,  # Enable mic-side fast mode (NEW)
    'flags': {
        'specular_mic_pickup': True,  # Required for mic weighting vector
        'mic_weighting_vector': [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],  # 6-element vector (one per wall)
        
        # Source-side parameters (if any) should be fixed
        'specular_source_injection': True,
        'source_weighting': 1.0,  # Fixed value
    }
}

# Example 3: Source-side SCALAR mode (LEGACY - existing behavior)
config_source_scalar = {
    'label': 'SourceWeightingTest',
    'cache_label': 'example_source_scalar',
    'use_fast_method': True,  # Enable source-side fast mode (LEGACY)
    'flags': {
        'specular_source_injection': True,
        'source_weighting': 2.5,  # The parameter being optimized
        
        # Mic-side parameters (if any) should be fixed
        'specular_mic_pickup': True,
        'mic_weighting': 1.0,  # Fixed value
    }
}

# Example 4: Source-side VECTOR mode (LEGACY - existing behavior)
config_source_vector = {
    'label': 'NodeWeightingTest',
    'cache_label': 'example_source_vector',
    'use_fast_method': True,  # Enable source-side fast mode (LEGACY)
    'flags': {
        'specular_source_injection': True,
        'node_weighting_vector': [2.0, 1.5, 1.0, 1.0, 0.5, 0.5],  # 6-element vector
        
        # Mic-side parameters (if any) should be fixed
        'specular_mic_pickup': True,
        'mic_weighting': 1.0,  # Fixed value
    }
}

"""
Usage with calculate_sdn_rir_fast():

from rir_calculators import calculate_sdn_rir_fast, enable_basis_disk_cache

# Optional: Enable disk caching for long optimization runs
enable_basis_disk_cache()

# Calculate RIR using mic-side fast mode
sdn, rir, label, is_default = calculate_sdn_rir_fast(
    room_parameters=room_params,
    test_name='OptimizeMicWeighting',
    room=room_object,
    duration=1.0,
    Fs=44100,
    config=config_mic_scalar  # or config_mic_vector
)

Key Points:
-----------
1. Backward Compatible: 
   - use_fast_method (existing) for source-side optimization
   - use_fast_mic_method (new) for mic-side optimization
   - Both default to False, preserving original behavior

2. Cache Efficiency:
   - Mic scalar mode: Computes 2 basis functions (mic_w=0, mic_w=1)
   - Mic vector mode: Computes 7 basis functions (mic_vec=[0,0,0,0,0,0] + 6 unit vectors)
   - Source scalar mode: Computes 2 basis functions (c=0, c=1)
   - Source vector mode: Computes 7 basis functions (c=[0,0,0,0,0,0] + 6 unit vectors)

3. Use Cases:
   - Use use_fast_mic_method=True when optimizing mic_weighting or mic_weighting_vector
   - Use use_fast_method=True when optimizing source_weighting or node_weighting_vector
   - Never use both fast methods simultaneously (enforced by assertion)
   - NEVER use collection_vector with fast method (it's arbitrary, not one-parameter family)

4. Requirements:
   - For use_fast_mic_method=True, specular_mic_pickup must be True in flags
   - For mic vector mode, mic_weighting_vector must be a 6-element list/array (one per wall)
   - For mic scalar mode, mic_weighting can be any float
   - collection_vector (5-element arbitrary) is NOT compatible with fast method

5. Validation Pattern (as mentioned by user):
   Compare results with and without fast method to verify correctness:
   
   # Config WITH fast method
   config_fast = {
       'use_fast_mic_method': True,
       'flags': {'specular_mic_pickup': True, 'mic_weighting': 2.5}
   }
   
   # Config WITHOUT fast method (ground truth)
   config_standard = {
       'use_fast_mic_method': False,  # or omit
       'flags': {'specular_mic_pickup': True, 'mic_weighting': 2.5}
   }
   
   # Both should produce identical RIRs (within numerical precision)
"""

