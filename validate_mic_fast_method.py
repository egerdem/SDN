"""
Validation script to compare use_fast_mic_method=True vs False

This verifies that the mic-side fast method produces identical results
to the standard method (within numerical precision).
"""

import numpy as np
from rir_calculators import calculate_sdn_rir, calculate_sdn_rir_fast
import geometry
import experiment_configs as exp_config

# Add your room setup imports here

def validate_mic_scalar_mode(room_parameters, room, duration=1.0, Fs=44100):
    """
    Validate mic scalar mode: mic_weighting with use_fast_mic_method
    """
    print("\n" + "="*70)
    print("VALIDATING MIC SCALAR MODE (mic_weighting)")
    print("="*70)
    
    test_mic_weightings = [0.0, 1.0, 2.5, 5.0]
    
    for mic_w in test_mic_weightings:
        print(f"\nTesting mic_weighting = {mic_w}")
        
        # Standard method (ground truth)
        config_standard = {
            'label': 'Standard',
            'flags': {
                'specular_mic_pickup': True,
                'mic_weighting': mic_w,
            }
        }
        _, rir_standard, _, _ = calculate_sdn_rir(
            room_parameters, 'Standard', room, duration, Fs, config_standard
        )
        
        # Fast method
        config_fast = {
            'label': 'Fast',
            'use_fast_mic_method': True,
            'flags': {
                'specular_mic_pickup': True,
                'mic_weighting': mic_w,
            }
        }
        _, rir_fast, _, _ = calculate_sdn_rir_fast(
            room_parameters, 'Fast', room, duration, Fs, config_fast
        )
        
        # Compare
        max_diff = np.max(np.abs(rir_standard - rir_fast))
        correlation = np.corrcoef(rir_standard, rir_fast)[0, 1]
        
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Correlation: {correlation:.15f}")
        
        if max_diff < 1e-10 and correlation > 0.999999:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL - Results differ!")

def validate_mic_vector_mode(room_parameters, room, duration=1.0, Fs=44100):
    """
    Validate mic vector mode: mic_weighting_vector with use_fast_mic_method
    """
    print("\n" + "="*70)
    print("VALIDATING MIC VECTOR MODE (mic_weighting_vector)")
    print("="*70)
    
    test_mic_weighting_vectors = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    for mic_vec in test_mic_weighting_vectors:
        print(f"\nTesting mic_weighting_vector = {mic_vec}")
        
        # Standard method (ground truth)
        config_standard = {
            'label': 'Standard',
            'flags': {
                'specular_mic_pickup': True,
                'mic_weighting_vector': mic_vec,
            }
        }
        _, rir_standard, _, _ = calculate_sdn_rir(
            room_parameters, 'Standard', room, duration, Fs, config_standard
        )
        
        # Fast method
        config_fast = {
            'label': 'Fast',
            'use_fast_mic_method': True,
            'flags': {
                'specular_mic_pickup': True,
                'mic_weighting_vector': mic_vec,
            }
        }
        _, rir_fast, _, _ = calculate_sdn_rir_fast(
            room_parameters, 'Fast', room, duration, Fs, config_fast
        )
        
        # Compare
        max_diff = np.max(np.abs(rir_standard - rir_fast))
        correlation = np.corrcoef(rir_standard, rir_fast)[0, 1]
        
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Correlation: {correlation:.15f}")
        
        if max_diff < 1e-10 and correlation > 0.999999:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL - Results differ!")

def compare_source_vs_mic_fast():
    """
    Demonstrate that source and mic fast methods work independently
    """
    print("\n" + "="*70)
    print("COMPARING SOURCE-SIDE vs MIC-SIDE FAST METHODS")
    print("="*70)
    
    print("\nBoth should work independently when optimizing different parameters")
    print("Source-side: use_fast_method=True with source_weighting")
    print("Mic-side: use_fast_mic_method=True with mic_weighting")

if __name__ == "__main__":
    # Parameters
    duration = 1  # seconds
    duration_in_ms = 1000 * duration  # Convert to milliseconds

    Fs = 44100
    num_samples = int(Fs * duration)
    rirs = {}
    default_rirs = set()  # Track which RIRs should be black
    rirs_analysis = {}
    # Assign source signals
    impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)

    # Set up your room parameters here
    room_parameters = exp_config.active_room
    room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
    room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
    room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                    signal="will be replaced", Fs=Fs)

    room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

    # Setup signal
    room.source.signal = impulse_dirac['signal']
    # room.source.signal = impulse_gaussian['signal']

    # Calculate reflection coefficient - will be overwritten for each method
    room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
    room.wallAttenuation = [room_parameters['reflection']] * 6
    
    # Uncomment to run validation:
    validate_mic_scalar_mode(room_parameters, room)
    validate_mic_vector_mode(room_parameters, room)
    compare_source_vs_mic_fast()

