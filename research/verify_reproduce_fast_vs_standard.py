
import numpy as np
import geometry
from rir_calculators import calculate_sdn_rir, calculate_sdn_rir_fast, rir_normalisation
from analysis import analysis as an

def fast_vs_standard():
    # Setup basic room
    room_dims = [7.0, 5.0, 3.0]
    source_pos = [2.0, 2.0, 1.5]
    mic_pos = [5.0, 3.0, 1.5]
    Fs = 44100
    duration = 0.5
    absorption = 0.2
    
    room = geometry.Room(room_dims[0], room_dims[1], room_dims[2])
    num_samples = int(Fs * duration)
    impulse = geometry.Source.generate_signal("dirac", num_samples)
    room.set_source(source_pos[0], source_pos[1], source_pos[2], signal=impulse['signal'], Fs=Fs)
    room.set_microphone(mic_pos[0], mic_pos[1], mic_pos[2])
    room.wallAttenuation = [np.sqrt(1 - absorption)] * 6
    
    # Config
    c_val = 3.0
    cfg = {
        "flags": {
            "specular_source_injection": True,
            "source_weighting": c_val
        },
        "label": "Test"
    }
    
    # Standard Method
    print("Calculating Standard RIR...")
    room_params = {"width": room_dims[0], "depth": room_dims[1], "height": room_dims[2], "absorption": absorption}
    room_params['reflection'] = np.sqrt(1 - absorption)
    
    _, rir_std, _, _ = calculate_sdn_rir(
        room_params,
        "Standard", room, duration, Fs, cfg
    )
    
    # Fast Method
    print("Calculating Fast RIR...")
    cfg_fast = cfg.copy()
    cfg_fast["use_fast_method"] = True
    cfg_fast["cache_label"] = "test_comparison"
    
    _, rir_fast, _, _ = calculate_sdn_rir_fast(
        room_params,
        "Fast", room, duration, Fs, cfg_fast
    )
    
    # Normalize both
    rir_std_norm = rir_normalisation(rir_std, room, Fs, normalize_to_first_impulse=True)["single_rir"]
    rir_fast_norm = rir_normalisation(rir_fast, room, Fs, normalize_to_first_impulse=True)["single_rir"]
    
    # Compare
    diff = rir_std_norm - rir_fast_norm
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    
    print("\nComparison (c=" + str(c_val) + "):")
    print("RMSE between Standard and Fast: " + str(rmse))
    print("Max Difference: " + str(max_diff))
    
    # Compare EDCs
    edc_std, _, _ = an.compute_edc(rir_std_norm, Fs, plot=False)
    edc_fast, _, _ = an.compute_edc(rir_fast_norm, Fs, plot=False)
    
    edc_rmse = an.compute_RMS(edc_std, edc_fast, range=50, Fs=Fs, skip_initial_zeros=True)
    print("EDC RMSE (50ms): " + str(edc_rmse))

if __name__ == "__main__":
    fast_vs_standard()
