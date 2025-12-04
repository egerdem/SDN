import numpy as np
import geometry
from rir_calculators import calculate_sdn_rir, rir_normalisation
from copy import deepcopy

# Setup from center source data
room_dims = [7.0, 5.0, 3.0]
source_pos = [2.0, 2.0, 1.5]  # Center
mic_pos = [0.5, 0.5, 1.5]  # First receiver
Fs = 44100
duration = 1.0
absorption = 0.2

room = geometry.Room(room_dims[0], room_dims[1], room_dims[2])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal("dirac", num_samples)
room.set_source(source_pos[0], source_pos[1], source_pos[2], signal=impulse['signal'], Fs=Fs)
room.set_microphone(mic_pos[0], mic_pos[1], mic_pos[2])

room_params = {
    "width": room_dims[0], 
    "depth": room_dims[1], 
    "height": room_dims[2], 
    "absorption": absorption,
    "reflection": np.sqrt(1 - absorption)
}
room.wallAttenuation = [room_params["reflection"]] * 6

base_cfg = {
    "flags": {
        "specular_source_injection": True,
    },
    "label": "Test"
}

# Calculate RIRs for c=0, c=1, c=2.998
results = {}
for c_val in [0.0, 1.0, 2.998]:
    cfg = deepcopy(base_cfg)
    cfg['flags']['source_weighting'] = c_val
    cfg['label'] = f"c={c_val}"
    
    _, rir, _, _ = calculate_sdn_rir(room_params, f"Test_{c_val}", room, duration, Fs, cfg)
    
    # Normalize
    rir_norm = rir_normalisation(rir, room, Fs, normalize_to_first_impulse=True)["single_rir"]
    results[c_val] = rir_norm
    
    print(f"c={c_val}: max={np.max(np.abs(rir_norm)):.6f}, samples={len(rir_norm)}")

# Test linearity: RIR(2.998) should equal RIR(0) + 2.998 * (RIR(1) - RIR(0))
rir_0 = results[0.0]
rir_1 = results[1.0]
rir_actual = results[2.998]

# Fast method prediction
rir_shape = rir_1 - rir_0
rir_predicted = rir_0 + 2.998 * rir_shape

# Compare
min_len = min(len(rir_actual), len(rir_predicted))
diff = rir_actual[:min_len] - rir_predicted[:min_len]
rmse = np.sqrt(np.mean(diff**2))
max_diff = np.max(np.abs(diff))

print(f"\nLinearity Test:")
print(f"RIR RMSE (actual vs predicted): {rmse:.10f}")
print(f"Max difference: {max_diff:.10f}")
print(f"Relative error: {rmse / np.max(np.abs(rir_actual)):.10f}")

if rmse > 1e-10:
    print("\nWARNING: Fast method assumption is INVALID!")
    print("The RIR is NOT linear in c for this configuration.")
    
    # Find where the biggest differences are
    abs_diff = np.abs(diff)
    top_indices = np.argsort(abs_diff)[-10:][::-1]
    print("\nTop 10 samples with largest differences:")
    for idx in top_indices:
        print(f"  Sample {idx}: actual={rir_actual[idx]:.10f}, predicted={rir_predicted[idx]:.10f}, diff={diff[idx]:.10f}")
else:
    print("\nFast method assumption is VALID (within numerical precision)")
