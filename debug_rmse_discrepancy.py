
import os
import numpy as np
import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, enable_basis_disk_cache

# Path to data
data_path = "results/paper_data/aes_room_spatial_edc_data_center_source.npz"
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")

print(f"Loading {data_path}...")
with np.load(data_path, allow_pickle=True) as data:
    room_params = data["room_params"][0]
    source_pos = data["source_pos"]
    receiver_positions = data["receiver_positions"]
    Fs = int(data["Fs"])
    duration = float(data["duration"])

    # Load Reference EDCs
    if "edcs_RIMPY-neg10" in data:
        ref_edcs = data["edcs_RIMPY-neg10"]
        print("Loaded ref_edcs (RIMPY-neg10)")
    else:
        print("RIMPY-neg10 EDCs not found in file")

    # Load Saved SDN EDCs
    if "edcs_SDN-Test2.998" in data:
        saved_sdn_edcs = data["edcs_SDN-Test2.998"]
        print("Loaded saved_sdn_edcs (SDN-Test2.998)")
    else:
        print("SDN-Test2.998 EDCs not found in file")
        saved_sdn_edcs = None

# 1. Compute RMSE using SAVED data
if saved_sdn_edcs is not None:
    print("\n--- RMSE from SAVED Data ---")
    total_rmse = 0
    for i in range(len(receiver_positions)):
        rmse = an.compute_RMS(
            saved_sdn_edcs[i], ref_edcs[i],
            range=50, Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True
        )
        total_rmse += rmse
        # print(f"Receiver {i}: {rmse:.6f}")
    print(f"Mean RMSE (Saved): {total_rmse / len(receiver_positions):.6f}")

# 2. Compute RMSE using ON-THE-FLY calculation
print("\n--- RMSE from ON-THE-FLY Calculation (Fast Method) ---")
enable_basis_disk_cache()

room = geometry.Room(room_params["width"], room_params["depth"], room_params["height"])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal("dirac", num_samples)
room.set_source(source_pos[0], source_pos[1], source_pos[2], signal=impulse['signal'], Fs=Fs)
room_params["reflection"] = np.sqrt(1 - room_params["absorption"])
room.wallAttenuation = [room_params["reflection"]] * 6

cfg = {
    "flags": {
        "specular_source_injection": True,
        "source_weighting": 2.998
    },
    "label": "Debug",
    "use_fast_method": True,
    "cache_label": "debug_center"
}

total_rmse_otf = 0
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params["mic z"])
    cfg['cache_label'] = f"Center Source_rx{i:02d}" # Match cache key format if possible

    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_params, "SDN-Opt", room, duration, Fs, cfg)

    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)

    rmse = an.compute_RMS(
        edc_sdn, ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    total_rmse_otf += rmse
    # print(f"Receiver {i}: {rmse:.6f}")

print(f"Mean RMSE (On-the-fly Fast): {total_rmse_otf / len(receiver_positions):.6f}")

# 3. Compute RMSE using ON-THE-FLY Calculation (Standard Method)
print("\n--- RMSE from ON-THE-FLY Calculation (Standard Method) ---")
from rir_calculators import calculate_sdn_rir

total_rmse_std = 0
for i, (rx, ry) in enumerate(receiver_positions):
    room.set_microphone(rx, ry, room_params["mic z"])

    # Standard method doesn't need cache label or use_fast_method
    cfg_std = {
        "flags": {
            "specular_source_injection": True,
            "source_weighting": 2.998
        },
        "label": "DebugStd"
    }

    _, rir_sdn, _, _ = calculate_sdn_rir(room_params, "SDN-Opt", room, duration, Fs, cfg_std)

    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)

    rmse = an.compute_RMS(
        edc_sdn, ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    total_rmse_std += rmse

print(f"Mean RMSE (On-the-fly Standard): {total_rmse_std / len(receiver_positions):.6f}")

# 4. Compare RIRs if available
print(f"Available keys in data: {data.files}")
if "rirs" in data:
    # Load the dictionary from the 0-d array
    rirs_dict = data["rirs"].item()
    if "rirs_SDN-Test2.998" in rirs_dict:
        saved_rirs = rirs_dict["rirs_SDN-Test2.998"]
        print("\n--- RIR Comparison ---")

        # Compare first receiver RIR
        saved_rir_0 = saved_rirs[0]

        # Re-calculate on-the-fly RIR for first receiver
        room.set_microphone(receiver_positions[0][0], receiver_positions[0][1], room_params["mic z"])
        cfg['cache_label'] = f"Center Source_rx00"
        _, rir_sdn_otf, _, _ = calculate_sdn_rir_fast(room_params, "SDN-Opt", room, duration, Fs, cfg)

        # Normalize OTF RIR
        rir_sdn_otf_norm = rir_normalisation(rir_sdn_otf, room, Fs, normalize_to_first_impulse=True)["single_rir"]

        # Compare
        # Ensure lengths match
        min_len = min(len(saved_rir_0), len(rir_sdn_otf_norm))
        diff = saved_rir_0[:min_len] - rir_sdn_otf_norm[:min_len]
        rir_rmse = np.sqrt(np.mean(diff**2))
        max_diff = np.max(np.abs(diff))

        print(f"RIR RMSE (Rx 0): {rir_rmse:.8f}")
        print(f"Max RIR Diff (Rx 0): {max_diff:.8f}")

        if max_diff > 1e-6:
            print("WARNING: RIRs are significantly different!")
            print(f"Saved RIR max: {np.max(np.abs(saved_rir_0))}")
            print(f"OTF RIR max: {np.max(np.abs(rir_sdn_otf_norm))}")
    else:
        print("SDN-Test2.998 RIRs not found in file")
else:
    print("RIRs dictionary not found in file")

