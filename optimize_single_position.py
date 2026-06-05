"""
Simple optimization for a single room/receiver position.
Two-stage: Stage 1 (optimize c), Stage 2 (optimize m)
"""
import numpy as np
from scipy.optimize import minimize_scalar
import geometry
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir_fast, calculate_rimpy_rir, rir_normalisation, enable_basis_disk_cache

# === CONFIGURATION ===
room_parameters = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 1, 'source y': 1, 'source z': 2,
    'mic x': 0.5, 'mic y': 0.5, 'mic z': 1.5,
    'absorption': 0.2,
}

duration = 1.0
Fs = 44100
err_duration_ms = 50

# Optimization settings
OPTIMIZE_SOURCE = True   # Stage 1
OPTIMIZE_MIC = True      # Stage 2
BOUNDS = (1.0, 7.0)

# Enable disk caching
enable_basis_disk_cache()

# Set random seed for reproducibility
np.random.seed(42)

# === SETUP ===
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
num_samples = int(Fs * duration)
impulse = geometry.Source.generate_signal('dirac', num_samples)
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal=impulse['signal'], Fs=Fs)
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# === CALCULATE REFERENCE ===
print("Calculating reference ISM...")
rir_ref, _ = calculate_rimpy_rir(room_parameters, duration, Fs, reflection_sign=-1, 
                                  tw_fractional_delay_length=0, randDist=0.1, rand_seed=42)
rir_ref_normed = rir_normalisation(rir_ref, room, Fs, normalize_to_first_impulse=True)['single_rir']
edc_ref, _, _ = an.compute_edc(rir_ref_normed, Fs, plot=False)

# === OBJECTIVE FUNCTION ===
def compute_rmse(c_val, mic_val=None):
    c = np.squeeze(c_val).item()
    
    cfg = {
        'enabled': True,
        'info': 'opt',
        'flags': {'specular_source_injection': True, 'source_weighting': c},
        'label': 'SDN-Opt'
    }
    
    if mic_val is not None:
        m = np.squeeze(mic_val).item()
        cfg['flags']['specular_mic_pickup'] = True
        cfg['flags']['mic_weighting'] = m
        cfg['use_fast_mic_method'] = True
        cfg['use_fast_method'] = False
        print(f"  Evaluating c={c:.4f}, m={m:.4f}", end="")
        print(f" [DEBUG: source_weighting={cfg['flags'].get('source_weighting', 'MISSING')}, specular_source_injection={cfg['flags'].get('specular_source_injection', 'MISSING')}]", end="")
    else:
        cfg['use_fast_method'] = True
        print(f"  Evaluating c={c:.4f}", end="")
    
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "SDN-Opt", room, duration, Fs, cfg)
    rir_sdn_normed = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)['single_rir']
    edc_sdn, _, _ = an.compute_edc(rir_sdn_normed, Fs, plot=False)
    
    rmse = an.compute_RMS(edc_sdn, edc_ref, range=int(err_duration_ms), Fs=Fs,
                          skip_initial_zeros=True, normalize_by_active_length=True)
    print(f" → RMSE = {rmse:.6f}")
    return rmse

# === BASELINE ===
print("\n" + "="*60)
print("BASELINE: c=1.0, m=1.0")
print("="*60)
baseline_rmse = compute_rmse(1.0)
print(f"Baseline RMSE: {baseline_rmse:.6f}")

# === STAGE 1: OPTIMIZE SOURCE ===
optimal_c = 1.0
stage1_rmse = baseline_rmse

if OPTIMIZE_SOURCE:
    print("\n" + "="*60)
    print("STAGE 1: Optimizing source_weighting (c)")
    print("="*60)
    res = minimize_scalar(compute_rmse, bounds=BOUNDS, method='bounded', 
                          options={'xatol': 1e-3, 'maxiter': 20})
    optimal_c = res.x
    stage1_rmse = res.fun
    print(f"\nStage 1 Result: c = {optimal_c:.4f}, RMSE = {stage1_rmse:.6f}")
    print(f"Improvement: {baseline_rmse - stage1_rmse:.6f} ({100*(baseline_rmse - stage1_rmse)/baseline_rmse:.2f}%)")
else:
    print("\nStage 1: SKIPPED (using c=1.0)")

# === STAGE 2: OPTIMIZE MIC ===
optimal_m = 1.0
stage2_rmse = stage1_rmse

if OPTIMIZE_MIC:
    print("\n" + "="*60)
    print(f"STAGE 2: Optimizing mic_weighting (m) with c={optimal_c:.4f} fixed")
    print("="*60)
    
    def mic_obj(m_val):
        return compute_rmse(optimal_c, m_val)
    
    res = minimize_scalar(mic_obj, bounds=BOUNDS, method='bounded',
                          options={'xatol': 1e-3, 'maxiter': 20})
    optimal_m = res.x
    stage2_rmse = res.fun
    print(f"\nStage 2 Result: m = {optimal_m:.4f}, RMSE = {stage2_rmse:.6f}")
    print(f"Improvement over Stage 1: {stage1_rmse - stage2_rmse:.6f} ({100*(stage1_rmse - stage2_rmse)/stage1_rmse:.2f}%)")
    print(f"Total improvement: {baseline_rmse - stage2_rmse:.6f} ({100*(baseline_rmse - stage2_rmse)/baseline_rmse:.2f}%)")
else:
    print("\nStage 2: SKIPPED (using m=1.0)")

# === SUMMARY ===
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Optimal c: {optimal_c:.4f}")
print(f"Optimal m: {optimal_m:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.6f}")
if OPTIMIZE_SOURCE:
    print(f"Stage 1 RMSE:  {stage1_rmse:.6f}")
if OPTIMIZE_MIC:
    print(f"Stage 2 RMSE:  {stage2_rmse:.6f}")
print("="*60)

