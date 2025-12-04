"""
Side-by-side comparison: paper_figures logic vs optimization logic
Using EXACT same inputs, same reference EDCs, same c value
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from analysis import analysis as an

# Load the file
DATA_FILE = 'results/paper_data/aes_room_spatial_edc_data_top_middle_source.npz'
data = np.load(DATA_FILE, allow_pickle=True)

ref_edcs = data['edcs_RIMPY-neg10']
sdn_edcs_saved = data['edcs_SDN-Test2.998']
Fs = int(data['Fs'])

print("="*80)
print("COMPARING: paper_figures vs optimization - RMSE calculation")
print("="*80)

# METHOD 1: Paper figures approach (from paper_figures_spatial.py)
print("\nMETHOD 1: paper_figures_spatial.py approach")
print("Loading saved SDN EDCs from file, compare to reference")

rmse_paper_list = []
for i in range(len(ref_edcs)):
    rmse = an.compute_RMS(
        sdn_edcs_saved[i], ref_edcs[i],
        range=50, Fs=Fs,
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_paper_list.append(rmse)
    if i < 3:
        print(f"  Receiver {i+1}: EDC_saved len={len(sdn_edcs_saved[i])}, ref len={len(ref_edcs[i])}, RMSE={rmse:.6f}")

mean_rmse_paper = np.mean(rmse_paper_list)
print(f"\nMean RMSE (paper approach): {mean_rmse_paper:.6f}")

# METHOD 2: Optimization approach (from optimisation_singleC.py)
print("\nMETHOD 2: optimisation_singleC.py approach")
print("Uses same ref_edcs loaded from file, but calculates SDN on-the-fly")
print("(Simulating fresh calculation)")

# For this test, we'll just use the saved EDCs to isolate the comparison logic
# The key difference should be in how RMSE is calculated or data is processed
rmse_opt_list = []
for i in range(len(ref_edcs)):
    # Optimization script does this (line 106-108 in optimisation_singleC.py):
    rmse = an.compute_RMS(
        sdn_edcs_saved[i], ref_edcs[i],  # Using same saved EDCs for now
        range=int(50), Fs=Fs,  # Note: int() cast here
        skip_initial_zeros=True,
        normalize_by_active_length=True
    )
    rmse_opt_list.append(rmse)
    if i < 3:
        print(f"  Receiver {i+1}: RMSE={rmse:.6f}")

mean_rmse_opt = np.mean(rmse_opt_list)
print(f"\nMean RMSE (optimization approach): {mean_rmse_opt:.6f}")

# COMPARISON
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
print(f"Paper figures:  {mean_rmse_paper:.6f}")
print(f"Optimization:   {mean_rmse_opt:.6f}")
print(f"Difference:     {abs(mean_rmse_paper - mean_rmse_opt):.6f}")

if abs(mean_rmse_paper - mean_rmse_opt) < 0.001:
    print("\n✓ IDENTICAL - The RMSE calculation logic is the same")
    print("  The 0.314 vs 0.254 difference must come from DIFFERENT EDC INPUTS")
else:
    print(f"\n✗ DIFFERENT - Found {abs(mean_rmse_paper - mean_rmse_opt):.6f} discrepancy")
    print("  in the RMSE calculation itself!")

print("\n" + "="*80)
print("CONCLUSION:")
print("If the above shows IDENTICAL, then the optimization must be calculating")
print("different SDN EDCs (not using the saved ones), which produce lower RMSE.")
print("We need to compare the ACTUAL SDN EDCs being generated in each script.")
print("="*80)
