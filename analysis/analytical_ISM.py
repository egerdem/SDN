import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to import experiment_configs and spatial_analysis
sys.path.insert(0, str(Path(__file__).parent.parent))
import experiment_configs as exp_cfg
from spatial_analysis import generate_source_positions

c = 343.0  # speed of sound (m/s)
import matplotlib
matplotlib.use('Qt5Agg')


def image_rir_times_and_amps(src, mic, L, max_order=5):
    """
    Compute image source distances and amplitudes for a rectangular room
    with rigid walls using a standard image-source construction.

    src, mic, L are 3D vectors (x,y,z).
    max_order controls the integer tile indices (n_x, n_y, n_z in [-max_order, max_order]).
    """
    src = np.array(src, dtype=float)
    mic = np.array(mic, dtype=float)
    L = np.array(L, dtype=float)

    times = []
    amps = []

    # integer lattice indices
    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                n = np.array([nx, ny, nz], dtype=float)

                # parity (reflection) bits in each dimension
                for qx in [0, 1]:
                    for qy in [0, 1]:
                        for qz in [0, 1]:
                            q = np.array([qx, qy, qz], dtype=float)
                            s = (-1.0) ** q  # +/-1 vector

                            # standard image-source coordinate
                            img = 2.0 * n * L + s * src # could also be + src and put s to the mic
                            R = np.linalg.norm(img - mic) # could be + s*mic
                            if R == 0:
                                continue

                            a = 1.0 / (4.0 * np.pi * R)
                            t = R / c
                            times.append(t)
                            amps.append(a)

    times = np.array(times)
    amps = np.array(amps)
    idx = np.argsort(times)
    return times[idx], amps[idx]

def analytical_edc_from_images(times, amps, tau_grid=None):
    """
    Analytical EDC for an image-source RIR:
        d(tau) ≈ sum_{k: t_k >= tau} a_k^2

    Parameters
    ----------
    times : array-like
        Arrival times t_k of each image source (seconds).
        Assumed 1D and not necessarily sorted.
    amps : array-like
        Corresponding amplitudes a_k = 1 / (4*pi*R_k).
    tau_grid : array-like or None
        Times tau at which to evaluate the EDC (seconds).
        If None, uses a copy of 'times' (sorted and unique).

    Returns
    -------
    tau_grid : np.ndarray
        Grid of times where d(tau) is evaluated (seconds).
    d_tau : np.ndarray
        Analytical EDC values d(tau) at tau_grid (same units as a_k^2).
    """
    times = np.asarray(times, dtype=float)
    amps = np.asarray(amps, dtype=float)

    # Sort by arrival time (earliest first)
    order = np.argsort(times)
    t_sorted = times[order]
    E_sorted = amps[order] ** 2  # a_k^2 = 1/(16*pi^2 R_k^2)

    # Backwards cumulative sum: for each index j,
    # cumE[j] = sum_{k >= j} E_sorted[k]
    cumE = np.cumsum(E_sorted[::-1])[::-1]

    # Default tau grid: just use the (sorted) arrival times
    if tau_grid is None:
        tau_grid = t_sorted.copy()
    tau_grid = np.asarray(tau_grid, dtype=float)

    # For each tau, find the first index j with t_sorted[j] >= tau
    # and pick cumE[j]. If tau is after all arrivals, EDC=0.
    idx = np.searchsorted(t_sorted, tau_grid, side="left")
    d_tau = np.zeros_like(tau_grid)

    valid = idx < len(cumE)
    d_tau[valid] = cumE[idx[valid]]

    return tau_grid, d_tau

def rir_from_impulses(times, amps, fs, rir_duration):
    """
    Build a discrete-time RIR from impulses at 'times' with amplitudes 'amps'.
    """
    n_samples = int(np.round(rir_duration * fs))
    h = np.zeros(n_samples)
    for t, a in zip(times, amps):
        if t >= rir_duration:
            continue
        n = int(np.round(t * fs))
        if 0 <= n < n_samples:
            h[n] += a
    return h


def edc_from_rir(h):
    """
    Schroeder EDC: backwards cumulative sum of h^2.
    """
    e = h ** 2
    edc = np.flip(np.cumsum(np.flip(e)))
    return edc


# --- Room and positions (from experiment_configs.py) ---

# Get active room configuration
active_room = exp_cfg.active_room

# Room dimensions (width, depth, height) in meters
L = (active_room['width'], active_room['depth'], active_room['height'])

# Microphone position (mic x, mic y, mic z)
mic = (active_room['mic x'], active_room['mic y'], active_room['mic z'])

# Get source positions using generate_source_positions with version v3
source_list = generate_source_positions(active_room, name="v3")

# Extract source positions from the list
# source_list contains tuples: (x, y, z, name)
sources = {}
for src_x, src_y, src_z, src_name in source_list:
    sources[src_name] = (src_x, src_y, src_z)

fs = exp_cfg.Fs
rir_duration = exp_cfg.duration  # Use duration from experiment_configs
n_samples = int(np.round(rir_duration * fs))
max_order = 10         # truncate image orders

rirs = {}
edcs = {}

plt.figure(figsize=(12, 8))
# subplot_idx = 1

for name, src in sources.items():
    times, amps = image_rir_times_and_amps(src, mic, L, max_order=max_order)
    
    # Diagnostic: Check for coincident arrivals
    sample_indices = np.round(times * fs).astype(int)
    unique_samples, counts = np.unique(sample_indices, return_counts=True)
    n_coincident = np.sum(counts > 1)
    max_coincident = np.max(counts)
    print(f"\n{name}:")
    print(f"  Total images: {len(times)}")
    print(f"  Samples with coincident arrivals: {n_coincident}")
    print(f"  Max images per sample: {max_coincident}")
    
    h = rir_from_impulses(times, amps, fs, rir_duration)
    d = edc_from_rir(h) # Schroeder EDC: backwards cumulative sum of h^2.
    d_norm = d / d[0]           # normalise for comparison
    rirs[name] = h
    edcs[name] = d_norm

    t_disc = np.arange(n_samples) / fs
    tau_grid, d_anal = analytical_edc_from_images(times, amps, tau_grid=t_disc)
    # Normalise both to 1 at tau=0 for easy comparison
    d_anal_norm = d_anal / d_anal[0]
    
    # Calculate relative difference
    rel_diff = np.abs(d_norm - d_anal_norm) / (d_anal_norm + 1e-20)
    mean_rel_diff = np.mean(rel_diff[d_anal_norm > 1e-3])  # Exclude tail
    print(f"  Mean relative difference: {mean_rel_diff:.4f}")

    # Create subplot for this source
    n_sources = len(sources)
    # plt.subplot(n_sources, 1, subplot_idx)
    
    plt.plot(t_disc * 1000, 10*np.log10(d_norm + 1e-20),
             label="Schroeder (discrete RIR)", linewidth=2, alpha=0.8)
    plt.plot(tau_grid * 1000, 10*np.log10(d_anal_norm + 1e-20),
             "--", label="Analytical IM EDC", linewidth=2, alpha=0.8)
    plt.xlabel("Time (ms)")
    plt.ylabel("EDC (dB, normalised)")
    plt.legend()
    plt.grid(True)
    plt.title(f"{name}: Analytical vs Discrete EDC (mean rel. diff: {mean_rel_diff:.4f})")
    # plt.xlim(0, min(200, rir_duration * 1000))  # Show first 200ms or full duration
    
    # subplot_idx += 1

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*60)
print("SUMMARY: Why the EDCs differ")
print("="*60)
print("1. Coherent vs Incoherent Summation:")
print("   - Discrete: Multiple images landing on same sample are")
print("     coherently summed: h[n] = a₁+a₂, then h[n]² = (a₁+a₂)²")
print("     This includes cross-terms: a₁² + a₂² + 2a₁a₂")
print("   - Analytical: d(τ) = Σa_k² (no cross-terms)")
print("\n2. Time Quantization:")
print("   - Discrete: Arrival times rounded to nearest sample")
print("   - Analytical: Uses exact continuous arrival times")
print("\n3. Expected: Discrete EDC ≥ Analytical EDC when cross-terms > 0")
print("="*60)
