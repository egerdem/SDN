import numpy as np
import os
from scipy import signal
import analysis as an
import plot_room as pp
import geometry
from rir_calculators import calculate_ho_sdn_rir, rir_normalisation
from plotting_utils import load_data, get_display_name, DISPLAY_NAME_MAP

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

# ----------------------------------------------------------------------
# ---------- 1. Log–spectral distance with band control ----------------
# ----------------------------------------------------------------------

def compute_LSD(rir1: np.ndarray,
                rir2: np.ndarray,
                Fs: int = 44100,
                nfft: int = 4096,
                f_max: float | None = None,
                third_oct: bool = False) -> float:
    """
    Log-spectral distance between two RIRs.

    Parameters
    ----------
    rir1, rir2 : np.ndarray
        Impulse responses (need not be same length).
    Fs : int
        Sample-rate (Hz).
    nfft : int
        FFT length (power of two recommended).
    f_max : float or None
        If given, LSD is evaluated only up to *f_max* Hz
        (use 4000 for the "LP-4 kHz" condition).
    third_oct : bool
        If True, return the **mean of 1/3-octave-band LSDs**
        instead of the raw bin-by-bin average.

    Returns
    -------
    float
        LSD in decibels.
    """
    # --- length equalisation & window -------------------------------

    # --- ensure 1-D float arrays ----------------------
    rir1 = np.asarray(rir1, dtype=float).ravel()
    rir2 = np.asarray(rir2, dtype=float).ravel()
    assert rir1.ndim == 1 and rir2.ndim == 1, "RIRs must be 1-D."

    L = max(len(rir1), len(rir2))
    w = hann(L, sym=False)
    x1 = np.pad(rir1, (0, L - len(rir1))) * w
    x2 = np.pad(rir2, (0, L - len(rir2))) * w

    # --- spectra (positive freqs) -----------------------------------
    H1 = np.fft.rfft(x1, n=nfft)
    H2 = np.fft.rfft(x2, n=nfft)
    mag1 = np.abs(H1) + 1e-12
    mag2 = np.abs(H2) + 1e-12
    freqs = np.fft.rfftfreq(nfft, 1 / Fs)

    # optional band-limit
    if f_max is not None:
        keep = freqs <= f_max
        mag1, mag2, freqs = mag1[keep], mag2[keep], freqs[keep]

    # --- bin-wise log error ----------------------------------------
    log_err = 20 * np.log10(mag1 / mag2)

    if not third_oct:
        return float(np.sqrt(np.mean(log_err ** 2)))

    # --- 1/3-octave averaging --------------------------------------
    # ISO centre frequencies from 100 Hz to Nyquist
    cfs = 100 * (2 ** (np.arange(0, 60) / 3))
    cfs = cfs[cfs < freqs[-1] * 0.9]

    band_err = []
    for fc in cfs:
        fl = fc / 2 ** (1 / 6)
        fu = fc * 2 ** (1 / 6)
        idx = (freqs >= fl) & (freqs <= fu)
        if idx.any():
            band_err.append(np.sqrt(np.mean(log_err[idx] ** 2)))
    return float(np.mean(band_err))


# ----------------------------------------------------------------------
# ---------- 2. Plot with third-octave LSD overlay --------------------
# ----------------------------------------------------------------------

def plot_spectral_comparison(rir1: np.ndarray,
                             rir2: np.ndarray,
                             Fs: int = 44_100,
                             label1: str = "RIR 1",
                             label2: str = "RIR 2",
                             nfft: int = 4096,
                             f_max_plot: float | None = None,
                             **kwargs):
    """
    Plot (i) raw FFT magnitude, (ii) optionally 1/3-octave smoothed magnitude,
    and (iii) the dB-difference + per-bin LSD.

    Extra keyword:
    ----------------
    octave : bool   (default False)
        If True draw an additional 1/3-octave trace on both spectra.
    """
    # ---------- parameters ----------------------------
    use_oct   = kwargs.get("octave", False)          # <-- NEW FLAG
    eps       = 1e-12

    # ---------- spectra (windowed FFT) ----------------
    rir1 = np.asarray(rir1, float).ravel()
    rir2 = np.asarray(rir2, float).ravel()
    L    = max(len(rir1), len(rir2))
    w    = hann(L, sym=False)

    H1   = np.fft.rfft(np.pad(rir1, (0, L-len(rir1))) * w, n=nfft)
    H2   = np.fft.rfft(np.pad(rir2, (0, L-len(rir2))) * w, n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1/Fs)

    if f_max_plot:                     # optional band-limit for plotting
        keep      = freqs <= f_max_plot
        freqs, H1, H2 = freqs[keep], H1[keep], H2[keep]

    mag1_db = 20*np.log10(np.abs(H1)+eps)
    mag2_db = 20*np.log10(np.abs(H2)+eps)
    diff_db = mag1_db - mag2_db

    # ---------- figure --------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].semilogx(freqs, mag1_db, label=label1, lw=0.9)
    ax[0].semilogx(freqs, mag2_db, label=label2, lw=0.9)

    # ---- 1∕3-octave smoothing (optional) -------------
    if use_oct:
        # centre bands (ISO): ... 125-250-500-1k-2k-4k-8k -- stop at Nyquist
        f_iso = np.array([31.5, 63, 125, 250, 500, 1000,
                          2000, 4000, 8000, 16000])
        f_iso = f_iso[f_iso < Fs/2]
        bw    = 2**(1/6)                    # 1/3-oct half-band ratio

        def octave_mean(H, f_c):
            f_lo, f_hi = f_c/bw, f_c*bw
            idx        = (freqs >= f_lo) & (freqs <= f_hi)
            return 20*np.log10(np.mean(np.abs(H[idx])+eps))

        oct1 = [octave_mean(H1, fc) for fc in f_iso]
        oct2 = [octave_mean(H2, fc) for fc in f_iso]
        ax[0].semilogx(f_iso, oct1, 'o-', lw=2, label=f"{label1} (1/3-oct)")
        ax[0].semilogx(f_iso, oct2, 'o-', lw=2, label=f"{label2} (1/3-oct)")

    ax[0].set_ylabel("Magnitude (dB)")
    ax[0].grid(True, which='both')
    ax[0].legend()

    # ---- per-bin LSD (full-band unless f_max_plot is set) --------------
    lsd_bin = 20*np.sqrt(np.mean((np.log10((np.abs(H1)+eps)/(np.abs(H2)+eps)))**2))
    ax[0].set_title(f"Full-band LSD = {lsd_bin:.2f} dB")

    # ---- difference panel ---------------------------------------------
    ax[1].semilogx(freqs, diff_db, color='k', lw=0.8)
    ax[1].set_ylabel("Difference (dB)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].grid(True, which='both')

    plt.tight_layout()
    plt.show()
    return lsd_bin


def plot_spectral_comparison_multi(rirs_dict: dict,
                                   ref_key: str,
                                   Fs: int = 44_100,
                                   nfft: int = 4096,
                                   f_max_plot: float | None = None,
                                   **kwargs):
    """
    Plots spectral comparison for multiple RIRs against a reference RIR.

    Parameters
    ----------
    rirs_dict : dict
        A dictionary where keys are labels (str) and values are RIRs (np.ndarray).
    ref_key : str
        The key in rirs_dict that corresponds to the reference RIR.
    Fs : int
        Sample-rate (Hz).
    nfft : int
        FFT length (power of two recommended).
    f_max_plot : float or None
        If given, plot is generated only up to *f_max* Hz.
    **kwargs
        octave (bool): if True, draw 1/3-octave smoothed traces.
    """
    use_oct = kwargs.get("octave", False)
    eps = 1e-12

    if ref_key not in rirs_dict:
        raise ValueError(f"Reference key '{ref_key}' not found in rirs_dict.")

    max_len = 0
    processed_rirs = {}
    for label, rir in rirs_dict.items():
        rir_arr = np.asarray(rir, dtype=float).ravel()
        assert rir_arr.ndim == 1, f"RIR for '{label}' must be 1-D."
        processed_rirs[label] = rir_arr
        max_len = max(max_len, len(rir_arr))

    w = hann(max_len, sym=False)
    freqs_full = np.fft.rfftfreq(nfft, 1 / Fs)
    
    spectra_h = {}
    for label, rir in processed_rirs.items():
        padded_rir = np.pad(rir, (0, max_len - len(rir))) * w
        spectra_h[label] = np.fft.rfft(padded_rir, n=nfft)

    freqs_plot = freqs_full
    if f_max_plot:
        keep = freqs_full <= f_max_plot
        freqs_plot = freqs_full[keep]

    spectra_db = {}
    for label, H in spectra_h.items():
        H_plot = H
        if f_max_plot:
            H_plot = H[freqs_full <= f_max_plot]
        spectra_db[label] = 20 * np.log10(np.abs(H_plot) + eps)
    
    ref_spec_db = spectra_db[ref_key]
    ref_h = spectra_h[ref_key]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for label, spec_db in spectra_db.items():
        ax.semilogx(freqs_plot, spec_db, label=label, lw=0.9)

    if use_oct:
        f_iso = np.array([31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        f_iso = f_iso[f_iso < Fs / 2 * 0.9]
        bw = 2**(1 / 6)

        def octave_mean(H, f_c):
            f_lo, f_hi = f_c / bw, f_c * bw
            idx = (freqs_full >= f_lo) & (freqs_full <= f_hi)
            return 20 * np.log10(np.mean(np.abs(H[idx]) + eps))

        for label, H in spectra_h.items():
            oct_vals = [octave_mean(H, fc) for fc in f_iso]
            ax.semilogx(f_iso, oct_vals, 'o-', lw=2, label=f"{label} (1/3-oct)")

    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which='both')
    ax.legend()

    lsd_texts = []
    for label, H in spectra_h.items():
        if label == ref_key:
            continue
        _h1, _h2 = ref_h, H
        if f_max_plot:
            keep = freqs_full <= f_max_plot
            _h1, _h2 = _h1[keep], _h2[keep]
        lsd_bin = 20 * np.sqrt(np.mean((np.log10((np.abs(_h1) + eps) / (np.abs(_h2) + eps)))**2))
        lsd_texts.append(f"LSD({ref_key} vs {label}) = {lsd_bin:.2f} dB")
    
    ax.set_title("Spectral Comparison\n" + " | ".join(lsd_texts))

    plt.tight_layout()
    plt.show()


def lowpass_filter(data, cutoff, fs, order=4):
    """Applies a low-pass Butterworth filter to the data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)

def print_comparison_table(final_edcs, ref_method, cutoff_freqs, exclude_methods=None):
    """Print comparison table showing RMSE for original vs multiple lowpass versions."""
    if exclude_methods is None:
        exclude_methods = []
        
    print(f"\n{'='*95}")
    print(f"EDC RMSE 50ms COMPARISON with Low-passed Versions: (Cutoffs: {cutoff_freqs}) | Ref: {ref_method}")
    print(f"{'Method':<20} | {'Original':<9} | {f'LP {cutoff_freqs[0]}Hz':<9} | {'Diff':<8} | {f'LP {cutoff_freqs[1]}Hz':<9} | {'Diff':<8}")
    print('-' * 95)
    
    methods = set()
    for k in final_edcs.keys():
        if ' (Original)' in k:
            methods.add(k.replace(' (Original)', ''))
        elif f' (Lowpass {cutoff_freqs[0]})' in k:
            methods.add(k.replace(f' (Lowpass {cutoff_freqs[0]})', ''))
        elif f' (Lowpass {cutoff_freqs[1]})' in k:
            methods.add(k.replace(f' (Lowpass {cutoff_freqs[1]})', ''))
    
    # Filter out excluded methods
    methods = {m for m in methods if m not in exclude_methods}
    
    ref_orig = final_edcs.get(f"{ref_method} (Original)", [])
    ref_lp1 = final_edcs.get(f"{ref_method} (Lowpass {cutoff_freqs[0]})", [])
    ref_lp2 = final_edcs.get(f"{ref_method} (Lowpass {cutoff_freqs[1]})", [])
    
    for method in sorted(methods):
        if method == ref_method:
            continue
            
        # Calculate RMSE for all versions
        rmses = {}
        for version, ref_edcs in [('Original', ref_orig), (f'Lowpass {cutoff_freqs[0]}', ref_lp1), (f'Lowpass {cutoff_freqs[1]}', ref_lp2)]:
            key = f"{method} ({version})"
            if key in final_edcs and ref_edcs:
                rmse_vals = []
                for i, edc in enumerate(final_edcs[key]):
                    if i < len(ref_edcs):
                        min_len = min(len(ref_edcs[i]), len(edc))
                        rmse = an.compute_RMS(ref_edcs[i][:min_len], edc[:min_len], range=50, normalize_by_active_length=True)
                        rmse_vals.append(rmse)
                rmses[version] = np.mean(rmse_vals) if rmse_vals else np.nan
            else:
                rmses[version] = np.nan
        
        # Format output
        display_name = DISPLAY_NAME_MAP.get(method, method)
        if isinstance(display_name, dict):
            display_name = display_name.get('name', method)
            
        orig_str = f"{rmses['Original']:.2f}" if not np.isnan(rmses['Original']) else "N/A"
        lp1_str = f"{rmses[f'Lowpass {cutoff_freqs[0]}']:.2f}" if not np.isnan(rmses[f'Lowpass {cutoff_freqs[0]}']) else "N/A"
        lp2_str = f"{rmses[f'Lowpass {cutoff_freqs[1]}']:.2f}" if not np.isnan(rmses[f'Lowpass {cutoff_freqs[1]}']) else "N/A"
        
        # Calculate differences against original
        if not np.isnan(rmses['Original']) and not np.isnan(rmses[f'Lowpass {cutoff_freqs[0]}']):
            diff1 = rmses[f'Lowpass {cutoff_freqs[0]}'] - rmses['Original']
            diff1_str = f"{diff1:+.2f}"
        else:
            diff1_str = "N/A"
            
        if not np.isnan(rmses['Original']) and not np.isnan(rmses[f'Lowpass {cutoff_freqs[1]}']):
            diff2 = rmses[f'Lowpass {cutoff_freqs[1]}'] - rmses['Original']
            diff2_str = f"{diff2:+.2f}"
        else:
            diff2_str = "N/A"
            
        print(f"{display_name:<20} | {orig_str:<9} | {lp1_str:<9} | {diff1_str:<8} | {lp2_str:<9} | {diff2_str:<8}")
    
    print('=' * 95)

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "results/paper_data"
    FILES_TO_PROCESS = ["aes_room_spatial_edc_data.npz"]
    METHODS_TO_PROCESS = ['RIMPY-neg10', "SDN-Test1",
                          "SDN-Test5","SDN-Test6",
                          "HO-SDN-N2", "HO-SDN-N3"]  # None = process all methods
    METHODS_TO_PROCESS = None  # Set to None to process all methods in the data file
    PROCESS_ORIGINAL = True
    PROCESS_LOWPASS = True
    REFERENCE_METHOD_NAME = 'RIMPY-neg10'
    EXCLUDE_METHODS = ['ISM-pra-rand10', "ISM", "RIMPY-neg"]  # Methods to exclude from the table
    position_index = 0  # 0 for single position, None for all
    cutoff_freqs = [2000, 4000]

    for filename in FILES_TO_PROCESS:
        data_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            continue

        print(f"\n--- Processing: {filename} ---")
        sim_data = load_data(data_path)

        # Setup room object
        room_params = sim_data['room_params']
        room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
        source_pos = sim_data['source_pos']
        room.set_source(source_pos[0], source_pos[1], source_pos[2])
        receiver_positions = sim_data['receiver_positions']
        Fs = sim_data['Fs']
        all_sim_rirs = sim_data['rirs']

        # Determine positions and methods to process
        positions_to_process = [position_index] if position_index is not None else range(len(receiver_positions))
        methods_to_use = METHODS_TO_PROCESS if METHODS_TO_PROCESS is not None else list(all_sim_rirs.keys())
        
        print(f"Processing {len(positions_to_process)} position(s), {len(methods_to_use)} method(s)")
        print(f"Applying {cutoff_freqs[0]}Hz and {cutoff_freqs[1]}Hz low-pass filters...")

        final_rirs = {}
        final_edcs = {}

        # Main processing loop
        for key in methods_to_use:
            if key not in all_sim_rirs:
                print(f"Warning: Method '{key}' not found. Skipping.")
                continue

            rirs_data = all_sim_rirs[key]
            
            for pos_idx in positions_to_process:
                if pos_idx >= len(rirs_data):
                    continue
                    
                original_rir = rirs_data[pos_idx]
                
                # Process versions based on flags
                versions = []
                if PROCESS_ORIGINAL:
                    versions.append(('Original', original_rir))
                if PROCESS_LOWPASS:
                    rx, ry = receiver_positions[pos_idx]
                    room.set_microphone(rx, ry, room_params['mic z'])
                    
                    # Process both cutoff frequencies
                    for cutoff in cutoff_freqs:
                        rir_lp = lowpass_filter(original_rir, cutoff, Fs)
                        rir_lp_norm = rir_normalisation(rir_lp, room, Fs, normalize_to_first_impulse=True)["single_rir"]
                        versions.append((f'Lowpass {cutoff}', rir_lp_norm))

                # Calculate EDCs for each version
                for version_name, rir_to_process in versions:
                    plot_key = f"{key} ({version_name})"
                    
                    if plot_key not in final_edcs:
                        final_edcs[plot_key] = []
                    if plot_key not in final_rirs:
                        final_rirs[plot_key] = []
                    
                    edc, _, _ = an.compute_edc(rir_to_process, Fs, plot=False)
                    final_edcs[plot_key].append(edc)
                    
                    # Store RIR for both original and lowpass versions
                    final_rirs[plot_key].append(rir_to_process)

        # Print comparison table
        print_comparison_table(final_edcs, REFERENCE_METHOD_NAME, cutoff_freqs, EXCLUDE_METHODS)

        """
        # Optional plotting for single position
        if position_index is not None and final_edcs:
            # Interactive RIR plot - use both original and lowpass RIRs
            if final_rirs:
                rirs_to_plot_interactive = {key: rirs[0] for key, rirs in final_rirs.items()}
                pp.create_unified_interactive_plot(rirs_to_plot_interactive, Fs)
            
            # Static EDC comparison plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.title(f'EDC Comparison (Position: {position_index})')
            
            for key, edc_list in final_edcs.items():
                if edc_list:
                    edc = edc_list[0]
                    time_axis = np.arange(len(edc)) / Fs
                    linestyle = '--' if 'Original' in key else '-'
                    plt.plot(time_axis, edc, label=key, alpha=0.8, linestyle=linestyle)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (dB)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(-80, 5)
            plt.show()

        """


    rir_ref = final_rirs.get(f"{REFERENCE_METHOD_NAME} (Original)")
    rir_sdn_orig = final_rirs.get('SDN-Test1 (Original)')
    rir_sdn_c5 = final_rirs.get('SDN-Test5 (Original)')

    rir_ref_lp2k = final_rirs.get(f"{REFERENCE_METHOD_NAME} (Lowpass 2000)")
    rir_sdn_orig_lp2k = final_rirs.get('SDN-Test1 (Lowpass 2000)')
    rir_sdn_c5_lp2k  = final_rirs.get('SDN-Test5 (Lowpass 2000)')

    rir_ref_lp4k = final_rirs.get(f"{REFERENCE_METHOD_NAME} (Lowpass 4000)")
    rir_sdn_orig_lp4k = final_rirs.get('SDN-Test1 (Lowpass 4000)')
    rir_sdn_c5_lp4k  = final_rirs.get('SDN-Test5 (Lowpass 4000)')

    # full band (dc–Nyquist)
    lsd_orig_full = compute_LSD(rir_ref, rir_sdn_orig)
    lsd_c5_full   = compute_LSD(rir_ref, rir_sdn_c5)

    # up to 4 kHz only
    lsd_orig_less4k  = compute_LSD(rir_ref, rir_sdn_orig, f_max=4_000)
    lsd_c5_less4k    = compute_LSD(rir_ref, rir_sdn_c5,   f_max=4_000)

    # up to 2 kHz only
    lsd_orig_less2k  = compute_LSD(rir_ref, rir_sdn_orig, f_max=2_000)
    lsd_c5_less2k    = compute_LSD(rir_ref, rir_sdn_c5,   f_max=2_000)

    print("orig vs orig_less4k :", lsd_orig_full-lsd_orig_less4k)
    print("orig vs orig_less2k :  ", lsd_orig_full-lsd_orig_less2k)

    # Interpretation
    # If lsd_orig_full − lsd_orig_≤4k is large, most of the mismatch is above 4 kHz.
    # Do the same subtraction for 2 kHz to see how much of that error is confined to the 2–4 kHz octave.

    lbin1 = plot_spectral_comparison(rir_ref, rir_sdn_orig, Fs=48_000,
                         label1="ISM‐rimpy", label2="SDN orig",
                         nfft=4096, f_max_plot=20_000, octave=True)

    lbin2 = plot_spectral_comparison(rir_ref, rir_sdn_c5, Fs=48_000,
                         label1="ISM‐rimpy", label2="SW‐SDN c=5",
                         nfft=4096, f_max_plot=20_000, octave=True)


    # compare each RIR with *its own* 2-kHz-LP version
    lsd_orig_vs_lp2k = compute_LSD(rir_sdn_orig, rir_sdn_orig_lp2k)
    lsd_c5_vs_lp2k   = compute_LSD(rir_sdn_c5,   rir_sdn_c5_lp2k)

    print("SDN orig vs LP 2kHz:", lsd_orig_vs_lp2k)
    print("SDN c5 vs LP 2kHz:  ", lsd_c5_vs_lp2k)

    # --- New multi-plot ---
    if rir_ref and rir_sdn_orig and rir_sdn_c5:
        plot_spectral_comparison_multi(
            rirs_dict={
                "ISM-rimpy": rir_ref,
                "SDN orig": rir_sdn_orig,
                "SW-SDN c=5": rir_sdn_c5,
            },
            ref_key="ISM-rimpy",
            Fs=44_100,
            nfft=4096,
            f_max_plot=20_000,
            octave=True
        )