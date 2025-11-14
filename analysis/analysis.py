import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def PRA_measure_rt60(h, fs=1, decay_db=60, energy_thres=1.0, plot=False, rt60_tgt=None, energy_db=None, schroder_energy=None):
    #I took it from pra's code, replacing h with powered h as I 
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    energy_thres: float
        This should be a value between 0.0 and 1.0.
        If provided, the fit will be done using a fraction energy_thres of the
        whole energy. This is useful when there is a long noisy tail for example.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    energy_db: array_like, optional
        Pre-calculated energy decay curve in dB. If provided, the function will skip
        the EDC calculation step and use this curve directly.
    raw_energy: array_like, optional
        Pre-calculated raw energy values (before dB conversion). Can be provided 
        along with energy_db for more accurate plotting.
    """

    h = np.array(h)
    fs = float(fs)

    # Skip EDC calculation if energy_db is provided
    if energy_db is None:
        # The power of the impulse response in dB
        power = h**2
        # Backward energy integration according to Schroeder
        energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

        if energy_thres < 1.0:
            assert 0.0 < energy_thres < 1.0
            energy -= energy[0] * (1.0 - energy_thres)
            energy = np.maximum(energy, 0.0)

        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]
    else:
        # Use provided energy_db directly
        power = h**2  # Still needed for plotting
        energy = schroder_energy  # Use provided raw energy if available

    min_energy_db = -np.min(energy_db)
    if min_energy_db - 5 < decay_db:
        decay_db = min_energy_db

    # -5 dB headroom
    try:
        i_5db = np.min(np.where(energy_db < -5)[0])
    except ValueError:
        return 0.0
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs
    # after decay
    try:
        i_decay = np.min(np.where(energy_db < -5 - decay_db)[0])
    except ValueError:
        i_decay = len(energy_db)
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    if plot:
        import matplotlib.pyplot as plt

        # If energy wasn't calculated or provided, we need to estimate energy_min for plotting
        if energy is None:
            energy_db_min = energy_db[-1]
            energy_min = 10**(energy_db_min/10)  # Convert from dB back to linear
        else:
            energy_min = energy[-1] if len(energy) > 0 else 0
            energy_db_min = energy_db[-1]
            
        # Remove clip power below to minimum energy (for plotting purpose mostly)
        power[power < energy_min] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            return np.arange(x.shape[0]) / fs - i_5db / fs

        T = get_time(power_db, fs)

        # plot power and energy
        plt.plot(get_time(energy_db, fs), energy_db, label="Energy")

        # now the linear fit
        plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
        plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
        plt.vlines(
            est_rt60, energy_db_min, 0, linestyles="dashed", label="Estimated RT60"
        )

        if rt60_tgt is not None:
            plt.vlines(rt60_tgt, energy_db_min, 0, label="Target RT60")

        plt.legend()

    return est_rt60


def calculate_rt60_from_rir(rir, fs, plot, pre_calculated_edc=None, schroder_energy=None):
    """Calculate RT60 from RIR using pyroomacoustics.

    Args:
        rir: Room impulse response
        fs: Sampling frequency
        plot: Whether to plot the RT60 calculation
        pre_calculated_edc: Pre-calculated energy decay curve (optional)
        raw_energy: Pre-calculated raw energy values (optional)

    Returns:
        rt60: Estimated RT60 value
    """
    # Normalize RIR
    rir = rir / np.max(np.abs(rir))

    # Estimate RT60 - use pre-calculated EDC if provided
    if pre_calculated_edc is not None:
        rt60 = PRA_measure_rt60(rir, fs, plot=plot, energy_db=pre_calculated_edc, schroder_energy=schroder_energy)
    else:
        rt60 = PRA_measure_rt60(rir, fs, plot=plot)
    return rt60

def EDC(rir):
    # eski koddan
    """
    Energy Decay Curve:
    Integral from t to infinity of the square of the impulse response,
    all divided by the integral from 0 to infinity of the square of the impulse response,
    presented in dB scale.
    from https://github.com/BrechtDeMan/pycoustics
    """
    rir = np.array(rir)
    print("EDC from rrdecay.py")
    cumul = 10 * np.log10(sum(rir**2))
    decay_curve = 10 * np.log10(np.flipud(np.cumsum(np.flipud(np.square(rir))))) - cumul
    return decay_curve

def EDC_timu(rir, Fs, label):
    # Calculate EDC exactly as in SDN_timu
    pEnergy = (np.cumsum(rir[::-1] ** 2) / np.sum(rir[::-1]))[::-1]
    pEdB = 10.0 * np.log10(pEnergy / np.max(pEnergy))
    plt.plot(np.arange(len(pEdB)) / Fs, pEdB, label=label, alpha=0.7)

def EDC_dp(impulse_response):
    # eski kod + deepseek
    """
    Energy Decay Curve:
    Integral from t to infinity of the square of the impulse response,
    divided by the integral from 0 to infinity of the square of the impulse response,
    presented in dB scale.
    """
    impulse_response = np.array(impulse_response)

    # Step 1: Compute the squared impulse response
    squared_ir = impulse_response ** 2

    # Step 2: Compute the cumulative sum of the reversed squared impulse response
    reversed_squared_ir = np.flipud(squared_ir)
    cumulative_energy = np.cumsum(reversed_squared_ir)

    # Step 3: Reverse the cumulative sum back to get the EDC
    edc = np.flipud(cumulative_energy)

    # Step 4: Normalize the EDC by the total energy
    total_energy = np.sum(squared_ir)
    normalized_edc = edc / total_energy

    # Step 5: Convert to dB scale (add small offset to avoid log(0))
    edc_dB = 10 * np.log10(normalized_edc + 1e-10)

    return edc_dB

def compute_edc(rir, Fs, label=None, plot=True, color=None, energy_thres=1.0):
    """Compute and optionally plot Energy Decay Curve.
    
    Args:
        rir (np.ndarray): Room impulse response
        Fs (int): Sampling frequency
        label (str, optional): Label for the EDC curve in plots
        plot (bool): Whether to plot the EDC curve
        color (str, optional): Color for the EDC curve in plots
        energy_thres (float): Energy threshold value between 0.0 and 1.0
            If less than 1.0, the fit will use a fraction of the energy
        
    Returns:
        tuple: (edc_db, raw_energy) where:
            - edc_db: Energy decay curve in dB, normalized and with proper zero tail handling
            - raw_energy: Raw energy values before dB conversion, for RT60 calculation
    """
    # Calculate squared RIR (power)
    squared_rir = rir ** 2
    
    # Backward energy integration according to Schroeder
    energy = np.flip(np.cumsum(np.flip(squared_rir)))
    
    # Apply energy threshold if specified (like in PRA_measure_rt60)
    if energy_thres < 1.0:
        assert 0.0 < energy_thres < 1.0
        energy -= energy[0] * (1.0 - energy_thres)
        energy = np.maximum(energy, 0.0)
    
    # Handle zero tail (like in PRA_measure_rt60)
    nonzero_indices = np.where(energy > 0)[0]
    if len(nonzero_indices) > 0:
        i_nz = np.max(nonzero_indices)
        energy = energy[:i_nz + 1]  # Include the last non-zero index CHANGED ...

    
    # Convert to dB and normalize
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]  # Normalize like PRA_measure_rt60
    
    # Create time array in seconds, starting from 0
    time_db = np.arange(len(energy_db)) / Fs
    
    # Plot only if requested
    if plot:
        plt.plot(time_db, energy_db, color=color, label=label)
        plt.show()
    
    # Return both the dB curve and the raw energy (for RT60 calculation)
    return energy_db, time_db, energy

def calculate_smoothed_energy(rir: np.ndarray, window_length: int = 30, range: int = None, Fs: int = 44100) -> tuple:
    """Calculate smoothed energy of RIR for the early part.
    
    Args:
        rir (np.ndarray): Room impulse response
        window_length (int): Length of smoothing window (default: 30 samples)
        range (int): Time range in milliseconds to analyze (default: 50ms)
        Fs (int): Sampling frequency (default: 44100 Hz)
        
    Returns:
        tuple: (energy, smoothed, err) where:
            - energy: Raw energy of the early RIR
            - smoothed: Smoothed energy of early RIR
    """
    if range is not None:
        # Calculate number of samples for the given time range
        range_samples = int((range / 1000) * Fs)  # Convert ms to samples
        # Trim RIR to the specified range
        # Calculate energy
        energy = rir[:range_samples] ** 2 #trimmed energy
    else:
        energy = rir ** 2  # Full RIR energy

    # Apply smoothing window
    window = signal.windows.hann(window_length)
    # window = window / np.sum(window)
    smoothed = signal.convolve(energy, window, mode='same')
    # smoothed = signal.convolve(energy, window, mode='full')

    return smoothed

# from scipy import signal
# energy = rir ** 2
# window_length = 30
# window = signal.windows.hann(window_length)
# smoothed = signal.convolve(energy, window, mode='same')
# plt.figure()
# plt.plot(energy, label='Energy')
# plt.plot(smoothed, label='Energy')
# plt.legend()
# plt.show()

def calculate_error_metric(rir1: np.ndarray, rir2: np.ndarray) -> float:
    """Calculate error metric between two RIRs.

    Args:
        rir1 (np.ndarray): First RIR
        rir2 (np.ndarray): Second RIR

    Returns:
        float: Error metric value
    """
    # Calculate smoothed energies
    smoothed1 = calculate_smoothed_energy(rir1)
    smoothed2 = calculate_smoothed_energy(rir2)

    # Calculate error (mean squared error of smoothed energies)
    error = np.mean((smoothed1 - smoothed2) ** 2)
    return error

def plot_smoothing_comparison(rir: np.ndarray, window_length: int = 100, Fs: int = 44100):
    """Plot original RIR and its smoothed version for comparison.
    
    Args:
        rir (np.ndarray): Room impulse response
        window_length (int): Length of smoothing window
        Fs (int): Sampling frequency
    """
    smoothed = calculate_smoothed_energy(rir, window_length=window_length, Fs=Fs)
    time = np.arange(len(rir)) / Fs * 1000  # Convert to milliseconds
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, rir, label='Original RIR', alpha=0.7)
    plt.plot(time, smoothed, label='Smoothed Energy', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'RIR and Smoothed Energy Comparison (window length: {window_length} samples)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def calculate_rms_envelope(signal: np.ndarray,
                           frame_size: int = 256,
                           hop_size: int = 128) -> np.ndarray:
    """
    Calculate a short-time RMS envelope of the input signal.

    Args:
        signal (np.ndarray): 1D audio signal.
        frame_size (int): Number of samples per frame (default=256).
        hop_size (int): Number of samples to advance between frames (default=128).

    Returns:
        np.ndarray: Array of RMS values, one per frame.
    """
    # Number of frames we can fit
    num_frames = 1 + (len(signal) - frame_size) // hop_size

    rms_values = np.zeros(num_frames, dtype=float)

    # Process each frame
    for i in range(num_frames):
        start = i * hop_size
        stop = start + frame_size
        frame = signal[start:stop]

        # Compute RMS: sqrt of the average of squared samples
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values[i] = rms

    return rms_values

def compute_RMS_OLD(sig1: np.ndarray, sig2: np.ndarray, range: int = None, Fs: int = 44100, method = "rmse", skip_initial_zeros: bool = False, normalize_by_active_length: bool = True) -> float:
    """
    Computes the difference between two signals using specified method.

    Args:
        sig1 (np.ndarray): First signal
        sig2 (np.ndarray): Second signal
        range (int, optional): Time range in milliseconds to analyze (default: None)
        Fs (int): Sampling frequency
        method (str): Method for computing the difference
        skip_initial_zeros (bool): Whether to skip initial zeros in the error calculation
        normalize_by_active_length (bool): Whether to normalize by the active length of the signals

    Returns:
        float: Computed difference
    """
    # Check if inputs are valid
    if sig1 is None or sig2 is None:
        print("WARNING: Null signal received in compute_RMS")
        return 0.0
        
    # Calculate samples for range (e.g., 50ms)
    # only trim if range is not None
    if range is not None:
        samples_range = int(range/1000 * Fs)  # Convert ms to samples
        # print("trimming signals for error calculation"

        # Trim signals to specified range if length is larger than range
        if len(sig1) > samples_range and len(sig2) > samples_range:
            sig1_early = sig1[:samples_range]
            sig2_early = sig2[:samples_range]
        else:
            # If range is larger than either signal length, trim both to the length of the shorter signal
            min_length = min(len(sig1), len(sig2))
            sig1_early = sig1[:min_length]
            sig2_early = sig2[:min_length]
    else:
        # If no range specified, ensure signals are the same length by trimming to the shorter one
        #min_length = min(len(sig1), len(sig2))
        #sig1_early = sig1[:min_length]
        #sig2_early = sig2[:min_length]
        sig1_early = sig1   
        sig2_early = sig2
    
    if skip_initial_zeros:
        # Get indices of non-zero samples with a small threshold
        threshold = 1e-10
        sig1_nonzeros = np.where(np.abs(sig1_early) > threshold)[0]
        sig2_nonzeros = np.where(np.abs(sig2_early) > threshold)[0]
        
        # If either signal has non-zero samples, find the earlier first non-zero sample
        if len(sig1_nonzeros) > 0 and len(sig2_nonzeros) > 0:
            # print("Skipping initial zeros in edc for rmse calc:", sig1_nonzeros[0], sig2_nonzeros[0])
            first_nonzero = min(sig1_nonzeros[0], sig2_nonzeros[0])
            # Trim both signals to start from the first non-zero sample
            sig1_early = sig1_early[first_nonzero:]
            sig2_early = sig2_early[first_nonzero:]
        elif len(sig1_nonzeros) > 0:
            # Only sig1 has non-zero samples
            sig1_early = sig1_early[sig1_nonzeros[0]:]
            sig2_early = sig2_early[sig1_nonzeros[0]:]
        elif len(sig2_nonzeros) > 0:
            # Only sig2 has non-zero samples
            sig1_early = sig1_early[sig2_nonzeros[0]:]
            sig2_early = sig2_early[sig2_nonzeros[0]:]

    # # Calculate difference based on method
    # if method == "rmse":
    #     # Root Mean Square Error (for linear scale)
    #     diff = np.sqrt(np.mean((sig1_early - sig2_early)**2))
    #
    # elif method == "sum":
    #     # Sum of absolute differences (total accumulated error)
    #     diff = np.sum(np.abs(sig1_early - sig2_early))
    #
    # elif method == "sum_of_raw_diff":
    #     # Sum of actual differences (total accumulated error)
    #     diff = np.sum(sig1_early - sig2_early)

    # Ensure signals have the same length after all processing
    if len(sig1_early) != len(sig2_early):
        print(f"Warning: Signals have different lengths after processing ({len(sig1_early)} vs {len(sig2_early)}). Truncating to shorter length.")
        min_len = min(len(sig1_early), len(sig2_early))
        sig1_early = sig1_early[:min_len]
        sig2_early = sig2_early[:min_len]
        
    diff = sig1_early - sig2_early
    
    N = len(diff)
    if normalize_by_active_length:
        # Find first index where the reference signal is not equal to its first value.
        # This marks the end of the initial plateau for an EDC.
        onset_idx = np.where(sig1_early != sig1_early[0])[0]
        if len(onset_idx) > 0:
            num_initial_zeros = onset_idx[0]
            # Adjust the normalization factor
            N = len(diff) - num_initial_zeros
            # print("num_initial_zeros:", num_initial_zeros)
        if N <= 0: # Avoid division by zero
            N = 1 

    if method == "rmse":
        # print("before, now", np.sqrt(np.sum(np.square(diff)) / len(diff)), np.sqrt(np.sum(np.square(diff)) / N))
        return np.sqrt(np.sum(np.square(diff)) / N)
    elif method == "mae":
        return np.sum(np.abs(diff)) / N
    elif method == "median":
        return np.median(np.abs(diff))
    elif method == "sum":
        return np.sum(np.abs(diff))
    else:
        raise ValueError(f"Unknown method: {method}")

def compute_RMS(sig1: np.ndarray,
                sig2: np.ndarray,
                range: int | None = None,          # [ms] window length
                Fs: int = 44100,
                method: str = "rmse",
                skip_initial_zeros: bool = False,
                normalize_by_active_length: bool = True) -> float:
    """
    Root-mean-square (or MAE / SUM / MEDIAN) distance between two 1-D signals.

    * If ``skip_initial_zeros`` is True, both signals are first aligned to the
      earliest non-zero sample (common onset); *only then* the ``range``-ms
      window is taken, so the comparison window always has the requested
      length.
    * If ``range`` is None the full signals are used (after optional alignment).

    Parameters
    ----------
    sig1, sig2 : np.ndarray
        Signals to compare (1-D real).
    range : int | None
        Window length to analyse in milliseconds.  None = full length.
    Fs : int
        Sampling rate [Hz].
    method : {"rmse","mae","median","sum"}
        Error metric.
    skip_initial_zeros : bool
        Align to first non-zero sample before cropping.
    normalize_by_active_length : bool
        For RMSE / MAE, divide by the number of *active* samples
        (after the onset) instead of the whole window.

    Returns
    -------
    float
        Error according to ``method``.
    """
    # --------- sanity checks --------------------------------------------------
    if sig1 is None or sig2 is None:
        raise ValueError("Null signal passed to compute_RMS")
    # sig1 = np.asarray(sig1, dtype=float).ravel()
    # sig2 = np.asarray(sig2, dtype=float).ravel()

    # --------- (1) optional onset alignment -----------------------------------
    if skip_initial_zeros:
        thresh = 1e-12
        nz1 = np.argmax(np.abs(sig1) > thresh)  # 0 if all-zeros
        nz2 = np.argmax(np.abs(sig2) > thresh)
        onset = min(nz1, nz2)
        # print("onset at sample:", onset, "for signals of lengths", len(sig1), "and", len(sig2))
        sig1, sig2 = sig1[onset:], sig2[onset:]

    # --------- (2) crop the requested window ----------------------------------
    if range is not None:
        Lwin = int(range * 1e-3 * Fs)          # samples in window
        sig1 = sig1[:Lwin]
        sig2 = sig2[:Lwin]

    # --------- make length equal ---------------------------------------------
    Lmin = min(len(sig1), len(sig2))
    if Lmin == 0:
        return 0.0
    sig1, sig2 = sig1[:Lmin], sig2[:Lmin]

    diff = sig1 - sig2

    # --------- normalisation length ------------------------------------------
    N = len(diff)
    if normalize_by_active_length and skip_initial_zeros:
        N = max(N, 1)                          # already aligned ⇒ no plateau
    elif normalize_by_active_length:
        plateau = np.argmax(sig1 != sig1[0])   # end of initial constant part
        if plateau:                            # plateau > 0 means found
            N = max(len(diff) - plateau, 1)

    # --------- metrics --------------------------------------------------------
    if method == "rmse":
        return np.sqrt(np.sum(diff ** 2) / N)
    if method == "mae":
        return np.sum(np.abs(diff)) / N
    if method == "median":
        return np.median(np.abs(diff))
    if method == "sum":
        return np.sum(np.abs(diff))
    raise ValueError(f"Unknown method: {method}")


def plot_edc_comparison(edc1: np.ndarray, edc2: np.ndarray, Fs: int = 44100, 
                       label1: str = "EDC 1", label2: str = "EDC 2"):
    """Plot two EDCs and their difference for visual comparison.
    
    Args:
        edc1 (np.ndarray): First energy decay curve in dB
        edc2 (np.ndarray): Second energy decay curve in dB
        Fs (int): Sampling frequency in Hz
        label1 (str): Label for first EDC
        label2 (str): Label for second EDC
    """
    min_length = min(len(edc1), len(edc2))
    time = np.arange(min_length) / Fs
    
    plt.figure(figsize=(12, 8))
    
    # Plot EDCs
    plt.subplot(2, 1, 1)
    plt.plot(time, edc1[:min_length], label=label1)
    plt.plot(time, edc2[:min_length], label=label2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel('Energy (dB)')
    plt.title('Energy Decay Curves Comparison')
    
    # Plot difference
    plt.subplot(2, 1, 2)
    plt.plot(time, edc1[:min_length] - edc2[:min_length], label='Difference')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Difference (dB)')
    plt.title('EDC Difference')
    
    plt.tight_layout()
    plt.show()

def compute_LSD(rir1: np.ndarray, rir2: np.ndarray, Fs: int = 44100, nfft: int = 2048) -> float:
    """Calculate Logarithmic Spectral Distance between two RIRs.
    
    LSD measures the average difference between the log-magnitude spectra of two RIRs:
    LSD = sqrt(1/|F| * sum((20*log10(|H1(f)|/|H2(f)|))^2))
    
    Args:
        rir1 (np.ndarray): First RIR
        rir2 (np.ndarray): Second RIR
        Fs (int): Sampling frequency (default: 44100 Hz)
        nfft (int): FFT size (default: 2048)
        
    Returns:
        float: LSD value in dB
    """
    # Compute FFT of both RIRs
    H1 = np.fft.fft(rir1, n=nfft)
    H2 = np.fft.fft(rir2, n=nfft)
    
    # Use only positive frequencies up to Nyquist
    f_pos = nfft // 2 + 1
    H1 = H1[:f_pos]
    H2 = H2[:f_pos]
    
    # Compute magnitude spectra (add small epsilon to avoid log(0))
    eps = 1e-10
    mag_H1 = np.abs(H1) + eps
    mag_H2 = np.abs(H2) + eps
    
    # Compute LSD according to the formula
    log_ratio = 20 * np.log10(mag_H1 / mag_H2)
    lsd = np.sqrt(np.mean(log_ratio**2))
    
    return lsd

def plot_spectral_comparison(rir1: np.ndarray, rir2: np.ndarray, 
                           Fs: int = 44100, 
                           label1: str = "RIR 1", 
                           label2: str = "RIR 2",
                           nfft: int = 2048):
    """Plot spectral comparison between two RIRs.
    
    Args:
        rir1 (np.ndarray): First RIR
        rir2 (np.ndarray): Second RIR
        Fs (int): Sampling frequency (default: 44100 Hz)
        label1 (str): Label for first RIR (default: "RIR 1")
        label2 (str): Label for second RIR (default: "RIR 2")
        nfft (int): FFT size (default: 2048)
    """
    # Input validation
    if not isinstance(Fs, (int, float)):
        raise TypeError(f"Fs must be a number, got {type(Fs)}")
    if not isinstance(nfft, int):
        raise TypeError(f"nfft must be an integer, got {type(nfft)}")
    
    # Compute FFT
    H1 = np.fft.fft(rir1, n=nfft)
    H2 = np.fft.fft(rir2, n=nfft)
    
    # Use positive frequencies up to Nyquist
    f_pos = nfft // 2 + 1
    freqs = np.linspace(0, Fs/2, f_pos)
    H1 = H1[:f_pos]
    H2 = H2[:f_pos]
    
    # Compute magnitude spectra in dB
    eps = 1e-10
    mag_H1_db = 20 * np.log10(np.abs(H1) + eps)
    mag_H2_db = 20 * np.log10(np.abs(H2) + eps)
    
    # Calculate LSD for title
    lsd = compute_LSD(rir1, rir2, Fs, nfft)
    
    plt.figure(figsize=(12, 8))
    
    # Plot magnitude spectra
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs, mag_H1_db, label=label1)
    plt.semilogx(freqs, mag_H2_db, label=label2)
    plt.grid(True)
    plt.legend()
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Spectral Comparison (LSD: {lsd:.2f} dB)')
    
    # Plot difference
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs, mag_H1_db - mag_H2_db, label='Difference')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Difference (dB)')
    
    plt.tight_layout()
    plt.show()
    
    return lsd  # Return the LSD value for reference

def compute_clarity_c50(rir: np.ndarray, Fs: int) -> float:
    """Calculate C50 clarity metric from a room impulse response.
    
    C50 is the ratio of early energy (0-50ms) to late energy (50ms-end) in dB:
    C50 = 10 * log10(early_energy / late_energy)
    
    Args:
        rir (np.ndarray): Room impulse response
        Fs (int): Sampling frequency
        
    Returns:
        float: C50 value in dB
    """
    # Convert 50ms to samples
    early_samples = int(0.05 * Fs)
    
    # Ensure we don't exceed the length of the RIR
    early_samples = min(early_samples, len(rir))
    
    # Calculate early and late energy
    early_energy = np.sum(rir[:early_samples]**2)
    late_energy = np.sum(rir[early_samples:]**2)

    # Calculate C50 in dB
    try:
        c50 = 10 * np.log10(early_energy / late_energy)
    except ZeroDivisionError:
        print("Warning: Division by zero in C50 calculation.")
    
    return c50

def compute_clarity_c80(rir: np.ndarray, Fs: int) -> float:
    """Calculate C80 clarity metric from a room impulse response.
    
    C80 is the ratio of early energy (0-80ms) to late energy (80ms-end) in dB:
    C80 = 10 * log10(early_energy / late_energy)
    
    Args:
        rir (np.ndarray): Room impulse response
        Fs (int): Sampling frequency
        
    Returns:
        float: C80 value in dB
    """
    # Convert 80ms to samples
    early_samples = int(0.08 * Fs)
    
    # Ensure we don't exceed the length of the RIR
    early_samples = min(early_samples, len(rir))
    
    # Calculate early and late energy
    early_energy = np.sum(rir[:early_samples]**2)
    late_energy = np.sum(rir[early_samples:]**2)

    # Calculate C80 in dB
    try:
        c80 = 10 * np.log10(early_energy / late_energy)
    except ZeroDivisionError:
        print("Warning: Division by zero in C50 calculation.")
    
    return c80

def compute_all_metrics(rir: np.ndarray, Fs: int = 44100) -> dict:
    """Compute all acoustic metrics for a room impulse response.
    
    Args:
        rir (np.ndarray): Room impulse response
        Fs (int): Sampling frequency (default: 44100 Hz)
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Compute EDC
    metrics['edc'] = compute_edc(rir, Fs, plot=False)
    
    # Compute clarity metrics
    metrics['c50'] = compute_clarity_c50(rir, Fs)
    metrics['c80'] = compute_clarity_c80(rir, Fs)
    
    # Compute LSD (comparing with itself, should be 0)
    metrics['lsd'] = compute_LSD(rir, rir, Fs)
    
    return metrics

def compare_metrics(rir1: np.ndarray, rir2: np.ndarray, Fs: int = 44100) -> dict:
    """Compare metrics between two room impulse responses.
    
    Args:
        rir1 (np.ndarray): First room impulse response
        rir2 (np.ndarray): Second room impulse response
        Fs (int): Sampling frequency (default: 44100 Hz)
        
    Returns:
        dict: Dictionary containing metrics for both RIRs and their differences
    """
    # Compute metrics for both RIRs
    metrics1 = compute_all_metrics(rir1, Fs)
    metrics2 = compute_all_metrics(rir2, Fs)
    
    # Compute differences
    differences = {}
    for key in metrics1:
        if key == 'edc':
            # For EDC, compute RMS difference
            differences[key] = np.sqrt(np.mean((metrics1[key] - metrics2[key])**2))
        else:
            # For scalar metrics, compute absolute difference
            differences[key] = abs(metrics1[key] - metrics2[key])
    
    return {
        'rir1': metrics1,
        'rir2': metrics2,
        'differences': differences
    }

def calculate_err(rir: np.ndarray, early_range: int = 50, Fs: int = 44100) -> float:
    """Calculate Energy Ratio (ERR) metric from a room impulse response.
    
    ERR is the ratio of early energy (0-early_range ms) to total energy:
    ERR = early_energy / total_energy
    
    Args:
        rir (np.ndarray): Room impulse response
        early_range (int): Time range in milliseconds for early energy (default: 50ms)
        Fs (int): Sampling frequency (default: 44100 Hz)
        
    Returns:
        float: ERR value (ratio between 0 and 1)
    """
    # Calculate total energy of the full RIR
    energy = rir**2

    # Calculate early energy (first early_range ms)
    early_samples = int((early_range / 1000) * Fs)  # Convert ms to samples
    early_energy = rir[:early_samples]**2

    # Calculate ERR
    ERR = np.sum(rir[:early_samples]**2) / np.sum(rir**2)
    
    return early_energy, energy, ERR


def calculate_rt60_theoretical(room_dim, absorption):
    """Calculate theoretical RT60 using Sabine and Eyring formulas.

    Args:
        room_dim: Room dimensions [width, depth, height] in meters
        absorption: Average absorption coefficient

    Returns:
        rt60_sabine: Reverberation time using Sabine's formula
        rt60_eyring: Reverberation time using Eyring's formula
    """
    # Room volume and surface area
    V = room_dim[0] * room_dim[1] * room_dim[2]  # Volume
    S = 2 * (room_dim[0] * room_dim[1] + room_dim[1] * room_dim[2] + room_dim[0] * room_dim[2])  # Surface area

    # Sabine's formula
    rt60_sabine = 0.161 * V / (S * absorption)

    # Eyring's formula
    rt60_eyring = 0.161 * V / (-S * np.log(1 - absorption))

    return rt60_sabine, rt60_eyring


def analyze_rir_pulses(rirs_dict, Fs, print_results=True):
    """
    Analyze pulse characteristics of RIRs (nonzero sample count, time span, etc.)
    
    For ISM-pra methods: uses peak detection
    For other methods: counts nonzero samples
    
    Args:
        rirs_dict: Dictionary of RIRs {label: rir_array}
        Fs: Sampling frequency
        print_results: If True, print analysis results
        
    Returns:
        dict: Analysis results for each RIR
    """
    from scipy.signal import find_peaks
    
    results = {}
    
    if print_results:
        print("\n=== RIR Pulse Analysis ===")
    
    for rir_label, rir in rirs_dict.items():
        rir_results = {}
        
        if print_results:
            print(f"\n{rir_label}:")
        
        if "ISM-pra" in rir_label:  # For PRA method, use peak detection
            # Find peaks that are at least 1% of the maximum amplitude
            threshold = 0.01 * np.max(np.abs(rir))
            peaks, _ = find_peaks(np.abs(rir), height=threshold)
            
            rir_results['method'] = 'peak_detection'
            rir_results['total_peaks'] = len(peaks)
            rir_results['first_peak_sample'] = peaks[0] if len(peaks) > 0 else 0
            rir_results['last_peak_sample'] = peaks[-1] if len(peaks) > 0 else 0
            rir_results['first_peak_time_ms'] = peaks[0] / Fs * 1000 if len(peaks) > 0 else 0
            rir_results['last_peak_time_ms'] = peaks[-1] / Fs * 1000 if len(peaks) > 0 else 0
            rir_results['time_span_ms'] = (peaks[-1] - peaks[0]) / Fs * 1000 if len(peaks) > 0 else 0
            
            if print_results:
                print(f"  Total significant peaks: {rir_results['total_peaks']}")
                print(f"  First peak at: {rir_results['first_peak_time_ms']:.2f} ms")
                print(f"  Last peak at: {rir_results['last_peak_time_ms']:.2f} ms")
                print(f"  Time span: {rir_results['time_span_ms']:.2f} ms")
        
        else:  # For other methods, use non-zero analysis
            # Count total nonzero pulses (using small threshold to account for floating point)
            threshold = 1e-10  # Adjust this threshold based on your needs
            nonzero_indices = np.where(np.abs(rir) > threshold)[0]
            nonzero_count = len(nonzero_indices)
            
            # Calculate percentage of nonzero samples
            percentage = (nonzero_count / len(rir)) * 100
            
            # Find first and last nonzero indices
            first_pulse = nonzero_indices[0] if nonzero_count > 0 else 0
            last_pulse = nonzero_indices[-1] if nonzero_count > 0 else 0
            
            rir_results['method'] = 'nonzero_analysis'
            rir_results['total_nonzero_pulses'] = nonzero_count
            rir_results['percentage_nonzero'] = percentage
            rir_results['first_pulse_sample'] = first_pulse
            rir_results['last_pulse_sample'] = last_pulse
            rir_results['first_pulse_time_ms'] = first_pulse / Fs * 1000
            rir_results['last_pulse_time_ms'] = last_pulse / Fs * 1000
            rir_results['time_span_ms'] = (last_pulse - first_pulse) / Fs * 1000
            
            if print_results:
                print(f"  Total (nonzero) pulses: {rir_results['total_nonzero_pulses']}")
                print(f"  Percentage of nonzero samples: {rir_results['percentage_nonzero']:.2f}%")
                print(f"  First pulse at: {rir_results['first_pulse_time_ms']:.2f} ms")
                print(f"  Last pulse at: {rir_results['last_pulse_time_ms']:.2f} ms")
                print(f"  Time span: {rir_results['time_span_ms']:.2f} ms")
        
        results[rir_label] = rir_results
    
    return results


def compute_rir_metrics_batch(rirs_dict, Fs):
    """
    Compute comprehensive metrics for all RIRs in a batch.
    
    Computes: smoothed_energy, early_energy, energy, ERR, C50, C80
    
    Args:
        rirs_dict: Dictionary of RIRs {label: rir_array}
        Fs: Sampling frequency
        
    Returns:
        dict: Nested dictionary {rir_label: {metric_name: value}}
    """
    rirs_analysis = {}
    
    for rir_label, rir in rirs_dict.items():
        smoothed = calculate_smoothed_energy(rir, window_length=30, range=50, Fs=Fs)
        early_energy, energy, ERR = calculate_err(rir, early_range=50, Fs=Fs)
        c50 = compute_clarity_c50(rir, Fs=Fs)
        c80 = compute_clarity_c80(rir, Fs=Fs)
        
        rirs_analysis[rir_label] = {
            "smoothed_energy": smoothed,
            "early_energy": early_energy,
            "energy": energy,
            "ERR": ERR,
            "c50": c50,
            "c80": c80
        }
    
    return rirs_analysis


def print_rir_metrics(rirs_analysis):
    """
    Print RIR metrics in a formatted way.
    
    Args:
        rirs_analysis: Dictionary from compute_rir_metrics_batch()
    """
    for rir_label, metrics in rirs_analysis.items():
        print(f"\nEnergy Total {rir_label} = {sum(metrics['energy']):.3f}")
        print(f"Energy 50ms {rir_label} = {sum(metrics['early_energy']):.3f}")
        print(f"ERR: Energy50ms/EnergyTotal {rir_label} = {metrics['ERR']:.3f}")
        print(f"C50 {rir_label} = {metrics['c50']:.3f}")
        print(f"C80 {rir_label} = {metrics['c80']:.3f}")


def compare_rir_pairs(rirs_analysis, method_pairs, comparison_type='early_energy'):
    """
    Compare RIR pairs using RMSE, MAE, and Median metrics.
    
    Args:
        rirs_analysis: Dictionary from compute_rir_metrics_batch()
        method_pairs: List of pair dictionaries with 'label1', 'label2', 'pair', 'info'
        comparison_type: 'early_energy' or 'smoothed_energy'
        
    Returns:
        list: List of comparison result dictionaries
    """
    comparison_results = []
    
    for pair_info in method_pairs:
        l1, l2 = pair_info['label1'], pair_info['label2']
        
        # Get the signals to compare
        signal1 = rirs_analysis[l1][comparison_type]
        signal2 = rirs_analysis[l2][comparison_type]
        
        # Compute metrics
        result = {
            'pair': pair_info['pair'],
            'info': pair_info['info'],
            'rmse': compute_RMS(signal1, signal2, method="rmse", skip_initial_zeros=True),
            'mae': compute_RMS(signal1, signal2, method="mae", skip_initial_zeros=True),
            'median': compute_RMS(signal1, signal2, method="median", skip_initial_zeros=True)
        }
        comparison_results.append(result)
    
    return comparison_results


def print_comparison_results(comparison_results, title):
    """
    Print comparison results in formatted table.
    
    Args:
        comparison_results: List of comparison dictionaries
        title: Title for the comparison table
    """
    print("\n" + "=" * 130)
    print(" " * 50 + title)
    print("=" * 130)
    print(f"{'Method Pair':<40} {'Rmse':>12} {'MAE':>12} {'Median':>12} {'Info':>20}")
    print("-" * 130)
    
    for result in comparison_results:
        print(f"{result['pair']:<40} {result['rmse']:12.6f} {result['mae']:12.6f} "
              f"{result['median']:12.6f}  {result['info']:>20}")
    
    print("=" * 130)


def compare_edc_pairs(rirs_dict, method_pairs, Fs):
    """
    Compare EDCs between RIR pairs.
    
    Args:
        rirs_dict: Dictionary of RIRs {label: rir_array}
        method_pairs: List of pair dictionaries with 'label1', 'label2'
        Fs: Sampling frequency
        
    Returns:
        dict: EDC comparison results {pair: rms_diff}
    """
    edc_comparisons = {}
    
    for pair_info in method_pairs:
        label1, label2 = pair_info['label1'], pair_info['label2']
        
        # Calculate EDCs for both RIRs without plotting
        edc1, _, _ = compute_edc(rirs_dict[label1], Fs, label1, plot=False)
        edc2, _, _ = compute_edc(rirs_dict[label2], Fs, label2, plot=False)
        
        # Compare EDCs
        rms_diff = compute_RMS(edc1, edc2, range=50, Fs=Fs, method="mae", skip_initial_zeros=True)
        
        pair_name = f"{label1} vs {label2}"
        edc_comparisons[pair_name] = rms_diff
    
    return edc_comparisons


def print_edc_comparisons(edc_comparisons):
    """
    Print EDC comparison results.
    
    Args:
        edc_comparisons: Dictionary from compare_edc_pairs()
    """
    print("\nEDC Comparison (First 50ms RMSE Differences):")
    print("-" * 50)
    
    for pair_name, rms_diff in edc_comparisons.items():
        print(f"{pair_name}: {rms_diff:.2f} dB RMSE difference")