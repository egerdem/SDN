import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

def compute_edc(rir, Fs, label=None, plot=True, color=None):
    """Compute and optionally plot Energy Decay Curve."""
    # Calculate EDC
    squared_rir = rir ** 2
    edc = np.flip(np.cumsum(np.flip(squared_rir)))
    
    # Add small epsilon to prevent log10(0)
    eps = 1e-10
    normalized_edc = edc / np.max(edc)
    edc_db = 10 * np.log10(normalized_edc + 1e-10)
    
    # Create time array in seconds, starting from 0
    time = np.arange(len(rir)) / Fs
    
    # Plot only if requested
    if plot:
        plt.plot(time, edc_db, color=color, label=label)
    
    return edc_db

def calculate_smoothed_energy(rir: np.ndarray, window_length: int = 30, range: int = 50, Fs: int = 44100) -> np.ndarray:
    """Calculate smoothed energy of RIR for the early part.
    
    Args:
        rir (np.ndarray): Room impulse response
        window_length (int): Length of smoothing window (default: 30 samples)
        range (int): Time range in milliseconds to analyze (default: 50ms)
        Fs (int): Sampling frequency (default: 44100 Hz)
        
    Returns:
        np.ndarray: Smoothed energy of early RIR
    """
    # Calculate number of samples for the given time range
    range_samples = int((range / 1000) * Fs)  # Convert ms to samples
    
    # Trim RIR to the specified range
    rir_trimmed = rir[:range_samples]
    
    # Calculate energy
    energy = rir_trimmed ** 2
    
    # Apply smoothing window
    window = signal.windows.hann(window_length)
    # window = window / np.sum(window)
    smoothed = signal.convolve(energy, window, mode='same')
    # smoothed = signal.convolve(energy, window, mode='full')

    return energy, smoothed

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

def compute_RMS(sig1: np.ndarray, sig2: np.ndarray, range: int = 50, Fs: int = 44100, method = "rmse") -> float:
    """Compare two energy decay curves or smoothed RIRs and compute difference using various metrics."""
    # Calculate samples for range (e.g., 50ms)
    samples_range = int(range/1000 * Fs)  # Convert ms to samples
    # print("trimming signals for error calculation")

    # Trim signals to specified range
    sig1_early = sig1[:samples_range]
    sig2_early = sig2[:samples_range]
    
    # Calculate difference based on method
    if method == "rmse":
        # Root Mean Square Error (for linear scale)
        diff = np.sqrt(np.mean((sig1_early - sig2_early)**2))

    elif method == "sum":
        # Sum of absolute differences (total accumulated error)
        diff = np.sum(np.abs(sig1_early - sig2_early))

    elif method == "sum_of_raw_diff":
        # Sum of actual differences (total accumulated error)
        diff = np.sum(sig1_early - sig2_early)

    elif method == "mae":
        # Mean Absolute Error (for linear scale)
        diff = np.mean(np.abs(sig1_early - sig2_early))

    elif method == "median":
        # Median of absolute differences
        diff = np.median(np.abs(sig1_early - sig2_early))
        
    elif method == "lsd":
        # Logarithmic Spectral Distance
        diff = compute_LSD(sig1_early, sig2_early, Fs)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return diff

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
