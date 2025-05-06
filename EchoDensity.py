#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# erfc(1/√2)
ERFC = 0.3173


"""
Returns the weighted standard deviation.
"""

def weightedStd(signal, window_func, use_local_avg):
    if use_local_avg:
        average = np.average(signal, weights=window_func)
        variance = np.average((signal-average)**2, weights=window_func)
    else:
        variance = np.average((signal)**2, weights=window_func)
    return np.sqrt(variance)


"""
Computes the Echo Density Profile as defined by Abel.
window_type should be one of ['rect', 'bart', 'blac', 'hamm', 'hann']
"""

def echoDensityProfile(rir,
                       window_lentgh_ms=30, window_type='hann', #was 30
                       fs=44100, use_local_avg=False, max_duration=0.5):
    """Calculate echo density profile of a room impulse response.
    
    Args:
        rir: Room impulse response
        window_lentgh_ms: Window length in milliseconds
        window_type: Type of window ('rect', 'hann', 'hamm', 'blac', 'bart')
        fs: Sampling frequency in Hz
        use_local_avg: Whether to use local average for standard deviation
        max_duration: Maximum duration in seconds to calculate NED (default: 0.5s)
                     Set to None to process the entire RIR
        target_fs: Target sampling frequency in Hz for downsampling (default: None)
                  If provided and lower than fs, the RIR will be downsampled
                  before processing to reduce computation time
    
    Returns:
        numpy.ndarray: Echo density profile
    """
    # First, limit RIR to max_duration if specified (do this only once)
    if max_duration is not None:
        max_samples = int(max_duration * fs)
        rir_limited = rir[:max_samples]
    else:
        rir_limited = rir
    # Calculate window length in frames based on the working sampling rate
    window_length_frames = int(window_lentgh_ms * fs/1000)

    if not window_length_frames % 2:
        window_length_frames += 1
    half_window = int((window_length_frames-1)/2)

    padded_rir = np.zeros(len(rir_limited) + 2*half_window)
    padded_rir[half_window:-half_window] = rir_limited
    output = np.zeros(len(rir_limited) + 2*half_window)

    if window_type == 'rect':
        window_func = (1/window_length_frames) * np.ones(window_length_frames)
    elif window_type == 'hann':
        window_func = np.hanning(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'hamm':
        window_func = np.hamming(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'blac':
        window_func = np.blackman(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'bart':
        window_func = np.bartlett(window_length_frames)
        window_func = window_func / sum(window_func)
    else:
        raise ValueError('Unavailable window type.')

    for cursor in range(len(rir_limited)):
        frame = padded_rir[cursor:cursor+window_length_frames]
        std = weightedStd(frame, window_func, use_local_avg)

        count = ((np.abs(frame) > std) * window_func).sum()

        output[cursor] = (1/ERFC) * count

    ned_profile = output[:-2*half_window]

    return ned_profile

"""
Computes the non-normalized Echo Density Profile.
This version returns the raw count of samples exceeding the standard deviation,
without normalizing by ERFC.
"""
def echoDensityProfileRaw(rir,
                         window_lentgh_ms=30, window_type='hann',
                         fs=44100, use_local_avg=False):
    window_length_frames = int(window_lentgh_ms * fs/1000)

    if not window_length_frames % 2:
        window_length_frames += 1
    half_window = int((window_length_frames-1)/2)

    padded_rir = np.zeros(len(rir) + 2*half_window)
    padded_rir[half_window:-half_window] = rir
    output = np.zeros(len(rir) + 2*half_window)

    if window_type == 'rect':
        window_func = (1/window_length_frames) * np.ones(window_length_frames)
    elif window_type == 'hann':
        window_func = np.hanning(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'hamm':
        window_func = np.hamming(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'blac':
        window_func = np.blackman(window_length_frames)
        window_func = window_func / sum(window_func)
    elif window_type == 'bart':
        window_func = np.bartlett(window_length_frames)
        window_func = window_func / sum(window_func)
    else:
        raise ValueError('Unavailable window type.')

    for cursor in range(len(rir)):
        frame = padded_rir[cursor:cursor+window_length_frames]
        std = weightedStd(frame, window_func, use_local_avg)

        count = ((np.abs(frame) > std) * window_func).sum()

        output[cursor] = count  # Raw count without ERFC normalization

    return output[:-2*window_length_frames]

import numpy as np
from scipy.special import erfc

def echo_density_profile_unweighted(rir, window_length_ms=30, fs=44100):
    """
    Compute the Normalized Echo Density (NED) of a Room Impulse Response (RIR)
    using an unweighted approach.

    Parameters:
    - rir: numpy array
        The room impulse response signal.
    - window_length_ms: float
        The length of the sliding window in milliseconds.
    - fs: int
        The sampling frequency in Hz.

    Returns:
    - ned_profile: numpy array
        The normalized echo density profile.
    """
    window_length_samples = int(window_length_ms * fs / 1000)
    half_window = window_length_samples // 2

    # Pad the RIR to handle edge cases
    padded_rir = np.pad(rir, (half_window, half_window), 'constant', constant_values=(0, 0))
    ned_profile = np.zeros(len(rir))

    # ERFC constant for Gaussian noise
    ERFC = erfc(1 / np.sqrt(2))

    for i in range(len(rir)):
        # Extract the current window
        window = padded_rir[i:i + window_length_samples]

        # Compute mean and standard deviation
        mean = np.mean(window)
        std = np.std(window)

        # Count samples exceeding one standard deviation from the mean
        count = np.sum(np.abs(window - mean) > std)

        # Normalize by window length and ERFC
        ned_profile[i] = (count / window_length_samples) / ERFC

    return ned_profile
