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

        output[cursor] = (1/ERFC) * count

    return output[:-2*window_length_frames]

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
