
import numpy as np
import matplotlib.pyplot as plt

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

def compute_edc(rir, Fs, label=None):
    # simplest edc but corrected (total energy ddivision) according to deepseek
    # Step 1: Compute the squared RIR
    rir_squared = rir ** 2

    # Step 2: Reverse the squared RIR and compute the cumulative sum
    reversed_rir_squared = rir_squared[::-1]
    cumulative_energy = np.cumsum(reversed_rir_squared)

    # Step 3: Reverse the cumulative sum back to get the EDC
    edc = cumulative_energy[::-1]

    # Step 4: Normalize the EDC by the total energy
    total_energy = np.sum(rir_squared)
    normalized_edc = edc / total_energy

    # Step 5: Convert to dB scale
    edc_dB = 10.0 * np.log10(normalized_edc + 1e-10)  # Add small offset to avoid log(0)

    # Step 6: Plot the EDC
    time_axis = np.arange(len(edc_dB)) / Fs
    plt.plot(time_axis, edc_dB, label=label, alpha=0.7)