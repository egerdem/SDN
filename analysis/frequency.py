from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt

def gaussian_impulse(num_samples, num_gaussian_samples = 50, std_dev = 10, plot=False):
    # std_dev of the Gaussian (controls the width)
    gaussian_pulse = gaussian(num_gaussian_samples, std_dev)
    # Normalize the pulse to have a maximum amplitude of 1
    gaussian_pulse = gaussian_pulse / np.max(gaussian_pulse)
    # Calculate the number of zeros to append
    num_zeros = num_samples - num_gaussian_samples
    # Append zeros to the Gaussian pulse
    gaussian_pulse = np.pad(gaussian_pulse, (0, num_zeros), mode='constant')
    if plot:
        # Plot the Gaussian pulse
        plt.plot(gaussian_pulse)
        plt.title('Smoothed Gaussian Impulse')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.show()
    return(gaussian_pulse)

def calculate_frequency_response(signal, sample_rate):
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    # Compute the FFT of the signal
    fft_result = np.fft.fft(windowed_signal)

    # Compute the magnitude of the FFT result
    magnitude = np.abs(fft_result)
    log_magnitude = 20*np.log10(magnitude)

    # Compute the corresponding frequency bins
    n = len(windowed_signal)
    freq_bins = np.fft.fftfreq(n, d=1 / sample_rate)

    # Only take the positive frequencies (since FFT output is symmetric)
    positive_freq = freq_bins[:n // 2]
    positive_magnitude = log_magnitude[:n // 2]

    return positive_freq, positive_magnitude


def plot_frequency_response(freq, magnitude, label='Frequency Response'):
    plt.figure(figsize=(10, 6))
    # plt.semilogx(freq, magnitude)  # Use semilogx for logarithmic frequency axis
    plt.plot(freq, magnitude)
    plt.title(label)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Example usage
    sample_rate = 44100  # Sample rate in Hz
    N = 1024  # Number of samples
    signal = np.random.randn(N)  # Replace with your actual signal

    freq, magnitude = calculate_frequency_response(signal, sample_rate)
    plot_frequency_response(freq, magnitude)