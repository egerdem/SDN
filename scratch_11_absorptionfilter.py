import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import freqz
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5
# ----------------------------
# 1. Define the sparse absorption coefficients and frequencies
# ----------------------------
# Absorption coefficients (alpha) for the octave bands
alpha = np.array([0.2, 0.23, 0.3, 0.45, 0.6, 0.6, 0.55, 0.6])
# Corresponding center frequencies (Hz) for the octave bands
freqs_sparse = np.array([125, 250, 500, 1000, 2000, 4000, 8000, 16000])

# Convert absorption coefficients to reflectance magnitudes:
# |R(jω)| = sqrt(1 - α)
reflectance_sparse = np.sqrt(1 - alpha)

# ----------------------------
# 2. Interpolate to create a dense target curve
# ----------------------------
# For better interpolation, work in log-frequency domain.
# We extend the curve slightly by extrapolating to f_min and f_max.
f_min = 50   # Extend below the lowest octave band
f_max = 24000  # Extend above the highest (Nyquist-ish for fs=48000)
# Create a logarithmically spaced frequency vector for the target
frequencies_dense = np.logspace(np.log10(f_min), np.log10(f_max), 500)

# Perform interpolation in log-frequency domain.
# First, take the log of the sparse frequencies.
log_freqs_sparse = np.log10(freqs_sparse)
log_f_dense = np.log10(frequencies_dense)
# Use linear interpolation in the log domain
reflectance_dense = np.interp(log_f_dense, log_freqs_sparse, reflectance_sparse)

# ----------------------------
# 3. Filter design via least-squares fitting
# ----------------------------
# We will design a third-order IIR filter of the form:
#    H(z) = (b0 + b1 z^-1 + b2 z^-2 + b3 z^-3) / (1 + a1 z^-1 + a2 z^-2 + a3 z^-3)
# The optimization variables are: [b0, b1, b2, b3, a1, a2, a3]
# We fix a0 = 1.

fs = 48000  # Sampling frequency (Hz)
# Convert our dense frequencies (Hz) to normalized angular frequencies (radians/sample)
w_dense = 2 * np.pi * frequencies_dense / fs

def filter_error(x, w, target):
    # Extract numerator and denominator coefficients
    b = x[0:4]
    a = np.r_[1, x[4:]]  # a0 fixed to 1
    # Compute frequency response using freqz
    w_out, H = freqz(b, a, worN=w)
    # We compare magnitudes (absolute values)
    H_mag = np.abs(H)
    return H_mag - target

# Initial guess: start with a filter that is all-pass (i.e., H=1)
x0 = np.zeros(7)
x0[0] = 1.0  # b0=1

# Run the least-squares optimization
result = least_squares(filter_error, x0, args=(w_dense, reflectance_dense))

# Retrieve optimized filter coefficients
b_opt = result.x[0:4]
a_opt = np.r_[1, result.x[4:]]

# Compute the frequency response of the optimized filter over our dense grid
w_out, H_opt = freqz(b_opt, a_opt, worN=w_dense)
H_opt_mag = np.abs(H_opt)

# ----------------------------
# 4. Plot the results
# ----------------------------

# Figure 1: Sparse points and interpolated target curve
plt.figure(figsize=(10, 5))
plt.semilogx(freqs_sparse, reflectance_sparse, 'o', label='Sparse Reflectance Points')
plt.semilogx(frequencies_dense, reflectance_dense, '-', label='Interpolated Target')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Reflectance |R(jω)|')
plt.title('Target Reflectance Curve (Sparse & Interpolated)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Figure 2: Target curve vs. Fitted Filter Frequency Response
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies_dense, reflectance_dense, '-', label='Interpolated Target')
plt.semilogx(frequencies_dense, H_opt_mag, '--', label='Fitted IIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fitted Filter Magnitude Response vs. Target Reflectance')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Figure 3: Error between the filter response and target curve
error = H_opt_mag - reflectance_dense
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies_dense, error, 'r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Error (Filter - Target)')
plt.title('Error between Fitted Filter and Target Reflectance')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# (Optional) Figure 4: Impulse response of the designed filter
from scipy.signal import dimpulse
# Create an FIR impulse (delta function)
n_points = 50
t, h_imp = dimpulse((b_opt, a_opt, 1), n=n_points)
h_imp = np.squeeze(h_imp)
plt.figure(figsize=(10, 5))
plt.stem(np.arange(n_points), h_imp)
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.title('Impulse Response of the Designed Filter')
plt.grid(True, ls="--")
plt.tight_layout()
plt.show()
