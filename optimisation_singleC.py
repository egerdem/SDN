"""
Optimises the six c_dom parameters of the SDN so that its
energy–decay curve matches an Image‑Source‑Method reference.

Run:  python optimize_cdom.py
"""
import geometry
import analysis as an
from rir_calculators import calculate_pra_rir
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import plot_room as pp
from sdn_core import DelayNetwork
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Build room and get reference EDC once
# ------------------------------------------------------------------
Fs = 44100
duration = 1  # seconds – long enough for early & mid decay
optimization_duration = 0.05  # seconds – short enough for fast optimization
print("optimization dur. [s]:", optimization_duration)
num_samples = int(Fs * duration)
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)

room_aes = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

room_waspaa = {
    'display_name': 'WASPAA Room',
    'width': 6, 'depth': 7, 'height': 4,
    # 'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
    'source x': 3.6, 'source y': 6, 'source z': 1.3,
    # 'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
    'mic x': 1.833333, 'mic y': 3, 'mic z': 2.4,
    'absorption': 0.1,
}

room_journal = {
    'display_name': 'Journal Room',
    'width': 3.2, 'depth': 4, 'height': 2.7,
    'source x': 2, 'source y': 3., 'source z': 2,
    'mic x': 1, 'mic y': 1, 'mic z': 1.5,
    'absorption': 0.1,
}


def construct_room_object(params, Fs=44100, source_signal=impulse_dirac['signal']):
    """Constructs a Room object from the given room parameters."""
    room = geometry.Room(params['width'], params['depth'], params['height'])
    room.set_microphone(params['mic x'], params['mic y'], params['mic z'])
    room.set_source(params['source x'], params['source y'], params['source z'],
                    signal="will be replaced", Fs=Fs)
    room_dim = np.array([params['width'], params['depth'], params['height']])
    room.source.signal = source_signal
    params['reflection'] = np.sqrt(1 - params['absorption'])
    room.wallAttenuation = [params['reflection']] * 6
    return room, room_dim


room_parameters = room_aes
room, room_dim = construct_room_object(room_aes, source_signal=impulse_dirac['signal'])
# room_parameters = room_aes  # Choose the room
# room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
# room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
# room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
#                 signal = "will be replaced", Fs = Fs)
# room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])
# Setup signal
# room.source.signal = impulse_dirac['signal']
# room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
# room.wallAttenuation = [room_parameters['reflection']] * 6

# ---- reference RIR & EDC via pyroomacoustics ISM -----------------
pra_rir, label = calculate_pra_rir(room_parameters, duration, Fs, 100)
rir_ref = pra_rir
edc_ref, _, _ = an.compute_edc(rir_ref, Fs)

# T60_ref = pra.experimental.rt60.measure_rt60(rir_ref, Fs)  # just to clip error window
# rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, room_parameters['absorption'])
# T60_ref = rt60_sabine
# cut_smpl = int(min(T60_ref, dur-0.01) * Fs)
cut_smpl = int(optimization_duration * Fs)

# ------------------------------------------------------------------
# 2. Create SDN once - reuse for all optimizations
# ------------------------------------------------------------------
# Initialize the SDN with default config
test_name = 'TestX'
config = {'enabled': True,
          'info': "",
          'flags': {
              'specular_source_injection': True,
              'source_weighting': 5,  # Initial value, will be updated
          },
          'label': "SDN"
          }

# Create the SDN network once
sdn = DelayNetwork(room, Fs=Fs, label=config['label'], **config['flags'])


# ------------------------------------------------------------------
# 3. Function to update c_vector and calculate RIR
# ------------------------------------------------------------------
def calculate_rir_with_c(sdn, c_scalar):
    """Calculate RIR with updated c_scalar without recreating the network"""
    # Update the injection_c_vector in the SDN
    sdn.source_weighting = c_scalar
    # Reset injection index to 0 for the new calculation
    sdn.injection_index = 0

    # Calculate RIR
    return sdn.calculate_rir(duration)


# ------------------------------------------------------------------
# 4. Objective function using the existing SDN
# ------------------------------------------------------------------
def compute_RMSE_optimizer(c, edc_ref, sdn):
    """RMSE between SDN EDC (with given c_vec) and reference EDC"""
    # The optimizer might pass c as an array, so extract the scalar value
    c_scalar = c[0] if isinstance(c, (np.ndarray, list)) else c

    # Calculate RIR with the updated c_vector
    rir_sdn = calculate_rir_with_c(sdn, c_scalar)

    # Compute EDC
    edc_sdn, _, _ = an.compute_edc(rir_sdn, Fs)
    # Cut EDCs up to cut_smpl
    edc_sdn = edc_sdn[:cut_smpl]
    edc_ref_cut = edc_ref[:cut_smpl]

    # Calculate RMS difference
    rms_diff = an.compute_RMS(edc_sdn, edc_ref_cut, range=50, Fs=Fs, method="rmse")
    print(f"c: {c_scalar:.4f}, RMSE: {rms_diff:.6f}")
    optimization_log.append((c_scalar, rms_diff))
    return rms_diff


# ------------------------------------------------------------------
# 5. Optimisation call ---------------------------------------------
# ------------------------------------------------------------------
# Set an initial guess for the optimizer
x0 = [0.0]
bounds = [(-3, 7.0)]
optimization_log = []

# Configure the local minimizer (L-BFGS-B) to be used by basinhopping.
# This includes passing the bounds and additional arguments for our objective function.
minimizer_kwargs = {
    "method": "L-BFGS-B",
    "bounds": bounds,
    "args": (edc_ref, sdn)
}

result = basinhopping(compute_RMSE_optimizer, x0,
                      minimizer_kwargs=minimizer_kwargs,
                      niter=20,
                      disp=True)


print("\nBest c_dom vector:", np.round(result.x, 3))
print("RMSE:", result.fun)

# ------------------------------------------------------------------
# 6. Plotting the optimization results
# ------------------------------------------------------------------
if optimization_log:
    # Sort the log by c value for a clean plot
    optimization_log.sort(key=lambda x: x[0])
    c_values, loss_values = zip(*optimization_log)

    plt.figure(figsize=(10, 6))
    plt.plot(c_values, loss_values, 'bo-', label='Optimization Path')
    plt.scatter(result.x, [result.fun], color='red', s=100, zorder=5, label=f'Minimum Found (c={np.round(result.x[0], 3)})')
    plt.xlabel("c value")
    plt.ylabel("RMSE Loss")
    plt.title("Optimization Error Surface for c")
    plt.legend()
    plt.grid(True)
    plt.show()
