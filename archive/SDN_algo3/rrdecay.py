import numpy as np
from scipy.signal.windows import hann
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy.signal import butter
from scipy.signal import firls
from scipy.ndimage import uniform_filter1d


def EDC(impulse_response):
    """
    Energy Decay Curve:
    Integral from t to infinity of the square of the impulse response,
    all divided by the integral from 0 to infinity of the square of the impulse response,
    presented in dB scale.
    from https://github.com/BrechtDeMan/pycoustics
    """
    impulse_response = np.array(impulse_response)
    print("EDC from rrdecay.py")
    cumul = 10 * np.log10(sum(impulse_response**2))
    decay_curve = 10 * np.log10(np.flipud(np.cumsum(np.flipud(np.square(impulse_response))))) - cumul
    return decay_curve


def EAC(impulse_response):
    """
    Energy Accumulation Curve:
    Integral from 0 to t of the square of the impulse response.
    """
    impulse_response = np.array(impulse_response)

    accumulation_curve = np.cumsum(np.square(impulse_response))
    return accumulation_curve


def EAD(impulse_response):
    """
    Energy Accumulation Development:
    Integral from 0 to t of the square of the impulse response, all divided by t.
    """
    accumulation_curve = EAC(impulse_response)
    time = np.array(range(1, 1+len(impulse_response)))
    return accumulation_curve / time


def WED(impulse_response, width, normalize=True):
    """
    Windowed Energy Development:
    Integral from t-width/2 to t+width/2 of the square of the impulse response.

    If normalize=True, the output is normalized by the window length,
    meaning that the result can be interpreted as the average energy per sample.
    """
    if width % 2 != 0:
        width += 1
    half_width = int(width/2)

    accumulation_curve = EAC(impulse_response)
    anticipated_accumulation_curve = accumulation_curve[half_width:]
    delayed_accumulation_curve = np.concatenate((np.zeros(half_width),
                                                 accumulation_curve[:-width]))
    windowed_curve = anticipated_accumulation_curve - delayed_accumulation_curve

    if normalize:
        windowed_curve /= width

    return windowed_curve


def rrdecay(ir, fs, flag=False, zeta=20, order=2):
    # fb = 125 * 2.^([-2:zeta:7])  # filterbank band center frequencies, Hz
    fb = 125 * np.logspace(-2, 7, zeta, base=2)  # filterbank band center frequencies, Hz
    nbands = len(fb)  # impulse response filterbank band count, bands
    beta = 10  # band energy smoothing filter duration, milliseconds
    eta = 20  # noise floor estimation window length, milliseconds
    delta1 = 10  # band decay rate estimation window end level, dB
    ntaps = len(ir)

    # irb = ir * np.ones((1, nbands))
    irb = np.repeat(ir[:, np.newaxis], 20, axis=1)
    for i in range(nbands):
        edges = fb[i] * 2**((9/zeta) * np.array([-2, 2]))
        b, a = butter(order, [min((0.9, edges[0] * 2/fs)),
                              min((0.9, edges[1] * 2/fs))],
                      btype='pass')
        irb[:, i] = filtfilt(b, a, irb[:, i])

    if not flag:
        staps = round(beta/2 * fs/1000)
        bS = hann(2*staps-1)/sum(hann(2*staps-1))

        irbs = np.real(np.sqrt(filtfilt(bS, [1], irb**2, axis=0)))
        irbs = irbs[2*staps-1 + np.array(range(min((ntaps, irbs.shape[0]-2*staps)))), :]

        etaps = round(eta * fs / 1000)
        # basis = [ones(etaps,1) [1:etaps]'/fs];
        temp = np.ones((etaps, 1))
        temp[:, 0] = np.array(range(etaps))/fs
        basis = np.concatenate((np.ones((etaps, 1)), temp), axis=1)
        # theta = basis \ (20*log10(irbs(end+[1-etaps:0],:)));
        theta = np.linalg.lstsq(basis, 20*np.log10(irbs[np.array(range(-etaps, 0)), :]))[0]
        nu = theta[0] + theta[1]*etaps / fs

        temp = np.unravel_index(np.argmax(irbs, axis=0), np.shape(irbs))[0]
        # preroll = round(median(temp))
        preroll = min((1, min(temp)))

        rt60 = np.zeros((nbands,1))
        for i in range(nbands):
            index0 = temp[i]
            indices = np.argwhere(20*np.log10(irbs[preroll:, i]) < nu[i] + delta1)
            if indices.shape[0] > 0:
                index1 = indices[0, 0] + preroll
            else:
                index1 = index0
            # index1 = find(20 * log10(irbs(preroll:end, i)) < nu(i) + delta1, 1) + preroll;
            index1 = min(index1, round(1000*fs/1000))

            basis = np.ones((max(0, index1-index0), 1))/fs
            # theta = basis \ (20*log10(irbs(index,i)));
            theta = np.linalg.lstsq(basis, 20*np.log10(irbs[index0:index1, i]))[0]

            rt60[i] = -60/theta[0]
    else:
        # TODO: fix this case
        """
        t60_low = [0.3, 0.5]  # t60 limits for 2 stage decay (in s)
        t60_high = [2, 4]
        rt60 = np.zeros((nbands,2))
        smbeta = 50
        
        for i in range(len(nbands)):
            _, start = max(mag2db(abs(irb[:, i])))
            rirEnv = energyEnvelope(irb[round(start+round(0.005*fs)):3*fs,i],fs,smbeta)
            t_env = (0:1/fs:(length(rirEnv)-1)/fs)
            taus, gm, ir_fit = fit_two_stage_decay(rirEnv,t_env,[t60_low, t60_high])
    
            rt60(i,:) = taus * log(1000)
        """
        pass

    return rt60, fb

def calculate_speed_of_sound(t, h, p):
    """
    Compute the speed of sound as a function of
    temperature, humidity and pressure

    Arguments
    ---------

    t: temperature [Celsius]
    h: relative humidity [%]
    p: atmospheric pressure [kpa]

    Return
    ------

    Speed of sound in [m/s]
    """

    # using crude approximation for now
    return 331.4 + 0.6 * t + 0.0124 * h


# print(calculate_speed_of_sound(t=20, h=50, p=100))

# EGE

def calculate_rir_error(rir1, rir2, fs, window_length=0.05, smoothing_window=10):
    """
    Calculate the error between two RIRs within a given time window.

    Parameters:
    - rir1, rir2: np.array
        The two RIRs (room impulse responses) to compare.
    - fs: int
        Sampling frequency in Hz.
    - window_length: float
        Length of the comparison window in seconds (default is 0.05s or 50ms).
    - smoothing_window: int
        Length of the smoothing window (in samples) for uniform smoothing.

    Returns:
    - error: float
        The error measure between the smoothed squared RIRs.
    """

    # Convert time window to sample index
    end_sample = int(window_length * fs)

    # Square the RIRs to get energy
    energy1 = rir1[:end_sample] ** 2
    energy2 = rir2[:end_sample] ** 2

    # Smooth the energy using a uniform filter
    smoothed_energy1 = uniform_filter1d(energy1, size=smoothing_window)
    smoothed_energy2 = uniform_filter1d(energy2, size=smoothing_window)

    # Compute the difference between smoothed energies
    error = np.sum((smoothed_energy1 - smoothed_energy2) ** 2)

    return error

def calculate_edc_error_db(edc_sdn, edc_ism, start_time, end_time, fs):
    """
    Calculate the error between SDN and ISM EDCs in dB over a specified time range.

    Parameters:
        edc_sdn (ndarray): EDC array (in dB) from the SDN simulation.
        edc_ism (ndarray): EDC array (in dB) from the ISM simulation.
        start_time (float): Start time of the error computation in seconds.
        end_time (float): End time of the error computation in seconds.
        fs (int): Sampling frequency in Hz.

    Returns:
        float: Error value between the two EDCs.
    """
    # Convert start and end times to sample indices
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)

    # Extract the relevant time range
    edc_sdn_section = edc_sdn[start_idx:end_idx]
    edc_ism_section = edc_ism[start_idx:end_idx]

    # Convert dB to linear scale
    edc_sdn_linear = 10 ** (edc_sdn_section / 10)
    edc_ism_linear = 10 ** (edc_ism_section / 10)

    # Compute the mean squared error in the linear domain
    mse = np.mean((edc_sdn_linear - edc_ism_linear) ** 2)

    # (RMSE) for better interpretability
    rmse = np.sqrt(mse)
    return rmse