import numpy as np
import pyroomacoustics as pra
from copy import deepcopy
import sys
sys.path.append('SDN-Simplest_Hybrid_HO-SDN/SDNPy')
from src import Simulation_HO_SDN_centroid as sim_HO_SDN_centroid
from src import Geometry as geom
from src import Signal as sig
from src import Source as src
from src import Microphone as mic
from sdn_core import DelayNetwork
from archive.sdn_base import calculate_sdn_base_rir
from scipy.signal import find_peaks

def rir_normalisation(rirs_dict, room, Fs, normalize_to_first_impulse=True):
    """
    Normalize RIRs either to maximum absolute value or to first impulse.
    
    Args:
        rirs_dict (dict or np.ndarray): Dictionary of RIRs to normalize or a single RIR array
        room: Room object containing source and receiver positions
        Fs: Sampling frequency
        normalize_to_first_impulse (bool): If True, normalize to direct sound value
                                         If False, normalize to maximum absolute value
    
    Returns:
        dict or np.ndarray: Dictionary of normalized RIRs or a single normalized RIR
    """
    normalized_rirs = {}

    # Check if input is a single RIR array
    if isinstance(rirs_dict, np.ndarray):
        rirs_dict = {'single_rir': rirs_dict}  # Convert to dict for uniform processing
        # print("Input is a single RIR array, converting to dictionary format.")

    if normalize_to_first_impulse:
        # print("Normalizing to direct- first impulse")
        # Calculate theoretical direct sound arrival time
        direct_distance = room.micPos.getDistance(room.source.srcPos)
        direct_time = direct_distance / 343.0  # speed of sound in m/s
        direct_sample = int(direct_time * Fs)

        # Use a small window around the theoretical arrival time to find the peak
        window_size = 20  # samples
        for label, rir in rirs_dict.items():
            end_idx = direct_sample + window_size
            window = rir[:end_idx]
            max_in_window_idx = np.argmax(np.abs(window))
            normalized_rirs[label] = rir / abs(rir[max_in_window_idx])
    else:
        print("Normalizing to maximum absolute value")
        # Normalize to maximum absolute value
        for label, rir in rirs_dict.items():
            normalized_rirs[label] = rir / np.max(np.abs(rir))
            
    return normalized_rirs

def pad_zeros_to_rir(rir, num_samples):
    if len(rir) < num_samples:
        # Pad with zeros to reach num_samples
        rir = np.pad(rir, (0, num_samples - len(rir)))
    else:
        # Truncate if longer
        rir = rir[:num_samples]
    return rir

def calculate_pra_rir(room_parameters, duration, Fs, use_rand_ism=False, max_rand_disp=0.1, max_order=100):
    """
    Calculate RIR using PRA (PyRoomAcoustics) with ISM.
    
    Args:
        room_params (dict): Room parameters including dimensions and absorption
        source_pos (tuple): Source position (x, y, z)
        mic_pos (tuple): Microphone position (x, y, z)
        duration (float): Duration of RIR in seconds
        Fs (int): Sampling frequency
        max_order (int): Maximum reflection order for ISM
        
    Returns:
        tuple: (normalized RIR, room object)
    """
    # Room dimensions for PRA
    room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])
    
    if room_parameters.get('air') is None:
        pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                materials=pra.Material(energy_absorption = room_parameters['absorption']),
                                max_order=max_order,
                                air_absorption=False, ray_tracing=False,use_rand_ism=use_rand_ism, max_rand_disp=max_rand_disp)
    else:
        print("air absorption True")
        pra_room = pra.ShoeBox(room_dim, fs=Fs,
                                materials=pra.Material(energy_absorption=room_parameters['absorption']),
                                max_order=max_order,
                                temperature=room_parameters['air']['temperature'],
                                humidity=room_parameters['air']['humidity'], air_absorption=True)
    pra_room.set_sound_speed(343)
    
    # Add source and receiver
    # Setup room for ISM with PRA package
    source_loc = np.array([room_parameters['source x'], room_parameters['source y'], room_parameters['source z']])
    mic_loc = np.array([room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']])

    pra_room.add_source(source_loc).add_microphone(mic_loc)
    
    # Compute RIR
    pra_room.compute_rir()
    pra_rir = pra_room.rir[0][0]
    
    # Process the RIR
    global_delay = pra.constants.get("frac_delay_length") // 2
    pra_rir = pra_rir[global_delay:]  # Shift left by removing the initial delay
    pra_rir = np.pad(pra_rir, (0, global_delay))  # Pad with zeros at the end to maintain length

    num_samples = int(Fs * duration)
    rir = pad_zeros_to_rir(pra_rir, num_samples)
    
    # Normalize
    # rir = rir / np.max(np.abs(rir))
    
    if use_rand_ism and max_rand_disp == 0.1:
        label = 'ISM-pra-rand10'
    elif use_rand_ism == False:
        label = 'ISM'
    
    return rir, label

def calculate_rimpy_rir(room_parameters, duration, Fs, reflection_sign, tw_fractional_delay_length=0, randDist=0.0):
    """
    Calculate RIR using rimPy ISM.
    
    Args:
        room_params (dict): Room parameters including dimensions and absorption
        source_pos (tuple): Source position (x, y, z)
        mic_pos (tuple): Microphone position (x, y, z)
        duration (float): Duration of RIR in seconds
        Fs (int): Sampling frequency
        reflection_sign (int): 1 for positive reflection coefficients, -1 for negative
        
    Returns:
        tuple: (normalized RIR, room object)
    """
    import sys
    sys.path.append('SDN-Simplest_Hybrid_HO-SDN/SDNPy')
    from rimPypack import rimPy as rimpy

    # Room dimensions for rimPy
    room_dim = np.array([room_parameters['width'], room_parameters['height'], room_parameters['depth']])
    
    # Add source and receiver
    source_loc = np.array([room_parameters['source x'],
                            room_parameters['source z'],
                            room_parameters['source y']])
    mic_loc = np.array([room_parameters['mic x'],
                            room_parameters['mic z'],
                            room_parameters['mic y']])
    
    # Calculate RIR duration in seconds
    rirDuration = duration
    
    # Set reflection coefficients
    reflection = room_parameters['reflection']
    if reflection_sign < 0 and randDist == 0.1:
        label = 'ISM (rimpy-neg10)'
        reflection = -reflection

    elif reflection_sign < 0:
        label = 'ISM (rimpy-neg)'
        reflection = -reflection

    elif reflection_sign > 0 and randDist == 0.1:
        label = 'ISM (rimpy-pos10)'
    else:
        label = 'ISM-(rimpy-pos)'

    # Run rimPy calculation
    beta = reflection * np.ones((2, 3))
    h_rimpy = rimpy.rimPy(
        micPos=mic_loc,
        sourcePos=source_loc,
        roomDim=room_dim,
        beta=beta,
        rirDuration=rirDuration,  # duration divided by 2 for faster computation
        fs=Fs,
        randDist=randDist,  # 0.1 --- 0.0 for standard image source method
        tw=tw_fractional_delay_length,  # =0 for no fractional delay
        fc=None,
        c=343
    )

    rimpy_rir = h_rimpy[:, 0]
    # rimpy_rir = rimpy_rir / np.max(np.abs(rimpy_rir))

    return rimpy_rir, label

def calculate_sdn_rir(room_parameters, test_name, room, duration, Fs, config):
    """
    Calculate RIR using SDN (Scattering Delay Network).
    
    Args:
        room_params (dict): Room parameters including dimensions and absorption
        source_pos (tuple): Source position (x, y, z)
        mic_pos (tuple): Microphone position (x, y, z)
        duration (float): Duration of RIR in seconds
        Fs (int): Sampling frequency
        config (dict): Configuration for SDN
        
    Returns:
        tuple: (normalized RIR, room object)
    """
    # Create room object
    # print("Running", test_name, "with:")
    if 'absorption' in config:
    # Override absorption
        room_parameters['absorption'] = config['absorption']
        room_parameters['reflection'] = np.sqrt(1 - config['absorption'])
    room.wallAttenuation = [room_parameters['reflection']] * 6

    flags = config.get('flags', {})
    # Create SDN instance with configured flags
    sdn = DelayNetwork(room, Fs=Fs, label=config['label'], **flags)

    # Calculate RIR
    rir = sdn.calculate_rir(duration)
    
    # Normalize
    # rir = rir / np.max(np.abs(rir))

    # Check if this is a default configuration (no flags)
    is_default = len(config.get('flags', {})) == 0

    # Store result with optional info
    if is_default:
        label = 'SDN-Original'
    else:
        label = f'SDN-{test_name}'
        
    if 'info' in config:
        label += f': {config["info"]}'
    
    return sdn, rir, label, is_default

def calculate_ho_sdn_rir(room_parameters, Fs, duration, source_signal='dirac', order=None):
    import geometry
    """
    Calculate RIR using HO-SDN (Higher-Order Scattering Delay Network).
    
    Args:
        room_params (dict): Room parameters including dimensions and absorption
        room (Room): Room object with source and microphone positions
        Fs (int): Sampling frequency
        duration (float): Duration of RIR in seconds
        source_signal (str): Type of source signal ('dirac' or 'gaussian')
        order (int): Order of HO-SDN (2 or 3)
        
    Returns:
        tuple: (normalized RIR, label)
    """

    label = f'HO-SDN N={order}'
    num_samples = int(Fs * duration)

    # Create room object
    ho_room = geom.Room()
    ho_room.shape = geom.Cuboid(room_parameters['width'], room_parameters['height'], room_parameters['depth'])
    ho_room.wallFilters = [None] * 6
    ho_room.wallAttenuation = [room_parameters['reflection']] * 6

    # Create signal based on source_signal parameter
    if source_signal == 'gaussian':
        signal_data = geometry.Source.generate_signal('gaussian', num_samples)['signal']
    else:  # Default to dirac
        signal_data = geometry.Source.generate_signal('dirac', num_samples)['signal']
    
    ho_signal = sig.Signal(Fs, signal_data)
    # Create source and mic objects
    ho_source = src.Source(geom.Point(room_parameters['source x'],
                                        room_parameters['source z'],
                                        room_parameters['source y']), #their notation is !!!! ....
                             ho_signal)
    ho_mic = mic.Microphone(geom.Point(room_parameters['mic x'],
                                        room_parameters['mic z'],
                                        room_parameters['mic y']))
    
    # Fixed parameters for HO-SDN
    ho_params = {
        'order': order,
        'connection': 'full',
        'node selection': 'all',
        'matrix': 'isotropic',
        'skeleton extras': 0
    }
    
    # Create simulation
    simulate_mod = sim_HO_SDN_centroid.Simulation(deepcopy(ho_room), ho_source, ho_mic, None, num_samples, ho_params['order'],
                                ho_params['connection'],
                                room_parameters.get('air'),
                                ho_params['skeleton extras'],
                                ho_params['matrix'])
    
    # Run simulation
    rir_ho_sdn, _, _ = simulate_mod.run() # multiAudio, sdnKickIn
    rir_ho_sdn = pad_zeros_to_rir(rir_ho_sdn, num_samples)

    # Normalize
    # rir_ho_sdn = rir_ho_sdn / np.max(np.abs(rir_ho_sdn))

    
    return rir_ho_sdn, label
