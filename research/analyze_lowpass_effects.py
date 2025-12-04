"""
Analyze the effect of low-pass filtering on RIR analysis metrics.

This script loads existing RIR data and applies various low-pass filtering approaches
to study their effect on EDC RMSE and other metrics. Filtering is done on-the-fly
without modifying the original data files.

Two filtering approaches:
1. Butterworth low-pass (sharp frequency cutoff)
2. Gaussian smoothing (time-domain local averaging - softer, more like air absorption)
"""

import numpy as np
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from analysis import analysis as an
from analysis import plot_room as pp
from analysis.plotting_utils import load_data, get_display_name, DISPLAY_NAME_MAP
from rir_calculators import rir_normalisation
import geometry
import matplotlib.pyplot as plt


def apply_butterworth_lowpass(rir, cutoff_hz, fs, order=4):
    """
    Apply Butterworth low-pass filter (sharp frequency cutoff).
    
    Parameters
    ----------
    rir : np.ndarray
        Input RIR
    cutoff_hz : float
        Cutoff frequency in Hz
    fs : int
        Sample rate
    order : int
        Filter order (higher = sharper cutoff)
        
    Returns
    -------
    np.ndarray
        Filtered RIR
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, rir)


def apply_gaussian_smoothing(rir, sigma_samples):
    """
    Apply Gaussian smoothing in time domain (local averaging effect).
    
    This creates a softer, more gradual effect similar to air absorption,
    without a sharp frequency cutoff. It's equivalent to convolving with
    a Gaussian kernel.
    
    Parameters
    ----------
    rir : np.ndarray
        Input RIR
    sigma_samples : float
        Standard deviation of Gaussian kernel in samples.
        Larger sigma = more smoothing (lower effective cutoff).
        Example: sigma=5 at 44.1kHz ≈ 0.11ms smoothing window
        
    Returns
    -------
    np.ndarray
        Smoothed RIR
    """
    return gaussian_filter1d(rir, sigma=sigma_samples, mode='nearest')


def compute_metrics_for_filtered_rirs(data_files, filter_configs, reference_method='RIMPY-neg10', methods_to_analyze=None):
    """
    Load RIR data, apply filters, and compute comparison metrics.
    
    Parameters
    ----------
    data_files : list of str
        List of .npz filenames to process
    filter_configs : list of dict
        Each dict should have:
        - 'type': 'butterworth' or 'gaussian'
        - 'params': parameters for the filter
        - 'label': display name
    reference_method : str
        Method to use as reference for error computation
    methods_to_analyze : list of str or None
        Specific methods to analyze. If None, analyzes all methods in the file.
        
    Returns
    -------
    dict
        Results with two types of keys:
        - '<filename>_metrics': EDC RMSE metrics for each method/filter
        - '<filename>_filtered_rirs': Normalized filtered RIRs for plotting
    """
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, "results", "paper_data")
    
    all_results = {}
    
    for filename in data_files:
        data_path = os.path.join(data_dir, filename)
        if not os.path.exists(data_path):
            print(f"⚠️  File not found: {data_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"📁 Processing: {filename}")
        print(f"{'='*80}")
        
        # Load data
        sim_data = load_data(data_path)
        all_rirs = sim_data['rirs']
        receiver_positions = sim_data['receiver_positions']
        Fs = sim_data['Fs']
        
        # Check reference method exists
        if reference_method not in all_rirs:
            print(f"❌ Reference method '{reference_method}' not found in {filename}")
            print(f"   Available methods: {list(all_rirs.keys())}")
            continue
        
        ref_rirs = all_rirs[reference_method]
        num_receivers = len(receiver_positions)
        
        # Setup room for normalization
        room_params = sim_data['room_params']
        room = geometry.Room(room_params['width'], room_params['depth'], room_params['height'])
        source_pos = sim_data['source_pos']
        room.set_source(source_pos[0], source_pos[1], source_pos[2])
        
        print(f"ℹ️  Reference: {reference_method}")
        print(f"ℹ️  Receivers: {num_receivers}")
        print(f"ℹ️  Sample rate: {Fs} Hz")
        print(f"ℹ️  Methods in file: {len(all_rirs)}")
        
        file_results = {}
        filtered_rirs_for_plot = {}  # Store filtered RIRs for interactive plotting
        
        # Process each test method (excluding reference)
        all_methods = [m for m in all_rirs.keys() if m != reference_method]
        
        # Filter by methods_to_analyze if specified
        if methods_to_analyze is not None:
            test_methods = [m for m in all_methods if m in methods_to_analyze]
            if not test_methods:
                print(f"⚠️  None of the specified methods found in {filename}")
                print(f"   Specified: {methods_to_analyze}")
                print(f"   Available: {all_methods}")
                continue
        else:
            test_methods = all_methods
        
        print(f"📊 Analyzing {len(test_methods)} methods: {test_methods}")
        
        for method_name in test_methods:
            method_rirs = all_rirs[method_name]
            
            # Original (unfiltered) metrics
            original_rmses = []
            for i in range(num_receivers):
                # Compute EDCs
                ref_edc, _, _ = an.compute_edc(ref_rirs[i], Fs, plot=False)
                test_edc, _, _ = an.compute_edc(method_rirs[i], Fs, plot=False)
                
                # Compute RMSE (50ms)
                rmse = an.compute_RMS(
                    ref_edc, test_edc,
                    range=50, Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True
                )
                original_rmses.append(rmse)
            
            original_mean = np.mean(original_rmses)
            
            # Store original
            file_results[method_name] = {
                'original': {
                    'rmses': original_rmses,
                    'mean': original_mean
                },
                'filtered': {}
            }
            
            # Store first receiver's RIR for plotting (original)
            if method_name not in filtered_rirs_for_plot:
                filtered_rirs_for_plot[method_name] = {}
            filtered_rirs_for_plot[method_name]['Original'] = method_rirs[0]
            
            # Apply each filter configuration
            for filter_cfg in filter_configs:
                filter_label = filter_cfg['label']
                filter_type = filter_cfg['type']
                filter_params = filter_cfg['params']
                
                filtered_rmses = []
                
                for i in range(num_receivers):
                    # Apply filter to BOTH reference and test RIRs
                    if filter_type == 'butterworth':
                        ref_filtered = apply_butterworth_lowpass(
                            ref_rirs[i], 
                            cutoff_hz=filter_params['cutoff_hz'],
                            fs=Fs,
                            order=filter_params.get('order', 4)
                        )
                        test_filtered = apply_butterworth_lowpass(
                            method_rirs[i],
                            cutoff_hz=filter_params['cutoff_hz'],
                            fs=Fs,
                            order=filter_params.get('order', 4)
                        )
                    elif filter_type == 'gaussian':
                        ref_filtered = apply_gaussian_smoothing(
                            ref_rirs[i],
                            sigma_samples=filter_params['sigma_samples']
                        )
                        test_filtered = apply_gaussian_smoothing(
                            method_rirs[i],
                            sigma_samples=filter_params['sigma_samples']
                        )
                    else:
                        print(f"⚠️  Unknown filter type: {filter_type}")
                        continue
                    
                    # Compute EDCs from filtered RIRs
                    ref_edc_filt, _, _ = an.compute_edc(ref_filtered, Fs, plot=False)
                    test_edc_filt, _, _ = an.compute_edc(test_filtered, Fs, plot=False)
                    
                    # Compute RMSE
                    rmse_filt = an.compute_RMS(
                        ref_edc_filt, test_edc_filt,
                        range=50, Fs=Fs,
                        skip_initial_zeros=True,
                        normalize_by_active_length=True
                    )
                    filtered_rmses.append(rmse_filt)
                
                filtered_mean = np.mean(filtered_rmses)
                
                # Store filtered results
                file_results[method_name]['filtered'][filter_label] = {
                    'rmses': filtered_rmses,
                    'mean': filtered_mean,
                    'delta': filtered_mean - original_mean
                }
                
                # Store first receiver's filtered RIR for plotting (with normalization)
                if filter_type == 'butterworth':
                    first_filtered = apply_butterworth_lowpass(
                        method_rirs[0],
                        cutoff_hz=filter_params['cutoff_hz'],
                        fs=Fs,
                        order=filter_params.get('order', 4)
                    )
                elif filter_type == 'gaussian':
                    first_filtered = apply_gaussian_smoothing(
                        method_rirs[0],
                        sigma_samples=filter_params['sigma_samples']
                    )
                
                # Normalize filtered RIR (set receiver position for normalization)
                rx, ry = receiver_positions[0]  # First receiver
                room.set_microphone(rx, ry, room_params['mic z'])
                normalized_dict = rir_normalisation(first_filtered, room, Fs, normalize_to_first_impulse=True)
                first_filtered_norm = normalized_dict['single_rir']
                
                filtered_rirs_for_plot[method_name][filter_label] = first_filtered_norm
        
        # Store with descriptive keys
        all_results[f"{filename}_metrics"] = file_results
        all_results[f"{filename}_filtered_rirs"] = (filtered_rirs_for_plot, Fs)
    
    return all_results


def print_summary_table(results, filter_configs):
    """Print formatted summary table of results."""
    
    for key, file_results in results.items():
        # Only process metrics entries (skip filtered RIR data)
        if not key.endswith('_metrics'):
            continue
        
        # Extract original filename for display
        filename = key.replace('_metrics', '')
            
        print(f"\n\n{'='*100}")
        print(f"📊 SUMMARY: {filename}")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Method':<25} | {'Original':>10}"
        for cfg in filter_configs:
            header += f" | {cfg['label']:>10}"
        print(header)
        print("-" * 100)
        
        # Sort methods for consistent output
        methods = sorted(file_results.keys())
        
        for method in methods:
            data = file_results[method]
            
            # Get display name
            display_name = get_display_name(method, {}, DISPLAY_NAME_MAP)
            if len(display_name) > 24:
                display_name = display_name[:21] + "..."
            
            # Build row
            row = f"{display_name:<25}"
            
            for cfg in filter_configs:
                label = cfg['label']
                if label in data['filtered']:
                    filt_data = data['filtered'][label]
                    row += f" | {filt_data['mean']:>10.4f}"
                else:
                    row += f" | {'N/A':>10} | {'N/A':>8}"
            
            print(row)
        
        print("=" * 100)

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Files to analyze
    FILES_TO_PROCESS = [
        "aes_room_spatial_edc_data_center_source.npz",
        # "aes_room_spatial_edc_data_top_middle_source.npz",
        # "aes_room_spatial_edc_data_upper_right_source.npz",
        # "aes_room_spatial_edc_data_lower_left_source.npz",
    ]
    
    # Reference method for error computation
    REFERENCE_METHOD = 'RIMPY-neg10'
    
    # Methods to analyze (None = all methods)
    METHODS_TO_ANALYZE = ['SDN-Test1', 'SDN-Test3', 'SDN-Test2.998']
    # METHODS_TO_ANALYZE = None  # Uncomment to analyze all methods
    
    # Filter configurations to test
    FILTER_CONFIGS = [
        {
            'type': 'butterworth',
            'params': {'cutoff_hz': 4000, 'order': 4},
            'label': 'Butter-4kHz'
        },
        {
            'type': 'butterworth',
            'params': {'cutoff_hz': 2000, 'order': 4},
            'label': 'Butter-2kHz'
        },
        {
            'type': 'gaussian',
            'params': {'sigma_samples': 5},  # ~0.11ms @ 44.1kHz
            'label': 'Gauss-σ5'
        },
        {
            'type': 'gaussian',
            'params': {'sigma_samples': 10},  # ~0.23ms @ 44.1kHz
            'label': 'Gauss-σ10'
        },
    ]
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    print("\n" + "="*80)
    print("🔬 LOW-PASS FILTER EFFECTS ANALYSIS")
    print("="*80)
    print(f"Reference Method: {REFERENCE_METHOD}")
    print(f"Filter Configurations: {len(FILTER_CONFIGS)}")
    for cfg in FILTER_CONFIGS:
        print(f"  • {cfg['label']}: {cfg['type']} with {cfg['params']}")
    
    # Compute metrics
    results = compute_metrics_for_filtered_rirs(
        data_files=FILES_TO_PROCESS,
        filter_configs=FILTER_CONFIGS,
        reference_method=REFERENCE_METHOD,
        methods_to_analyze=METHODS_TO_ANALYZE
    )
    
    # Print summary
    print_summary_table(results, FILTER_CONFIGS)
    
    # Interactive visualization for first file
    if FILES_TO_PROCESS:
        first_file = FILES_TO_PROCESS[0]
        rirs_key = f"{first_file}_filtered_rirs"
        
        if rirs_key in results:
            filtered_rirs_dict, Fs = results[rirs_key]
            
            print("\n" + "="*80)
            print("🎨 INTERACTIVE VISUALIZATION (First Receiver Position)")
            print("="*80)
            
            # Flatten the nested dict for plotting
            rirs_to_plot = {}
            for method_name, versions in filtered_rirs_dict.items():
                for version_label, rir in versions.items():
                    plot_key = f"{method_name} ({version_label})"
                    rirs_to_plot[plot_key] = rir
            
            print(f"Plotting {len(rirs_to_plot)} RIR versions...")
            
            # Create unified interactive plot
            pp.create_unified_interactive_plot(rirs_to_plot, Fs)
    
    print("\n✅ Analysis complete!")
