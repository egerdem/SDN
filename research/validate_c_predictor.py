"""
Validate C Predictor Performance

Goal:
-----
Compare SDN performance using:
1. Baseline C=1 (original SDN)
2. Predicted C (from c_predictor.py model)
3. Optimal C (from monte_carlo_proximity.py optimization)

This validates how well the predictor generalizes to the Monte Carlo dataset.

Workflow:
---------
STEP 1: Generate Monte Carlo data
    $ python research/monte_carlo_proximity.py
    
    This creates:
    - results/monte_carlo_experiments/<EXPERIMENT_ID>/results.json
    - results/monte_carlo_experiments/<EXPERIMENT_ID>/rimpy_cache/
    
    Where <EXPERIMENT_ID> is:
    - "default" (no seed specified)
    - "seed_42" (set MC_EXPERIMENT_ID=seed_42 before running)
    - custom name via environment variable

STEP 2: Validate C predictor
    $ python research/validate_c_predictor.py
    
    This uses the MC data and creates:
    - results/monte_carlo_experiments/<EXPERIMENT_ID>/validation/*.json
    - results/monte_carlo_experiments/<EXPERIMENT_ID>/validation/*.png

When to re-run:
---------------
- Changed room dimension ranges → re-run STEP 1 (new experiment)
- Changed RIMPY seed → re-run STEP 1 with regenerate flag
- Changed C predictor model → re-run STEP 2 only (uses cached MC data)
- Changed predictor experiment type → re-run STEP 2 only

Usage:
------
    from research.validate_c_predictor import validate_predictor
    
    results = validate_predictor(
        experiment='8src_diagonal',    # Which C predictor model
        model_type='linear',           # linear/polynomial/power
        mc_experiment_id='default',    # Which MC experiment to use
        max_runs=None,                 # None for all runs
        use_cached_rirs=True,          # Use pre-cached RIMPY RIRs
        regenerate_rirs=False,         # Set True to recalculate with seed
    )

Outputs:
--------
- Scatter plots: RMSE(baseline) vs RMSE(predicted) vs RMSE(optimal)
- Performance metrics: improvement vs baseline, degradation vs optimal
- Results saved to: 
  results/monte_carlo_experiments/<ID>/validation/c_predictor_validation_<exp>_<model>.json
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import geometry
from analysis import analysis as an
from c_predictor import predict_c_from_source, EXPERIMENT_MODELS
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, calculate_rimpy_rir, enable_basis_disk_cache

# ============================================================================
# CONFIGURATION
# ============================================================================

# Error window for EDC comparison
ERR_DURATION_MS = 50

# Monte Carlo experiment folder
MC_EXPERIMENT_ROOT = os.path.join(project_root, "results", "monte_carlo_experiments")

# ============================================================================
# HELPER FUNCTIONS (copied from monte_carlo_proximity.py for compatibility)
# ============================================================================

def _rimpy_cache_paths(cache_dir: str,
                       room_dims: tuple,
                       src_pos: tuple,
                       rx_pos: tuple,
                       duration_s: float,
                       Fs: int,
                       reflection_sign: int,
                       rand_dist: float) -> tuple:
    """
    Compute stable cache paths for a RIMPY RIR by geometry.
    Returns: (npy_path, json_meta_path)
    
    NOTE: This must match the implementation in monte_carlo_proximity.py
    """
    import hashlib
    payload = {
        "room_dims": [float(x) for x in room_dims],
        "src_pos": [float(x) for x in src_pos],
        "rx_pos": [float(x) for x in rx_pos],
        "duration_s": float(duration_s),
        "Fs": int(Fs),
        "reflection_sign": int(reflection_sign),
        "rand_dist": float(rand_dist),
    }
    key = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:12]
    base = f"rimpy_{h}"
    return (os.path.join(cache_dir, f"{base}.npy"), os.path.join(cache_dir, f"{base}.json"))


# ============================================================================
# MAIN VALIDATION LOGIC
# ============================================================================

def validate_predictor(
    experiment: str = '8src_diagonal',
    model_type: str = 'linear',
    mc_experiment_id: str = 'default',
    max_runs: Optional[int] = None,
    use_fast_sdn: bool = True,
    use_cached_rirs: bool = True,
    regenerate_rirs: bool = False,
    rimpy_seed: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Validate C predictor on Monte Carlo dataset.
    
    Args:
        experiment: Which experiment's model to use (e.g., '8src_diagonal', '13src_grid')
        model_type: Model type ('linear', 'polynomial', 'power')
        mc_experiment_id: Monte Carlo experiment folder name (e.g., 'default', 'mc_generic')
        max_runs: Maximum number of runs to process (None = all)
        use_fast_sdn: Use fast SDN method with basis caching
        use_cached_rirs: Try to use pre-cached RIMPY RIRs
        regenerate_rirs: Force recalculation of RIRs (useful for known seed)
        rimpy_seed: Random seed for RIMPY if regenerating (None = unseeded)
        verbose: Print per-run details
    
    Returns:
        Dictionary with validation results
    """
    if use_fast_sdn:
        enable_basis_disk_cache()
        print("  ✓ Basis function disk cache enabled (shared with monte_carlo_proximity.py)")
    
    # Locate Monte Carlo experiment (simple and explicit)
    exp_dir = os.path.join(MC_EXPERIMENT_ROOT, mc_experiment_id)
    MONTE_CARLO_RESULTS = os.path.join(exp_dir, "results.json")
    RIR_CACHE_DIR = os.path.join(exp_dir, "rimpy_cache")
    
    if not os.path.exists(MONTE_CARLO_RESULTS):
        raise FileNotFoundError(
            f"Monte Carlo experiment '{mc_experiment_id}' not found!\n"
            f"Expected: {MONTE_CARLO_RESULTS}\n"
            f"Available: {os.listdir(MC_EXPERIMENT_ROOT) if os.path.exists(MC_EXPERIMENT_ROOT) else 'none'}"
        )
    
    with open(MONTE_CARLO_RESULTS, 'r') as f:
        mc_results = json.load(f)
    
    print("=" * 80)
    print("C PREDICTOR VALIDATION")
    print("=" * 80)
    print(f"Monte Carlo results: {MONTE_CARLO_RESULTS}")
    print(f"RIMPY cache: {RIR_CACHE_DIR}")
    print(f"Total runs available: {len(mc_results)}")
    print(f"Experiment model: {experiment}")
    print(f"Model type: {model_type}")
    print(f"Using fast SDN: {use_fast_sdn}")
    
    if max_runs:
        mc_results = mc_results[:max_runs]
        print(f"Processing: {len(mc_results)} runs (limited)")
    else:
        print(f"Processing: {len(mc_results)} runs (all)")
    print()
    
    # Display model equation
    exp_models = EXPERIMENT_MODELS[experiment]
    model_params = exp_models[model_type]
    
    print("Model equation:")
    if model_type == 'linear':
        eq = f"C = {model_params['intercept']:.4f} + {model_params['coefficient']:.4f} * H_src_norm"
    elif model_type == 'polynomial':
        eq = f"C = {model_params['intercept']:.4f} + {model_params['coef_h']:.4f}*H + {model_params['coef_h2']:.4f}*H²"
    elif model_type == 'power':
        eq = f"C = {model_params['a']:.4f} * H_src_norm^{model_params['b']:.4f}"
    
    print(f"  {eq}")
    print(f"  R² = {model_params['r2_score']:.4f}")
    print()
    
    # Validation loop
    validation_results = []
    
    for i, mc_run in enumerate(mc_results):
        run_id = mc_run.get('run_id', mc_run.get('run_key', i))
        
        # Safety check: skip runs missing critical data
        if 'optimal_c' not in mc_run or 'min_rmse' not in mc_run:
            print(f"⚠️  Skipping run {run_id}: missing optimal_c or min_rmse")
            continue
        
        # Extract geometry
        src_pos = tuple(mc_run['src_pos'])
        mic_pos = tuple(mc_run['mic_pos'])
        room_dims = tuple(mc_run['room_dims'])
        w, d, h = room_dims
        
        # Get optimal C (ground truth from optimization)
        c_optimal = float(mc_run['optimal_c'])
        rmse_optimal = float(mc_run['min_rmse'])
        
        # Predict C using our model
        c_predicted = predict_c_from_source(
            src_pos, room_dims,
            model_type=model_type,
            experiment=experiment,
            verbose=False
        )
        
        # Get H_src_norm for analysis
        h_src_norm = float(mc_run['h_src_norm'])
        
        if verbose:
            print(f"[Run {i+1}/{len(mc_results)}] {run_id}")
            print(f"  H_src_norm: {h_src_norm:.4f}")
            print(f"  C_optimal: {c_optimal:.3f} (RMSE: {rmse_optimal:.4f})")
            print(f"  C_predicted: {c_predicted:.3f}")
            print(f"  C_baseline: 1.0")
        
        # Calculate RMSE with predicted C
        # Need to: 1) get reference RIR, 2) run SDN with c_predicted, 3) compute EDC RMSE
        
        # Setup room
        absorption = mc_run.get('room_params', {}).get('absorption', 0.2)
        if isinstance(absorption, dict):  # Handle old format
            absorption = 0.2
        
        reflection = np.sqrt(1.0 - absorption)
        
        room_params = {
            'width': w,
            'depth': d,
            'height': h,
            'absorption': absorption,
            'reflection': reflection,
            'source x': src_pos[0],
            'source y': src_pos[1],
            'source z': src_pos[2],
            'mic x': mic_pos[0],
            'mic y': mic_pos[1],
            'mic z': mic_pos[2],
        }
        
        # Determine Fs and duration
        Fs = mc_run.get('Fs', 44100)
        duration = mc_run.get('duration_s', 1.0)  # Fixed: was 'duration', should be 'duration_s'
        
        # Build room object
        room = geometry.Room(w, d, h)
        impulse = geometry.Source.generate_signal('dirac', int(Fs * duration))
        room.set_source(*src_pos, signal=impulse['signal'], Fs=Fs)
        room.set_microphone(*mic_pos)
        room.wallAttenuation = [reflection] * 6
        
        # Get reference RIR - auto-detect cache naming scheme
        # Legacy experiment uses integer-based naming, modern uses hash-based
        if mc_experiment_id == 'legacy_4x20_50cornerplacement_noseed':
            # Legacy: integer-based naming (rimpy_run_N.npy)
            rir_path = os.path.join(RIR_CACHE_DIR, f"rimpy_run_{run_id}.npy")
            meta_path = None
        else:
            # Modern: hash-based naming (rimpy_abc123.npy)
            rir_path, meta_path = _rimpy_cache_paths(
                RIR_CACHE_DIR,
                tuple(mc_run['room_dims']),
                src_pos,
                mic_pos,
                duration,
                Fs,
                reflection_sign=-1,
                rand_dist=0.1
            )
        
        ref_rir = None
        if use_cached_rirs and not regenerate_rirs and os.path.exists(rir_path):
            # Load from cache (hash-based or legacy)
            try:
                ref_rir = np.load(rir_path)
                if verbose:
                    print(f"  Loaded cached RIR: {os.path.basename(rir_path)}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Failed to load cached RIR: {e}, recalculating...")
                ref_rir = None
        
        # If not cached or regenerating, calculate fresh
        if ref_rir is None:
            if verbose and use_cached_rirs:
                print(f"  Cached RIR not found, calculating fresh...")
            
            # Set seed if provided (for reproducibility)
            if rimpy_seed is not None:
                np.random.seed(rimpy_seed)
            
            # Calculate reference (RIMPY-neg10)
            ref_rir, _ = calculate_rimpy_rir(
                room_params, duration, Fs,
                reflection_sign=-1, randDist=0.1
            )
            
            # Save to cache if regenerating
            if regenerate_rirs:
                os.makedirs(RIR_CACHE_DIR, exist_ok=True)
                np.save(rir_path, ref_rir)
                # Also save metadata for hash-based cache (debugging)
                if meta_path:
                    try:
                        with open(meta_path, 'w') as f:
                            json.dump({
                                "room_dims": list(mc_run['room_dims']),
                                "src_pos": list(src_pos),
                                "rx_pos": list(mic_pos),
                                "duration_s": float(duration),
                                "Fs": int(Fs),
                                "reflection_sign": -1,
                                "rand_dist": 0.1,
                            }, f, indent=2)
                    except Exception:
                        pass  # Meta is optional
                if verbose:
                    print(f"  Saved regenerated RIR to cache: {os.path.basename(rir_path)}")
        
        # Normalize reference and compute EDC
        norm_ref = rir_normalisation(ref_rir, room, Fs, normalize_to_first_impulse=True)['single_rir']
        ref_edc, _, _ = an.compute_edc(norm_ref, Fs, plot=False)
        
        # Run SDN with predicted C
        sdn_config_pred = {
            'enabled': True,
            'calculator': 'sdn',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': float(c_predicted),
            },
            'label': 'SDN',
            'cache_label': f'validation_pred_{run_id}',
        }
        
        if use_fast_sdn:
            _, rir_sdn_pred, _, _ = calculate_sdn_rir_fast(
                room_params, "ValidationPred", room, duration, Fs, sdn_config_pred
            )
        else:
            from rir_calculators import calculate_sdn_rir
            _, rir_sdn_pred, _, _ = calculate_sdn_rir(
                room_params, "ValidationPred", room, duration, Fs, sdn_config_pred
            )
        
        # Normalize and compute EDC for predicted C
        norm_sdn_pred = rir_normalisation(rir_sdn_pred, room, Fs, normalize_to_first_impulse=True)['single_rir']
        edc_sdn_pred, _, _ = an.compute_edc(norm_sdn_pred, Fs, plot=False)
        
        # Compute RMSE with predicted C
        rmse_predicted = float(
            an.compute_RMS(
                edc_sdn_pred, ref_edc,
                range=int(ERR_DURATION_MS),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True,
            )
        )
        
        # Run SDN with baseline C=1
        sdn_config_c1 = {
            'enabled': True,
            'calculator': 'sdn',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 1.0,
            },
            'label': 'SDN',
            'cache_label': f'validation_c1_{run_id}',
        }
        
        if use_fast_sdn:
            _, rir_sdn_c1, _, _ = calculate_sdn_rir_fast(
                room_params, "ValidationC1", room, duration, Fs, sdn_config_c1
            )
        else:
            _, rir_sdn_c1, _, _ = calculate_sdn_rir(
                room_params, "ValidationC1", room, duration, Fs, sdn_config_c1
            )
        
        # Normalize and compute EDC for C=1
        norm_sdn_c1 = rir_normalisation(rir_sdn_c1, room, Fs, normalize_to_first_impulse=True)['single_rir']
        edc_sdn_c1, _, _ = an.compute_edc(norm_sdn_c1, Fs, plot=False)
        
        # Compute RMSE with C=1
        rmse_c1 = float(
            an.compute_RMS(
                edc_sdn_c1, ref_edc,
                range=int(ERR_DURATION_MS),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True,
            )
        )
        
        # Performance metrics
        degradation_vs_optimal = rmse_predicted - rmse_optimal
        degradation_pct_vs_optimal = 100.0 * degradation_vs_optimal / rmse_optimal if rmse_optimal > 0 else 0.0
        
        improvement_vs_baseline = rmse_c1 - rmse_predicted
        improvement_pct_vs_baseline = 100.0 * improvement_vs_baseline / rmse_c1 if rmse_c1 > 0 else 0.0
        
        if verbose:
            print(f"  RMSE_baseline (c=1): {rmse_c1:.4f}")
            print(f"  RMSE_predicted: {rmse_predicted:.4f}")
            print(f"  Improvement vs baseline: {improvement_vs_baseline:+.4f} ({improvement_pct_vs_baseline:+.1f}%)")
            print(f"  Degradation vs optimal: {degradation_vs_optimal:+.4f} ({degradation_pct_vs_optimal:+.1f}%)")
            print()
        
        validation_results.append({
            'run_id': run_id,
            'room_dims': room_dims,
            'src_pos': src_pos,
            'mic_pos': mic_pos,
            'h_src_norm': h_src_norm,
            'c_baseline': 1.0,
            'c_optimal': c_optimal,
            'c_predicted': c_predicted,
            'c_error': c_predicted - c_optimal,
            'rmse_baseline': rmse_c1,
            'rmse_optimal': rmse_optimal,
            'rmse_predicted': rmse_predicted,
            'rmse_degradation_vs_optimal': degradation_vs_optimal,
            'rmse_degradation_pct_vs_optimal': degradation_pct_vs_optimal,
            'rmse_improvement_vs_baseline': improvement_vs_baseline,
            'rmse_improvement_pct_vs_baseline': improvement_pct_vs_baseline,
        })
    
    # Compute summary statistics
    c_errors = np.array([r['c_error'] for r in validation_results])
    rmse_baseline = np.array([r['rmse_baseline'] for r in validation_results])
    rmse_optimal = np.array([r['rmse_optimal'] for r in validation_results])
    rmse_predicted = np.array([r['rmse_predicted'] for r in validation_results])
    
    rmse_degradations_vs_optimal = np.array([r['rmse_degradation_vs_optimal'] for r in validation_results])
    rmse_degradations_pct_vs_optimal = np.array([r['rmse_degradation_pct_vs_optimal'] for r in validation_results])
    
    rmse_improvements_vs_baseline = np.array([r['rmse_improvement_vs_baseline'] for r in validation_results])
    rmse_improvements_pct_vs_baseline = np.array([r['rmse_improvement_pct_vs_baseline'] for r in validation_results])
    
    summary = {
        'n_runs': len(validation_results),
        'c_error': {
            'mean': float(np.mean(c_errors)),
            'std': float(np.std(c_errors)),
            'median': float(np.median(c_errors)),
            'mae': float(np.mean(np.abs(c_errors))),
        },
        'rmse_baseline': {
            'mean': float(np.mean(rmse_baseline)),
            'std': float(np.std(rmse_baseline)),
            'median': float(np.median(rmse_baseline)),
        },
        'rmse_optimal': {
            'mean': float(np.mean(rmse_optimal)),
            'std': float(np.std(rmse_optimal)),
            'median': float(np.median(rmse_optimal)),
        },
        'rmse_predicted': {
            'mean': float(np.mean(rmse_predicted)),
            'std': float(np.std(rmse_predicted)),
            'median': float(np.median(rmse_predicted)),
        },
        'improvement_vs_baseline': {
            'mean': float(np.mean(rmse_improvements_vs_baseline)),
            'std': float(np.std(rmse_improvements_vs_baseline)),
            'median': float(np.median(rmse_improvements_vs_baseline)),
            'mean_pct': float(np.mean(rmse_improvements_pct_vs_baseline)),
            'median_pct': float(np.median(rmse_improvements_pct_vs_baseline)),
        },
        'degradation_vs_optimal': {
            'mean': float(np.mean(rmse_degradations_vs_optimal)),
            'std': float(np.std(rmse_degradations_vs_optimal)),
            'median': float(np.median(rmse_degradations_vs_optimal)),
            'mean_pct': float(np.mean(rmse_degradations_pct_vs_optimal)),
            'median_pct': float(np.median(rmse_degradations_pct_vs_optimal)),
        }
    }
    
    return {
        'experiment': experiment,
        'model_type': model_type,
        'model_equation': eq,
        'model_r2': float(model_params['r2_score']),
        'summary': summary,
        'results': validation_results,
    }


def plot_validation_results(validation_data: Dict, save_path: str = None):
    """Generate validation plots"""
    results = validation_data['results']
    
    h_src_norms = np.array([r['h_src_norm'] for r in results])
    c_optimal = np.array([r['c_optimal'] for r in results])
    c_predicted = np.array([r['c_predicted'] for r in results])
    rmse_baseline = np.array([r['rmse_baseline'] for r in results])
    rmse_optimal = np.array([r['rmse_optimal'] for r in results])
    rmse_predicted = np.array([r['rmse_predicted'] for r in results])
    c_errors = np.array([r['c_error'] for r in results])
    rmse_degradations_vs_optimal = np.array([r['rmse_degradation_vs_optimal'] for r in results])
    rmse_improvements_vs_baseline = np.array([r['rmse_improvement_vs_baseline'] for r in results])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: C_predicted vs C_optimal
    ax = axes[0, 0]
    ax.scatter(c_optimal, c_predicted, alpha=0.6, c=h_src_norms, cmap='viridis')
    lim_min = min(c_optimal.min(), c_predicted.min())
    lim_max = max(c_optimal.max(), c_predicted.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('C_optimal (from optimization)')
    ax.set_ylabel('C_predicted (from model)')
    ax.set_title('Predicted vs Optimal C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='H_src_norm')
    
    # Plot 2: C error vs H_src_norm
    ax = axes[0, 1]
    sc = ax.scatter(h_src_norms, c_errors, alpha=0.6, c=rmse_optimal, cmap='plasma')
    ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('H_src_norm')
    ax.set_ylabel('C_error = C_predicted - C_optimal')
    ax.set_title('C Prediction Error vs Wall Proximity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='RMSE_optimal')
    
    # Plot 3: RMSE comparison - three methods (baseline, predicted, optimal)
    ax = axes[0, 2]
    # Baseline vs Predicted
    ax.scatter(rmse_baseline, rmse_predicted, alpha=0.5, c='blue', s=40, label='Baseline→Predicted', edgecolors='none')
    # Optimal vs Predicted
    ax.scatter(rmse_optimal, rmse_predicted, alpha=0.5, c='green', s=40, label='Optimal→Predicted', edgecolors='none')
    
    # Add diagonal line
    all_rmse = np.concatenate([rmse_baseline, rmse_optimal, rmse_predicted])
    lim_min = all_rmse.min()
    lim_max = all_rmse.max()
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, alpha=0.7, label='Equal performance')
    
    ax.set_xlabel('RMSE (Baseline c=1 / Optimal c*)')
    ax.set_ylabel('RMSE with C_predicted')
    ax.set_title('SDN Performance Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement vs baseline & degradation vs optimal
    ax = axes[1, 0]
    sc = ax.scatter(h_src_norms, rmse_improvements_vs_baseline, alpha=0.6, c='blue', s=60, label='vs Baseline (c=1)', edgecolors='black', linewidths=0.5)
    ax.scatter(h_src_norms, -rmse_degradations_vs_optimal, alpha=0.6, c='green', s=60, label='vs Optimal (c*)', marker='s', edgecolors='black', linewidths=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('H_src_norm')
    ax.set_ylabel('RMSE Improvement (positive = better)')
    ax.set_title('Performance vs Baseline & Optimal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Histogram comparison
    ax = axes[1, 1]
    bins = np.linspace(
        min(rmse_improvements_vs_baseline.min(), -rmse_degradations_vs_optimal.min()),
        max(rmse_improvements_vs_baseline.max(), -rmse_degradations_vs_optimal.max()),
        25
    )
    ax.hist(rmse_improvements_vs_baseline, bins=bins, alpha=0.6, 
            label='Improvement vs Baseline', color='blue', edgecolor='black')
    ax.hist(-rmse_degradations_vs_optimal, bins=bins, alpha=0.6, 
            label='Improvement vs Optimal', color='green', edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('RMSE Improvement (positive = better)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Improvements')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = validation_data['summary']
    n_better_than_baseline = int(np.sum(rmse_improvements_vs_baseline > 0))
    n_worse_than_baseline = int(np.sum(rmse_improvements_vs_baseline < 0))
    n_equal_baseline = int(np.sum(rmse_improvements_vs_baseline == 0))
    n_better_than_optimal = int(np.sum(rmse_degradations_vs_optimal < 0))
    n_worse_than_optimal = int(np.sum(rmse_degradations_vs_optimal > 0))
    n_equal_optimal = int(np.sum(rmse_degradations_vs_optimal == 0))
    
    summary_text = f"""
VALIDATION SUMMARY
{'='*40}

Model: {validation_data['model_type']}
Experiment: {validation_data['experiment']}
R²: {validation_data['model_r2']:.4f}
Runs: {summary['n_runs']}

C Prediction Error:
  Mean: {summary['c_error']['mean']:+.3f}
  MAE:  {summary['c_error']['mae']:.3f}

RMSE Performance:
  Baseline (c=1):   {summary['rmse_baseline']['mean']:.4f}
  Predicted:        {summary['rmse_predicted']['mean']:.4f}
  Optimal (c*):     {summary['rmse_optimal']['mean']:.4f}

vs Baseline (c=1):
  Improvement: {summary['improvement_vs_baseline']['mean']:+.4f}
  ({summary['improvement_vs_baseline']['mean_pct']:+.1f}%)
  Better: {n_better_than_baseline}
  Equal:  {n_equal_baseline}
  Worse:  {n_worse_than_baseline}
  Total:  {n_better_than_baseline + n_equal_baseline + n_worse_than_baseline}/{summary['n_runs']}

vs Optimal (c*):
  Degradation: {summary['degradation_vs_optimal']['mean']:+.4f}
  ({summary['degradation_vs_optimal']['mean_pct']:+.1f}%)
  Better: {n_better_than_optimal}
  Equal:  {n_equal_optimal}
  Worse:  {n_worse_than_optimal}
  Total:  {n_better_than_optimal + n_equal_optimal + n_worse_than_optimal}/{summary['n_runs']}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_validation_example():
    """
    Example usage of the validation function.
    Run this to validate the C predictor on the Monte Carlo dataset.
    """
    print("=" * 80)
    print("C PREDICTOR VALIDATION - EXAMPLE")
    print("=" * 80)
    print()
    
    # Configuration
    experiment = '7src_3Ddiagonal'   # Which C predictor model: '8src_diagonal', '13src_grid', '7src_3Ddiagonal' etc.
    model_type = 'linear'          # Model type: 'linear', 'polynomial', 'power'
    mc_experiment_id = '10x60_seed42_0cornerplacement_big'   # MC experiment folder: 'default', 'mc_generic', 'seed_42', etc.
    max_runs = None                # None = all runs, or set to 10 for quick test
    
    print(f"C Predictor Model: {experiment} ({model_type})")
    print(f"MC Experiment: {mc_experiment_id}")
    print(f"Max runs: {max_runs or 'all'}")
    print()
    
    # Run validation
    validation_data = validate_predictor(
        experiment=experiment,
        model_type=model_type,
        mc_experiment_id=mc_experiment_id,  # REQUIRED: Which MC experiment to validate
        max_runs=max_runs,
        use_fast_sdn=True,
        use_cached_rirs=True,      # Use pre-cached RIMPY RIRs
        regenerate_rirs=False,     # Set True to recalculate with seed
        rimpy_seed=None,           # Set to int for reproducible RIMPY
        verbose=True
        # Note: Legacy cache naming (rimpy_run_N.npy) auto-detected for 'legacy_4x20_50cornerplacement_noseed'
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    summary = validation_data['summary']
    print(f"\nModel: {experiment} {model_type}")
    print(f"Equation: {validation_data['model_equation']}")
    print(f"Model R²: {validation_data['model_r2']:.4f}")
    print(f"Runs validated: {summary['n_runs']}")
    print(f"\nC Prediction Error:")
    print(f"  Mean: {summary['c_error']['mean']:+.3f}")
    print(f"  MAE:  {summary['c_error']['mae']:.3f}")
    print(f"  Std:  {summary['c_error']['std']:.3f}")
    print(f"\nRMSE Performance:")
    print(f"  Baseline (c=1):   {summary['rmse_baseline']['mean']:.4f}")
    print(f"  Predicted:        {summary['rmse_predicted']['mean']:.4f}")
    print(f"  Optimal (c*):     {summary['rmse_optimal']['mean']:.4f}")
    print(f"\nImprovement vs Baseline (c=1):")
    print(f"  Mean: {summary['improvement_vs_baseline']['mean']:+.4f} ({summary['improvement_vs_baseline']['mean_pct']:+.1f}%)")
    print(f"  Median: {summary['improvement_vs_baseline']['median']:+.4f} ({summary['improvement_vs_baseline']['median_pct']:+.1f}%)")
    print(f"\nDegradation vs Optimal (c*):")
    print(f"  Mean: {summary['degradation_vs_optimal']['mean']:+.4f} ({summary['degradation_vs_optimal']['mean_pct']:+.1f}%)")
    print(f"  Median: {summary['degradation_vs_optimal']['median']:+.4f} ({summary['degradation_vs_optimal']['median_pct']:+.1f}%)")
    
    # Save results to validation subfolder within the experiment
    exp_folder = os.path.join(MC_EXPERIMENT_ROOT, mc_experiment_id)
    output_dir = os.path.join(exp_folder, "validation")
    os.makedirs(output_dir, exist_ok=True)
    
    output_base = f"c_predictor_validation_{experiment}_{model_type}"
    
    json_path = os.path.join(output_dir, f"{output_base}.json")
    with open(json_path, 'w') as f:
        json.dump(validation_data, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Generate plots
    plot_path = os.path.join(output_dir, f"{output_base}.png")
    plot_validation_results(validation_data, save_path=plot_path)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    
    return validation_data


if __name__ == "__main__":
    # Run the example validation
    run_validation_example()

