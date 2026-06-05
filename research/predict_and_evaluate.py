"""
Unified C Prediction and Evaluation Script

Workflow:
1. Load legacy grid files (fullgrid or quarter)
2. For each file's source, predict C using model parameters
3. Evaluate prediction on that file's receivers against RIMPY reference

Usage:
    python3 research/predict_and_evaluate.py --receiver-grid fullgrid --model linear
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import geometry
from analysis import analysis as an
from analysis.plotting_utils import load_data
from rir_calculators import calculate_sdn_rir, rir_normalisation
from c_predictor import predict_c_from_source

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIGS = {
    '13src_grid': {
        'experiment_name': 'aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair',
        'description': '13 srcs, 16 receivers per source'
    },
    '8src_diagonal': {
        'experiment_name': 'aes_fullgrid_8src_diagonal',
        'description': '8 diagonal sources, 16 receivers'
    },

    '7src_3Ddiagonal': {
        'experiment_name': 'aes_fullgrid_7src_3Ddiagonal',
        'description': '7 3d diagonal sources, 16 receivers'
    },

    '10x60_seed42_0cornerplacement_big': {
        'experiment_name': '10x60_seed42_0cornerplacement_big',
        'description': '10 rooms × 60 pairs, seed 42, 0% corner placement'
    },
    '10x60_seed42_25cornerplacement_big': {
        'experiment_name': '10x60_seed42_25cornerplacement_big',
        'description': '10 rooms × 60 pairs, seed 42, 25% corner placement'
    },
    'legacy_4x20_50cornerplacement_noseed': {
        'experiment_name': 'legacy_4x20_50cornerplacement_noseed',
        'description': '4 rooms × 20 pairs, 50% corner placement, no seed'
    }
}

GRID_FILES = {
    'fullgrid': [
        # "aes_FULLGRID_center_source.npz",
        # "aes_FULLGRID_top_middle_source.npz",
        # "aes_FULLGRID_upper_right_source.npz",
        # "aes_FULLGRID_lower_left_source.npz",
        # "aes_FULLGRID_corner_sourcev3.npz",
        "journal_FULLGRID_center_source.npz",
        "journal_FULLGRID_top_middle_source.npz",
        "journal_FULLGRID_upper_right_source.npz",
        "journal_FULLGRID_lower_left_source.npz",
    ],
    'quarter': [
        "aes_quarter_center_source.npz",
        "aes_quarter_top_middle_source.npz",
        "aes_quarter_upper_right_source.npz",
        "aes_quarter_lower_left_source.npz",
        "aes_quarter_corner_sourcev3.npz",
    ]
}

REFERENCE_METHOD = 'RIMPY-neg10'
ERR_DURATION_MS = 50


def make_sdn_config(source_weighting: float) -> dict:
    """Build SDN config dict"""
    return {
        "enabled": True,
        "calculator": "sdn",
        "flags": {
            "specular_source_injection": True,
            "source_weighting": source_weighting,
        },
        "label": "SDN",
    }


def main():
    parser = argparse.ArgumentParser(description="Predict and evaluate C parameters")
    parser.add_argument(
        '--experiment',
        type=str,
        # default='8src_diagonal',
        # default='13src_grid',
        # default='4x20_random_pair',
        # default='legacy_4x20_50cornerplacement_noseed',
        # default='10x60_seed42_25cornerplacement_big',
        # default='10x60_seed42_0cornerplacement_big',
        # default='10x60_seed42_0cornerplacement_big_V2',
        default='7src_3Ddiagonal',
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help='Experiment to use for model parameters (fit equation)'
    )
    parser.add_argument(
        '--receiver-grid',
        type=str,
        default='fullgrid',
        choices=['fullgrid', 'quarter'],
        help='Receiver grid type: fullgrid or quarter'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='linear',
        choices=['linear', 'polynomial', 'power'],
        help='Model type for prediction',
    )
    
    args = parser.parse_args()
    
    # Get legacy grid files
    grid_files = GRID_FILES[args.receiver_grid]
    grid_dir = os.path.join(project_root, "results", "paper_data")
    
    # Check which files exist
    available_files = [f for f in grid_files if os.path.exists(os.path.join(grid_dir, f))]
    if not available_files:
        print(f"❌ ERROR: No {args.receiver_grid} grid files found in {grid_dir}")
        return
    
    print("=" * 80)
    print("C PREDICTION AND EVALUATION")
    print("=" * 80)
    print(f"Model from: {args.experiment}")
    print(f"Model type: {args.model}")
    print(f"Receiver grid: {args.receiver_grid}")
    print(f"Files to process: {len(available_files)}")
    print(f"Reference: {REFERENCE_METHOD}")
    print()
    
    # Display model equation
    from c_predictor import EXPERIMENT_MODELS
    exp_models = EXPERIMENT_MODELS[args.experiment]
    model_params = exp_models[args.model]
    
    print("=" * 80)
    print(f"MODEL: {args.model.upper()}")
    print("=" * 80)
    
    if args.model == 'linear':
        eq = f"C = {model_params['intercept']:.4f} + {model_params['coefficient']:.4f} * H_src_norm"
    elif args.model == 'polynomial':
        eq = f"C = {model_params['intercept']:.4f} + {model_params['coef_h']:.4f}*H + {model_params['coef_h2']:.4f}*H²"
    elif args.model == 'power':
        eq = f"C = {model_params['a']:.4f} * H_src_norm^{model_params['b']:.4f}"
    
    print(f"Equation: {eq}")
    print(f"R² score: {model_params['r2_score']:.4f}")
    print()
    
    # Process each grid file
    print("=" * 80)
    print("PREDICTING AND EVALUATING")
    print("=" * 80)
    
    all_results = []
    
    for grid_file in available_files:
        grid_path = os.path.join(grid_dir, grid_file)
        grid_sim = load_data(grid_path)
        
        # Get source position from this file
        source_pos = tuple(float(x) for x in grid_sim["source_pos"])
        room_params = grid_sim["room_params"]
        room_dims = (room_params["width"], room_params["depth"], room_params["height"])
        
        # Calculate H_src_norm
        from deprecated import wall_proximity
        sx, sy, sz = source_pos
        w, d, h = room_dims
        dists = wall_proximity.get_wall_distances(sx, sy, sz, w, d, h)
        h_src = wall_proximity.harmonic_mean(dists)
        volume = w * d * h
        char_len = volume ** (1/3)
        h_src_norm = h_src / char_len
        
        # Predict C
        c_pred = predict_c_from_source(
            source_pos, room_dims,
            model_type=args.model,
            experiment=args.experiment,
            verbose=False
        )
        
        print(f"\n[{grid_file}]")
        print(f"  Source: ({sx:.2f}, {sy:.2f}, {sz:.2f})")
        print(f"  H_src_norm: {h_src_norm:.4f}")
        print(f"  C_predicted: {c_pred:.3f}")
        
        # Evaluate on this file's receivers
        Fs = int(grid_sim["Fs"])
        duration = float(grid_sim["duration"])
        receiver_positions = grid_sim["receiver_positions"]
        
        if REFERENCE_METHOD not in grid_sim["edcs"]:
            print(f"  ⚠️  No RIMPY reference, skipping")
            continue
        
        ref_edcs = grid_sim["edcs"][REFERENCE_METHOD]
        
        # Setup room
        room = geometry.Room(w, d, h)
        impulse = geometry.Source.generate_signal("dirac", int(Fs * duration))
        room.set_source(*source_pos, signal=impulse["signal"], Fs=Fs)
        room.wallAttenuation = [np.sqrt(1.0 - room_params["absorption"])] * 6
        
        room_parameters = dict(room_params)
        room_parameters["reflection"] = np.sqrt(1.0 - room_params["absorption"])
        
        cfg = make_sdn_config(c_pred)
        
        rmses = []
        for i, pos in enumerate(receiver_positions):
            rx, ry, rz = float(pos[0]), float(pos[1]), float(pos[2])
            room.set_microphone(rx, ry, rz)
            
            _, rir_sdn, _, _ = calculate_sdn_rir(room_parameters, "Eval", room, duration, Fs, cfg)
            rir_sdn_norm = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
            edc_sdn, _, _ = an.compute_edc(rir_sdn_norm, Fs, plot=False)
            
            ref_edc = ref_edcs[i]
            rmse = an.compute_RMS(
                edc_sdn, ref_edc,
                range=int(ERR_DURATION_MS),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True,
            )
            rmses.append(float(rmse))
        
        rmses_np = np.array(rmses)
        mean_rmse = float(np.mean(rmses))
        
        print(f"  Receivers: {len(receiver_positions)}")
        print(f"  Mean RMSE: {mean_rmse:.4f}")
        print(f"  RMSE range: [{rmses_np.min():.4f}, {rmses_np.max():.4f}]")
        print(f"  RMSE std: {rmses_np.std():.4f}")
        
        all_results.append({
            'file': grid_file,
            'source_pos': source_pos,
            'h_src_norm': h_src_norm,
            'c_predicted': c_pred,
            'n_receivers': len(receiver_positions),
            'mean_rmse': mean_rmse,
            'rmse_std': float(rmses_np.std()),
            'rmse_min': float(rmses_np.min()),
            'rmse_max': float(rmses_np.max())
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_mean_rmses = [r['mean_rmse'] for r in all_results]
    overall_mean = np.mean(all_mean_rmses)
    overall_std = np.std(all_mean_rmses)
    
    print(f"\nOverall Performance:")
    print(f"  Mean RMSE across all sources: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"  Best: {min(all_results, key=lambda x: x['mean_rmse'])['file']} (RMSE: {min(all_mean_rmses):.4f})")
    print(f"  Worst: {max(all_results, key=lambda x: x['mean_rmse'])['file']} (RMSE: {max(all_mean_rmses):.4f})")
    
    # Export
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)
    
    import json
    
    export_data = {
        'model_experiment': args.experiment,
        'receiver_grid': args.receiver_grid,
        'model_type': args.model,
        'model_equation': eq,
        'model_r2': model_params['r2_score'],
        'results': all_results,
        'summary': {
            'overall_mean_rmse': overall_mean,
            'overall_std_rmse': overall_std,
            'best_file': min(all_results, key=lambda x: x['mean_rmse'])['file'],
            'best_rmse': min(all_mean_rmses),
            'worst_file': max(all_results, key=lambda x: x['mean_rmse'])['file'],
            'worst_rmse': max(all_mean_rmses)
        }
    }
    
    output_dir = grid_dir
    filename_base = f"c_prediction_{args.experiment}_{args.receiver_grid}_{args.model}"
    
    json_file = os.path.join(output_dir, f"{filename_base}.json")
    txt_file = os.path.join(output_dir, f"{filename_base}.txt")
    
    with open(json_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")
    
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("C PREDICTION AND EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {args.experiment} {args.model}\n")
        f.write(f"Grid: {args.receiver_grid}\n")
        f.write(f"Equation: {eq}\n")
        f.write(f"R²: {model_params['r2_score']:.4f}\n\n")
        
        for r in all_results:
            f.write(f"{r['file']}: C={r['c_predicted']:.3f}, RMSE={r['mean_rmse']:.4f}\n")
        
        f.write(f"\nOverall: {overall_mean:.4f} ± {overall_std:.4f}\n")
    
    print(f"✓ Saved text: {txt_file}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
