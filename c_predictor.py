"""
C Parameter Predictor for SDN

Predicts the optimal C parameter (scattering coefficient) based on source position
and room dimensions using empirically-derived relationships from Monte Carlo analysis.

Supports multiple experiment configurations with different source grid patterns.

Usage:
    from c_predictor import predict_c_from_source
    
    source_pos = (2.0, 3.0, 1.5)  # x, y, z in meters
    room_dims = (6.0, 8.0, 3.0)    # width, depth, height in meters
    
    # Use default 4x20 random pair experiment
    c_optimal = predict_c_from_source(source_pos, room_dims)
    
    # Or specify a different experiment
    c_optimal = predict_c_from_source(source_pos, room_dims, experiment='3x5_grid')
"""

import numpy as np
import os
import json


def harmonic_mean(distances):
    """
    Calculate harmonic mean of distances.
    Harmonic mean is dominated by the smallest value, making it sensitive to wall proximity.
    
    Args:
        distances: list or array of distances (must be positive)
    
    Returns:
        float: harmonic mean value
    """
    valid_dists = [d for d in distances if d > 0]
    if not valid_dists:
        return float('inf')
    return len(valid_dists) / sum(1.0 / d for d in valid_dists)


def get_wall_distances(x, y, z, width, depth, height):
    """
    Calculate distances from a point to all 6 walls of a rectangular room.
    
    Args:
        x, y, z: Point coordinates in meters
        width, depth, height: Room dimensions in meters
    
    Returns:
        list: [dist_to_west, dist_to_east, dist_to_south, dist_to_north, dist_to_floor, dist_to_ceiling]
    """
    return [
        x,              # Distance to x=0 wall (West)
        width - x,      # Distance to x=W wall (East)
        y,              # Distance to y=0 wall (South)
        depth - y,      # Distance to y=D wall (North)
        z,              # Distance to z=0 wall (Floor)
        height - z      # Distance to z=H wall (Ceiling)
    ]


# =====================================================================
# EXPERIMENT MODEL PARAMETERS
# =====================================================================

EXPERIMENT_MODELS = {

    'legacy_4x20_50cornerplacement_noseed': {
        'description': '4 rooms × 20 pairs, 50% corner placement, no seed',
        'n_samples': 80,
        'outliers_removed': 5,
        'linear': {
            'intercept': -1.5655534214459408,
            'coefficient': 13.275925025172917,
            'r2_score': 0.8341541321049044,
            'mae': 0.3428781097794205
        },
        'polynomial': {
            'intercept': -1.7760451118411789,
            'coef_h': 14.662826598032463,
            'coef_h2': -2.1546539978453896,
            'r2_score': 0.834294913596452,
            'mae': 0.3421755082874279
        },
        'power': {
            'a': 18.86308469005158,
            'b': 1.7576917414447608,
            'r2_score': 0.8197259143840665,
            'mae': 0.3643821043971566
        }
    },
    '4x20_random_pair': {
        'description': 'Original 80-run Monte Carlo (4 rooms × 20 random pairs)',
        'n_samples': 80,
        'outliers_removed': 4,
        'linear': {
            'intercept': -0.355819,
            'coefficient': 13.145891,
            'r2_score': 0.87
        },
        'polynomial': {
            'intercept': -1.945634,
            'coef_h': 26.338472,
            'coef_h2': -18.156293,
            'r2_score': 0.89
        },
        'power': {
            'a': 2.456791,
            'b': 1.583425,
            'r2_score': 0.88
        }
    },
    '13src_grid': {
        'description': '13 srcs, 16 receivers per source',
        'n_samples': 208,
        'outliers_removed': 4,
        'linear': {
            'intercept': -3.096649567228145,
            'coefficient': 17.59127414999333,
            'r2_score': 0.793633458298416
        },
        'polynomial': {
            'intercept': 2.862216714178102,
            'coef_h': -18.802875178610073,
            'coef_h2': 53.126066051116794,
            'r2_score': 0.8073220172049285
        },
        'power': {
            'a': 32.036060920925486,
            'b': 2.3191418858421726,
            'r2_score': 0.8006552402745857
        }
    },
    '8src_diagonal': {
        'description': '8 diagonal sources, 16 receivers per source',
        'n_samples': 112,
        'outliers_removed': 0,
        'linear': {
            'intercept': -3.2595686198268106,
            'coefficient': 17.744564442977666,
            'r2_score': 0.8278416598732914
        },
        'polynomial': {
            'intercept': -0.23683172544892495,
            'coef_h': 1.0056307878811166,
            'coef_h2': 22.46117236150021,
            'r2_score': 0.831007924849763
        },
        'power': {
            'a': 27.219392409783147,
            'b': 2.1781107498952963,
            'r2_score': 0.8280868057099284
        }
    },

    '7src_3Ddiagonal': {
        'description': '7 3d diagonal sources, 16 receivers per source',
        'n_samples': 112,
        'outliers_removed': 0,
        'linear': {
            "intercept": -2.442631440043548,
          "coefficient": 15.428518797386236,
          "r2_score": 0.9263844070598752,
        },
        'polynomial': {
            "intercept": -3.750727981758488,
          "coef_h": 23.150529130321065,
          "coef_h2": -10.746409793849088,
          "r2_score": 0.9282889772029899,

        },
        'power': {
            "a": 24.038611774404707,
          "b": 2.0746416348056913,
          "r2_score": 0.9018663394368239,
        }
    },

# "equation": "C = -1.6112 + 13.1135 * H_src_norm",
    '10x60_seed42_0cornerplacement_big_V2': {
            "best_model": "polynomial",
            'description': '10 rooms × 60 pairs, seed 42, 0% corner placement V2',
            'n_samples': 60,
            'outliers_removed': 0,
        'linear': {
          'intercept': -1.6111543006495945,
          'coefficient': 13.113469257616064,
          'r2_score': 0.7474157747828505,
          'mae': 0.3949889610191548
                },
        'polynomial': { #"equation": "C = -0.4156 + 5.8684*H + 10.6026*H",
            "intercept": -0.41560507142630865,
              'coef_h': 5.868393271650666,
              'coef_h2': 10.602633731048837,
              'r2_score': 0.7498233506778578,
              'mae': 0.39653550083289274
                    },
                    #"equation": "C = 17.3260 * H_src_norm^1.7099",
        'power': {
            "a": 17.32604613345441,
              "b": 1.7099033525805343,
              "r2_score": 0.7451984988112004,
              "mae": 0.3989348628559244
}
    },

    '10x60_seed42_0cornerplacement_big': {
        'description': '10 rooms × 60 pairs, seed 42, 0% corner placement',
        'n_samples': 60,
        'outliers_removed': 1,
        'linear': {
            'intercept': -2.6115105461479016,
            'coefficient': 15.670504557631078,
            'r2_score': 0.7780394474347803,
            'mae': 0.37224865014482206
        },
        'polynomial': {
            'intercept': -4.239872122013528,
            'coef_h': 24.69161497996228,
            'coef_h2': -12.189496246730721,
            'r2_score': 0.7802255060944882,
            'mae': 0.3615219795488345
        },
        'power': {
            'a': 24.19808075847763,
            'b': 2.08738339591621,
            'r2_score': 0.750433632649027,
            'mae': 0.4141591973994427
        }
    },
    '10x60_seed42_25cornerplacement_big': {
        'description': '10 rooms × 60 pairs, seed 42, 25% corner placement',
        'n_samples': 60,
        'outliers_removed': 3,
        'linear': {
            'intercept': -1.3612952208783828,
            'coefficient': 12.845789135505381,
            'r2_score': 0.7526838698129632,
            'mae': 0.4659723387066763
        },
        'polynomial': {
            'intercept': -0.7395606722424968,
            'coef_h': 8.691428266938297,
            'coef_h2': 6.479539436121281,
            'r2_score': 0.754091681446555,
            'mae': 0.4632982108534563
        },
        'power': {
            'a': 15.384679373052057,
            'b': 1.5642570223282717,
            'r2_score': 0.748201033780676,
            'mae': 0.47782192508652377
        }
    },
    
}


def predict_c_from_source(source_pos, room_dims, model_type='linear', experiment='4x20_random_pair', verbose=False):
    """
    Predict optimal C parameter from source position and room dimensions.
    
    The prediction is based on the normalized harmonic mean of wall distances (H_src_norm),
    which captures how close the source is to room boundaries. Sources in corners have
    low H_src_norm values and require higher C values.
    
    Args:
        source_pos: tuple (x, y, z) - source position in meters
        room_dims: tuple (width, depth, height) - room dimensions in meters
        model_type: str - 'linear' (default), 'polynomial', or 'power'
        experiment: str - which experiment's model to use:
                         '4x20_random_pair' (default, original 80-run Monte Carlo)
                         '3x5_grid' (15 sources, 3x5 grid with corners)
                         '8src_diagonal' (8 diagonal sources)
        verbose: bool - if True, print intermediate calculations
    
    Returns:
        float: predicted optimal C value (clamped to [1.0, 7.0])
    
    Example:
        >>> source_pos = (1.2, 1.5, 0.7)  # Near corner
        >>> room_dims = (5.5, 7.8, 4.33)
        >>> c_opt = predict_c_from_source(source_pos, room_dims, experiment='3x5_grid')
        >>> print(f"Recommended C: {c_opt:.2f}")
        Recommended C: 2.23
    """
    # Calculate room volume and characteristic length
    volume = room_dims[0] * room_dims[1] * room_dims[2]
    char_len = volume ** (1.0 / 3.0)
    
    # Calculate wall distances
    x, y, z = source_pos
    w, d, h = room_dims
    dists = get_wall_distances(x, y, z, w, d, h)
    
    # Harmonic mean (dominated by smallest distance, i.e., wall proximity)
    h_src = harmonic_mean(dists)
    
    # Normalize by characteristic length for scale invariance
    h_src_norm = h_src / char_len
    
    if verbose:
        print(f"Source position: ({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"Room dimensions: ({w:.2f}, {d:.2f}, {h:.2f})")
        print(f"Characteristic length: {char_len:.3f} m")
        print(f"Wall distances: {[f'{d:.3f}' for d in dists]}")
        print(f"H_src (harmonic mean): {h_src:.4f} m")
        print(f"H_src_norm (normalized): {h_src_norm:.4f}")
        print(f"Using experiment: {experiment}")
    
    # Get model parameters for selected experiment
    if experiment not in EXPERIMENT_MODELS:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from: {list(EXPERIMENT_MODELS.keys())}")
    
    exp_models = EXPERIMENT_MODELS[experiment]
    
    if model_type not in exp_models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from: {list(exp_models.keys())}")
    
    params = exp_models[model_type]
    
    if model_type == 'linear':
        # Linear Model: C = a + b * H_src_norm
        c_pred = params['intercept'] + params['coefficient'] * h_src_norm
        
    elif model_type == 'polynomial':
        # Polynomial Model: C = a + b*H + c*H²
        c_pred = params['intercept'] + params['coef_h'] * h_src_norm + params['coef_h2'] * h_src_norm**2
        
    elif model_type == 'power':
        # Power Law Model: C = a * H_src_norm^b
        c_pred = params['a'] * h_src_norm ** params['b']
    
    # Clamp to valid SDN range
    c_pred = np.clip(c_pred, 1.0, 7.0)
    
    if verbose:
        print(f"Predicted C ({model_type} model): {c_pred:.3f}")
        print(f"  Model R²: {params.get('r2_score', 'N/A'):.3f}")
        
        # Warning for extreme cases
        if h_src_norm < 0.20:
            print("⚠ WARNING: Source is very close to walls/corners (H_src_norm < 0.20)")
            print("  SDN may not perform well in such extreme geometries.")
            print("  Consider repositioning source or using ISM for reference.")
        elif c_pred >= 6.8:
            print("⚠ WARNING: Predicted C is near upper bound (≥6.8)")
            print("  This suggests difficult acoustic conditions.")
    
    return c_pred


def predict_c_for_source_grid(source_positions, room_dims, model_type='linear', experiment='4x20_random_pair'):
    """
    Predict optimal C for multiple source positions.
    
    Args:
        source_positions: list of (x, y, z) tuples
        room_dims: tuple (width, depth, height)
        model_type: str - 'linear', 'polynomial', or 'power'
        experiment: str - which experiment's model to use
    
    Returns:
        dict: {source_idx: {'position': (x,y,z), 'c_predicted': float, 'h_src_norm': float}}
    """
    results = {}
    for idx, src_pos in enumerate(source_positions):
        # Calculate H_src_norm
        x, y, z = src_pos
        w, d, h = room_dims
        volume = w * d * h
        char_len = volume ** (1/3)
        dists = get_wall_distances(x, y, z, w, d, h)
        h_src = harmonic_mean(dists)
        h_src_norm = h_src / char_len
        
        # Predict C
        c_pred = predict_c_from_source(src_pos, room_dims, model_type, experiment, verbose=False)
        
        results[idx] = {
            'position': src_pos,
            'h_src_norm': h_src_norm,
            'c_predicted': c_pred
        }
    
    return results


def export_predictions_to_json(predictions, output_file, experiment_info=None):
    """
    Export C predictions to JSON file.
    
    Args:
        predictions: dict from predict_c_for_source_grid()
        output_file: path to output JSON file
        experiment_info: optional dict with experiment metadata
    """
    output = {
        'experiment_info': experiment_info or {},
        'predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved predictions to: {output_file}")


def load_fitted_model_params(model_file=None):
    """
    Load model parameters from JSON file if available.
    
    Args:
        model_file: path to JSON file with model parameters
                   (default: looks in results/c_prediction_model_params.json)
    
    Returns:
        dict: model parameters including type, coefficients, and performance metrics
    """
    if model_file is None:
        # Try default location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(script_dir, "results", "c_prediction_model_params.json")
    
    if not os.path.exists(model_file):
        return None
    
    with open(model_file, 'r') as f:
        return json.load(f)


# =====================================================================
# COMMAND-LINE INTERFACE
# =====================================================================

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("C PARAMETER PREDICTOR FOR SDN")
    print("="*70)
    
    # Example 1: Corner source (challenging)
    print("\n[Example 1: Corner Source - 4x20 Random Pair Model]")
    src1 = (1.2, 1.5, 0.7)
    room1 = (5.5, 7.8, 4.33)
    c1 = predict_c_from_source(src1, room1, model_type='linear', experiment='4x20_random_pair', verbose=True)
    print(f"\n➜ Recommended C: {c1:.2f}")
    
    # Example 2: Center source (easier)
    print("\n" + "-"*70)
    print("\n[Example 2: Center Source - 3x5 Grid Model]")
    src2 = (2.75, 3.9, 2.2)
    room2 = (5.5, 7.8, 4.33)
    c2 = predict_c_from_source(src2, room2, model_type='polynomial', experiment='3x5_grid', verbose=True)
    print(f"\n➜ Recommended C: {c2:.2f}")
    
    # Example 3: Test all models
    print("\n" + "-"*70)
    print("\n[Example 3: Model Comparison - 3x5 Grid Experiment]")
    src3 = (2.0, 3.0, 1.5)
    room3 = (6.08, 4.85, 3.5)
    print(f"Source: {src3}, Room: {room3}")
    for model in ['linear', 'polynomial', 'power']:
        c = predict_c_from_source(src3, room3, model_type=model, experiment='3x5_grid', verbose=False)
        print(f"  {model.capitalize():12s} model: C = {c:.3f}")
    
    print("\n" + "="*70)
    print("\nAvailable experiments:")
    for exp_name, exp_data in EXPERIMENT_MODELS.items():
        print(f"  - {exp_name}: {exp_data['description']}")
    
    print("\n" + "="*70)
    
    # Interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        print("\nINTERACTIVE MODE")
        print("Enter source position and room dimensions to get C prediction.")
        print("(Press Ctrl+C to exit)\n")
        
        try:
            while True:
                print("-"*70)
                x = float(input("Source X (m): "))
                y = float(input("Source Y (m): "))
                z = float(input("Source Z (m): "))
                w = float(input("Room Width (m): "))
                d = float(input("Room Depth (m): "))
                h = float(input("Room Height (m): "))
                
                print("\nChoose experiment:")
                for i, exp_name in enumerate(EXPERIMENT_MODELS.keys(), 1):
                    print(f"  {i}. {exp_name}")
                exp_choice = int(input("Enter number: "))
                exp_name = list(EXPERIMENT_MODELS.keys())[exp_choice - 1]
                
                c_pred = predict_c_from_source((x, y, z), (w, d, h), experiment=exp_name, verbose=True)
                print(f"\n✓ Predicted Optimal C: {c_pred:.2f}\n")
        except KeyboardInterrupt:
            print("\n\nExiting.")
        except ValueError as e:
            print(f"Error: {e}")
