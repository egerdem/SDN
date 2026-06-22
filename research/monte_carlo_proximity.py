"""
Monte Carlo Experiment Generator for SDN C Parameter Optimization

What this script does:
----------------------
1. Generates random (or fixed) room configurations
2. Places source(s) and receiver(s) in systematic grids
3. Computes reference RIRs (RIMPY) and caches them
4. Finds optimal C parameter for SDN that minimizes RMSE vs reference
5. Computes geometric metrics (H_src_norm, d_sm_norm, etc.)

Outputs:
--------
results/monte_carlo_experiments/<EXPERIMENT_ID>/
  ├── results.json        # Optimization results (optimal_c, min_rmse, geometry)
  ├── rimpy_cache/        # Cached RIMPY RIRs (expensive to recalculate)
  │   └── rimpy_*.npy
  ├── validation/         # (created later by validate_c_predictor.py)
  └── config.json         # Experiment configuration snapshot

results.json contains:
----------------------
For each (room, source, receiver) configuration:
- room_dims, src_pos, rx_pos
- optimal_c: Best C value found via optimization
- min_rmse: RMSE at optimal_c
- h_src_norm: Normalized harmonic mean wall distance (source proximity)
- d_sm_norm: Normalized source-receiver distance
- Other geometric metrics

Usage:
------
# Default experiment (no seed)
$ python research/monte_carlo_proximity.py

# Reproducible experiment with seed
$ export MC_EXPERIMENT_ID=seed_42
$ python research/monte_carlo_proximity.py

# Custom experiment name
$ export MC_EXPERIMENT_ID=my_custom_experiment
$ python research/monte_carlo_proximity.py

Configuration:
--------------
Edit PRESET at top of file to choose experiment type:
- "aes_diagonal_8src_fullgrid": Fixed AES room, diagonal grid sources
- "aes_fullgrid_perpair": Fixed room, custom source grids
- "mc_fullgrid_perpair": Random rooms, parametric source grid
- "legacy_pair_single_mic": Legacy single-receiver pairs

Next step:
----------
After running this script, use validate_c_predictor.py to evaluate
how well your C predictor model performs on this dataset.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import hashlib
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

import geometry
from analysis import analysis as an
from deprecated import wall_proximity
from optimisation_singleC import find_optimal_c
from rir_calculators import rir_normalisation, calculate_rimpy_rir, enable_basis_disk_cache, calculate_sdn_rir_fast
from analysis.spatial_analysis import (
    generate_fixed_source_grid,
    generate_fixed_source_grid_simple,
    print_source_grid,
)

###############################################################################
# Experiment configuration (preset-driven)
###############################################################################
#
# The goal is to avoid "hundreds of knobs" at module scope.
# You typically change only PRESET (and optionally EXPERIMENT_NAME).
#
# Presets:
# - "aes_fullgrid_perpair": 1 fixed room via `experiment_configs.active_room`, fixed duration, per-(src,rx) c*
# - "mc_fullgrid_perpair": Monte Carlo rooms, Sabine-adaptive duration, per-(src,rx) c*
# - "legacy_pair_single_mic": legacy Monte Carlo pairs, fixed duration, single receiver
#PRESET = "legacy_pair_single_mic"
PRESET = os.environ.get("MC_PRESET", "aes_diagonal3D_8src_fullgrid")
# Optional: put outputs into `results/experiments/<EXPERIMENT_NAME>/`
# If None, a descriptive name is auto-generated from the preset + key params.
EXPERIMENT_NAME: Optional[str] = None

# Outlier rejection is now applied INLINE during experiment generation
# using the apply_outlier_rejection() function. Results are automatically
# corrected and saved with 'outlier_corrected' flags.
# See apply_outlier_rejection() function below for implementation.

# Constants that are truly global (rarely changed)
ERR_DURATION_MS = 50
PADDING = 0.5  # Minimum distance from walls

if TYPE_CHECKING:
    # Type-only stubs so IDEs/linters understand these globals (they are set by `_apply_preset()` at import).
    MODE: str
    FULLGRID_OPTIMIZE_SCOPE: str
    NUM_ROOMS: int
    PAIRS_PER_ROOM: int
    RECEIVER_GRID_N_X: int
    RECEIVER_GRID_N_Y: int
    RECEIVER_GRID_MARGIN: float
    SOURCE_GRID_N_X: int
    SOURCE_GRID_N_Y: int
    SOURCE_GRID_INCLUDE_CORNERS: bool
    SOURCE_GRID_CORNER_OFFSET: float
    SOURCE_GENERATION_SPECS: Optional[list]
    FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS: bool
    WIDTH_RANGE: Tuple[float, float]
    DEPTH_RANGE: Tuple[float, float]
    HEIGHT_RANGE: Tuple[float, float]
    FS: int
    DURATION: float
    ABSORPTION: float
    TOTAL_RUNS: int


def _preset_table() -> Dict[str, Dict[str, Any]]:
    return {
        "aes_diagonal_8src_fullgrid": {
            "MODE": "fullgrid_sources",
            "FULLGRID_OPTIMIZE_SCOPE": "per_pair",
            "NUM_ROOMS": 1,
            "RECEIVER_GRID_N_X": 4,
            "RECEIVER_GRID_N_Y": 4,
            "RECEIVER_GRID_MARGIN": 0.5,
            # Single source generation spec with diagonals
            "SOURCE_GENERATION_SPECS": [
                {
                    "func": "generate_fixed_source_grid",
                    "padding": 0.5,
                    "n_x": 15,
                    "n_y": 15,
                    "z_mode": "fixed_1p5",
                    "include_corners": True,
                    "corner_offset": 0.5,
                    "diagonals": True  # 2D diagonal
                }
            ],
            # These are needed for _recompute_total_runs() even when using SOURCE_GENERATION_SPECS
            "SOURCE_GRID_N_X": 15,
            "SOURCE_GRID_N_Y": 15,
            "SOURCE_GRID_INCLUDE_CORNERS": True,
            "FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS": True,
            "USE_ADAPTIVE_DURATION": False,  # Use fixed duration from experiment_configs
        },

        "aes_diagonal3D_8src_fullgrid": {
            "MODE": "fullgrid_sources",
            "FULLGRID_OPTIMIZE_SCOPE": "per_pair",
            "NUM_ROOMS": 1,
            "RECEIVER_GRID_N_X": 4,
            "RECEIVER_GRID_N_Y": 4,
            "RECEIVER_GRID_MARGIN": 0.5,
            # Single source generation spec with 3D diagonals
            "SOURCE_GENERATION_SPECS": [
                {
                    "func": "generate_fixed_source_grid",
                    "padding": 0.5,
                    "n_x": 15,
                    "n_y": 15,
                    "z_mode": "fixed_1p5",
                    "include_corners": True,
                    "corner_offset": 0.5,
                    "diagonals": "3d"  # 3D diagonal
                }
            ],
            # These are needed for _recompute_total_runs() even when using SOURCE_GENERATION_SPECS
            "SOURCE_GRID_N_X": 15,
            "SOURCE_GRID_N_Y": 15,
            "SOURCE_GRID_INCLUDE_CORNERS": True,
            "FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS": True,
            "USE_ADAPTIVE_DURATION": False,  # Use fixed duration from experiment_configs
        },

        "aes_fullgrid_perpair": {
            "MODE": "fullgrid_sources",
            "FULLGRID_OPTIMIZE_SCOPE": "per_pair",
            "NUM_ROOMS": 1,
            "RECEIVER_GRID_N_X": 4,
            "RECEIVER_GRID_N_Y": 4,
            "RECEIVER_GRID_MARGIN": 0.5,
            # Custom source generation: combine two specific grids
            "SOURCE_GENERATION_SPECS": [
                {"func": "generate_fixed_source_grid", "padding": 0.5, "n_x": 3, "n_y": 3, "z_mode": "fixed_1p5", "include_corners": True, "corner_offset": 1},
                {"func": "generate_fixed_source_grid_simple", "padding": 2, "n_x": 2, "n_y": 2, "z_mode": "fixed_1p5"},
            ],
            # Parametric source grid params (unused when SOURCE_GENERATION_SPECS is set, but kept for clarity)
            "SOURCE_GRID_N_X": 3,
            "SOURCE_GRID_N_Y": 5,
            "SOURCE_GRID_INCLUDE_CORNERS": True,
            "SOURCE_GRID_CORNER_OFFSET": 2.0,
            # Fixed room + duration come from experiment_configs.active_room / experiment_configs.duration
            "FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS": True,
            "USE_ADAPTIVE_DURATION": False,  # Use fixed duration from experiment_configs
        },
        "mc_fullgrid_perpair": {
            "MODE": "fullgrid_sources",
            "FULLGRID_OPTIMIZE_SCOPE": "per_pair",
            "NUM_ROOMS": 5,
            "RECEIVER_GRID_N_X": 4,
            "RECEIVER_GRID_N_Y": 4,
            "RECEIVER_GRID_MARGIN": 0.5,
            # Parametric source generation (no manual specs)
            "SOURCE_GENERATION_SPECS": None,
            "SOURCE_GRID_N_X": 3,
            "SOURCE_GRID_N_Y": 5,
            "SOURCE_GRID_INCLUDE_CORNERS": True,
            "SOURCE_GRID_CORNER_OFFSET": 2.0,
            "FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS": False,
            # Room dimension ranges (Monte Carlo)
            "WIDTH_RANGE": (3.0, 9.5),
            "DEPTH_RANGE": (4.0, 7.5),
            "HEIGHT_RANGE": (2.6, 4.2),
            "FS": 44100,
            "ABSORPTION": 0.2,
            "USE_ADAPTIVE_DURATION": True,  # Adapt to RT60 (Monte Carlo rooms)
        },
        "legacy_pair_single_mic": {
            "MODE": "pair_single_mic",
            "NUM_ROOMS": 10,
            "PAIRS_PER_ROOM": 6,
            "SOURCE_GENERATION_SPECS": None,
            "WIDTH_RANGE": (3.0, 9.0),
            "DEPTH_RANGE": (3.0, 8.0),
            "HEIGHT_RANGE": (2.5, 7.0),
            "FS": 44100,
            "USE_ADAPTIVE_DURATION": True,  # Adapt duration to RT60 (1.2*RT60, clipped 1.0-2.5s)
            # "DURATION": 1.0,  # Base duration (used if USE_ADAPTIVE_DURATION=False)
            "ABSORPTION": 0.2,
            "CORNER_BIAS_PROBABILITY": 0.0,  # 0 = uniform, 0.25 = 25% corner bias
            # Note: np.random.seed(42) at startup makes rooms, positions, AND RIMPY reproducible
        },
    }


def _apply_preset(preset: str) -> None:
    presets = _preset_table()
    if preset not in presets:
        raise ValueError(f"Unknown PRESET '{preset}'. Available: {sorted(presets.keys())}")
    for k, v in presets[preset].items():
        globals()[k] = v


def _auto_experiment_name() -> str:
    scope = globals().get("FULLGRID_OPTIMIZE_SCOPE", "na")
    nx = globals().get("SOURCE_GRID_N_X", "x")
    ny = globals().get("SOURCE_GRID_N_Y", "y")
    corner = globals().get("SOURCE_GRID_CORNER_OFFSET", "c")
    return f"{PRESET}_srcgrid{nx}x{ny}_corner{corner}_{scope}"


def _recompute_total_runs() -> int:
    mode = globals()["MODE"]
    num_rooms = int(globals()["NUM_ROOMS"])
    if mode == "pair_single_mic":
        pairs_per_room = int(globals()["PAIRS_PER_ROOM"])
        return num_rooms * pairs_per_room
    
    # fullgrid_sources: need to actually generate sources to get accurate count
    # (especially when using diagonals or other filters)
    optimize_scope = globals()["FULLGRID_OPTIMIZE_SCOPE"]
    est_receivers = int(globals()["RECEIVER_GRID_N_X"]) * int(globals()["RECEIVER_GRID_N_Y"])
    
    # Try to get actual source count by generating them
    source_generation_specs = globals().get("SOURCE_GENERATION_SPECS")
    if source_generation_specs is not None:
        # Generate sources using the specs to get accurate count
        try:
            # We need a dummy room_params to generate sources
            # Use experiment_configs.active_room if available
            import experiment_configs
            room_params = experiment_configs.active_room
            
            all_sources = []
            for spec in source_generation_specs:
                func_name = spec["func"]
                kwargs = {k: v for k, v in spec.items() if k != "func"}
                kwargs["room_params"] = room_params
                
                if func_name == "generate_fixed_source_grid":
                    from analysis.spatial_analysis import generate_fixed_source_grid
                    all_sources.append(generate_fixed_source_grid(**kwargs))
                elif func_name == "generate_fixed_source_grid_simple":
                    from analysis.spatial_analysis import generate_fixed_source_grid_simple
                    all_sources.append(generate_fixed_source_grid_simple(**kwargs))
            
            # Deduplicate
            sources_set = set()
            sources = []
            for grid in all_sources:
                for src in grid:
                    key = (round(src[0], 2), round(src[1], 2), round(src[2], 2))
                    if key not in sources_set:
                        sources_set.add(key)
                        sources.append(src)
            
            est_sources = len(sources)
        except Exception as e:
            print(f"Warning: Could not generate sources for TOTAL_RUNS calculation: {e}")
            # Fallback to estimate
            n_x = int(globals()["SOURCE_GRID_N_X"])
            n_y = int(globals()["SOURCE_GRID_N_Y"])
            include_corners = bool(globals()["SOURCE_GRID_INCLUDE_CORNERS"])
            est_sources = (n_x * n_y) + (4 if include_corners else 0)
    else:
        # Parametric generation estimate
        n_x = int(globals()["SOURCE_GRID_N_X"])
        n_y = int(globals()["SOURCE_GRID_N_Y"])
        include_corners = bool(globals()["SOURCE_GRID_INCLUDE_CORNERS"])
        est_sources = (n_x * n_y) + (4 if include_corners else 0)
    
    return num_rooms * (est_sources if optimize_scope == "per_source" else est_sources * est_receivers)


# Apply preset at import time (so the rest of the file can keep using MODE/FS/etc).
_apply_preset(PRESET)

# Per-run absorption override (frequency-independent sweep).
_abs_env = os.environ.get("MC_ABSORPTION")
if _abs_env is not None:
    ABSORPTION = float(_abs_env)
    print(f"[absorption] {ABSORPTION}")

# Pair mode reads the fixed 60-config geometry so positions are identical across absorptions.
FIXED_GEOMETRY = None
if MODE == "pair_single_mic":
    _fixed_geo_path = os.environ.get("MC_FIXED_GEOMETRY",
        os.path.join(project_root, "results", "monte_carlo_experiments", "mc60_geometry.json"))
    if os.path.exists(_fixed_geo_path):
        _fg = json.load(open(_fixed_geo_path))
        FIXED_GEOMETRY = {(c["room_idx"], c["pair_idx"]): c for c in _fg["configs"]}
        NUM_ROOMS, PAIRS_PER_ROOM = int(_fg["num_rooms"]), int(_fg["pairs_per_room"])
        print(f"[FIXED_GEOMETRY] {len(FIXED_GEOMETRY)} configs from {_fixed_geo_path}")

if EXPERIMENT_NAME is None:
    EXPERIMENT_NAME = _auto_experiment_name()
TOTAL_RUNS = _recompute_total_runs()

# Minimal runtime validation removed - Python will raise NameError if variables are truly missing
# This allows for more flexible preset configurations

# =====================================================================
# EXPERIMENT FOLDER ORGANIZATION
# =====================================================================
# All Monte Carlo outputs now go into organized experiment folders:
#   results/monte_carlo_experiments/{EXPERIMENT_ID}/
#     - results.json
#     - rimpy_cache/
#     - validation/
#
# EXPERIMENT_ID can be:
#   - "default" (if no seed specified)
#   - "seed_{N}" (if seed specified)
#   - custom name via environment variable: MC_EXPERIMENT_ID

import datetime

def _get_experiment_id() -> str:
    """Determine experiment ID for this Monte Carlo run."""
    # Option 1: Custom ID via environment variable
    custom_id = os.environ.get('MC_EXPERIMENT_ID')
    if custom_id:
        return custom_id
    
    # Option 2: Seed-based ID (if you set np.random.seed in your code)
    # For now, we'll use "default" but you can modify this logic
    return "default"

EXPERIMENT_ID = _get_experiment_id()
EXPERIMENT_ROOT = os.path.join(project_root, "results", "monte_carlo_experiments", EXPERIMENT_ID)
OUTPUT_FILE = os.path.join(EXPERIMENT_ROOT, "results.json")


def _atomic_json_dump(path: str, payload) -> None:
    """Write JSON atomically (safe for stop/restart)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _resolve_experiment_dir(default_dir: str) -> str:
    """
    Resolve where to write outputs for this run.
    If EXPERIMENT_NAME is set, use results/experiments/<EXPERIMENT_NAME>/, otherwise use default_dir.
    """
    if EXPERIMENT_NAME:
        return os.path.join(project_root, "results", "experiments", str(EXPERIMENT_NAME))
    return default_dir


def _write_experiment_config(experiment_dir: str, payload: dict) -> None:
    """Persist a small config snapshot next to results for later traceability."""
    try:
        os.makedirs(experiment_dir, exist_ok=True)
        _atomic_json_dump(os.path.join(experiment_dir, "config.json"), payload)
    except Exception:
        # Best-effort only; don't block long runs if config write fails.
        pass


def _rimpy_cache_paths(cache_dir: str,
                       room_dims: tuple[float, float, float],
                       src_pos: tuple[float, float, float],
                       rx_pos: tuple[float, float, float],
                       duration_s: float,
                       Fs: int,
                       reflection_sign: int,
                       rand_dist: float,
                       absorption: float) -> tuple[str, str]:
    """
    Compute stable cache paths for a RIMPY RIR by geometry + absorption.
    Returns: (npy_path, json_meta_path)
    """
    payload = {
        "room_dims": [float(x) for x in room_dims],
        "src_pos": [float(x) for x in src_pos],
        "rx_pos": [float(x) for x in rx_pos],
        "duration_s": float(duration_s),
        "Fs": int(Fs),
        "reflection_sign": int(reflection_sign),
        "rand_dist": float(rand_dist),
        "absorption": float(absorption),
    }
    key = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:12]
    base = f"rimpy_{h}"
    return (os.path.join(cache_dir, f"{base}.npy"), os.path.join(cache_dir, f"{base}.json"))

def _euclidean_distance(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))


def estimate_rt60_sabine(w: float, d: float, h: float, alpha: float) -> float:
    """
    Very simple Sabine RT60 estimate for a shoebox room with uniform absorption.
    RT60 ≈ 0.161 * V / Aeq, where Aeq = alpha * S_total.
    """
    v = w * d * h
    s_total = 2.0 * (w * d + w * h + d * h)
    aeq = max(alpha * s_total, 1e-9)
    return float(0.161 * v / aeq)


def choose_duration_for_room(w: float, d: float, h: float, alpha: float) -> float:
    """
    Pick a duration long enough for stable EDC comparison (captures late tail).
    We base it on a Sabine RT60 estimate but clamp to keep compute bounded.
    """
    rt60 = estimate_rt60_sabine(w, d, h, alpha)
    # Heuristic: 1.2x RT60 provides sufficient decay capture
    # Keep within practical bounds for Monte Carlo.
    dur = 1.2 * rt60
    return float(np.clip(dur, 1.0, 2.5))


def apply_outlier_rejection(opt_c: float, min_rmse: float, rmse_at_c1: float, 
                            bounds: tuple = (1.0, 7.0), eps: float = 5e-3) -> tuple:
    """
    Apply outlier rejection to optimal_c.
    
    If optimizer hits bounds (1.0 or 7.0 ± eps) AND c=1 has better (or equal) RMSE,
    replace optimal_c with 1.0.
    
    Args:
        opt_c: Optimized c value
        min_rmse: RMSE at optimal_c
        rmse_at_c1: RMSE at c=1
        bounds: Optimization bounds (lower, upper)
        eps: Tolerance for detecting bound hits
    
    Returns:
        (c_final, rmse_final, was_rejected): Tuple of final c, final RMSE, and rejection flag
    """
    lower, upper = bounds
    hit_upper = abs(opt_c - upper) <= eps
    hit_lower = abs(opt_c - lower) <= eps
    
    if hit_upper or hit_lower:
        # Optimizer hit a bound - check if c=1 is better
        if rmse_at_c1 <= min_rmse + 1e-6:
            # c=1 is better (or equal), use it instead
            return 1.0, rmse_at_c1, True
    
    return opt_c, min_rmse, False


def _ensure_derived_metrics(result_dict):
    """
    Backward/forward compatibility helper.
    If older cached JSON entries are missing newly-added derived metrics,
    compute and insert them in-place.
    """
    # Always prefer a descriptive run key over a sequential id.
    # run_id is kept for backward compatibility.
    if "run_key" not in result_dict:
        # If room/pair indices exist, use them.
        if "room_idx" in result_dict and "pair_idx" in result_dict:
            result_dict["run_key"] = f"room{int(result_dict['room_idx']) + 1}_pair{int(result_dict['pair_idx']) + 1}"
        # Else infer from legacy run_id if possible.
        elif "run_id" in result_dict:
            rid = int(result_dict["run_id"])
            room_idx = (rid - 1) // PAIRS_PER_ROOM
            pair_idx = (rid - 1) % PAIRS_PER_ROOM
            result_dict["room_idx"] = room_idx
            result_dict["pair_idx"] = pair_idx
            result_dict["run_key"] = f"room{room_idx + 1}_pair{pair_idx + 1}"

    if "src_pos" in result_dict and "mic_pos" in result_dict and "char_len" in result_dict:
        src_pos = result_dict["src_pos"]
        mic_pos = result_dict["mic_pos"]
        char_len = float(result_dict["char_len"])

        if "src_mic_dist" not in result_dict:
            result_dict["src_mic_dist"] = _euclidean_distance(src_pos, mic_pos)
        if "src_mic_dist_norm" not in result_dict:
            result_dict["src_mic_dist_norm"] = float(result_dict["src_mic_dist"]) / char_len

        # Composite candidates (dimensionless)
        if "h_src_norm" in result_dict:
            h = float(result_dict["h_src_norm"])
            d = float(result_dict["src_mic_dist_norm"])
            
            # New candidates (match hypothesis: larger d_sm_norm -> lower C)
            if "h_minus_d_norm" not in result_dict:
                result_dict["h_minus_d_norm"] = h - d
            if "h_over_d_norm" not in result_dict:
                eps = 1e-6
                result_dict["h_over_d_norm"] = h / (d + eps)

        # Node-distance metrics can be recomputed from geometry if missing.
        # Note: node locations depend on BOTH source and microphone positions.
        if (
            "room_dims" in result_dict
            and (
                "mean_node_dist_all_norm" not in result_dict
                or "hmean_node_dist_norm" not in result_dict
                or "mean_node_dist_norm" not in result_dict
            )
        ):
            try:
                w, d_room, h_room = result_dict["room_dims"]
                temp_room = geometry.Room(w, d_room, h_room)
                temp_source = geometry.Source(*src_pos, signal=None, Fs=FS)
                temp_room.source = temp_source
                temp_room.set_microphone(*mic_pos)

                node_dists = []
                if temp_room.sdn_nodes_calculated:
                    for wall in temp_room.walls.values():
                        np_pos = wall.node_positions
                        dist = temp_room.source.srcPos.getDistance(np_pos)
                        node_dists.append(float(dist))

                if node_dists:
                    node_dists.sort()
                    mean3 = float(np.mean(node_dists[:3]))
                    mean_all = float(np.mean(node_dists))
                    inv_sum = sum(1.0 / dd for dd in node_dists if dd > 0)
                    hmean_all = float(len(node_dists) / inv_sum) if inv_sum > 0 else float("inf")

                    # Store both raw and normalized versions
                    result_dict.setdefault("mean_node_dist", mean3)
                    result_dict.setdefault("mean_node_dist_norm", mean3 / char_len)
                    result_dict.setdefault("mean_node_dist_all", mean_all)
                    result_dict.setdefault("mean_node_dist_all_norm", mean_all / char_len)
                    result_dict.setdefault("hmean_node_dist", hmean_all)
                    result_dict.setdefault("hmean_node_dist_norm", hmean_all / char_len)
                else:
                    # Fallback if nodes not calculated for some reason
                    result_dict.setdefault("mean_node_dist_all_norm", np.nan)
                    result_dict.setdefault("hmean_node_dist_norm", np.nan)
            except Exception:
                # Keep existing dict; plotting code will handle NaNs.
                result_dict.setdefault("mean_node_dist_all_norm", np.nan)
                result_dict.setdefault("hmean_node_dist_norm", np.nan)
    return result_dict

def generate_random_room():
    w = np.random.uniform(*WIDTH_RANGE)
    d = np.random.uniform(*DEPTH_RANGE)
    h = np.random.uniform(*HEIGHT_RANGE)
    return w, d, h

def generate_random_position(w, d, h, padding, mode='uniform', corner_bias_prob=0.5):
    if mode == 'corner_biased' and np.random.rand() < corner_bias_prob:
        # corner_bias_prob chance to force a position deep into a random corner
        # Pick a random corner (0 or Max Dimension)
        x_corner = 0 if np.random.rand() < 0.5 else w
        y_corner = 0 if np.random.rand() < 0.5 else d
        z_corner = 0 if np.random.rand() < 0.5 else h
        
        # Add small random offset (0.1m to 1.5m) ensuring it is inside room
        # We need to move Inwards from the corner
        dx = np.random.uniform(0.1, 1.5)
        dy = np.random.uniform(0.1, 1.5)
        dz = np.random.uniform(0.1, 1.5)
        
        x = x_corner + dx if x_corner == 0 else x_corner - dx
        y = y_corner + dy if y_corner == 0 else y_corner - dy
        z = z_corner + dz if z_corner == 0 else z_corner - dz
        
        # Clamp to be safe (though logic above should handle it if w,d,h > 1.5)
        x = np.clip(x, padding, w - padding)
        y = np.clip(y, padding, d - padding)
        z = np.clip(z, padding, h - padding)
        
        return x, y, z
    else:
        # Standard uniform generation
        x = np.random.uniform(padding, w - padding)
        y = np.random.uniform(padding, d - padding)
        z = np.random.uniform(padding, h - padding)
        return x, y, z

def run_monte_carlo():
    if MODE != "pair_single_mic":
        raise RuntimeError("run_monte_carlo() is legacy. Use run_monte_carlo_fullgrid_sources() for MODE='fullgrid_sources'.")

    print(f"--- Starting Monte Carlo Experiment (N={TOTAL_RUNS}) ---")
    
    # 0. Enable Basis Cache (shared with validate_c_predictor.py)
    enable_basis_disk_cache()
    print("✓ Basis function disk cache enabled (shared across scripts)")
    
    # 1. Set Seed for Deterministic "Random" Rooms, Positions, and RIMPY
    # This single seed(42) makes ALL randomness reproducible:
    # - Room dimensions (Monte Carlo)
    # - Source/receiver positions
    # - RIMPY image source perturbations (randDist=0.1)
    np.random.seed(42)
    
    # 2. Cache Directory (within experiment folder)
    RIR_CACHE_DIR = os.path.join(EXPERIMENT_ROOT, "rimpy_cache")
    os.makedirs(RIR_CACHE_DIR, exist_ok=True)
    
    print(f"Experiment folder: {EXPERIMENT_ROOT}")
    print(f"RIR cache: {RIR_CACHE_DIR}")
    print(f"Global RNG seed: 42 (affects rooms, positions, RIMPY)")
    
    # 3. Load Existing Results (keyed by run_key for readability/stability)
    existing_results_map = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                data = json.load(f)
                for r in data:
                    _ensure_derived_metrics(r)
                    existing_results_map[r.get("run_key", str(r.get("run_id")))] = r
            print(f"Loaded {len(existing_results_map)} existing results from {OUTPUT_FILE} (auto-upgraded schema)")
        except Exception as e:
            print(f"Error loading existing results: {e}")

    results = [] # Final list to save
    runs_added_this_session = set()  # Track runs added in THIS session (not from file)
    
    start_time = time.time()
    
    for room_idx in range(NUM_ROOMS):
        # 4. Room dimensions: fixed dataset or random
        if FIXED_GEOMETRY is not None:
            w, d, h = [float(x) for x in FIXED_GEOMETRY[(room_idx, 0)]["room_dims"]]
        else:
            w, d, h = generate_random_room()
            # Round dimensions for cleaner logs
            w, d, h = round(w, 2), round(d, 2), round(h, 2)
        
        room_params = {
            'width': w, 'depth': d, 'height': h,
            'absorption': ABSORPTION, 'reflection': np.sqrt(1 - ABSORPTION),
            'mic z': 1.5, # Placeholder
            # Air absorption parameters (standard conditions)
            'air': {'temperature': 20, 'humidity': 50} 
        }
        
        # Adaptive duration based on RT60 (if enabled)
        use_adaptive_duration = globals().get('USE_ADAPTIVE_DURATION', False)
        if use_adaptive_duration:
            rt60_est = estimate_rt60_sabine(w, d, h, ABSORPTION)
            duration_s = choose_duration_for_room(w, d, h, ABSORPTION)
            print(f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}, RT60~{rt60_est:.2f}s, duration={duration_s:.2f}s")
        else:
            rt60_est = None
            duration_s = globals().get('DURATION', 1.0)  # Fallback to 1.0s if not specified
            print(f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}, duration={duration_s:.2f}s (fixed)")
        
        # 5. Generate Pairs
        for pair_idx in range(PAIRS_PER_ROOM):
            run_id = room_idx * PAIRS_PER_ROOM + pair_idx + 1  # legacy numeric id
            run_key = f"room{room_idx + 1}_pair{pair_idx + 1}"
            
            # --- RESUME LOGIC ---
            if run_key in existing_results_map:
                existing = existing_results_map[run_key]
                
                # Only consume RNG if this was loaded from file (not just added this session)
                if run_key not in runs_added_this_session:
                    print(f"  Run {run_key} ({run_id}/{TOTAL_RUNS}): SKIPPING (loaded from file, consuming RNG)")
                    # Ensure any newly added derived fields exist before re-saving
                    _ensure_derived_metrics(existing)
                    results.append(existing)
                    
                    # Consume RNG to keep sync (RNG mode only)
                    if FIXED_GEOMETRY is None:
                        corner_bias = globals().get('CORNER_BIAS_PROBABILITY', 0.)
                        src_mode = 'uniform' if corner_bias == 0.0 else 'corner_biased'
                        _ = generate_random_position(w, d, h, PADDING, mode=src_mode, corner_bias_prob=corner_bias)
                        _ = generate_random_position(w, d, h, PADDING, mode='uniform')
                else:
                    print(f"  Run {run_key} ({run_id}/{TOTAL_RUNS}): SKIPPING (just added, no RNG consumption)")
                    results.append(existing)
                
                continue
            
            if FIXED_GEOMETRY is not None:
                _cfg = FIXED_GEOMETRY[(room_idx, pair_idx)]
                src_pos = tuple(float(x) for x in _cfg["src_pos"])
                mic_pos = tuple(float(x) for x in _cfg["mic_pos"])
            else:
                corner_bias = globals().get('CORNER_BIAS_PROBABILITY')
                # Optimize: if no corner bias, use uniform mode directly (skips redundant RNG call)
                src_mode = 'uniform' if corner_bias == 0.0 else 'corner_biased'
                src_pos = generate_random_position(w, d, h, PADDING, mode=src_mode, corner_bias_prob=corner_bias)
                mic_pos = generate_random_position(w, d, h, PADDING, mode='uniform') # Generate random Z for mic too
            
            # Update params for rir_calculators
            room_params['source x'], room_params['source y'], room_params['source z'] = src_pos
            room_params['mic x'], room_params['mic y'], room_params['mic z'] = mic_pos
            
            # Volume and Characteristic Length
            vol = w * d * h
            char_len = vol ** (1.0/3.0)

            # Source–Mic distance (normalized) — dimensionless, scale invariant
            src_mic_dist = _euclidean_distance(src_pos, mic_pos)
            src_mic_dist_norm = src_mic_dist / char_len
            
            # --- Geometric Analysis ---
            # 1. Wall Proximity (Harmonic Mean)
            src_dists = wall_proximity.get_wall_distances(*src_pos, w, d, h)
            mic_dists = wall_proximity.get_wall_distances(*mic_pos, w, d, h)
            
            h_src = wall_proximity.harmonic_mean(src_dists)
            h_rcv = wall_proximity.harmonic_mean(mic_dists)
            combined_score = (1.0 / h_src) + (1.0 / h_rcv)
            
            # Normalized Metrics (Scale Invariant)
            h_src_norm = h_src / char_len
            
            # 2. Node Distance (Mean 3 Nearest)
            # Create temporary room to calculate nodes
            temp_room = geometry.Room(w, d, h)
            temp_source = geometry.Source(*src_pos, signal=None, Fs=FS)
            temp_room.source = temp_source # Manually attach source
            temp_room.set_microphone(*mic_pos)
            
            node_dists = []
            if temp_room.sdn_nodes_calculated:
                for wall in temp_room.walls.values():
                    np_pos = wall.node_positions
                    dist = temp_room.source.srcPos.getDistance(np_pos)
                    node_dists.append(dist)
                node_dists.sort()
                mean_node_dist = sum(node_dists[:3]) / 3.0
            else:
                mean_node_dist = 0.0 # Should not happen
            
            mean_node_dist_norm = mean_node_dist / char_len

            # --- Additional node-distance metrics (use all 6 nodes) ---
            # Mean over all node distances (less "nearest-3" specific)
            if node_dists:
                mean_node_dist_all = float(np.mean(node_dists))
                mean_node_dist_all_norm = mean_node_dist_all / char_len

                # Harmonic mean over node distances (emphasizes nearest nodes, but uses all)
                inv_sum = sum(1.0 / d for d in node_dists if d > 0)
                hmean_node_dist = float(len(node_dists) / inv_sum) if inv_sum > 0 else float("inf")
                hmean_node_dist_norm = hmean_node_dist / char_len
            else:
                mean_node_dist_all = 0.0
                mean_node_dist_all_norm = 0.0
                hmean_node_dist = 0.0
                hmean_node_dist_norm = 0.0
            
            # --- Acoustic Simulation ---
            # 1. Setup Room
            sim_room = geometry.Room(w, d, h)
            impulse = geometry.Source.generate_signal('dirac', int(FS * duration_s))
            sim_room.set_source(*src_pos, signal=impulse['signal'], Fs=FS)
            sim_room.set_microphone(*mic_pos)
            sim_room.wallAttenuation = [room_params['reflection']] * 6
            
            # 2. Calculate Reference & Normalize
            # Check Cache
            rir_filename = f"rimpy_{run_key}.npy"
            # rir_path = os.path.join(RIR_CACHE_DIR, rir_filename)
            rir_path, meta_path = _rimpy_cache_paths(
                RIR_CACHE_DIR,
                (w, d, h),
                src_pos,
                mic_pos,
                duration_s,
                FS,
                reflection_sign=-1,
                rand_dist=0.1,
                absorption=room_params['absorption'],
            )
            loaded_valid_rir = False
            norm_ref = None
            
            if os.path.exists(rir_path) and os.path.exists(meta_path):
                # Load from cache
                try:
                    ref_rir = np.load(rir_path)
                    
                    # Robustness Check 1: Energy
                    if np.max(np.abs(ref_rir)) < 1e-9:
                        raise ValueError("Silent RIR")
                        
                    # Robustness Check 2: Normalization (Direct Sound)
                    # We basically try to normalize it. If it fails (divide by zero) or gives NaNs, it's bad.
                    with np.errstate(divide='raise', invalid='raise'):
                        norm_dict = rir_normalisation(ref_rir, sim_room, FS, normalize_to_first_impulse=True)
                        norm_ref = norm_dict['single_rir']
                        
                    if np.any(np.isnan(norm_ref)):
                        raise ValueError("Normalization produced NaNs")
                        
                    loaded_valid_rir = True
                    # print(f"  [Run {run_id}] Loaded cached RIMPY RIR")
                    
                except Exception as e:
                    print(f"  [Run {run_id}] Cached RIR invalid ({e}). Re-calculating.")
                    loaded_valid_rir = False

            if not loaded_valid_rir:
                # Calculate
                # NEW: RIMPY-neg10 (Matches main paper data)
                # Note: RIMPY seed was set once at startup (line 595)
                # Don't reset seed here - it would break position generation RNG!
                
                _rimpy_seed = (10000 + run_id) if FIXED_GEOMETRY is not None else None
                ref_rir, _ = calculate_rimpy_rir(
                    room_params, duration_s, FS,
                    reflection_sign=-1, randDist=0.1, rand_seed=_rimpy_seed
                )
                
                # Verify newly calculated one too
                try:
                     with np.errstate(divide='raise', invalid='raise'):
                        norm_dict = rir_normalisation(ref_rir, sim_room, FS, normalize_to_first_impulse=True)
                        norm_ref = norm_dict['single_rir']
                except Exception as e:
                     print(f"  [Run {run_id}] WARNING: Calculated RIR failed normalization too ({e})!")
                     # Fallback? or just let it crash/nan later
                
                # Save to cache
                np.save(rir_path, ref_rir)
                print(f"  [Run {run_id}] Calculated & Cached RIMPY RIR")
            
            # At this point norm_ref is set (or we errored out)
            ref_edc, _, _ = an.compute_edc(norm_ref, FS)
            
            # 3. Optimize SDN
            # find_optimal_c expects `ref_edcs` as a LIST (for multiple receivers).
            # Here we have 1 receiver per pair. So pass `[ref_edc]`.
            receiver_positions = [mic_pos] # List of 1
            
            # Widen bounds to allow finding true optimum even if < 1.0 or > 7.0
            opt_c, min_rmse, _ = find_optimal_c(
                sim_room, room_params, [ref_edc], receiver_positions, 
                duration_s, FS, ERR_DURATION_MS, source_name=f"R{room_idx}_P{pair_idx}",
                bounds=(0.99, 7.0)
            )
            
            # 4. Compute baseline RMSE at C=1 (for principled outlier detection)
            # FastSDN basis at C=1 is already cached during optimization, so this is fast!
            sdn_config_c1 = {
                'enabled': True,
                'calculator': 'sdn',
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 1.0,
                },
                'cache_label': f'mc_baseline_c1_{run_key}',
            }
            _, rir_sdn_c1, _, _ = calculate_sdn_rir_fast(
                room_params, "BaselineC1", sim_room, duration_s, FS, sdn_config_c1
            )
            rir_sdn_c1_norm = rir_normalisation(rir_sdn_c1, sim_room, FS, normalize_to_first_impulse=True)['single_rir']
            edc_sdn_c1, _, _ = an.compute_edc(rir_sdn_c1_norm, FS, plot=False)
            rmse_c1 = float(
                an.compute_RMS(
                    edc_sdn_c1, ref_edc,
                    range=int(ERR_DURATION_MS),
                    Fs=FS,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True,
                )
            )
            
            # Apply outlier correction
            opt_c_original = opt_c
            min_rmse_original = min_rmse
            opt_c, min_rmse, was_outlier_corrected = apply_outlier_rejection(
                opt_c, min_rmse, rmse_c1
            )
            
            # Store Result
            result = {
                'run_id': run_id,
                'run_key': run_key,
                'room_idx': room_idx,
                'pair_idx': pair_idx,
                'room_dims': (w, d, h),
                'room_vol': vol,
                'char_len': char_len,
                'absorption': float(ABSORPTION),
                'src_pos': src_pos,
                'mic_pos': mic_pos,
                'h_src': h_src,
                'h_src_norm': h_src_norm,
                'h_rcv': h_rcv,
                'combined_score': combined_score,
                'mean_node_dist': mean_node_dist,
                'mean_node_dist_norm': mean_node_dist_norm,
                'mean_node_dist_all': mean_node_dist_all,
                'mean_node_dist_all_norm': mean_node_dist_all_norm,
                'hmean_node_dist': hmean_node_dist,
                'hmean_node_dist_norm': hmean_node_dist_norm,
                'src_mic_dist': src_mic_dist,
                'src_mic_dist_norm': src_mic_dist_norm,
                # New candidates (match hypothesis: larger d_sm_norm -> lower C)
                'h_minus_d_norm': float(h_src_norm - src_mic_dist_norm),
                'h_over_d_norm': float(h_src_norm / (src_mic_dist_norm + 1e-6)),
                'duration_s': float(duration_s),
                'optimal_c': opt_c,
                'min_rmse': min_rmse,
                'rmse_c1': rmse_c1,  # Baseline RMSE at C=1 (for principled outlier detection)
                'outlier_corrected': was_outlier_corrected,
            }
            
            # Store original values if outlier was corrected
            if was_outlier_corrected:
                result['optimal_c_original'] = opt_c_original
                result['min_rmse_original'] = min_rmse_original

            # Add RT60 if adaptive duration is enabled
            if rt60_est is not None:
                result['rt60_est_sabine'] = float(rt60_est)
            
            results.append(result)
            runs_added_this_session.add(run_key)  # Mark as added in this session
            
            print(f"  Run {run_key} ({run_id}/{TOTAL_RUNS}): H_src_norm={h_src_norm:.2f}, NodeDist_norm={mean_node_dist_norm:.2f}, "
                  f"C_opt={opt_c:.2f}, RMSE(C*)={min_rmse:.4f}, RMSE(C=1)={rmse_c1:.4f}")
            
            # Incremental save: write results after each run for crash recovery
            _ensure_derived_metrics(result)
            def _sort_key(r):
                if "room_idx" in r and "pair_idx" in r:
                    return (int(r["room_idx"]), int(r["pair_idx"]))
                return (999999, int(r.get("run_id", 999999)))
            results_sorted = sorted(results, key=_sort_key)
            _atomic_json_dump(OUTPUT_FILE, results_sorted)

    # Save Results (auto-upgrade: ensure *all* derived fields exist in the output JSON)
    for r in results:
        _ensure_derived_metrics(r)

    # Keep stable ordering in the JSON (room/pair order) for easy diffs/reading
    def _sort_key(r):
        if "room_idx" in r and "pair_idx" in r:
            return (int(r["room_idx"]), int(r["pair_idx"]))
        return (999999, int(r.get("run_id", 999999)))

    results_sorted = sorted(results, key=_sort_key)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")
    
    # --- Plotting ---
    plot_results(results_sorted)


def _export_npz_per_source(results, room_params_base, receiver_positions, sources, Fs, duration_s, experiment_dir):
    """
    Export NPZ files per source position, compatible with paper_figures_spatial.py and analyze_edc_source_variation.py.
    Includes: RIMPY (reference), SDN-Test1 (c=1), and SDN-C* (optimized c per source/pair).
    """
    from rir_calculators import calculate_sdn_rir
    
    print("\n--- Exporting NPZ files per source ---")
    
    # Group results by source_idx
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        src_idx = r.get("src_idx")
        if src_idx is not None:
            by_source[src_idx].append(r)
    
    # Build room object and parameters for SDN calculations
    w, d, h = room_params_base["width"], room_params_base["depth"], room_params_base["height"]
    room = geometry.Room(w, d, h)
    num_samples = int(Fs * duration_s)
    impulse = geometry.Source.generate_signal('dirac', num_samples)
    room.wallAttenuation = [room_params_base["reflection"]] * 6
    
    # Base parameters dict for SDN calculations
    room_parameters = room_params_base.copy()
    
    for src_idx in sorted(by_source.keys()):
        src_results = by_source[src_idx]
        if not src_results:
            continue
        
        # Get source position
        src_data = src_results[0]
        src_pos = src_data["src_pos"]
        sx, sy, sz = src_pos
        
        # Determine source name
        if src_idx < len(sources) and len(sources[src_idx]) >= 4:
            src_name = sources[src_idx][3]
        else:
            src_name = f"source_{src_idx+1}"
        
        print(f"\n  Source {src_idx+1}/{len(by_source)}: {src_name} @ ({sx:.2f}, {sy:.2f}, {sz:.2f})")
        
        # Prepare RIR storage
        n_rx = len(receiver_positions)
        rimpy_rirs = np.zeros((n_rx, num_samples))
        sdn_c1_rirs = np.zeros((n_rx, num_samples))
        sdn_copt_rirs = np.zeros((n_rx, num_samples))
        
        # Set source
        room.set_source(sx, sy, sz, signal=impulse['signal'], Fs=Fs)
        
        # Per-receiver calculations
        for rx_i, rx_pos in enumerate(receiver_positions):
            rx, ry, rz = rx_pos[:2] + (room_params_base.get("mic z", 1.5),)
            room.set_microphone(rx, ry, rz)
            
            # 1. Load RIMPY RIR (already cached)
            run_key = f"room{src_data['room_idx']+1}_src{src_idx+1}_rx{rx_i:02d}"
            cache_key_geo = f"{w:.2f}x{d:.2f}x{h:.2f}_s{sx:.2f}_{sy:.2f}_{sz:.2f}_r{rx:.2f}_{ry:.2f}_{rz:.2f}_dur{duration_s:.2f}_fs{Fs}"
            hash_str = hashlib.sha1(cache_key_geo.encode()).hexdigest()[:12]
            rimpy_cache_dir = os.path.join(experiment_dir, "rimpy_cache")
            rimpy_rir_path = os.path.join(rimpy_cache_dir, f"rimpy_{hash_str}.npy")
            
            if os.path.exists(rimpy_rir_path):
                rimpy_rir = np.load(rimpy_rir_path)
                rimpy_rirs[rx_i, :len(rimpy_rir)] = rimpy_rir[:num_samples]
            
            # 2. Calculate SDN with c=1 (original)
            current_params = room_parameters.copy()
            current_params.update({'mic x': rx, 'mic y': ry, 'source x': sx, 'source y': sy, 'source z': sz})

            # Build method configs
            method_configs = {
                'RIMPY-neg10': {'calculator': 'rimpy', 'params': {'reflection_sign': -1, 'rand_dist_mic': 0.10},
                                'enabled': True},
                'SDN-Test1': {
                            'enabled': True,
                            'info': "c1 original",
                            'calculator': 'sdn',
                            'flags': {
                                'specular_source_injection': True,
                                'source_weighting': 1,
                            }, 'label': "SDN" },

                'SDN-C*': {'calculator': 'sdn', 'flags': {'source_weighting': 'optimized per receiver'},
                           'enabled': True},
            }

            _, rir_c1, _, _ = calculate_sdn_rir(current_params, "SDN-Test1", room, duration_s, Fs, method_configs['SDN-Test1'])
            sdn_c1_rirs[rx_i, :len(rir_c1)] = rir_c1[:num_samples]
            
            # 3. Calculate SDN with optimized c*
            # c_opt = 1.0
            # for r in src_results:
            #     if r.get("rx_idx") == rx_i:
            #         c_opt = r.get("optimal_c", 1.0)
            #         break
            # config_copt = {'flags': {'source_weighting': c_opt}, 'calculator': 'sdn'}
            # _, rir_copt, _, _ = calculate_sdn_rir(current_params, f"SDN-C{c_opt:.2f}", room, duration_s, Fs, config_copt)
            # sdn_copt_rirs[rx_i, :len(rir_copt)] = rir_copt[:num_samples]
            #
            # print(f"    Receiver {rx_i+1}/{n_rx}: c*={c_opt:.2f}", end='\r')
        
        print(f"    Completed all {n_rx} receivers for {src_name}")
        
        # Compute EDCs
        def compute_edcs_array(rirs_array):
            edcs = []
            for rir in rirs_array:
                edc, _, _ = an.compute_edc(rir, Fs, plot=False)
                edcs.append(edc)
            max_len = max(len(e) for e in edcs)
            padded = [np.pad(e, (0, max_len - len(e)), 'constant', constant_values=e[-1]) for e in edcs]
            return np.array(padded)
        
        rimpy_edcs = compute_edcs_array(rimpy_rirs)
        sdn_c1_edcs = compute_edcs_array(sdn_c1_rirs)
        # sdn_copt_edcs = compute_edcs_array(sdn_copt_rirs)

        
        # Save NPZ
        output_filename = f"{EXPERIMENT_NAME}_{src_name.lower().replace(' ', '_')}.npz"
        output_path = os.path.join(experiment_dir, output_filename)
        
        # Create 'edcs' dictionary for compatibility with analysis scripts
        edcs_dict = {
            'RIMPY-neg10': rimpy_edcs,
            'SDN-Test1': sdn_c1_edcs,
        }
        
        np.savez(
            output_path,
            receiver_positions=np.array(receiver_positions),
            room_params=np.array([room_params_base]),
            source_pos=np.array(src_pos),
            Fs=np.array(Fs),
            duration=np.array(duration_s),
            method_configs=np.array([method_configs]),
            # Backward compatibility: keep flat keys
            rirs_RIMPY_neg10=rimpy_rirs,
            edcs_RIMPY_neg10=rimpy_edcs,
            rirs_SDN_Test1=sdn_c1_rirs,
            edcs_SDN_Test1=sdn_c1_edcs,
            # NEW: Add proper 'edcs' dictionary for analysis scripts
            edcs=edcs_dict,
            # **{f'rirs_SDN_C_optimized': sdn_copt_rirs, f'edcs_SDN_C_optimized': sdn_copt_edcs}
        )
        
        print(f"  Saved: {output_path}")
    
    print(f"\n--- NPZ export complete: {len(by_source)} files created ---")


def run_monte_carlo_fullgrid_sources():
    """
    Fullgrid source-sweep mode:
    - For each random room, create a fixed 4x4 full receiver grid (margin=0.5).
    - Sample many random source positions.
    - For each source, compute reference EDCs for ALL receivers and optimize ONE c over the grid.
    """
    # Import dynamically to avoid hard dependency on one module path (and keep linters happy).
    import importlib
    try:
        generate_full_receiver_grid = importlib.import_module("analysis.spatial_analysis").generate_full_receiver_grid
    except Exception:
        generate_full_receiver_grid = importlib.import_module("spatial_analysis").generate_full_receiver_grid

    print(f"--- Starting Monte Carlo FULLGRID Sources Experiment ---")
    
    # Enable Basis Cache (shared with validate_c_predictor.py)
    enable_basis_disk_cache()
    print("✓ Basis function disk cache enabled (shared across scripts)")
    
    # Still seed rooms (room dimensions) for determinism; sources are deterministic grids.
    np.random.seed(42)

    # Use the global EXPERIMENT_ROOT for consistency
    experiment_dir = EXPERIMENT_ROOT
    RIR_CACHE_DIR = os.path.join(experiment_dir, "rimpy_cache")
    os.makedirs(RIR_CACHE_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(experiment_dir, "results.json")
    
    print(f"Experiment folder: {experiment_dir}")
    print(f"RIR cache: {RIR_CACHE_DIR}")

    _write_experiment_config(
        experiment_dir,
        {
            "mode": MODE,
            "optimize_scope": FULLGRID_OPTIMIZE_SCOPE,
            "num_rooms": int(NUM_ROOMS),
            "fixed_room_from_experiment_configs": bool(FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS),
            "receiver_grid": {
                "n_x": int(RECEIVER_GRID_N_X),
                "n_y": int(RECEIVER_GRID_N_Y),
                "margin": float(RECEIVER_GRID_MARGIN),
            },
            "source_grid": {
                "n_x": int(SOURCE_GRID_N_X),
                "n_y": int(SOURCE_GRID_N_Y),
                "include_corners": bool(SOURCE_GRID_INCLUDE_CORNERS),
                "corner_offset": float(globals().get("SOURCE_GRID_CORNER_OFFSET", 1.5)),
            },
            "duration_policy": "fixed (experiment_configs/fixed_duration_s) OR sabine_adaptive_0.3+1.5*RT60_clipped_1.0..2.5",
            "err_duration_ms": int(ERR_DURATION_MS),
        },
    )

    # Load existing results (keyed by run_key)
    existing_results_map = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
            for r in data:
                # Minimal schema fields for this mode
                if "run_key" in r:
                    existing_results_map[r["run_key"]] = r
            print(f"Loaded {len(existing_results_map)} existing results from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error loading existing results: {e}")

    results = []
    processed_pairs = 0

    def _persist_partial_results() -> None:
        def _sort_key(r):
            return (
                int(r.get("room_idx", 999999)),
                int(r.get("src_idx", 999999)),
                int(r.get("rx_idx", -1)),
            )

        results_sorted = sorted(results, key=_sort_key)
        _atomic_json_dump(OUTPUT_FILE, results_sorted)

    for room_idx in range(NUM_ROOMS):
        # Room selection:
        # - experiment_configs.active_room (fixed, reproducible; preferred for AES-only experiments)
        # - else random room (Monte Carlo)
        active_room = None
        fixed_absorption = None
        fixed_mic_z = None
        fixed_fs = None
        fixed_duration = None

        if FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS:
            try:
                import experiment_configs as expcfg
                active_room = dict(expcfg.active_room)
                fixed_fs = int(expcfg.Fs)
                fixed_duration = float(expcfg.duration)
                fixed_absorption = float(active_room["absorption"])
                fixed_mic_z = float(active_room["mic z"])
                w = float(active_room["width"])
                d = float(active_room["depth"])
                h = float(active_room["height"])
            except Exception as e:
                raise RuntimeError(
                    f"FULLGRID_FIXED_ROOM_FROM_EXPERIMENT_CONFIGS=True but failed to load experiment_configs: {e}"
                )
        else:
            w, d, h = generate_random_room()
        w, d, h = round(w, 2), round(d, 2), round(h, 2)

        # Effective parameters for this room (allow per-room overrides for fixed-room mode)
        Fs_eff = fixed_fs if fixed_fs is not None else FS
        absorption_eff = fixed_absorption if fixed_absorption is not None else ABSORPTION
        mic_z_eff = fixed_mic_z if fixed_mic_z is not None else 1.5

        room_params = {
            "width": w,
            "depth": d,
            "height": h,
            "absorption": absorption_eff,
            "reflection": float(np.sqrt(1 - absorption_eff)),
            "mic z": mic_z_eff,
            "display_name": active_room.get("display_name", f"Room_{room_idx+1}") if active_room else f"Room_{room_idx+1}",
        }

        vol = w * d * h
        char_len = vol ** (1.0 / 3.0)

        # Duration:
        # - fixed-room from experiment_configs: use experiment_configs.duration (no Sabine)
        # - random rooms: Sabine-adaptive duration
        if fixed_duration is not None:
            duration_s = float(fixed_duration)
        else:
            duration_s = choose_duration_for_room(w, d, h, absorption_eff)

        # Fixed receiver grid for this room
        receiver_positions = generate_full_receiver_grid(
            w, d, room_params["mic z"], RECEIVER_GRID_N_X, RECEIVER_GRID_N_Y, RECEIVER_GRID_MARGIN
        )
        rt60_est = estimate_rt60_sabine(w, d, h, absorption_eff)
        print(
            f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}, receivers={len(receiver_positions)}, "
            f"RT60~{rt60_est:.2f}s, duration={duration_s:.2f}s"
        )

        # Deterministic source set for this room (repeatable coverage)
        if SOURCE_GENERATION_SPECS is not None:
            # Custom source generation: combine multiple grids
            all_sources = []
            for spec in SOURCE_GENERATION_SPECS:
                func_name = spec["func"]
                kwargs = {k: v for k, v in spec.items() if k != "func"}
                kwargs["room_params"] = room_params
                
                if func_name == "generate_fixed_source_grid":
                    all_sources.append(generate_fixed_source_grid(**kwargs))
                elif func_name == "generate_fixed_source_grid_simple":
                    all_sources.append(generate_fixed_source_grid_simple(**kwargs))
                else:
                    raise ValueError(f"Unknown source generation function: {func_name}")
            
            # Union of all grids (remove duplicates by rounding coordinates)
            sources_set = set()
            sources = []
            for grid in all_sources:
                for src in grid:
                    # src is (x, y, z, name) tuple
                    key = (round(src[0], 2), round(src[1], 2), round(src[2], 2))
                    if key not in sources_set:
                        sources_set.add(key)
                        sources.append(src)
            print(f"  Sources: {len(sources)} (custom: {len(SOURCE_GENERATION_SPECS)} grids merged, scope={FULLGRID_OPTIMIZE_SCOPE})")
        else:
            # Parametric source generation (fallback)
            sources = generate_fixed_source_grid(
                room_params,
                PADDING,
                n_x=SOURCE_GRID_N_X,
                n_y=SOURCE_GRID_N_Y,
                z_mode="mid",
                include_corners=SOURCE_GRID_INCLUDE_CORNERS,
                corner_offset=SOURCE_GRID_CORNER_OFFSET,
            )
            print(f"  Sources: {len(sources)} (parametric grid, scope={FULLGRID_OPTIMIZE_SCOPE})")
        
        # Save layout visualization at the start of the run
        if room_idx == 0:  # Only save once for the first room
            print(f"  Saving source-receiver grid layout...")
            exp_dir = _resolve_experiment_dir(os.path.join(project_root, "results"))
            os.makedirs(exp_dir, exist_ok=True)
            layout_save_path = os.path.join(exp_dir, f"{EXPERIMENT_NAME}_layout.png")
            print_source_grid(
                sources,
                receiver_positions,
                room_params,
                save=True,
                grid_name=layout_save_path,
            )
            print(f"  Layout saved to: {layout_save_path}")

        for src_idx, src_pos in enumerate(sources):
            run_key_base = f"room{room_idx + 1}_src{src_idx + 1}"

            sx, sy, sz = src_pos
            print(f"  [Source {src_idx+1}/{len(sources)}] src_pos=({sx:.2f}, {sy:.2f}, {sz:.2f})")

            # Source proximity metric
            src_dists = wall_proximity.get_wall_distances(sx, sy, sz, w, d, h)
            h_src = wall_proximity.harmonic_mean(src_dists)
            h_src_norm = float(h_src / char_len)

            # Setup sim room (reused across receivers for normalization and for SDN optimization)
            sim_room = geometry.Room(w, d, h)
            impulse = geometry.Source.generate_signal("dirac", int(Fs_eff * duration_s))
            sim_room.set_source(*src_pos, signal=impulse["signal"], Fs=Fs_eff)
            sim_room.wallAttenuation = [room_params["reflection"]] * 6

            # Compute reference EDCs for each receiver
            ref_edcs = []
            d_sm_norms = []
            for rx_i, rx_pos in enumerate(receiver_positions):
                rx, ry, rz = rx_pos
                # "pair" progress is defined as (source, receiver) evaluations for this mode
                processed_pairs += 1
                print(
                    f"    [Pair {processed_pairs}/{TOTAL_RUNS}] "
                    f"room {room_idx+1}/{NUM_ROOMS}, src {src_idx+1}/{len(sources)}, rx {rx_i+1}/{len(receiver_positions)} "
                    f"src=({sx:.2f},{sy:.2f},{sz:.2f}) rx=({rx:.2f},{ry:.2f},{rz:.2f})"
                )
                d_sm_norm = _euclidean_distance(src_pos, rx_pos) / char_len
                d_sm_norms.append(float(d_sm_norm))

                # Cache RIMPY RIR by geometry (stable across grid changes)
                rir_path, meta_path = _rimpy_cache_paths(
                    RIR_CACHE_DIR,
                    (w, d, h),
                    src_pos,
                    rx_pos,
                    duration_s,
                    Fs_eff,
                    reflection_sign=-1,
                    rand_dist=0.1,
                    absorption=room_params['absorption'],
                )

                if os.path.exists(rir_path) and os.path.exists(meta_path):
                    ref_rir = np.load(rir_path)
                else:
                    current_params = room_params.copy()
                    current_params.update(
                        {
                            "source x": sx,
                            "source y": sy,
                            "source z": sz,
                            "mic x": rx,
                            "mic y": ry,
                            "mic z": rz,
                        }
                    )
                    ref_rir, _ = calculate_rimpy_rir(current_params, duration_s, Fs_eff, reflection_sign=-1, randDist=0.1)
                    np.save(rir_path, ref_rir)
                    try:
                        with open(meta_path, "w") as f:
                            json.dump(
                                {
                                    "room_dims": [float(w), float(d), float(h)],
                                    "src_pos": [float(x) for x in src_pos],
                                    "rx_pos": [float(x) for x in rx_pos],
                                    "duration_s": float(duration_s),
                                    "Fs": int(Fs_eff),
                                    "reflection_sign": -1,
                                    "rand_dist": 0.1,
                                },
                                f,
                                indent=2,
                            )
                    except Exception:
                        # Meta is a convenience for validity checks; cache still works without it.
                        pass

                # Normalize per receiver (consistent with other pipelines)
                sim_room.set_microphone(rx, ry, rz)
                norm_ref = rir_normalisation(ref_rir, sim_room, Fs_eff, normalize_to_first_impulse=True)["single_rir"]
                ref_edc, _, _ = an.compute_edc(norm_ref, Fs_eff)
                ref_edcs.append(ref_edc)

            # Optimization:
            # - per_source: one c* that minimizes mean RMSE across all receivers
            # - per_pair:   one c* per receiver (pairwise), saved as separate rows
            d_sm_norms_arr = np.asarray(d_sm_norms, dtype=float)
            if FULLGRID_OPTIMIZE_SCOPE == "per_source":
                run_key = run_key_base
                if run_key in existing_results_map:
                    results.append(existing_results_map[run_key])
                    _persist_partial_results()
                    continue

                opt_c, min_rmse, _ = find_optimal_c(
                    sim_room,
                    room_params,
                    ref_edcs,
                    receiver_positions,
                    duration_s,
                    Fs_eff,
                    ERR_DURATION_MS,
                    source_name=run_key,
                    bounds=(1.0, 7.0),
                )

                # Compute baseline RMSE at C=1 (reuses cached basis)
                sdn_config_c1 = {
                    'enabled': True,
                    'calculator': 'sdn',
                    'flags': {
                        'specular_source_injection': True,
                        'source_weighting': 1.0,
                    },
                    'cache_label': f'mc_baseline_c1_{run_key}',
                }
                _, rir_sdn_c1, _, _ = calculate_sdn_rir_fast(
                    room_params, "BaselineC1", sim_room, duration_s, Fs_eff, sdn_config_c1
                )
                rir_sdn_c1_norm = rir_normalisation(rir_sdn_c1, sim_room, Fs_eff, normalize_to_first_impulse=True)['single_rir']
                edc_sdn_c1, _, _ = an.compute_edc(rir_sdn_c1_norm, Fs_eff, plot=False)
                
                # Compute RMSE against all receivers (mean)
                rmses_c1 = []
                for ref_edc in ref_edcs:
                    rmse_i = an.compute_RMS(
                        edc_sdn_c1, ref_edc,
                        range=int(ERR_DURATION_MS),
                        Fs=Fs_eff,
                        skip_initial_zeros=True,
                        normalize_by_active_length=True,
                    )
                    rmses_c1.append(rmse_i)
                rmse_c1 = float(np.mean(rmses_c1))

                # Apply outlier correction
                opt_c_original = opt_c
                min_rmse_original = min_rmse
                opt_c, min_rmse, was_outlier_corrected = apply_outlier_rejection(
                    opt_c, min_rmse, rmse_c1
                )

                # Distance summary across the grid (dimensionless)
                d_mean = float(np.mean(d_sm_norms_arr))
                d_min = float(np.min(d_sm_norms_arr))
                d_med = float(np.median(d_sm_norms_arr))

                result = {
                    "run_key": run_key,
                    "mode": "fullgrid_sources",
                    "optimize_scope": "per_source",
                    "room_idx": room_idx,
                    "src_idx": src_idx,
                    "room_dims": (w, d, h),
                    "room_vol": vol,
                    "char_len": char_len,
                    "src_pos": src_pos,
                    "h_src": h_src,
                    "h_src_norm": h_src_norm,
                    "d_sm_norm_mean": d_mean,
                    "d_sm_norm_min": d_min,
                    "d_sm_norm_median": d_med,
                    "duration_s": float(duration_s),
                    "rt60_est_sabine": float(rt60_est),
                    "optimal_c": float(opt_c),
                    "min_rmse": float(min_rmse),
                    "rmse_c1": rmse_c1,
                    "n_receivers": len(receiver_positions),
                    "outlier_corrected": was_outlier_corrected,
                }
                
                # Store original values if outlier was corrected
                if was_outlier_corrected:
                    result['optimal_c_original'] = opt_c_original
                    result['min_rmse_original'] = min_rmse_original

                results.append(result)
                _persist_partial_results()

                print(
                    f"  {run_key}: H_src_norm={h_src_norm:.3f}, "
                    f"d_mean={d_mean:.3f}, c*={opt_c:.2f}, RMSE={min_rmse:.3f}"
                )
            elif FULLGRID_OPTIMIZE_SCOPE == "per_pair":
                for rx_i, rx_pos in enumerate(receiver_positions):
                    run_key = f"{run_key_base}_rx{rx_i:02d}"
                    if run_key in existing_results_map:
                        results.append(existing_results_map[run_key])
                        _persist_partial_results()
                        continue

                    opt_c, min_rmse, _ = find_optimal_c(
                        sim_room,
                        room_params,
                        [ref_edcs[rx_i]],
                        [rx_pos],
                        duration_s,
                        Fs_eff,
                        ERR_DURATION_MS,
                        source_name=run_key,
                        bounds=(1.0, 7.0),
                    )

                    # Compute baseline RMSE at C=1 (reuses cached basis)
                    sdn_config_c1 = {
                        'enabled': True,
                        'calculator': 'sdn',
                        'flags': {
                            'specular_source_injection': True,
                            'source_weighting': 1.0,
                        },
                        'cache_label': f'mc_baseline_c1_{run_key}',
                    }
                    _, rir_sdn_c1, _, _ = calculate_sdn_rir_fast(
                        room_params, "BaselineC1", sim_room, duration_s, Fs_eff, sdn_config_c1
                    )
                    rir_sdn_c1_norm = rir_normalisation(rir_sdn_c1, sim_room, Fs_eff, normalize_to_first_impulse=True)['single_rir']
                    edc_sdn_c1, _, _ = an.compute_edc(rir_sdn_c1_norm, Fs_eff, plot=False)
                    rmse_c1 = float(
                        an.compute_RMS(
                            edc_sdn_c1, ref_edcs[rx_i],
                            range=int(ERR_DURATION_MS),
                            Fs=Fs_eff,
                            skip_initial_zeros=True,
                            normalize_by_active_length=True,
                        )
                    )

                    # Apply outlier correction
                    opt_c_original = opt_c
                    min_rmse_original = min_rmse
                    opt_c, min_rmse, was_outlier_corrected = apply_outlier_rejection(
                        opt_c, min_rmse, rmse_c1
                    )

                    result = {
                        "run_key": run_key,
                        "mode": "fullgrid_sources",
                        "optimize_scope": "per_pair",
                        "room_idx": room_idx,
                        "src_idx": src_idx,
                        "rx_idx": int(rx_i),
                        "room_dims": (w, d, h),
                        "room_vol": vol,
                        "char_len": char_len,
                        "src_pos": src_pos,
                        "rx_pos": rx_pos,
                        "h_src": h_src,
                        "h_src_norm": h_src_norm,
                        "d_sm_norm": float(d_sm_norms_arr[rx_i]),
                        "duration_s": float(duration_s),
                        "rt60_est_sabine": float(rt60_est),
                        "optimal_c": float(opt_c),
                        "min_rmse": float(min_rmse),
                        "rmse_c1": rmse_c1,
                        "n_receivers": 1,
                        "outlier_corrected": was_outlier_corrected,
                    }
                    
                    # Store original values if outlier was corrected
                    if was_outlier_corrected:
                        result['optimal_c_original'] = opt_c_original
                        result['min_rmse_original'] = min_rmse_original
                    
                    results.append(result)
                    _persist_partial_results()


                    print(
                        f"  {run_key}: H_src_norm={h_src_norm:.3f}, "
                        f"d_sm_norm={float(d_sm_norms_arr[rx_i]):.3f}, c*={opt_c:.2f}, RMSE={min_rmse:.3f}"
                    )
            else:
                raise ValueError(f"Unknown FULLGRID_OPTIMIZE_SCOPE: {FULLGRID_OPTIMIZE_SCOPE}")

    # Save
    _persist_partial_results()
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")

    # Export NPZ files per source (compatible with paper_figures_spatial.py / analyze_edc_source_variation.py)
    experiment_dir = _resolve_experiment_dir(os.path.join(project_root, "results"))
    _export_npz_per_source(results, room_params, receiver_positions, sources, Fs_eff, duration_s, experiment_dir)

    # Quick plots for this mode (scope-dependent)
    # Reuse the in-memory list (already persisted). Avoid relying on a local `results_sorted` symbol.
    plot_rows = [r for r in results if r.get("optimize_scope") == FULLGRID_OPTIMIZE_SCOPE]
    if not plot_rows:
        plot_rows = results

    h = np.array([r["h_src_norm"] for r in plot_rows], dtype=float)
    c = np.array([r["optimal_c"] for r in plot_rows], dtype=float)
    rmse = np.array([r["min_rmse"] for r in plot_rows], dtype=float)

    plt.figure(figsize=(14, 5))
    if FULLGRID_OPTIMIZE_SCOPE == "per_source":
        dmean = np.array([r["d_sm_norm_mean"] for r in plot_rows], dtype=float)
        dmin = np.array([r["d_sm_norm_min"] for r in plot_rows], dtype=float)

        plt.subplot(1, 3, 1)
        plt.scatter(h, c, c=rmse, cmap="viridis", alpha=0.75)
        plt.xlabel("H_src_norm")
        plt.ylabel("optimal c* (grid)")
        plt.title("c* vs H_src_norm (color = RMSE)")
        plt.colorbar(label="min RMSE")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.scatter(dmean, c, c=rmse, cmap="viridis", alpha=0.75)
        plt.xlabel("mean d_sm_norm over grid")
        plt.ylabel("optimal c* (grid)")
        plt.title("c* vs mean distance (color = RMSE)")
        plt.colorbar(label="min RMSE")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.scatter(dmin, c, c=rmse, cmap="viridis", alpha=0.75)
        plt.xlabel("min d_sm_norm over grid")
        plt.ylabel("optimal c* (grid)")
        plt.title("c* vs min distance (color = RMSE)")
        plt.colorbar(label="min RMSE")
        plt.grid(True, alpha=0.3)
    else:
        d_sm = np.array([r["d_sm_norm"] for r in plot_rows], dtype=float)

        plt.subplot(1, 3, 1)
        plt.scatter(h, c, c=rmse, cmap="viridis", alpha=0.75)
        plt.xlabel("H_src_norm")
        plt.ylabel("optimal c* (pair)")
        plt.title("c* vs H_src_norm (color = RMSE)")
        plt.colorbar(label="min RMSE")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.scatter(d_sm, c, c=rmse, cmap="viridis", alpha=0.75)
        plt.xlabel("d_sm_norm")
        plt.ylabel("optimal c* (pair)")
        plt.title("c* vs d_sm_norm (color = RMSE)")
        plt.colorbar(label="min RMSE")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.scatter(d_sm, rmse, alpha=0.75)
        plt.xlabel("d_sm_norm")
        plt.ylabel("RMSE")
        plt.title("RMSE vs d_sm_norm")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(experiment_dir, "correlations.png")
    plt.savefig(out, dpi=150)
    print(f"Plots saved to {out}")

def plot_results(results):
    h_srcs_norm = [r['h_src_norm'] for r in results]
    combined_scores = [r['combined_score'] for r in results]
    node_dists_norm = [r['mean_node_dist_norm'] for r in results]
    opt_cs = [r['optimal_c'] for r in results]
    rmses = [r['min_rmse'] for r in results]

    # Robustness: support older cached JSON entries (compute derived metrics if missing)
    for r in results:
        _ensure_derived_metrics(r)

    src_mic_dists_norm = [r.get('src_mic_dist_norm', np.nan) for r in results]
    h_minus_d_norm = [r.get('h_minus_d_norm', np.nan) for r in results]
    h_over_d_norm = [r.get('h_over_d_norm', np.nan) for r in results]

    # Node-distance variants (all nodes / harmonic mean)
    mean_node_dist_all_norm = [r.get('mean_node_dist_all_norm', np.nan) for r in results]
    hmean_node_dist_norm = [r.get('hmean_node_dist_norm', np.nan) for r in results]

    # Convert to numpy for stats
    h_srcs_norm_np = np.asarray(h_srcs_norm, dtype=float)
    d_sm_norm_np = np.asarray(src_mic_dists_norm, dtype=float)
    opt_cs_np = np.asarray(opt_cs, dtype=float)
    
    plt.figure(figsize=(18, 10))
    
    # 1. Normalized Harmonic Mean Source vs Opt C
    plt.subplot(2, 3, 1)
    plt.scatter(h_srcs_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('Normalized H_src (H_src / V^(1/3)) [Lower=Corner]')
    plt.ylabel('Optimal C')
    plt.title('Normalized Source Proximity vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    # 2. Combined Score vs Opt C
    plt.subplot(2, 3, 2)
    plt.scatter(combined_scores, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('Combined Proximity Score (1/Hs + 1/Hr)')
    plt.ylabel('Optimal C')
    plt.title('Total Geometric Confinement vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    # 3. Normalized Node Distance vs Opt C
    plt.subplot(2, 3, 3)
    plt.scatter(node_dists_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('Normalized Mean 3-Nearest Node Dist (Dist / V^(1/3))')
    plt.ylabel('Optimal C')
    plt.title('Normalized Image Source Density vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)

    # 4. H_src_norm vs Opt C, color-coded by normalized src–mic distance
    plt.subplot(2, 3, 4)
    sc = plt.scatter(h_srcs_norm, opt_cs, alpha=0.7, c=src_mic_dists_norm, cmap='plasma')
    plt.xlabel('Normalized H_src (H_src / V^(1/3))')
    plt.ylabel('Optimal C')
    plt.title('H_src_norm vs Optimal C (Color = d_sm_norm)')
    plt.colorbar(sc, label='d_sm_norm = ||x_s - x_m|| / V^(1/3)')
    plt.grid(True)

    # 5. Composite metric: H + d (dimensionless)
    plt.subplot(2, 3, 5)
    plt.scatter(h_minus_d_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('H_src_norm - d_sm_norm')
    plt.ylabel('Optimal C')
    plt.title('Composite (H - d) vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)

    # 6. Composite metric: H / d (dimensionless)
    plt.subplot(2, 3, 6)
    plt.scatter(h_over_d_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('H_src_norm / (d_sm_norm + ε)')
    plt.ylabel('Optimal C')
    plt.title('Composite (H / d) vs Optimal C')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    _corr_path = os.path.join(EXPERIMENT_ROOT, "monte_carlo_correlations_norm.png")
    plt.savefig(_corr_path)
    print(f"Plots saved to {_corr_path}")

    # ------------------------------------------------------------------
    # Extra: compare node-distance metrics (nearest-3 vs all vs harmonic)
    # ------------------------------------------------------------------
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(node_dists_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('mean(3 nearest nodes) / V^(1/3)')
    plt.ylabel('Optimal C')
    plt.title('Node Dist Metric: mean-3')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.scatter(mean_node_dist_all_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('mean(all nodes) / V^(1/3)')
    plt.ylabel('Optimal C')
    plt.title('Node Dist Metric: mean-all')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.scatter(hmean_node_dist_norm, opt_cs, alpha=0.7, c=rmses, cmap='viridis')
    plt.xlabel('hmean(all nodes) / V^(1/3)')
    plt.ylabel('Optimal C')
    plt.title('Node Dist Metric: harmonic-mean')
    plt.colorbar(label='Min RMSE')
    plt.grid(True)

    plt.tight_layout()
    node_plot_path = os.path.join(EXPERIMENT_ROOT, "monte_carlo_node_distance_metrics.png")
    plt.savefig(node_plot_path, dpi=150, bbox_inches='tight')
    print(f"Node-distance comparison plot saved to {node_plot_path}")

    # ------------------------------------------------------------------
    # Extra: lambda sweep for metric m_lambda = H_src_norm - lambda * d_sm_norm
    # This matches hypothesis: larger d_sm_norm -> lower optimal C.
    # ------------------------------------------------------------------
    # Auto lambda 1: variance matching so terms have comparable spread.
    # (If std(d) is large, lambda shrinks, and vice versa.)
    valid = np.isfinite(h_srcs_norm_np) & np.isfinite(d_sm_norm_np) & np.isfinite(opt_cs_np)
    h_v = h_srcs_norm_np[valid]
    d_v = d_sm_norm_np[valid]
    c_v = opt_cs_np[valid]

    std_h = float(np.std(h_v))
    std_d = float(np.std(d_v))
    lambda_var = (std_h / std_d) if std_d > 1e-12 else 1.0

    # Auto lambda 2: 2D linear regression c ≈ a + b*h + g*d (least squares),
    # then collapse to 1D metric h - lambda*d where lambda = -g/b.
    # This tells you how many "d units" compensate one "h unit" in the fitted plane.
    X = np.column_stack([np.ones_like(h_v), h_v, d_v])  # [1, h, d]
    try:
        a_hat, b_hat, g_hat = np.linalg.lstsq(X, c_v, rcond=None)[0]
        lambda_reg = (-g_hat / b_hat) if abs(b_hat) > 1e-12 else lambda_var
    except Exception:
        lambda_reg = lambda_var

    print("\n--- Lambda suggestions for metric: m = H_src_norm - λ * d_sm_norm ---")
    print(f"std(H_src_norm)={std_h:.4f}, std(d_sm_norm)={std_d:.4f}")
    print(f"λ_var (variance-match) = {lambda_var:.3f}")
    print(f"λ_reg (2D fit collapse) = {lambda_reg:.3f}")

    # Sweep around suggested lambdas + a few simple baselines
    lambdas = [
        0.0,
        0.5,
        1.0,
        2.0,
        float(lambda_var),
        float(lambda_reg),
    ]
    # De-duplicate while preserving order
    lambdas_unique = []
    for lam in lambdas:
        lam_r = float(np.round(lam, 3))
        if lam_r not in lambdas_unique:
            lambdas_unique.append(lam_r)

    # Make a small multi-panel plot for quick visual inspection
    n = len(lambdas_unique)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(6 * cols, 4.5 * rows))
    for i, lam in enumerate(lambdas_unique, start=1):
        m = h_v - lam * d_v
        ax = plt.subplot(rows, cols, i)
        ax.scatter(m, c_v, alpha=0.7, c=np.asarray(rmses)[valid], cmap='viridis')
        ax.set_xlabel(f'H_src_norm - {lam:.3f}·d_sm_norm')
        ax.set_ylabel('Optimal C')
        # Pearson r as a quick "linearity" indicator (not a full model score)
        r = np.corrcoef(m, c_v)[0, 1] if len(m) > 2 else np.nan
        ax.set_title(f'λ={lam:.3f}, r={r:.3f}')
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='Min RMSE')
    plt.tight_layout()
    out_path = os.path.join(EXPERIMENT_ROOT, "monte_carlo_lambda_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Lambda sweep plot saved to {out_path}")

if __name__ == "__main__":
    # Ensure results dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if MODE == "pair_single_mic":
        run_monte_carlo()
    elif MODE == "fullgrid_sources":
        run_monte_carlo_fullgrid_sources()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
