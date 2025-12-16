
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.append(project_root)

import geometry
from analysis import analysis as an
from deprecated import wall_proximity
from optimisation_singleC import find_optimal_c
from rir_calculators import rir_normalisation, calculate_rimpy_rir
from analysis.spatial_analysis import generate_fixed_source_grid

# --- Configuration ---
#
# MODE OPTIONS
# ------------
# - "pair_single_mic": legacy Monte Carlo (one random mic per source), optimizes c for that single receiver.
# - "fullgrid_sources": fixed full receiver grid per room, deterministic source grid per room.
#    - If FULLGRID_OPTIMIZE_SCOPE="per_source": optimizes ONE c per source over the whole receiver grid.
#    - If FULLGRID_OPTIMIZE_SCOPE="per_pair": optimizes ONE c per (source, receiver) pair.
MODE = "fullgrid_sources"

# Fullgrid optimization scope control
FULLGRID_OPTIMIZE_SCOPE = "per_pair"  # options: "per_source", "per_pair"

NUM_ROOMS = 5
PAIRS_PER_ROOM = 15  # legacy name used by schema upgrader; ignored in fullgrid_sources mode

# Receiver grid (MODE="fullgrid_sources")
RECEIVER_GRID_N_X = 4
RECEIVER_GRID_N_Y = 4
RECEIVER_GRID_MARGIN = 0.5

# Source grid (MODE="fullgrid_sources")
SOURCE_GRID_N_X = 3
SOURCE_GRID_N_Y = 5
SOURCE_GRID_INCLUDE_CORNERS = True
SOURCE_GRID_CORNER_OFFSET = 2.0
MAX_SOURCES_PER_ROOM = None  # e.g. 9 or 12 for quick pilots; None = use all generated sources

_est_sources_per_room = (SOURCE_GRID_N_X * SOURCE_GRID_N_Y) + (4 if SOURCE_GRID_INCLUDE_CORNERS else 0)
_est_sources_per_room = min(_est_sources_per_room, MAX_SOURCES_PER_ROOM) if MAX_SOURCES_PER_ROOM else _est_sources_per_room
_est_receivers_per_room = RECEIVER_GRID_N_X * RECEIVER_GRID_N_Y
TOTAL_RUNS = NUM_ROOMS * (
    PAIRS_PER_ROOM
    if MODE == "pair_single_mic"
    else (_est_sources_per_room if FULLGRID_OPTIMIZE_SCOPE == "per_source" else (_est_sources_per_room * _est_receivers_per_room))
)

# Room Dimensions Ranges (meters)
# Mode "B" (close to AES / Journal / WASPAA):
# - AES:     (9, 7, 4)
# - WASPAA:  (6, 7, 4)
# - Journal: (3.2, 4, 2.7)
# We pick tight-ish ranges that cover these rooms without going overly broad.
WIDTH_RANGE = (3.0, 9.5)
DEPTH_RANGE = (4.0, 7.5)
HEIGHT_RANGE = (2.6, 4.2)

# Constants
FS = 44100
# NOTE: For EDC-based metrics, RIR duration matters even for early windows because
# EDC uses backward integration to the end of the signal. In MODE="fullgrid_sources"
# we therefore use an adaptive duration based on a Sabine RT60 estimate.
DURATION = 1.0  # seconds (legacy default for MODE="pair_single_mic")
ERR_DURATION_MS = 50
ABSORPTION = 0.2 # Fixed absorption for consistency
PADDING = 0.5 # Minimum distance from walls

OUTPUT_FILE = os.path.join(project_root, "results", "monte_carlo_proximity_results.json")

# Separate output file for the grid-optimized mode (keeps datasets clean)
OUTPUT_FILE_FULLGRID = os.path.join(project_root, "results", "monte_carlo_fullgrid_sources_results.json")

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
    # Heuristic: capture well past RT60 to stabilize EDC shape at early times.
    # Keep within practical bounds for Monte Carlo.
    dur = 0.3 + 1.5 * rt60
    return float(np.clip(dur, 1.0, 2.5))

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

def generate_random_position(w, d, h, padding, mode='uniform'):
    if mode == 'corner_biased' and np.random.rand() < 0.5:
        # 50% chance to force a position deep into a random corner
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
    
    # 0. Set Seed for Deterministic "Random" Rooms
    np.random.seed(42)
    
    # Cache Directory
    RIR_CACHE_DIR = os.path.join(project_root, "results", "monte_carlo_rirs")
    os.makedirs(RIR_CACHE_DIR, exist_ok=True)
    
    # Load Existing Results (keyed by run_key for readability/stability)
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
    
    start_time = time.time()
    
    for room_idx in range(NUM_ROOMS):
        # 1. Generate Random Room
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
        
        print(f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}")
        
        # 2. Generate Pairs
        for pair_idx in range(PAIRS_PER_ROOM):
            run_id = room_idx * PAIRS_PER_ROOM + pair_idx + 1  # legacy numeric id
            run_key = f"room{room_idx + 1}_pair{pair_idx + 1}"
            
            # --- RESUME LOGIC ---
            if run_key in existing_results_map:
                print(f"  Run {run_key} ({run_id}/{TOTAL_RUNS}): SKIPPING (Result exists)")
                existing = existing_results_map[run_key]
                # Ensure any newly added derived fields exist before re-saving
                _ensure_derived_metrics(existing)
                results.append(existing)
                
                # Actually, since we set Seed=42 at start, if we skip, we MUST consume the RNG calls
                # OR rely on the fact that we restart the script every time?
                # If we restart script, Seed=42 ensures sequence.
                # If we skip result, we still need to iterate RNG?
                # YES. If we don't call generate_random_position, the Next Room's dimensions will be wrong (shifted).
                _ = generate_random_position(w, d, h, PADDING, mode='corner_biased')
                _ = generate_random_position(w, d, h, PADDING, mode='uniform')
                continue
            
            src_pos = generate_random_position(w, d, h, PADDING, mode='corner_biased')
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
            impulse = geometry.Source.generate_signal('dirac', int(FS * DURATION))
            sim_room.set_source(*src_pos, signal=impulse['signal'], Fs=FS)
            sim_room.set_microphone(*mic_pos)
            sim_room.wallAttenuation = [room_params['reflection']] * 6
            
            # 2. Calculate Reference & Normalize
            # Check Cache
            rir_filename = f"rimpy_{run_key}.npy"
            rir_path = os.path.join(RIR_CACHE_DIR, rir_filename)
            
            loaded_valid_rir = False
            norm_ref = None
            
            if os.path.exists(rir_path):
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
                ref_rir, _ = calculate_rimpy_rir(
                    room_params, DURATION, FS, 
                    reflection_sign=-1, randDist=0.1
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
                DURATION, FS, ERR_DURATION_MS, source_name=f"R{room_idx}_P{pair_idx}",
                bounds=(1, 7.0)
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
                'optimal_c': opt_c,
                'min_rmse': min_rmse
            }
            results.append(result)
            
            print(f"  Run {run_key} ({run_id}/{TOTAL_RUNS}): H_src_norm={h_src_norm:.2f}, NodeDist_norm={mean_node_dist_norm:.2f}, C_opt={opt_c:.2f}, RMSE={min_rmse:.4f}")

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
    # Still seed rooms (room dimensions) for determinism; sources are deterministic grids.
    np.random.seed(42)

    OUTPUT_FILE = OUTPUT_FILE_FULLGRID
    RIR_CACHE_DIR = os.path.join(project_root, "results", "monte_carlo_fullgrid_rirs")
    os.makedirs(RIR_CACHE_DIR, exist_ok=True)

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

    for room_idx in range(NUM_ROOMS):
        w, d, h = generate_random_room()
        w, d, h = round(w, 2), round(d, 2), round(h, 2)

        room_params = {
            "width": w,
            "depth": d,
            "height": h,
            "absorption": ABSORPTION,
            "reflection": float(np.sqrt(1 - ABSORPTION)),
            "mic z": 1.5,
        }

        vol = w * d * h
        char_len = vol ** (1.0 / 3.0)

        # Adaptive duration (EDC tail sensitivity)
        duration_s = choose_duration_for_room(w, d, h, ABSORPTION)

        # Fixed receiver grid for this room
        receiver_positions = generate_full_receiver_grid(
            w, d, room_params["mic z"], RECEIVER_GRID_N_X, RECEIVER_GRID_N_Y, RECEIVER_GRID_MARGIN
        )
        rt60_est = estimate_rt60_sabine(w, d, h, ABSORPTION)
        print(
            f"\n[Room {room_idx+1}/{NUM_ROOMS}] Dims: {w}x{d}x{h}, receivers={len(receiver_positions)}, "
            f"RT60~{rt60_est:.2f}s, duration={duration_s:.2f}s"
        )

        # Deterministic source set for this room (repeatable coverage)
        sources = generate_fixed_source_grid(
            room_params,
            PADDING,
            n_x=SOURCE_GRID_N_X,
            n_y=SOURCE_GRID_N_Y,
            z_mode="mid",
            include_corners=SOURCE_GRID_INCLUDE_CORNERS,
            corner_offset=SOURCE_GRID_CORNER_OFFSET,
        )
        if MAX_SOURCES_PER_ROOM is not None:
            sources = sources[: int(MAX_SOURCES_PER_ROOM)]
        print(f"  Sources: {len(sources)} (fixed grid, scope={FULLGRID_OPTIMIZE_SCOPE})")

        for src_idx, src_pos in enumerate(sources):
            run_key_base = f"room{room_idx + 1}_src{src_idx + 1}"

            sx, sy, sz = src_pos

            # Source proximity metric
            src_dists = wall_proximity.get_wall_distances(sx, sy, sz, w, d, h)
            h_src = wall_proximity.harmonic_mean(src_dists)
            h_src_norm = float(h_src / char_len)

            # Setup sim room (reused across receivers for normalization and for SDN optimization)
            sim_room = geometry.Room(w, d, h)
            impulse = geometry.Source.generate_signal("dirac", int(FS * duration_s))
            sim_room.set_source(*src_pos, signal=impulse["signal"], Fs=FS)
            sim_room.wallAttenuation = [room_params["reflection"]] * 6

            # Compute reference EDCs for each receiver
            ref_edcs = []
            d_sm_norms = []
            for rx_i, rx_pos in enumerate(receiver_positions):
                rx, ry, rz = rx_pos
                d_sm_norm = _euclidean_distance(src_pos, rx_pos) / char_len
                d_sm_norms.append(float(d_sm_norm))

                # Cache rimpy RIR per (run_key, receiver)
                rir_filename = f"rimpy_{run_key_base}_rx{rx_i:02d}.npy"
                rir_path = os.path.join(RIR_CACHE_DIR, rir_filename)

                if os.path.exists(rir_path):
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
                    ref_rir, _ = calculate_rimpy_rir(current_params, duration_s, FS, reflection_sign=-1, randDist=0.1)
                    np.save(rir_path, ref_rir)

                # Normalize per receiver (consistent with other pipelines)
                sim_room.set_microphone(rx, ry, rz)
                norm_ref = rir_normalisation(ref_rir, sim_room, FS, normalize_to_first_impulse=True)["single_rir"]
                ref_edc, _, _ = an.compute_edc(norm_ref, FS)
                ref_edcs.append(ref_edc)

            # Optimization:
            # - per_source: one c* that minimizes mean RMSE across all receivers
            # - per_pair:   one c* per receiver (pairwise), saved as separate rows
            d_sm_norms_arr = np.asarray(d_sm_norms, dtype=float)
            if FULLGRID_OPTIMIZE_SCOPE == "per_source":
                run_key = run_key_base
                if run_key in existing_results_map:
                    results.append(existing_results_map[run_key])
                    continue

                opt_c, min_rmse, _ = find_optimal_c(
                    sim_room,
                    room_params,
                    ref_edcs,
                    receiver_positions,
                    duration_s,
                    FS,
                    ERR_DURATION_MS,
                    source_name=run_key,
                    bounds=(1.0, 7.0),
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
                    "n_receivers": len(receiver_positions),
                }
                results.append(result)

                print(
                    f"  {run_key}: H_src_norm={h_src_norm:.3f}, "
                    f"d_mean={d_mean:.3f}, c*={opt_c:.2f}, RMSE={min_rmse:.3f}"
                )
            elif FULLGRID_OPTIMIZE_SCOPE == "per_pair":
                for rx_i, rx_pos in enumerate(receiver_positions):
                    run_key = f"{run_key_base}_rx{rx_i:02d}"
                    if run_key in existing_results_map:
                        results.append(existing_results_map[run_key])
                        continue

                    opt_c, min_rmse, _ = find_optimal_c(
                        sim_room,
                        room_params,
                        [ref_edcs[rx_i]],
                        [rx_pos],
                        duration_s,
                        FS,
                        ERR_DURATION_MS,
                        source_name=run_key,
                        bounds=(1.0, 7.0),
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
                        "n_receivers": 1,
                    }
                    results.append(result)

                    print(
                        f"  {run_key}: H_src_norm={h_src_norm:.3f}, "
                        f"d_sm_norm={float(d_sm_norms_arr[rx_i]):.3f}, c*={opt_c:.2f}, RMSE={min_rmse:.3f}"
                    )
            else:
                raise ValueError(f"Unknown FULLGRID_OPTIMIZE_SCOPE: {FULLGRID_OPTIMIZE_SCOPE}")

    # Save
    def _sort_key(r):
        return (
            int(r.get("room_idx", 999999)),
            int(r.get("src_idx", 999999)),
            int(r.get("rx_idx", -1)),
        )

    results_sorted = sorted(results, key=_sort_key)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")

    # Quick plots for this mode (scope-dependent)
    plot_rows = [r for r in results_sorted if r.get("optimize_scope") == FULLGRID_OPTIMIZE_SCOPE]
    if not plot_rows:
        plot_rows = results_sorted

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
    out = os.path.join(project_root, "results", "monte_carlo_fullgrid_sources_correlations.png")
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
    plt.savefig(os.path.join(project_root, "results", "monte_carlo_correlations_norm.png"))
    print("Plots saved to results/monte_carlo_correlations_norm.png")

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
    node_plot_path = os.path.join(project_root, "results", "monte_carlo_node_distance_metrics.png")
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
    out_path = os.path.join(project_root, "results", "monte_carlo_lambda_sweep.png")
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
