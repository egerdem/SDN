


"""
AES FULLGRID Pairwise C Analysis
--------------------------------
Goal: For the AES room FULLGRID data that already exists in results/paper_data,
analyze per-(source, receiver) behavior:

1) "Original SDN error" at c=1 (method key: SDN-Test1) vs geometry metric
2) Per-pair optimized c* (single receiver) vs the same geometry metric

This is an AES-only diagnostic step to validate that (H_src_norm, d_sm_norm) (or a
simple combination) explains systematic SDN error and/or optimal c.

Outputs
-------
- results/aes_pairwise_c_analysis.csv
- results/aes_fullgrid_rmse_c1_vs_distance_colored_by_h.png
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.append(_project_root)

import geometry
from analysis.plotting_utils import load_data
from analysis import analysis as an
from rir_calculators import calculate_sdn_rir_fast, rir_normalisation, enable_basis_disk_cache
from deprecated import wall_proximity
from research.optimisation_singleC import optimize_minimize_scalar, BOUNDS as DEFAULT_C_BOUNDS
from research.data_config import DataConfig


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# === DATA SOURCE CONFIGURATION ===
# Choose one mode: "legacy" or "experiment"

# Option 1: Legacy mode (flat file structure)
# config = DataConfig(mode="legacy")

# Option 2: Experiment mode (uncomment to use)
# config = DataConfig(
#     mode="experiment",
#     experiment_name="aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair",
    # experiment_name="aes_fullgrid_8src_diagonal",
# )

# Option 3: Legacy mode with custom file selection
config = DataConfig(
    mode="legacy",
    legacy_files=[
        "aes_FULLGRID_center_source.npz",
        "aes_FULLGRID_top_middle_source.npz",
        "aes_FULLGRID_upper_right_source.npz",
        "aes_FULLGRID_lower_left_source.npz",  # Use lower_left for testing
        # "aes_FULLGRID_corner_sourcev3.npz",

    # "journal_FULLGRID_center_source.npz",
    #     "journal_FULLGRID_top_middle_source.npz",
    #     "journal_FULLGRID_upper_right_source.npz",
    #     "journal_FULLGRID_lower_left_source.npz",
    ]
)

# Get paths from config
DATA_DIR = config.get_data_dir()
FILES_TO_PROCESS = config.get_files()

# Output directory: use experiment folder if in experiment mode, otherwise use results/
if config.mode == "experiment":
    OUTPUT_DIR = DATA_DIR  # Save to experiment directory
else:
    OUTPUT_DIR = os.path.join(_project_root, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_METHOD = "RIMPY-neg10"
ORIGINAL_SDN_METHOD = "SDN-Test1"  # c=1 original (saved in NPZ)

# RMSE window (ms)
ERR_DURATION_MS = 50

# Per-pair optimization bounds for c
# Keep consistent with research/optimisation_singleC.py
# (You can override if you want, but defaults should match the canonical optimizer.)
C_BOUNDS = DEFAULT_C_BOUNDS

# Per-pair optimization modes
RUN_SOURCE_ONLY = True   # Optimize c only
RUN_MIC_ONLY = True      # Optimize m only
RUN_SOURCE_MIC = True    # Optimize c, then m (two-stage)

# Outlier rejection
OUTLIER_REJECTION = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _euclidean_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _build_fast_sdn_cfg(c_value: float, cache_label: str) -> dict:
    # calculate_sdn_rir_fast expects flags['source_weighting']
    return {
        "enabled": True,
        "use_fast_method": True,
        "calculator": "sdn",
        "label": "SDN",
        "cache_label": cache_label,
        "flags": {
            "specular_source_injection": True,
            "source_weighting": float(c_value),
        },
    }


def _optimize_c_for_single_receiver(
    room: geometry.Room,
    room_parameters: dict,
    rx_pos: Tuple[float, float, float],
    ref_edc: np.ndarray,
    Fs: int,
    duration: float,
    err_duration_ms: int,
    cache_label_prefix: str,
) -> Tuple[float, float]:
    """
    Returns (c_opt, rmse_at_c_opt) for a single receiver.
    Uses FAST SDN with caching; basis functions are cached per (src,mic).
    """
    rx, ry, rz = rx_pos
    room.set_microphone(rx, ry, rz)

    def obj(c_val: float) -> float:
        cfg = _build_fast_sdn_cfg(c_val, cache_label=f"{cache_label_prefix}")
        _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "PairOpt", room, duration, Fs, cfg)
        rir_sdn_norm = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
        edc_sdn, _, _ = an.compute_edc(rir_sdn_norm, Fs, plot=False)
        return float(
            an.compute_RMS(
                edc_sdn,
                ref_edc,
                range=int(err_duration_ms),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True,
            )
        )

    # Use the same bounded optimizer helper as research/optimisation_singleC.py
    res = optimize_minimize_scalar(obj, C_BOUNDS)
    c_opt = float(res.x)
    rmse_opt = float(res.fun)

    if OUTLIER_REJECTION:
        # If optimizer hits a bound, it typically indicates a monotonic objective in [bounds].
        # In that case, prefer c=1 if it is actually better (or equal within tolerance).
        lower, upper = float(C_BOUNDS[0]), float(C_BOUNDS[1])
        eps = 5e-3
        hit_upper = abs(c_opt - upper) <= eps
        hit_lower = abs(c_opt - lower) <= eps

        if hit_upper or hit_lower:
            rmse_c1 = obj(1.0)
            if rmse_c1 <= rmse_opt + 1e-6:
                return 1.0, float(rmse_c1)

    return c_opt, rmse_opt


def _optimize_m_for_single_receiver(
    room: geometry.Room,
    room_parameters: dict,
    rx_pos: Tuple[float, float, float],
    ref_edc: np.ndarray,
    Fs: int,
    duration: float,
    err_duration_ms: int,
    cache_label_prefix: str,
    fixed_c: float = 1.0,
) -> Tuple[float, float]:
    """
    Returns (m_opt, rmse_at_m_opt) for a single receiver.
    Uses MIC-SIDE FAST method with caching.
    """
    rx, ry, rz = rx_pos
    room.set_microphone(rx, ry, rz)

    def obj(m_val: float) -> float:
        cfg = {
            "enabled": True,
            "use_fast_mic_method": True,
            "calculator": "sdn",
            "label": "SDN",
            "cache_label": cache_label_prefix,
            "flags": {
                "specular_source_injection": True,
                "source_weighting": float(fixed_c),
                "specular_mic_pickup": True,
                "mic_weighting": float(m_val),
            },
        }
        _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "PairOpt", room, duration, Fs, cfg)
        rir_sdn_norm = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
        edc_sdn, _, _ = an.compute_edc(rir_sdn_norm, Fs, plot=False)
        return float(
            an.compute_RMS(
                edc_sdn,
                ref_edc,
                range=int(err_duration_ms),
                Fs=Fs,
                skip_initial_zeros=True,
                normalize_by_active_length=True,
            )
        )

    res = optimize_minimize_scalar(obj, C_BOUNDS)
    m_opt = float(res.x)
    rmse_opt = float(res.fun)

    if OUTLIER_REJECTION:
        lower, upper = float(C_BOUNDS[0]), float(C_BOUNDS[1])
        eps = 5e-3
        hit_upper = abs(m_opt - upper) <= eps
        hit_lower = abs(m_opt - lower) <= eps

        if hit_upper or hit_lower:
            rmse_m1 = obj(1.0)
            if rmse_m1 <= rmse_opt + 1e-6:
                return 1.0, float(rmse_m1)

    return m_opt, rmse_opt


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # Set random seed for full reproducibility (though optimization should be deterministic)
    np.random.seed(42)
    
    enable_basis_disk_cache()

    rows: List[Dict[str, object]] = []

    for filename in FILES_TO_PROCESS:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[SKIP] Missing file: {path}")
            continue

        sim = load_data(path)
        Fs = int(sim["Fs"])
        duration = float(sim["duration"])
        room_params = sim["room_params"]
        receiver_positions = sim["receiver_positions"]

        # source_pos stored as np array in file (shape (3,))
        src_pos = tuple(float(x) for x in np.asarray(sim["source_pos"]).ravel().tolist())
        w, d_room, h_room = float(room_params["width"]), float(room_params["depth"]), float(room_params["height"])
        volume = w * d_room * h_room
        char_len = volume ** (1.0 / 3.0)

        # H_src_norm (source-only)
        sx, sy, sz = src_pos
        src_dists = wall_proximity.get_wall_distances(sx, sy, sz, w, d_room, h_room)
        h_src = wall_proximity.harmonic_mean(src_dists)
        h_src_norm = float(h_src / char_len)

        # Load EDC arrays
        if REFERENCE_METHOD not in sim["edcs"]:
            raise KeyError(f"{filename}: missing reference EDC '{REFERENCE_METHOD}'")
        if ORIGINAL_SDN_METHOD not in sim["edcs"]:
            raise KeyError(f"{filename}: missing SDN EDC '{ORIGINAL_SDN_METHOD}'")

        ref_edcs = sim["edcs"][REFERENCE_METHOD]
        sdn_edcs_orig = sim["edcs"][ORIGINAL_SDN_METHOD]

        # Build room for optimization (reuse per source; mic varies)
        room = geometry.Room(w, d_room, h_room)
        impulse = geometry.Source.generate_signal("dirac", int(Fs * duration))
        room.set_source(*src_pos, signal=impulse["signal"], Fs=Fs)
        room.wallAttenuation = [np.sqrt(1.0 - room_params["absorption"])] * 6

        room_parameters = dict(room_params)
        room_parameters["reflection"] = np.sqrt(1.0 - room_params["absorption"])

        # Determine a readable source tag from filename using config
        source_tag = config.extract_source_tag(filename)

        print(f"\n=== {filename} ===")
        print(f"source_tag={source_tag}, src_pos={src_pos}, H_src_norm={h_src_norm:.4f}, receivers={len(receiver_positions)}")
        
        # Verify reference is deterministic (print hash for first receiver)
        ref_hash = hash(ref_edcs[0].tobytes())
        print(f"Reference EDC hash (rx0): {ref_hash} [should be identical across runs]")

        for rx_idx, rx_pos in enumerate(receiver_positions):
            rx_pos = tuple(float(x) for x in np.asarray(rx_pos).ravel().tolist())
            d_sm = _euclidean_distance(src_pos, rx_pos)
            d_sm_norm = float(d_sm / char_len)

            # RMSE at original SDN (already saved)
            rmse_orig = float(
                an.compute_RMS(
                    sdn_edcs_orig[rx_idx],
                    ref_edcs[rx_idx],
                    range=int(ERR_DURATION_MS),
                    Fs=Fs,
                    skip_initial_zeros=True,
                    normalize_by_active_length=True,
                )
            )

            # Per-pair optimization (three modes)
            cache_label = f"aes_fullgrid_{source_tag}_rx{rx_idx:02d}"

            # Mode 1: Source-only optimization
            c_opt_src = float("nan")
            rmse_src = float("nan")
            if RUN_SOURCE_ONLY:
                c_opt_src, rmse_src = _optimize_c_for_single_receiver(
                    room=room,
                    room_parameters=room_parameters,
                    rx_pos=rx_pos,
                    ref_edc=ref_edcs[rx_idx],
                    Fs=Fs,
                    duration=duration,
                    err_duration_ms=ERR_DURATION_MS,
                    cache_label_prefix=cache_label,
                )

            # Mode 2: Mic-only optimization (c=1.0 fixed)
            m_opt_mic = float("nan")
            rmse_mic = float("nan")
            if RUN_MIC_ONLY:
                m_opt_mic, rmse_mic = _optimize_m_for_single_receiver(
                    room=room,
                    room_parameters=room_parameters,
                    rx_pos=rx_pos,
                    ref_edc=ref_edcs[rx_idx],
                    Fs=Fs,
                    duration=duration,
                    err_duration_ms=ERR_DURATION_MS,
                    cache_label_prefix=cache_label,
                    fixed_c=1.0,
                )

            # Mode 3: Two-stage optimization (c then m)
            c_opt_both = float("nan")
            m_opt_both = float("nan")
            rmse_both_stage1 = float("nan")
            rmse_both_stage2 = float("nan")
            if RUN_SOURCE_MIC:
                # Stage 1: optimize c
                c_opt_both, rmse_both_stage1 = _optimize_c_for_single_receiver(
                    room=room,
                    room_parameters=room_parameters,
                    rx_pos=rx_pos,
                    ref_edc=ref_edcs[rx_idx],
                    Fs=Fs,
                    duration=duration,
                    err_duration_ms=ERR_DURATION_MS,
                    cache_label_prefix=cache_label,
                )
                # Stage 2: optimize m with c fixed
                m_opt_both, rmse_both_stage2 = _optimize_m_for_single_receiver(
                    room=room,
                    room_parameters=room_parameters,
                    rx_pos=rx_pos,
                    ref_edc=ref_edcs[rx_idx],
                    Fs=Fs,
                    duration=duration,
                    err_duration_ms=ERR_DURATION_MS,
                    cache_label_prefix=cache_label,
                    fixed_c=c_opt_both,
                )

            rows.append(
                {
                    "file": filename,
                    "source_tag": source_tag,
                    "src_x": sx,
                    "src_y": sy,
                    "src_z": sz,
                    "rx_idx": rx_idx,
                    "rx_x": rx_pos[0],
                    "rx_y": rx_pos[1],
                    "rx_z": rx_pos[2],
                    "room_w": w,
                    "room_d": d_room,
                    "room_h": h_room,
                    "char_len": char_len,
                    "H_src": h_src,
                    "H_src_norm": h_src_norm,
                    "d_sm": d_sm,
                    "d_sm_norm": d_sm_norm,
                    "rmse_orig_c1": rmse_orig,
                    # Mode 1: Source-only
                    "c_opt_src": c_opt_src,
                    "rmse_src": rmse_src,
                    # Mode 2: Mic-only
                    "m_opt_mic": m_opt_mic,
                    "rmse_mic": rmse_mic,
                    # Mode 3: Two-stage
                    "c_opt_both": c_opt_both,
                    "m_opt_both": m_opt_both,
                    "rmse_both_stage1": rmse_both_stage1,
                    "rmse_both_stage2": rmse_both_stage2,
                }
            )

            print(f"  rx{rx_idx:02d}: d_sm_norm={d_sm_norm:.3f}, RMSE(c=1,m=1)={rmse_orig:.3f}")
            if RUN_SOURCE_ONLY:
                print(f"    Source-only: c={c_opt_src:.3f}, RMSE={rmse_src:.3f}")
            if RUN_MIC_ONLY:
                print(f"    Mic-only:    m={m_opt_mic:.3f}, RMSE={rmse_mic:.3f}")
            if RUN_SOURCE_MIC:
                print(f"    Two-stage:   c={c_opt_both:.3f}, m={m_opt_both:.3f}, Stage1={rmse_both_stage1:.3f}, Stage2={rmse_both_stage2:.3f}")

    if not rows:
        print("No rows generated (missing input files?).")
        return

    # Save CSV (simple, no pandas)
    csv_path = os.path.join(OUTPUT_DIR, "aes_pairwise_optimization_comparison.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nSaved: {csv_path}")

    # Print comparison summary
    if RUN_SOURCE_ONLY and RUN_MIC_ONLY:
        print("\n" + "="*80)
        print("COMPARISON: Source-only vs Mic-only Optimization")
        print("="*80)
        
        tolerance = 1e-2
        src_better = 0
        mic_better = 0
        equivalent = 0
        
        for r in rows:
            rx_idx = r["rx_idx"]
            rmse_src = r["rmse_src"]
            rmse_mic = r["rmse_mic"]
            c_src = r["c_opt_src"]
            m_mic = r["m_opt_mic"]
            if np.isfinite(rmse_src) and np.isfinite(rmse_mic):
                diff = rmse_src - rmse_mic  # Positive = Mic better
                
                if diff > tolerance:
                    winner = "MIC better"
                    mic_better += 1
                elif diff < -tolerance:
                    winner = "SRC better"
                    src_better += 1
                else:
                    winner = "equivalent"
                    equivalent += 1
                
                print(f"rx{rx_idx:02d}: Source(c={c_src:.3f})={rmse_src:.6f}, Mic(m={m_mic:.3f})={rmse_mic:.6f}, Δ={diff:+.6f} ({winner})")
        
        print(f"\nTolerance: {tolerance:.1e} dB")
        print(f"Source better: {src_better}/{len(rows)} receivers")
        print(f"Mic better:    {mic_better}/{len(rows)} receivers")
        print(f"Equivalent:    {equivalent}/{len(rows)} receivers")
        print("="*80)

    if RUN_SOURCE_MIC:
        print("\n" + "="*80)
        print("TWO-STAGE OPTIMIZATION: Stage 2 Performance Analysis")
        print("="*80)
        
        # Tolerance for meaningful difference
        tolerance = 1e-2
        
        stage2_improved = 0
        stage2_degraded = 0
        stage2_unchanged = 0
        
        for r in rows:
            rx_idx = r["rx_idx"]
            s1 = r["rmse_both_stage1"]
            s2 = r["rmse_both_stage2"]
            if np.isfinite(s1) and np.isfinite(s2):
                improvement = s1 - s2  # Positive = Stage 2 better
                
                if improvement > tolerance:
                    status = "IMPROVED"
                    stage2_improved += 1
                elif improvement < -tolerance:
                    status = "DEGRADED"
                    stage2_degraded += 1
                else:
                    status = "unchanged"
                    stage2_unchanged += 1
                
                print(f"rx{rx_idx:02d}: Stage1={s1:.6f}, Stage2={s2:.6f}, Δ={improvement:+.6f} ({status})")
        
        print(f"\nTolerance: {tolerance:.1e} dB")
        print(f"Stage 2 improved: {stage2_improved}/{len(rows)} receivers")
        print(f"Stage 2 degraded: {stage2_degraded}/{len(rows)} receivers")
        print(f"Stage 2 unchanged: {stage2_unchanged}/{len(rows)} receivers")
        print("="*80)

    # ------------------------------------------------------------------
    # Plot A: RMSE(c=1) vs d_sm_norm, color = H_src_norm
    # This targets your main diagnostic question:
    # "For a fixed source, when receiver is close, does original SDN error increase?"
    # ------------------------------------------------------------------
    y_rmse_orig = np.array([r["rmse_orig_c1"] for r in rows], dtype=float)
    H = np.array([r["H_src_norm"] for r in rows], dtype=float)
    D = np.array([r["d_sm_norm"] for r in rows], dtype=float)

    source_tags = [r["source_tag"] for r in rows]
    unique_sources = list(dict.fromkeys(source_tags))  # preserve order
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(unique_sources)}

    # Subplots: left = per-source fits, right = global fit
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    # --- Left: per-source fits ---
    ax = axes[0]
    last_sc = None
    for s in unique_sources:
        idx = np.array([st == s for st in source_tags])
        last_sc = ax.scatter(
            D[idx],
            y_rmse_orig[idx],
            c=H[idx],
            cmap="viridis",
            alpha=0.85,
            marker=marker_map[s],
            edgecolors="none",
            label=s,
        )

        # Per-source linear trend: rmse ≈ p0 + p1*d
        if np.sum(idx) >= 3:
            d_s = D[idx]
            y_s = y_rmse_orig[idx]
            p1, p0 = np.polyfit(d_s, y_s, 1)
            xs = np.linspace(d_s.min(), d_s.max(), 50)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, linewidth=1.8, alpha=0.7)

    ax.set_xlabel("d_sm_norm = ||x_s - x_m|| / V^(1/3)")
    ax.set_ylabel(f"RMSE(EDC) vs {REFERENCE_METHOD} at c=1 ({ORIGINAL_SDN_METHOD})")
    ax.set_title("Per-source fit\n(color = H_src_norm)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="source_tag", fontsize="small", framealpha=0.9)
    if last_sc is not None:
        fig.colorbar(last_sc, ax=ax, label="H_src_norm")

    # --- Right: global fit ---
    ax = axes[1]
    sc = ax.scatter(D, y_rmse_orig, c=H, cmap="viridis", alpha=0.85, edgecolors="none")
    # Global linear fit
    p1_g, p0_g = np.polyfit(D, y_rmse_orig, 1)
    xs = np.linspace(D.min(), D.max(), 200)
    ys = p0_g + p1_g * xs
    ax.plot(xs, ys, "k-", linewidth=2.2, alpha=0.85, label=f"global fit slope={p1_g:+.3f}")
    rho_g, p_g = spearmanr(D, y_rmse_orig)
    ax.set_xlabel("d_sm_norm = ||x_s - x_m|| / V^(1/3)")
    ax.set_ylabel(f"RMSE(EDC) vs {REFERENCE_METHOD} at c=1 ({ORIGINAL_SDN_METHOD})")
    ax.set_title(f"Global fit\nSpearman ρ={rho_g:+.3f} (p={p_g:.1e})")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.colorbar(sc, ax=ax, label="H_src_norm")

    out = os.path.join(OUTPUT_DIR, "aes_fullgrid_rmse_c1_vs_distance_colored_by_h.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # Extra diagnostic: replicate the old "RMSE vs H" view, and add RMSE vs distance,
    # both WITHOUT colormaps (markers per source only).
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax = axes2[0]
    for s in unique_sources:
        idx = np.array([st == s for st in source_tags])
        ax.scatter(H[idx], y_rmse_orig[idx], alpha=0.85, marker=marker_map[s], label=s)
    ax.set_xlabel("H_src_norm")
    ax.set_ylabel(f"RMSE(EDC) vs {REFERENCE_METHOD} at c=1")
    ax.set_title("RMSE(c=1) vs H_src_norm (no colormap)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="source_tag", fontsize="small", framealpha=0.9)

    ax = axes2[1]
    for s in unique_sources:
        idx = np.array([st == s for st in source_tags])
        ax.scatter(D[idx], y_rmse_orig[idx], alpha=0.85, marker=marker_map[s], label=s)
    ax.set_xlabel("d_sm_norm")
    ax.set_ylabel(f"RMSE(EDC) vs {REFERENCE_METHOD} at c=1")
    ax.set_title("RMSE(c=1) vs d_sm_norm (no colormap)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="source_tag", fontsize="small", framealpha=0.9)

    out2 = os.path.join(OUTPUT_DIR, "aes_fullgrid_rmse_c1_vs_h_and_distance_no_colormap.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved: {out2}")

    # ------------------------------------------------------------------
    # Analysis: distance dependence (global + per source), and optional 2D fit
    # ------------------------------------------------------------------
    print("\n--- Analysis: RMSE(c=1) vs d_sm_norm ---")
    rho_g, p_g = spearmanr(D, y_rmse_orig)
    print(f"Global Spearman rho={rho_g:+.3f}, p={p_g:.2e}")

    for s in unique_sources:
        idx = np.array([st == s for st in source_tags])
        d_s = D[idx]
        y_s = y_rmse_orig[idx]
        if len(d_s) < 3:
            continue
        slope, _intercept = np.polyfit(d_s, y_s, 1)
        rho, p = spearmanr(d_s, y_s)
        print(f"{s:>18s}: slope(d)->rmse = {slope:+.3f}, Spearman rho={rho:+.3f} (p={p:.2e})")

    # 2D descriptive linear model: RMSE(c=1) ≈ a + bH + gD
    X_rmse = np.column_stack([np.ones_like(H), H, D])
    a_r, b_r, g_r = np.linalg.lstsq(X_rmse, y_rmse_orig, rcond=None)[0]
    print("\n2D linear fit (descriptive): RMSE(c=1) ≈ a + b*H_src_norm + g*d_sm_norm")
    print(f"  a={a_r:.3f}, b={b_r:.3f}, g={g_r:.3f}")

    # ------------------------------------------------------------------
    # Plot: m_opt_both vs d_sm_norm (mic weighting optimization results)
    # ------------------------------------------------------------------
    if RUN_SOURCE_MIC:
        print("\n--- Plotting m_opt_both vs d_sm_norm ---")
        m_opt = np.array([r["m_opt_both"] for r in rows], dtype=float)
        c_opt = np.array([r["c_opt_both"] for r in rows], dtype=float)
        
        # Filter out any NaN values
        valid_mask = np.isfinite(m_opt) & np.isfinite(D) & np.isfinite(c_opt)
        m_opt_valid = m_opt[valid_mask]
        D_valid = D[valid_mask]
        c_opt_valid = c_opt[valid_mask]
        source_tags_valid = [source_tags[i] for i in range(len(source_tags)) if valid_mask[i]]
        
        # Create figure with 2 subplots
        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        
        # --- Left: m_opt_both vs d_sm_norm (colored by c_opt_both) ---
        ax = axes3[0]
        sc = ax.scatter(D_valid, m_opt_valid, c=c_opt_valid, cmap="coolwarm", 
                        alpha=0.85, s=80, edgecolors="black", linewidth=0.5)
        
        # Add linear fit
        if len(D_valid) >= 3:
            p1, p0 = np.polyfit(D_valid, m_opt_valid, 1)
            xs = np.linspace(D_valid.min(), D_valid.max(), 100)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, 'k--', linewidth=2, alpha=0.7, label=f'Linear fit: m={p0:.2f}+{p1:.2f}*d')
            
            # Compute correlation
            rho_m, p_m = spearmanr(D_valid, m_opt_valid)
            ax.text(0.05, 0.95, f'Spearman ρ={rho_m:+.3f}\np={p_m:.2e}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='m=1 (no mic weighting)')
        ax.set_xlabel("d_sm_norm = ||x_s - x_m|| / V^(1/3)", fontsize=12)
        ax.set_ylabel("Optimal m (mic_weighting)", fontsize=12)
        ax.set_title("Mic Weighting Optimization vs Distance\n(color = optimal c)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        fig3.colorbar(sc, ax=ax, label="c_opt_both")
        
        # --- Right: m_opt_both vs d_sm_norm (per-source markers) ---
        ax = axes3[1]
        for s in unique_sources:
            idx_valid = np.array([st == s for st in source_tags_valid])
            if np.sum(idx_valid) > 0:
                ax.scatter(D_valid[idx_valid], m_opt_valid[idx_valid], 
                          alpha=0.85, s=80, marker=marker_map[s], label=s,
                          edgecolors="black", linewidth=0.5)
                
                # Per-source fit if enough points
                if np.sum(idx_valid) >= 3:
                    d_s = D_valid[idx_valid]
                    m_s = m_opt_valid[idx_valid]
                    p1_s, p0_s = np.polyfit(d_s, m_s, 1)
                    xs_s = np.linspace(d_s.min(), d_s.max(), 50)
                    ys_s = p0_s + p1_s * xs_s
                    ax.plot(xs_s, ys_s, linewidth=1.5, alpha=0.6)
        
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='m=1 (baseline)')
        ax.set_xlabel("d_sm_norm = ||x_s - x_m|| / V^(1/3)", fontsize=12)
        ax.set_ylabel("Optimal m (mic_weighting)", fontsize=12)
        ax.set_title("Mic Weighting Optimization vs Distance\n(per-source trends)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(title="source_tag", fontsize="small", framealpha=0.9)
        
        out3 = os.path.join(OUTPUT_DIR, "aes_fullgrid_m_opt_vs_distance.png")
        fig3.savefig(out3, dpi=150)
        print(f"Saved: {out3}")
        
        # Print correlation analysis
        print("\n--- Analysis: m_opt_both vs d_sm_norm ---")
        if len(D_valid) >= 3:
            rho_global, p_global = spearmanr(D_valid, m_opt_valid)
            print(f"Global Spearman rho={rho_global:+.3f}, p={p_global:.2e}")
            
            for s in unique_sources:
                idx_valid = np.array([st == s for st in source_tags_valid])
                if np.sum(idx_valid) >= 3:
                    d_s = D_valid[idx_valid]
                    m_s = m_opt_valid[idx_valid]
                    slope, intercept = np.polyfit(d_s, m_s, 1)
                    rho, p = spearmanr(d_s, m_s)
                    print(f"{s:>18s}: slope(d)->m = {slope:+.3f}, intercept={intercept:.3f}, Spearman rho={rho:+.3f} (p={p:.2e})")

    # ------------------------------------------------------------------
    # Plot: Compare Source vs Mic Optimization Results
    # ------------------------------------------------------------------
    if RUN_SOURCE_ONLY and RUN_MIC_ONLY:
        print("\n--- Plotting Source vs Mic Optimization Comparison ---")
        c_opt_src_arr = np.array([r["c_opt_src"] for r in rows], dtype=float)
        m_opt_mic_arr = np.array([r["m_opt_mic"] for r in rows], dtype=float)
        rmse_src_arr = np.array([r["rmse_src"] for r in rows], dtype=float)
        rmse_mic_arr = np.array([r["rmse_mic"] for r in rows], dtype=float)
        
        # Filter valid values
        valid_mask = (np.isfinite(c_opt_src_arr) & np.isfinite(m_opt_mic_arr) & 
                      np.isfinite(rmse_src_arr) & np.isfinite(rmse_mic_arr))
        c_valid = c_opt_src_arr[valid_mask]
        m_valid = m_opt_mic_arr[valid_mask]
        rmse_src_valid = rmse_src_arr[valid_mask]
        rmse_mic_valid = rmse_mic_arr[valid_mask]
        D_valid_comp = D[valid_mask]
        source_tags_valid_comp = [source_tags[i] for i in range(len(source_tags)) if valid_mask[i]]
        
        # Create 2x2 figure
        fig4, axes4 = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
        
        # ========== TOP LEFT: c_opt vs d_sm_norm ==========
        ax = axes4[0, 0]
        for s in unique_sources:
            idx = np.array([st == s for st in source_tags_valid_comp])
            if np.sum(idx) > 0:
                ax.scatter(D_valid_comp[idx], c_valid[idx], 
                          alpha=0.85, s=80, marker=marker_map[s], label=s,
                          edgecolors="black", linewidth=0.5)
        
        if len(D_valid_comp) >= 3:
            p1, p0 = np.polyfit(D_valid_comp, c_valid, 1)
            xs = np.linspace(D_valid_comp.min(), D_valid_comp.max(), 100)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, 'k--', linewidth=2, alpha=0.7, label=f'Fit: c={p0:.2f}+{p1:.2f}*d')
            rho, p_val = spearmanr(D_valid_comp, c_valid)
            ax.text(0.05, 0.95, f'Spearman ρ={rho:+.3f}\np={p_val:.2e}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='c=1 (baseline)')
        ax.set_xlabel("d_sm_norm", fontsize=12)
        ax.set_ylabel("Optimal c (source_weighting)", fontsize=12)
        ax.set_title("Source Optimization: c_opt vs Distance", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, framealpha=0.9, loc='best')
        
        # ========== TOP RIGHT: m_opt vs d_sm_norm ==========
        ax = axes4[0, 1]
        for s in unique_sources:
            idx = np.array([st == s for st in source_tags_valid_comp])
            if np.sum(idx) > 0:
                ax.scatter(D_valid_comp[idx], m_valid[idx], 
                          alpha=0.85, s=80, marker=marker_map[s], label=s,
                          edgecolors="black", linewidth=0.5)
        
        if len(D_valid_comp) >= 3:
            p1, p0 = np.polyfit(D_valid_comp, m_valid, 1)
            xs = np.linspace(D_valid_comp.min(), D_valid_comp.max(), 100)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, 'k--', linewidth=2, alpha=0.7, label=f'Fit: m={p0:.2f}+{p1:.2f}*d')
            rho, p_val = spearmanr(D_valid_comp, m_valid)
            ax.text(0.05, 0.95, f'Spearman ρ={rho:+.3f}\np={p_val:.2e}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='m=1 (baseline)')
        ax.set_xlabel("d_sm_norm", fontsize=12)
        ax.set_ylabel("Optimal m (mic_weighting)", fontsize=12)
        ax.set_title("Mic Optimization: m_opt vs Distance", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, framealpha=0.9, loc='best')
        
        # ========== BOTTOM LEFT: RMSE vs c_opt (colored by distance) ==========
        ax = axes4[1, 0]
        sc = ax.scatter(c_valid, rmse_src_valid, c=D_valid_comp, cmap="viridis", 
                        alpha=0.85, s=80, edgecolors="black", linewidth=0.5)
        
        if len(c_valid) >= 3:
            p1, p0 = np.polyfit(c_valid, rmse_src_valid, 1)
            xs = np.linspace(c_valid.min(), c_valid.max(), 100)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, 'r--', linewidth=2, alpha=0.7, label=f'Fit: RMSE={p0:.3f}+{p1:.3f}*c')
            rho, p_val = spearmanr(c_valid, rmse_src_valid)
            ax.text(0.05, 0.95, f'Spearman ρ={rho:+.3f}\np={p_val:.2e}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("Optimal c (source_weighting)", fontsize=12)
        ax.set_ylabel("Achieved RMSE (source-only)", fontsize=12)
        ax.set_title("Source Optimization: RMSE vs c_opt\n(color = distance)", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        fig4.colorbar(sc, ax=ax, label="d_sm_norm")
        
        # ========== BOTTOM RIGHT: RMSE vs m_opt (colored by distance) ==========
        ax = axes4[1, 1]
        sc = ax.scatter(m_valid, rmse_mic_valid, c=D_valid_comp, cmap="viridis", 
                        alpha=0.85, s=80, edgecolors="black", linewidth=0.5)
        
        if len(m_valid) >= 3:
            p1, p0 = np.polyfit(m_valid, rmse_mic_valid, 1)
            xs = np.linspace(m_valid.min(), m_valid.max(), 100)
            ys = p0 + p1 * xs
            ax.plot(xs, ys, 'r--', linewidth=2, alpha=0.7, label=f'Fit: RMSE={p0:.3f}+{p1:.3f}*m')
            rho, p_val = spearmanr(m_valid, rmse_mic_valid)
            ax.text(0.05, 0.95, f'Spearman ρ={rho:+.3f}\np={p_val:.2e}',
                    transform=ax.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel("Optimal m (mic_weighting)", fontsize=12)
        ax.set_ylabel("Achieved RMSE (mic-only)", fontsize=12)
        ax.set_title("Mic Optimization: RMSE vs m_opt\n(color = distance)", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.9)
        fig4.colorbar(sc, ax=ax, label="d_sm_norm")
        
        out4 = os.path.join(OUTPUT_DIR, "aes_fullgrid_source_vs_mic_comparison.png")
        fig4.savefig(out4, dpi=150)
        print(f"Saved: {out4}")
        
        # Print correlation statistics
        print("\n--- Correlation Analysis: Optimal Parameters vs Achieved RMSE ---")
        if len(c_valid) >= 3:
            rho_c, p_c = spearmanr(c_valid, rmse_src_valid)
            print(f"Source: c_opt vs RMSE_src: Spearman rho={rho_c:+.3f}, p={p_c:.2e}")
        if len(m_valid) >= 3:
            rho_m, p_m = spearmanr(m_valid, rmse_mic_valid)
            print(f"Mic:    m_opt vs RMSE_mic: Spearman rho={rho_m:+.3f}, p={p_m:.2e}")
        
        # Cross-comparison: Do high c and high m correlate?
        if len(c_valid) >= 3:
            rho_cm, p_cm = spearmanr(c_valid, m_valid)
            print(f"\nCross:  c_opt vs m_opt:     Spearman rho={rho_cm:+.3f}, p={p_cm:.2e}")
            
        # Which optimization is better?
        print("\n--- Which Optimization Performs Better? ---")
        tolerance = 1e-2
        diff_src_mic = rmse_src_valid - rmse_mic_valid  # Positive = Mic better
        
        better_mic = np.sum(diff_src_mic > tolerance)
        better_src = np.sum(diff_src_mic < -tolerance)
        equivalent = np.sum(np.abs(diff_src_mic) <= tolerance)
        
        print(f"Tolerance: {tolerance:.1e} dB")
        print(f"Source better: {better_src}/{len(rmse_src_valid)} receivers")
        print(f"Mic better:    {better_mic}/{len(rmse_mic_valid)} receivers")
        print(f"Equivalent:    {equivalent}/{len(rmse_src_valid)} receivers")
        print(f"Mean RMSE - Source: {rmse_src_valid.mean():.6f} dB")
        print(f"Mean RMSE - Mic:    {rmse_mic_valid.mean():.6f} dB")
        improvement = rmse_src_valid.mean() - rmse_mic_valid.mean()
        print(f"Mic advantage:      {improvement:.6f} dB ({100*improvement/rmse_src_valid.mean():.1f}%)")

    # ------------------------------------------------------------------
    # Plot: Stage 1 vs Stage 2 RMSE (Does Stage 2 improve?)
    # ------------------------------------------------------------------
    if RUN_SOURCE_MIC:
        print("\n--- Plotting Stage 1 vs Stage 2 RMSE Comparison ---")
        s1_arr = np.array([r["rmse_both_stage1"] for r in rows], dtype=float)
        s2_arr = np.array([r["rmse_both_stage2"] for r in rows], dtype=float)
        
        valid_s12 = np.isfinite(s1_arr) & np.isfinite(s2_arr)
        s1 = s1_arr[valid_s12]
        s2 = s2_arr[valid_s12]
        
        fig5, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        
        # Tolerance for "meaningful" difference
        # - Tolerance = 1e-3 dB (0.001 dB)
        # - Below this, differences are due to numerical precision/optimization noise
        # - ISM random noise (~0.1 dB) >> tolerance, so this is conservative
        # - Avoids false positives from floating-point comparison
        tolerance = 1e-2
        
        # Color by improvement/degradation with tolerance
        improvement_per_rx = s1 - s2  # Positive = Stage 2 better (lower RMSE)
        
        # Classify each receiver
        improved = improvement_per_rx > tolerance
        degraded = improvement_per_rx < -tolerance
        unchanged = np.abs(improvement_per_rx) <= tolerance
        
        colors = np.where(improved, 'green', np.where(degraded, 'red', 'gray'))
        alphas = np.where(unchanged, 0.3, 
                         np.abs(improvement_per_rx) / np.abs(improvement_per_rx).max() * 0.7 + 0.3)
        
        # Scatter plot
        for i in range(len(s1)):
            ax.scatter(s1[i], s2[i], c=colors[i], alpha=alphas[i], s=100, 
                      edgecolors='black', linewidth=0.5)
        
        # Diagonal line (Stage 1 = Stage 2, no change)
        lims = [min(s1.min(), s2.min()), max(s1.max(), s2.max())]
        ax.plot(lims, lims, 'k--', linewidth=2, alpha=0.7, label='No change (Stage 1 = Stage 2)')
        
        # Statistics
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(s1, s2)
        mean_improvement = (s1 - s2).mean()
        
        n_improved = np.sum(improved)
        n_degraded = np.sum(degraded)
        n_unchanged = np.sum(unchanged)
        
        # Add text box with statistics
        textstr = f'Tolerance: {tolerance:.1e} dB\n'
        textstr += f'Mean Stage 1: {s1.mean():.6f} dB\n'
        textstr += f'Mean Stage 2: {s2.mean():.6f} dB\n'
        textstr += f'Mean Δ: {mean_improvement:.6f} dB\n'
        textstr += f'Improved: {n_improved}/{len(s1)} receivers\n'
        textstr += f'Degraded: {n_degraded}/{len(s1)} receivers\n'
        textstr += f'Unchanged: {n_unchanged}/{len(s1)} receivers\n'
        textstr += f'Paired t-test: p={p_val:.3e}\n'
        if p_val < 0.05:
            textstr += '→ Significant improvement'
        else:
            textstr += '→ No significant difference'
        
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # Labels and styling
        ax.set_xlabel('Stage 1 RMSE (c_opt only)', fontsize=13)
        ax.set_ylabel('Stage 2 RMSE (c_opt + m_opt)', fontsize=13)
        ax.set_title(f'Two-Stage Optimization: Does Adding m Improve RMSE?\n(Green = improved > {tolerance:.1e} dB, Red = degraded > {tolerance:.1e} dB, Gray = unchanged)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, framealpha=0.9)
        ax.set_aspect('equal', adjustable='box')
        
        out5 = os.path.join(OUTPUT_DIR, "aes_fullgrid_stage1_vs_stage2_rmse.png")
        fig5.savefig(out5, dpi=150)
        print(f"Saved: {out5}")

    print("\nDone.")


if __name__ == "__main__":
    main()



