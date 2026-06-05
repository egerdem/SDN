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
        "aes_FULLGRID_lower_left_source.npz",
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

# Optional: also compute per-pair optimized c* (slow). Keep off for the "SDN error only" study.
# NOTE: c_opt_pair column in CSV is filled when RUN_PAIRWISE_C_OPT=True.
#       If you already have optimal_c values from monte_carlo_proximity.py results.json,
#       use the populate_csv_from_json.py script to fill this column instead.
RUN_PAIRWISE_C_OPT = True

# Outlier rejection (only used if RUN_PAIRWISE_C_OPT=True)
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
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

            # Optional per-pair c optimization (disabled by default)
            c_opt = float("nan")
            rmse_opt = float("nan")
            if RUN_PAIRWISE_C_OPT:
                cache_label = f"aes_fullgrid_{source_tag}_rx{rx_idx:02d}"
                c_opt, rmse_opt = _optimize_c_for_single_receiver(
                    room=room,
                    room_parameters=room_parameters,
                    rx_pos=rx_pos,
                    ref_edc=ref_edcs[rx_idx],
                    Fs=Fs,
                    duration=duration,
                    err_duration_ms=ERR_DURATION_MS,
                    cache_label_prefix=cache_label,
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
                    "c_opt_pair": c_opt,
                    "rmse_opt_pair": rmse_opt,
                    "rmse_improvement": (rmse_orig - rmse_opt) if np.isfinite(rmse_opt) else float("nan"),
                }
            )

            print(
                f"  rx{rx_idx:02d}: d_sm_norm={d_sm_norm:.3f}, "
                f"RMSE(c=1)={rmse_orig:.3f}"
            )

    if not rows:
        print("No rows generated (missing input files?).")
        return

    # Save CSV (simple, no pandas)
    csv_path = os.path.join(OUTPUT_DIR, "aes_pairwise_c_analysis.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nSaved: {csv_path}")

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

    print("\nDone.")


if __name__ == "__main__":
    main()



