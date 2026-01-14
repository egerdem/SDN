"""
Evaluate manually-assigned per-source C (source_weighting) values.

Goal
----
You can manually set different C values for each *source position* (e.g. Center vs
Lower-Left), run SDN with those values, and compare the resulting EDCs against
the same reference EDCs used elsewhere (typically `RIMPY-neg10`) for the exact
same receiver grid already stored in the `paper_data/*.npz` files.

This script does NOT save any new .npz files. It just runs and prints metrics.

How it works
------------
1) Loads each data file listed in FILES_TO_PROCESS from results/paper_data/
2) Extracts:
   - room params (dims, absorption, mic z, source xyz)
   - receiver positions (grid)
   - reference EDCs (REFERENCE_METHOD)
3) Runs SDN for every receiver position using:
   DelayNetwork(..., specular_source_injection=True, source_weighting=<your C>)
4) Computes EDCs and RMSE vs reference over ERR_DURATION_MS (ms)
5) Prints a summary per source file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.append(_project_root)

import geometry
from analysis import analysis as an
from analysis.plotting_utils import load_data
from rir_calculators import calculate_sdn_rir, calculate_sdn_rir_fast, rir_normalisation
from research.data_config import DataConfig

# -----------------------------------------------------------------------------
# Configuration (keep consistent with generate_paper_data.py / analyze_edc_source_variation.py)
# -----------------------------------------------------------------------------

_project_root = os.path.dirname(_script_dir)

# === DATA SOURCE CONFIGURATION ===
# Option 1: Legacy mode
config = DataConfig(mode="legacy", legacy_files=[
    "aes_FULLGRID_center_source.npz",
    "aes_FULLGRID_top_middle_source.npz",
    "aes_FULLGRID_upper_right_source.npz",
    "aes_FULLGRID_lower_left_source.npz",
    "aes_FULLGRID_corner_sourcev3.npz",

    # Quarter grid (legacy):
    # "aes_quarter_center_source.npz",
    # "aes_quarter_top_middle_source.npz",
    # "aes_quarter_upper_right_source.npz",
    # "aes_quarter_lower_left_source.npz",
    # "aes_quarter_corner_sourcev3.npz",
])

# Option 2: Experiment mode (uncomment to use)
# config = DataConfig(mode="experiment", experiment_name="aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair")
# NOTE: If using experiment mode, update MANUAL_C_BY_FILE dict below with experiment filenames

DATA_DIR = config.get_data_dir()
FILES_TO_PROCESS = config.get_files()

REFERENCE_METHOD = "RIMPY-neg10"

# -----------------------------------------------------------------------------
# Manual per-source C assignments (edit these)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ManualCEntry:
    enabled: bool
    c_value: float
    info: str = ""


# Key: filename in FILES_TO_PROCESS
# Value: manual C assignment to use for that file
MANUAL_C_BY_FILE: Dict[str, ManualCEntry] = {

    # "aes_quarter_center_source.npz": ManualCEntry(enabled=True, c_value=5.84, info="manual: center"),
    # "aes_quarter_lower_left_source.npz": ManualCEntry(enabled=True, c_value=3.44, info="manual: lower-left"),
    # "aes_quarter_upper_right_source.npz": ManualCEntry(enabled=True, c_value=2.56, info="manual: "),
    # "aes_quarter_top_middle_source.npz": ManualCEntry(enabled=True, c_value=4.43, info="manual: "),
    # "aes_quarter_corner_sourcev3.npz": ManualCEntry(enabled=True, c_value=1.07, info="manual: corner (disabled)"),

    "aes_FULLGRID_center_source.npz": ManualCEntry(enabled=False, c_value=1, info="manual: center"),
    "aes_FULLGRID_lower_left_source.npz": ManualCEntry(enabled=False, c_value=3.44, info="manual: lower-left"),
    "aes_FULLGRID_upper_right_source.npz": ManualCEntry(enabled=False, c_value=2.56, info="manual: upper-right"),
    "aes_FULLGRID_top_middle_source.npz": ManualCEntry(enabled=True, c_value=1, info="manual: top-middle"),
    "aes_FULLGRID_corner_sourcev3.npz": ManualCEntry(enabled=False, c_value=1.07, info="manual: corner"),
}

# Default fallback if a file is in FILES_TO_PROCESS but missing from MANUAL_C_BY_FILE
DEFAULT_C = 1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_sdn_config(source_weighting: float, info: str) -> dict:
    """
    Build a config dict in the same *shape* as experiment_configs entries.
    `calculate_sdn_rir()` passes `config['flags']` directly into DelayNetwork(...).
    """
    return {
        "enabled": True,
        "info": info,
        "calculator": "sdn",
        "flags": {
            "specular_source_injection": True,
            "source_weighting": source_weighting,
        },
        "label": "SDN",
    }


def _compute_mean_rmse_for_source_file(
    data_file: str,
    c_value: float,
    err_duration_ms: int,
    use_fast: bool,
) -> Tuple[float, List[float]]:
    """
    Runs SDN (with given c_value) for all receivers in a paper_data file and
    returns mean RMSE and per-receiver RMSE list.
    """
    data_path = os.path.join(DATA_DIR, data_file)
    sim = load_data(data_path)

    Fs = int(sim["Fs"])
    duration = float(sim["duration"])
    receiver_positions = sim["receiver_positions"]
    room_params = sim["room_params"]
    source_pos = tuple(float(x) for x in sim["source_pos"])

    if REFERENCE_METHOD not in sim["edcs"]:
        raise KeyError(
            f"Reference method '{REFERENCE_METHOD}' not found in {data_file}. "
            f"Available: {list(sim['edcs'].keys())}"
        )
    ref_edcs = sim["edcs"][REFERENCE_METHOD]

    # Setup room for SDN evaluation (source fixed, mic varies)
    room = geometry.Room(room_params["width"], room_params["depth"], room_params["height"])
    impulse = geometry.Source.generate_signal("dirac", int(Fs * duration))
    room.set_source(*source_pos, signal=impulse["signal"], Fs=Fs)
    room.wallAttenuation = [np.sqrt(1.0 - room_params["absorption"])] * 6

    room_parameters = dict(room_params)
    room_parameters["reflection"] = np.sqrt(1.0 - room_params["absorption"])

    cfg = _make_sdn_config(c_value, info=f"manual_c={c_value:.3f}")

    rmses: List[float] = []
    num_receivers = len(receiver_positions)
    for i, pos in enumerate(receiver_positions):
        rx, ry = float(pos[0]), float(pos[1])
        room.set_microphone(rx, ry, room_params["mic z"])

        if use_fast:
            # For FAST SDN, calculate_sdn_rir_fast expects cfg['flags']['source_weighting'].
            # test_name only affects the label string.
            _, rir_sdn, _, _ = calculate_sdn_rir_fast(room_parameters, "Eval", room, duration, Fs, cfg)
        else:
            _, rir_sdn, _, _ = calculate_sdn_rir(room_parameters, "Eval", room, duration, Fs, cfg)

        rir_sdn_norm = rir_normalisation(rir_sdn, room, Fs, normalize_to_first_impulse=True)["single_rir"]
        edc_sdn, _, _ = an.compute_edc(rir_sdn_norm, Fs, plot=False)

        ref_edc = ref_edcs[i]
        rmse = an.compute_RMS(
            edc_sdn,
            ref_edc,
            range=int(err_duration_ms),
            Fs=Fs,
            skip_initial_zeros=True,
            normalize_by_active_length=True,
        )
        rmses.append(float(rmse))

        print(f"    receiver {i}/{num_receivers}: RMSE={rmse:.4f}")

    return float(np.mean(rmses)), rmses


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("EVALUATE MANUAL C PREDICTIONS (per source)")
    print("=" * 80)
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"REFERENCE_METHOD: {REFERENCE_METHOD}")
    print(f"ERR_DURATION_MS: {ERR_DURATION_MS} ms")
    print(f"USE_FAST_SDN: {USE_FAST_SDN}")
    print()

    for data_file in FILES_TO_PROCESS:
        entry = MANUAL_C_BY_FILE.get(data_file, ManualCEntry(enabled=True, c_value=DEFAULT_C, info="default"))
        if not entry.enabled:
            print(f"[SKIP] {data_file} (disabled)")
            continue

        print("-" * 80)
        print(f"[RUN ] {data_file}")
        print(f"  manual C (source_weighting): {entry.c_value:.4f}  ({entry.info})")

        mean_rmse, rmses = _compute_mean_rmse_for_source_file(
            data_file=data_file,
            c_value=entry.c_value,
            err_duration_ms=ERR_DURATION_MS,
            use_fast=USE_FAST_SDN,
        )

        rmses_np = np.array(rmses)
        print(f"  mean RMSE: {mean_rmse:.4f}")
        print(f"  RMSE stats: min={rmses_np.min():.4f}  max={rmses_np.max():.4f}  std={rmses_np.std():.4f}")

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()


