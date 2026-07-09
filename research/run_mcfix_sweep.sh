#!/usr/bin/env bash
# Monte Carlo absorption sweep: one experiment folder (mcfix_a<alpha>) per absorption.
# All runs read the same fixed geometry (results/monte_carlo_experiments/mc60_geometry.json).
#
# Usage:
#   ./run_mcfix_sweep.sh             # default remaining absorptions
#   ./run_mcfix_sweep.sh 0.4 0.5     # only the given absorptions
set -euo pipefail
cd "$(dirname "$0")"   # research/

ABS=("$@")
[ ${#ABS[@]} -eq 0 ] && ABS=(0.1 0.2 0.4 0.5 0.6)

for A in "${ABS[@]}"; do
    echo "=== absorption ${A} -> mcfix_a${A} ==="
    MC_PRESET=legacy_pair_single_mic MC_ABSORPTION="${A}" MC_EXPERIMENT_ID="mcfix_a${A}" \
        python3 monte_carlo_proximity.py
done
