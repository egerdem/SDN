# Monte Carlo Workflow Guide

## Overview

This guide explains the workflow for generating Monte Carlo experiments and validating the C parameter predictor.

---

## File Structure

```
results/monte_carlo_experiments/
  ├── default/                      # Default experiment
  │   ├── results.json              # MC optimization results
  │   ├── rimpy_cache/              # Cached RIMPY RIRs
  │   │   └── rimpy_*.npy
  │   ├── validation/               # C predictor validation
  │   │   ├── c_predictor_validation_8src_diagonal_linear.json
  │   │   └── c_predictor_validation_8src_diagonal_linear.png
  │   └── config.json               # Experiment config snapshot
  │
  ├── seed_42/                      # Reproducible experiment
  │   ├── results.json
  │   ├── rimpy_cache/
  │   └── validation/
  │
  └── README.md
```

---

## Workflow

### Step 1: Generate Monte Carlo Data

**Script**: `research/monte_carlo_proximity.py`

**What it does**:
- Generates room configurations (random or fixed)
- Places sources and receivers in systematic grids
- Computes reference RIRs using RIMPY (cached for speed)
- Finds optimal C parameter via optimization
- Computes geometric metrics (H_src_norm, d_sm_norm, etc.)

**Output**: `results.json` containing:
```json
{
  "run_key": "room1_src1_rx00",
  "room_dims": [6.5, 5.2, 3.1],
  "src_pos": [2.1, 1.5, 1.5],
  "rx_pos": [4.2, 3.8, 1.5],
  "optimal_c": 2.234,
  "min_rmse": 0.1823,
  "h_src_norm": 0.2935,
  "d_sm_norm": 0.8234,
  ...
}
```

**Run**:
```bash
# Default experiment
python research/monte_carlo_proximity.py

# Reproducible (with seed)
export MC_EXPERIMENT_ID=seed_42
python research/monte_carlo_proximity.py

# Custom name
export MC_EXPERIMENT_ID=my_experiment
python research/monte_carlo_proximity.py
```

**Resume capability**: If interrupted, re-run the same command - it will skip completed runs and continue where it left off.

---

### Step 2: Validate C Predictor

**Script**: `research/validate_c_predictor.py`

**What it does**:
- Loads MC data from Step 1
- Uses C predictor model to predict C values
- Runs SDN with: (1) baseline C=1, (2) predicted C, (3) optimal C
- Compares RMSE performance
- Generates plots and statistics

**Output**: Saved to `validation/` subfolder
- `c_predictor_validation_<exp>_<model>.json`
- `c_predictor_validation_<exp>_<model>.png`

**Run**:
```bash
# Auto-discover default experiment
python research/validate_c_predictor.py

# Or edit run_validation_example() to specify:
# - experiment: '8src_diagonal', '13src_grid', etc.
# - model_type: 'linear', 'polynomial', 'power'
# - mc_experiment_id: 'default', 'seed_42', etc.
```

---

## When to Re-run What?

| Change | Re-run Step 1? | Re-run Step 2? |
|--------|----------------|----------------|
| Changed room dimension ranges | ✅ Yes (new experiment) | ✅ Yes (on new data) |
| Changed RIMPY seed | ✅ Yes (regenerate RIRs) | ✅ Yes (on new RIRs) |
| Changed source/receiver grid | ✅ Yes (new geometry) | ✅ Yes (on new data) |
| Changed C predictor model | ❌ No | ✅ Yes (test new model) |
| Changed predictor experiment type | ❌ No | ✅ Yes (different model) |
| Just want to re-plot validation | ❌ No | ✅ Yes (fast, uses cache) |

---

## Multiple Experiments

You can run multiple experiments and keep them organized:

```bash
# Experiment 1: Default settings
python research/monte_carlo_proximity.py

# Experiment 2: With seed 42
export MC_EXPERIMENT_ID=seed_42
python research/monte_carlo_proximity.py

# Experiment 3: Different room sizes
export MC_EXPERIMENT_ID=large_rooms
# (Edit WIDTH_RANGE, DEPTH_RANGE in monte_carlo_proximity.py)
python research/monte_carlo_proximity.py

# Now validate each experiment separately
# Edit run_validation_example() to set mc_experiment_id='default', 'seed_42', 'large_rooms'
python research/validate_c_predictor.py
```

Each experiment is self-contained in its own folder.

---

## Understanding `results.json`

**Name**: `results.json` refers to **Monte Carlo generation results**, not validation.

**Contains**:
- Optimization results: `optimal_c`, `min_rmse`
- Room geometry: `room_dims`, `src_pos`, `rx_pos`
- Geometric metrics: `h_src_norm`, `d_sm_norm`, etc.
- Metadata: `run_key`, `room_idx`, `src_idx`, `rx_idx`

**Does NOT contain**:
- C predictor validation results (those are in `validation/` subfolder)
- RIMPY RIRs (those are in `rimpy_cache/` subfolder)

---

## Advanced: Regenerating RIRs with Known Seed

If your original MC data was generated without a RIMPY seed (or you lost track of it), you can regenerate with a known seed:

```python
from research.validate_c_predictor import validate_predictor

results = validate_predictor(
    experiment='8src_diagonal',
    model_type='linear',
    mc_experiment_id='default',
    regenerate_rirs=True,    # Force recalculation
    rimpy_seed=42,           # Known seed for reproducibility
)
```

This will:
1. Recalculate all RIMPY RIRs with seed=42
2. Overwrite the cached `.npy` files
3. Run validation with the new RIRs

---

## Quick Reference

**Generate MC data**:
```bash
python research/monte_carlo_proximity.py
```

**Validate predictor**:
```bash
python research/validate_c_predictor.py
```

**Output locations**:
- MC results: `results/monte_carlo_experiments/<ID>/results.json`
- RIMPY cache: `results/monte_carlo_experiments/<ID>/rimpy_cache/`
- Validation: `results/monte_carlo_experiments/<ID>/validation/`

**Experiment ID**:
- Default: `MC_EXPERIMENT_ID` not set
- Custom: `export MC_EXPERIMENT_ID=my_name`
- Recommended for reproducibility: `export MC_EXPERIMENT_ID=seed_42`

---

## Troubleshooting

**Q: Validation script can't find my MC data**

A: The script auto-discovers experiments. Check that:
- `results/monte_carlo_experiments/default/results.json` exists
- Or specify `mc_experiment_id='your_folder_name'` explicitly

**Q: I want to run a new experiment without overwriting the old one**

A: Set a unique `MC_EXPERIMENT_ID`:
```bash
export MC_EXPERIMENT_ID=new_experiment_v2
python research/monte_carlo_proximity.py
```

**Q: How do I know which experiment was used for validation?**

A: Check the validation JSON output - it includes the full path to the MC results file in the console output.

**Q: Can I validate multiple C predictor models on the same MC data?**

A: Yes! Just run `validate_c_predictor.py` multiple times with different `model_type` values. Each creates a separate output file.

---

## Summary

1. **Generate** MC data once with `monte_carlo_proximity.py`
2. **Validate** predictor models as needed with `validate_c_predictor.py`
3. **Organize** experiments by using unique `MC_EXPERIMENT_ID` values
4. **Resume** interrupted runs by re-running the same command
5. **Reproduce** results by setting consistent seeds and experiment IDs

