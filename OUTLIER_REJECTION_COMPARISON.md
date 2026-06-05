# Outlier Rejection: Old vs New Approach

## 📋 Summary

We've transitioned from **data-modifying post-processing** to **analysis-time filtering** for outlier rejection. This document explains the differences and why the new approach is better.

---

## 🔴 OLD Approach: `apply_outlier_rejection_to_json.py`

### How It Worked:
1. **Post-processing script** (runs AFTER Monte Carlo)
2. Loads `results.json`, **modifies it in-place** (with backup)
3. Logic: If `optimal_c` hits bounds (1.0 or 7.0 ± 0.005) AND `rmse_c1 <= min_rmse`:
   - Replace `optimal_c` → 1.0
   - Replace `min_rmse` → `rmse_c1`
4. Loads `rmse_c1` from NPZ files (experiment-specific format)

### ❌ Problems:

1. **Permanently modifies source data**
   - Can't distinguish "optimizer found 1.0" vs "outlier replaced with 1.0"
   - Loses information about what optimizer actually found
   - Breaks reproducibility

2. **Experiment-specific & brittle**
   - Hardcoded to load RMSE from NPZ files
   - Won't work for different experiments without modification
   - Requires specific file format (`aes_fullgrid_perpair_...`)

3. **Checks BOTH bounds**
   - Rejects `C ≈ 1.0` OR `C ≈ 7.0`
   - But `C ≈ 1.0` might be legitimate (source near walls)!

4. **Flawed logic: `rmse_c1 <= min_rmse`**
   - Rejects if baseline is **equal** or better
   - Should only reject if baseline is **strictly better** (`<`)
   - Equal performance doesn't mean optimization failed!

### Example (Old Approach):
```json
// Before (original data):
{
  "optimal_c": 6.9987,
  "min_rmse": 0.35
}

// After (modified):
{
  "optimal_c": 1.0,        // ← REPLACED!
  "min_rmse": 0.34         // ← REPLACED!
}
// Information about optimizer behavior is LOST!
```

---

## ✅ NEW Approach: Analysis-Time Filtering

### How It Works:
1. **During model fitting** (`predict_c_from_proximity.py`)
2. Reads `results.json` **without modification**
3. Logic: If `optimal_c >= 6.9` AND `rmse_c1 < min_rmse`:
   - Mark as outlier
   - Exclude from model fitting
4. Uses `rmse_c1` directly from `results.json`

### ✅ Advantages:

1. **Preserves original data**
   - JSON remains unchanged
   - Full information retained for debugging
   - Better reproducibility

2. **Works for all experiments**
   - No NPZ file dependencies
   - Automatic fallback if `rmse_c1` missing
   - Generic and maintainable

3. **Only checks upper bound**
   - `C ≥ 6.9` only (optimizer hit ceiling)
   - `C ≈ 1.0` is NOT rejected (could be legitimate)

4. **Correct logic: `rmse_c1 < min_rmse`**
   - Only rejects if baseline is **strictly better**
   - Principled outlier criterion

### Example (New Approach):
```json
// JSON (unchanged):
{
  "optimal_c": 6.9987,
  "min_rmse": 0.35,
  "rmse_c1": 0.34        // ← Baseline is better!
}
// During analysis: marked as outlier, excluded from fitting
// Original data preserved!
```

---

## 🔄 Transition Guide

### 1. **Delete or Archive Old Script**
```bash
# Archive for reference
mkdir -p archive
mv research/apply_outlier_rejection_to_json.py archive/

# Or just delete it
rm research/apply_outlier_rejection_to_json.py
```

### 2. **Ensure `rmse_c1` in All Results**

#### Option A: Re-run Monte Carlo (cleanest)
```bash
python research/monte_carlo_proximity.py
# New runs automatically include rmse_c1
```

#### Option B: Backfill Existing Results (faster)
```bash
python research/backfill_rmse_c1.py default
# Retroactively computes rmse_c1 for entries missing it
```

### 3. **Use New Workflow**
```bash
# 1. Generate data (includes rmse_c1 automatically)
python research/monte_carlo_proximity.py

# 2. Fit models (outlier filtering happens here)
python research/predict_c_from_proximity.py

# 3. Validate predictor
python research/validate_c_predictor.py
```

---

## 📊 Comparison Table

| Feature | Old (Post-Processing) | New (Analysis-Time) |
|---------|----------------------|---------------------|
| **Data Integrity** | ❌ Modifies JSON | ✅ Preserves JSON |
| **Reproducibility** | ❌ Hard to trace changes | ✅ Full transparency |
| **Bounds Checked** | ❌ Both (1.0 and 7.0) | ✅ Upper only (≥6.9) |
| **RMSE Logic** | ❌ `<=` (flawed) | ✅ `<` (correct) |
| **Experiment Portability** | ❌ Hardcoded NPZ loading | ✅ Generic JSON-based |
| **Maintenance** | ❌ Complex, brittle | ✅ Simple, robust |
| **When Applied** | After Monte Carlo | During model fitting |

---

## 🎯 Key Principle

**Outlier rejection is an ANALYSIS decision, not a DATA-PROCESSING step.**

✅ **Good:** Filter during analysis, keep original data  
❌ **Bad:** Modify source data, lose information

---

## 📝 Technical Details

### Outlier Criterion (New):

```python
outlier = (optimal_c >= 6.9) AND (rmse_c1 < min_rmse)
```

**Meaning:**
- Optimizer hit upper bound (couldn't find true minimum)
- **AND** baseline C=1 performs strictly better
- → Clear sign of failed optimization

### Why Only Upper Bound?

- **Lower bound (C ≈ 1.0):** Legitimate for sources near walls
- **Upper bound (C ≈ 7.0):** Indicates monotonic objective (optimization failed)

### Why Strict Inequality (`<`)?

- Equal performance (`rmse_c1 == min_rmse`) doesn't prove failure
- Only strictly better baseline (`rmse_c1 < min_rmse`) indicates problem

---

## 🔧 Backfill Script Details

**`research/backfill_rmse_c1.py`** retroactively computes `rmse_c1` for entries missing it.

### Features:
- ✅ Uses existing RIMPY RIR cache (fast)
- ✅ Uses existing basis function cache (C=1 likely cached)
- ✅ Atomic JSON write (safe)
- ✅ Creates backup automatically
- ✅ Verbose progress reporting

### Usage:
```bash
# Default experiment
python research/backfill_rmse_c1.py

# Specific experiment
python research/backfill_rmse_c1.py legacy_4x20_50cornerplacement_noseed
```

### Output:
```
================================================================================
BACKFILL MISSING rmse_c1 VALUES
================================================================================

Experiment: default
✓ Basis function disk cache enabled
✓ Created backup: results_backup_20260119_143052.json

Loading: results/monte_carlo_experiments/default/results.json
  Total entries: 43
  Missing rmse_c1: 28
  Already have rmse_c1: 15

Processing 28 entries...

[1/28] room1_pair1
  ✓ rmse_c1 = 0.3490 (vs optimal: 0.1633)
...
================================================================================
SUMMARY
================================================================================
Backup created: results_backup_20260119_143052.json
Successfully backfilled: 28 / 28
Failed: 0 / 28

✅ Updated results/monte_carlo_experiments/default/results.json
   All entries now have rmse_c1 for principled outlier detection!
```

---

## ✅ Recommendation

1. **Archive or delete** `apply_outlier_rejection_to_json.py`
2. **Run backfill script** on partial results: `python research/backfill_rmse_c1.py default`
3. **Use new workflow** going forward (outlier filtering in `predict_c_from_proximity.py`)

This ensures:
- ✅ Better data integrity
- ✅ Clearer analysis pipeline
- ✅ Easier maintenance
- ✅ Full reproducibility

