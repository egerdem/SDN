# Understanding Optimization Results vs Main.py Results

## Your Question

**Optimization Results** (Two-stage, Lower Left Source, Receiver 1):
- Stage 1 (optimize c): RMSE = 0.54
- Stage 2 (optimize m with c=2.21): RMSE = 0.65
- Optimal parameters: c=2.21, m=2.33

**Main.py Results** (Same parameters, same source/receiver):
- Config with c=2.21, m=2.33
- RMSE = 0.52

**Question**: Why are these different? Should successive optimization give the same result as setting both at once?

## Short Answer

**YES**, two-stage optimization should give the SAME result as setting both parameters simultaneously, **IF AND ONLY IF** all conditions are identical.

The discrepancy (0.65 vs 0.52) suggests different conditions between the two runs.

## Likely Causes of Discrepancy

### 1. **Most Likely: Random ISM Reference (rand10)**

**Optimization script** uses:
```python
REFERENCE_METHOD = 'RIMPY-neg10'  # This is rand10 (randDist=0.1)
```

**Main.py** might use:
```python
'ISM (rimpy-neg10)'  # Different random realization!
```

**Why this matters**:
- `rand10` adds ±10cm random displacement to each reflection point
- Each run generates a **different random ISM**
- Your optimization used one random realization
- Main.py computed a different random realization
- Different reference → different RMSE

**Solution**: Use `RIMPY-pos` (no randomness) for exact reproducibility

### 2. **Pre-computed vs On-the-fly Reference**

**Optimization script**:
```python
# Loads EDCs from .npz file (pre-computed earlier)
all_edcs = dict(np.load(data_path))['edcs_RIMPY-neg10']
```

**Main.py**:
```python
# Computes ISM on-the-fly when you run it
rir_ref, _ = calculate_rimpy_rir(room_parameters, ...)
```

Even with `rand10`, if you use the **same pre-computed** EDCs in both scripts, they should match.

### 3. **Different RMSE Calculation Methods**

Check if main.py uses the same RMSE calculation:

**Optimization**:
```python
rmse = an.compute_RMS(edc_sdn, ref_edc, 
                      range=int(50),  # First 50ms
                      Fs=Fs,
                      skip_initial_zeros=True,
                      normalize_by_active_length=True)
```

**Main.py** (check line ~426 in your file):
```python
edc_comparisons = an.compare_edc_pairs(rirs, get_method_pairs(), Fs)
```

Look inside `compare_edc_pairs()` - does it use the same parameters?

## Mathematical Equivalence: Two-Stage vs Simultaneous

### Two-Stage Optimization

```
Stage 1: Optimize c (with m=1.0 default)
  → Find c* that minimizes RMSE(c, m=1.0)
  → Result: c* = 2.21

Stage 2: Optimize m (with c=2.21 fixed)
  → Find m* that minimizes RMSE(c=2.21, m)
  → Result: m* = 2.33
  
Final RMSE = RMSE(c=2.21, m=2.33)
```

### Simultaneous Setting

```
Set both: c=2.21, m=2.33
Calculate: RMSE(c=2.21, m=2.33)
```

**These MUST give the same RMSE** if:
1. Same c and m values
2. Same reference (same ISM realization)
3. Same normalization
4. Same RMSE calculation method
5. Same receiver position

## Diagnostic Steps

### Step 1: Run Verification Script

```bash
python verify_optimization_vs_main.py
```

This will show you which RMSE values you get with:
- 50ms EDC window (same as optimization)
- Full EDC window
- Main.py style calculation

### Step 2: Check Reference Method

In `optimisation_singleC.py` (line 80):
```python
REFERENCE_METHOD = 'RIMPY-neg10'  # ← Is this rand10?
```

In `main.py`, check which ISM is enabled:
```python
PLOT_ISM_rimPy_neg_rand10 = True  # ← Is this enabled?
```

### Step 3: Compare Exact EDCs

Add this to your main.py after calculating RIRs:

```python
# After calculating both ISM and SDN
edc_ism, _, _ = an.compute_edc(rirs['ISM (rimpy-neg10)'], Fs, plot=False)
edc_sdn, _, _ = an.compute_edc(rirs['SDN c=2.21 m=2.33'], Fs, plot=False)

# Calculate RMSE exactly as optimization does
rmse_50ms = an.compute_RMS(
    edc_sdn, edc_ism,
    range=int(50),  # First 50ms
    Fs=Fs,
    skip_initial_zeros=True,
    normalize_by_active_length=True
)
print(f"RMSE (50ms EDC, same as optimization): {rmse_50ms:.6f}")
```

### Step 4: Use Same Pre-Computed Reference

The most reliable way to compare:

```python
# In main.py, load the SAME reference EDCs used in optimization
data_path = "results/spatial_data/aes_FULLGRID_lower_left_source.npz"
with np.load(data_path, allow_pickle=True) as data:
    ref_edcs = data['edcs_RIMPY-neg10']
    # Use ref_edcs[0] for receiver 1
```

## Why Stage 1 RMSE (0.54) < Stage 2 RMSE (0.65)?

**This is WRONG!** Stage 2 should have **lower** or **equal** RMSE than Stage 1, never higher!

Stage 1: Optimize c only → Best possible RMSE with m=1.0
Stage 2: Optimize m with c=2.21 → Should improve or stay same

**Possible explanations**:
1. **Stage 1 and Stage 2 used different references** (different rand10 realizations)
2. **Typo in your numbers** - maybe Stage 1 is 0.65 and Stage 2 is 0.54?
3. **Different receiver positions** between stages

## Your Individual Optimizations

You mentioned:
- Source-only optimization: c ≈ 2.21 (similar)
- Mic-only optimization: m ≈ 2.33 (similar but not exact)

**This is expected!** They should be similar but not identical because:

**Source-only** optimizes:
```
min RMSE(c, m=1.0)  → c* ≈ 2.2
```

**Mic-only** optimizes:
```
min RMSE(c=1.0, m)  → m* ≈ 2.3
```

**Two-stage** optimizes:
```
Step 1: min RMSE(c, m=1.0)    → c* = 2.21
Step 2: min RMSE(c=2.21, m)   → m* = 2.33
```

The values should be close but not identical because the optimization landscapes are different.

## Recommended Fix

### For Reproducible Results:

1. **Use non-random ISM**:
```python
# In optimisation_singleC.py
REFERENCE_METHOD = 'RIMPY-pos'  # Change from RIMPY-neg10

# In experiment_configs.py
PLOT_ISM_rimPy_pos = True
PLOT_ISM_rimPy_neg_rand10 = False
```

2. **Or use the same pre-computed reference**:
Save the reference EDCs and use them in both scripts

3. **Verify receiver positions match exactly**:
```python
# Optimization uses:
receiver_positions[0]  # From .npz file

# Main.py uses:
room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z']
```

Print both and ensure they're identical to 3+ decimal places.

## Summary

| Aspect | Optimization | Main.py | Should Match? |
|--------|-------------|---------|---------------|
| Parameters | c=2.21, m=2.33 | c=2.21, m=2.33 | ✓ YES |
| Reference | Pre-computed rand10 | On-the-fly rand10 | ✗ NO (different realization!) |
| RMSE method | 50ms EDC | Depends on code | ? Check |
| Normalization | normalize_to_first_impulse | normalize_to_first_impulse | ? Check |
| Result | RMSE=0.65 | RMSE=0.52 | Should match if above match |

**Most likely issue**: Different random ISM realizations (rand10)

**Quick fix**: Use `RIMPY-pos` (no randomness) in both scripts

