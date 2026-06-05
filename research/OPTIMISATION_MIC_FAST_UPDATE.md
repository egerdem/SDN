# Mic-Side Fast Method Integration in optimisation_singleC.py

## Summary

Updated `optimisation_singleC.py` to use the new **mic-side fast method** (`use_fast_mic_method`) for Stage 2 optimization, dramatically improving performance when optimizing `mic_weighting`.

## Changes Made

### 1. Enable Mic-Side Fast Method (Line ~117)

**Before:**
```python
if mic_weighting_val is not None:
    cfg['flags']['specular_mic_pickup'] = True
    cfg['flags']['mic_weighting'] = mic_scalar
    # Disable fast method for mic weighting (fast method support not yet implemented)
    cfg['use_fast_method'] = False
```

**After:**
```python
if mic_weighting_val is not None:
    cfg['flags']['specular_mic_pickup'] = True
    cfg['flags']['mic_weighting'] = mic_scalar
    # Use mic-side fast method for mic weighting optimization
    cfg['use_fast_mic_method'] = True
    cfg['use_fast_method'] = False  # Disable source-side fast method
```

### 2. Update RIR Calculator Selection (Line ~148)

**Before:**
```python
if cfg.get('use_fast_method', False):
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(...)
else:
    _, rir_sdn, _, _ = calculate_sdn_rir(...)
```

**After:**
```python
# Use appropriate RIR calculator based on fast method flags
if cfg.get('use_fast_method', False) or cfg.get('use_fast_mic_method', False):
    _, rir_sdn, _, _ = calculate_sdn_rir_fast(...)
else:
    _, rir_sdn, _, _ = calculate_sdn_rir(...)
```

### 3. Updated Documentation

- Function docstring now explains both source-side and mic-side fast methods
- Module docstring updated to reflect two-stage optimization with both fast methods
- Added explanatory comments throughout the code

## Performance Improvement

### Stage 1 (Source Injection Optimization)
- **Method**: Source-side fast method
- **Basis Functions**: 2 (c=0, c=1) per receiver
- **Cache Key**: Geometry + Duration + Absorption + Source-side params
- **Speedup**: ~10-20x after first evaluation

### Stage 2 (Mic Weighting Optimization) - **NEW**
- **Method**: Mic-side fast method (was slow standard method)
- **Basis Functions**: 2 (mic_w=0, mic_w=1) per receiver
- **Cache Key**: Geometry + Duration + Absorption + Mic-side params + Fixed c value
- **Speedup**: ~10-20x after first evaluation

## Example: Optimization with 25 Receivers

### Old Behavior (Standard Method for Stage 2):
```
Stage 1 (Source-side FAST): 
  - First eval: 25 receivers × 2 basis = 50 RIR computations
  - Next evals: Instant (from cache)

Stage 2 (Standard Method):  ← SLOW!
  - EVERY eval: 25 receivers × 1 RIR = 25 RIR computations
  - 20 iterations = 500 RIR computations
```

### New Behavior (Mic-side FAST for Stage 2):
```
Stage 1 (Source-side FAST):
  - First eval: 25 receivers × 2 basis = 50 RIR computations
  - Next evals: Instant (from cache)

Stage 2 (Mic-side FAST):  ← FAST!
  - First eval: 25 receivers × 2 basis = 50 RIR computations
  - Next evals: Instant (from cache)
  - 20 iterations = ~50 total RIR computations (vs 500!)
```

**Overall speedup for Stage 2: ~10x**

## Console Output Example

When running with `OPTIMIZE_MIC_WEIGHTING = True`:

```
--- STAGE 1: Optimizing Source Injection (c parameter) ---
🚀 SOURCE-SIDE FAST METHOD ACTIVE: SDN-Opt
  ⚙️  [FastSDN-Scalar] Cache Miss - Pre-computing 2 Basis Functions...
📊 STANDARD SDN METHOD: Basis_0
📊 STANDARD SDN METHOD: Basis_1
(... optimization runs, using cached basis ...)
Stage 1 Result: optimal_c = 2.998, RMSE = 0.045231

--- STAGE 2: Optimizing Mic Weighting (fixed c=2.998) ---
Using MIC-SIDE FAST METHOD with caching for efficient optimization
🚀 MIC-SIDE FAST METHOD ACTIVE: SDN-Opt
  ⚙️  [FastSDN-MicScalar] Cache Miss - Pre-computing 2 Basis Functions...
📊 STANDARD SDN METHOD: MicBasis_0
📊 STANDARD SDN METHOD: MicBasis_1
(... optimization runs, using cached basis ...)
Stage 2 Result: optimal_mic_weighting = 3.654, RMSE = 0.042187
Improvement over Stage 1: 0.003044 (6.72%)
Total improvement over baseline: 0.008765 (17.26%)
```

## Configuration Flags

Set these at the top of `optimisation_singleC.py`:

```python
# --- Two-Stage Optimization ---
OPTIMIZE_SOURCE_INJECTION = True   # Stage 1: Optimize source_weighting with source-side FAST
OPTIMIZE_MIC_WEIGHTING = True      # Stage 2: Optimize mic_weighting with mic-side FAST
```

## Cache Behavior

**Important**: Each receiver position gets its own cache entry!

- **Stage 1 cache**: Keyed by (geometry, absorption, duration, source position, receiver position, source-side params)
- **Stage 2 cache**: Keyed by (geometry, absorption, duration, source position, receiver position, mic-side params, **fixed c value**)

Cache files stored in: `results/basis_cache/`

## Backward Compatibility

- All existing Stage 1 caches remain valid
- New Stage 2 caches are independent (different cache key)
- Can disable Stage 2 by setting `OPTIMIZE_MIC_WEIGHTING = False`
- Can disable fast method by manually setting `use_fast_mic_method = False` in config

## Verification

To verify that the fast method is being used, look for these prints:

1. **Stage 2 start**: `Using MIC-SIDE FAST METHOD with caching for efficient optimization`
2. **First evaluation**: `🚀 MIC-SIDE FAST METHOD ACTIVE` + `⚙️ Cache Miss - Pre-computing 2 Basis Functions`
3. **Subsequent evaluations**: `🚀 MIC-SIDE FAST METHOD ACTIVE` + `✓ Using Cached Basis Functions`

If you see `📊 STANDARD SDN METHOD` for every evaluation in Stage 2, the fast method is NOT being used!

## Future Improvements

Potential future enhancements:
- Support for `mic_weighting_vector` (6-element per-wall pattern) in Stage 2
- Joint optimization of both source and mic parameters
- Multi-objective optimization (RT60 + EDC matching)

