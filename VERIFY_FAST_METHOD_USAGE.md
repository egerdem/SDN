# How to Verify Fast Method is Actually Being Used

## Added Print Statements

I've added distinctive print statements in `rir_calculators.py` to verify which method is being called:

### Location in Code:

**Standard SDN Method** (`calculate_sdn_rir`, line ~320):
```python
print(f"📊 STANDARD SDN METHOD: {test_name}")
```

**Fast SDN Methods** (`calculate_sdn_rir_fast`, line ~420-550):

**Mic-Side Fast Mode** (line ~424):
```python
print(f"🚀 MIC-SIDE FAST METHOD ACTIVE: {test_name}")
```

**Source-Side Fast Mode** (line ~490):
```python
print(f"🚀 SOURCE-SIDE FAST METHOD ACTIVE: {test_name}")
```

**Cache Status Messages**:
- `✓ [FastSDN-MicVec] Using Cached Basis Functions` - Mic vector mode using cache
- `⚙️  [FastSDN-MicVec] Cache Miss - Pre-computing 7 Basis Functions...` - Mic vector computing basis
- `✓ [FastSDN-MicScalar] Using Cached Basis Functions` - Mic scalar mode using cache
- `⚙️  [FastSDN-MicScalar] Cache Miss - Pre-computing 2 Basis Functions...` - Mic scalar computing basis
- Similar messages for source-side modes

## New Test Configs Added to experiment_configs.py

### Pair 1: Mic Scalar Mode (mic_weighting)

**`SDN-mic_365_standard`**:
- Flag: `RUN_SDN_mic_365_standard = True`
- Uses: Standard method (`calculate_sdn_rir`)
- Config: `mic_weighting: 3.65`

**`SDN-mic_365_fast`**:
- Flag: `RUN_SDN_mic_365_fast = True`
- Uses: Fast method (`calculate_sdn_rir_fast` with `use_fast_mic_method=True`)
- Config: `mic_weighting: 3.65`

### Pair 2: Mic Vector Mode (mic_weighting_vector)

**`SDN-mic_vec_standard`**:
- Flag: `RUN_SDN_mic_vec_standard = True`
- Uses: Standard method (`calculate_sdn_rir`)
- Config: `mic_weighting_vector: [3.2, 2.8, 4.1, 1.5, 2.9, 3.7]`

**`SDN-mic_vec_fast`**:
- Flag: `RUN_SDN_mic_vec_fast = True`
- Uses: Fast method (`calculate_sdn_rir_fast` with `use_fast_mic_method=True`)
- Config: `mic_weighting_vector: [3.2, 2.8, 4.1, 1.5, 2.9, 3.7]`

## Expected Console Output When Running main.py

```
📊 STANDARD SDN METHOD: mic_365_standard
(... processing ...)

🚀 MIC-SIDE FAST METHOD ACTIVE: mic_365_fast
  ⚙️  [FastSDN-MicScalar] Cache Miss - Pre-computing 2 Basis Functions...
📊 STANDARD SDN METHOD: Basis_0
(... computing basis 0 with mic_w=0 ...)
📊 STANDARD SDN METHOD: Basis_1
(... computing basis 1 with mic_w=1 ...)
(... reconstructing RIR using basis functions ...)

📊 STANDARD SDN METHOD: mic_vec_standard
(... processing ...)

🚀 MIC-SIDE FAST METHOD ACTIVE: mic_vec_fast
  ⚙️  [FastSDN-MicVec] Cache Miss - Pre-computing 7 Basis Functions...
📊 STANDARD SDN METHOD: MicBasis_0
(... computing basis 0 with mic_vec=[0,0,0,0,0,0] ...)
📊 STANDARD SDN METHOD: MicBasis_1
(... computing basis 1 with mic_vec=[1,0,0,0,0,0] ...)
📊 STANDARD SDN METHOD: MicBasis_2
(... computing basis 2 with mic_vec=[0,1,0,0,0,0] ...)
...
📊 STANDARD SDN METHOD: MicBasis_6
(... computing basis 6 with mic_vec=[0,0,0,0,0,1] ...)
(... reconstructing RIR using 6 basis slopes ...)
```

## On Subsequent Runs (Cache Hit)

```
📊 STANDARD SDN METHOD: mic_365_standard
(... processing ...)

🚀 MIC-SIDE FAST METHOD ACTIVE: mic_365_fast
  ✓ [FastSDN-MicScalar] Using Cached Basis Functions
(... instant reconstruction from cache ...)

📊 STANDARD SDN METHOD: mic_vec_standard
(... processing ...)

🚀 MIC-SIDE FAST METHOD ACTIVE: mic_vec_fast
  ✓ [FastSDN-MicVec] Using Cached Basis Functions
(... instant reconstruction from cache ...)
```

## Key Points to Verify

1. **Different Functions Called**:
   - Standard configs show `📊 STANDARD SDN METHOD`
   - Fast configs show `🚀 MIC-SIDE FAST METHOD ACTIVE`

2. **Basis Function Computation** (first run only):
   - Fast scalar mode computes 2 basis functions (mic_w=0, mic_w=1)
   - Fast vector mode computes 7 basis functions (baseline + 6 walls)
   - Each basis computation calls the standard method internally

3. **RIR Comparison**:
   - `SDN-mic_365_standard` vs `SDN-mic_365_fast` should be **identical** (max diff < 1e-10)
   - `SDN-mic_vec_standard` vs `SDN-mic_vec_fast` should be **identical** (max diff < 1e-10)

4. **Performance** (second run onwards):
   - Standard method: Full computation every time
   - Fast method: Instant reconstruction from cached basis functions

## Manual Check Points

If you want to verify manually without print statements:

1. **In `main.py` or calling script**, check which function is invoked:
   - Look for calls to `calculate_sdn_rir_fast` vs `calculate_sdn_rir`
   - Search for: `config.get('use_fast_mic_method')`

2. **In `rir_calculators.py`**:
   - Line ~320: `calculate_sdn_rir` (standard method)
   - Line ~352: `calculate_sdn_rir_fast` (fast method entry point)
   - Line ~424: Mic-side fast mode branch (if `use_fast_mic_method=True`)
   - Line ~485: Source-side fast mode branch (if `use_fast_method=True`)

3. **Cache Directory**:
   - Check `results/basis_cache/` for cached basis function files
   - Files named `basis_*.pkl` indicate cached basis functions
   - Delete cache to force recomputation and verify basis generation

## Verification Steps

1. **Clear cache** (if it exists):
   ```bash
   rm -rf results/basis_cache/
   ```

2. **Run main.py**:
   ```bash
   python main.py
   ```

3. **Check console output** for distinctive emoji prints:
   - 📊 = Standard method
   - 🚀 = Fast method
   - ⚙️ = Computing basis functions
   - ✓ = Using cached basis

4. **Compare RIRs visually** in the plot:
   - Standard and fast should overlay perfectly
   - If they differ, something is wrong

5. **Run again** to verify cache is working:
   - Should see `✓ Using Cached Basis Functions`
   - Should be much faster (no basis computation)

