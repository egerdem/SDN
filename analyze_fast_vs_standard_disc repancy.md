# Fast vs Standard SDN Method Discrepancy Analysis

## Problem Summary

The Fast caching method (`calculate_sdn_rir_fast`) produces **different RMSE results** than the Standard method (`calculate_sdn_rir`) for the same configuration:

- **Standard Method** (used by `generate_paper_data.py`): RMSE = **1.450706**
- **Fast Method** (used by `optimisation_globalC.py`): RMSE = **1.214129**
- **Difference**: ~16% lower with Fast method

## Evidence

From `debug_rmse_discrepancy.py` output for Center Source with c=2.998:
```
--- RMSE from SAVED Data ---
Mean RMSE (Saved): 1.450706

--- RMSE from ON-THE-FLY Calculation (Fast Method) ---
Mean RMSE (On-the-fly Fast): 1.214129

--- RMSE from ON-THE-FLY Calculation (Standard Method) ---
Mean RMSE (On-the-fly Standard): 1.450706
```

## Root Cause Analysis

### Fast Method's Linearity Assumption

The Fast method assumes:
```python
RIR(c) = RIR(c=0) + c * (RIR(c=1) - RIR(c=0))
```

This assumes that the RIR is a **linear function of c**.

### Actual SDN Implementation (sdn_core.py)

In `sdn_core.py` (lines 624-654), the source injection distribution is:
- **Dominant node** (best reflection target): gets `c * psk`
- **Non-dominant nodes** (other nodes): each gets `cn * psk` where `cn = (5 - c) / 4`

### Linearity Check

For a given source pressure `psk`:

| c value | Dominant contribution | Each non-dominant contribution | Total (1 dominant + 5 non-dominant) |
|---------|----------------------|-------------------------------|--------------------------------------|
| c = 0   | 0 * psk = 0         | (5-0)/4 * psk = 1.25 * psk    | 0 + 5*1.25*psk = **6.25 * psk**      |
| c = 1   | 1 * psk             | (5-1)/4 * psk = 1.00 * psk    | 1*psk + 5*1.0*psk = **6.00 * psk**   |
| c = 3   | 3 * psk             | (5-3)/4 * psk = 0.50 * psk    | 3*psk + 5*0.5*psk = **5.50 * psk**   |

The relationship is:
```
Total(c) = c*psk + 5*(5-c)/4*psk
         = c*psk + (25-5c)/4*psk
         = c*psk + 6.25*psk - 1.25*c*psk
         = 6.25*psk - 0.25*c*psk
```

This IS linear in c! So the Fast method's assumption should be valid.

## Hypothesis: The Issue is NOT in the Linearity

The Fast method's linear interpolation seems mathematically correct. So why the discrepancy?

### Possible Issues:

1. **Numerical precision in caching**: Small differences accumulate
2. **Room object state mutations**: The room object might be modified differently  
3. **Signal regeneration**: The source signal might be regenerated differently
4. **Normalization timing**: Normalization might happen at different stages
5. **Hidden non-linear flag**: There might be another flag or parameter that introduces non-linearity

## Next Steps

1. Compare the actual RIR arrays (not just EDCs) between Fast and Standard methods
2. Check if there are any hidden flags in the configuration that differ
3. Verify the basis function calculation in `calculate_sdn_rir_fast`
4. Test multiple c values to see if the error is consistent or varies
