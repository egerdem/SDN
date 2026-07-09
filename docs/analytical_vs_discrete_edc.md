# Analytical vs Discrete EDC in Image Source Method

## Mathematical Foundation

Both EDC calculation methods are derived from the same theoretical framework, but they differ in their implementation due to discretization.

### Analytical EDC (Continuous Time)

From the equation for the EDC:

```
d(τ) = ∫_τ^∞ p²(t) dt
```

With the image-source RIR:

```
p(t) = Σ_k a_k δ(t - t_k)
```

Where:
- `a_k = 1/(4πR_k)` is the amplitude of image source k
- `t_k = R_k/c` is the arrival time
- `R_k` is the distance from image source k to receiver

Squaring p(t) formally gives:

```
p²(t) = (Σ_k a_k δ(t-t_k)) × (Σ_ℓ a_ℓ δ(t-t_ℓ))
      = Σ_{k,ℓ} a_k a_ℓ δ(t-t_k) δ(t-t_ℓ)
```

**Key Approximation:** Assuming non-overlapping Dirac pulses, only k=ℓ terms contribute:

```
p²(t) ≈ Σ_k a_k² δ(t-t_k)
```

This gives the **analytical EDC**:

```
d(τ) ≈ Σ_{k: t_k ≥ τ} a_k²
```

### Discrete EDC (Sampled Time)

The discrete implementation follows these steps:

#### Step 1: Create Discrete RIR

```python
for t_k, a_k in zip(times, amps):
    n_k = round(t_k × f_s)  # Round to nearest sample
    h[n_k] += a_k            # Add amplitude to that sample
```

**Critical Issue:** Multiple images can have arrival times that round to the same sample index:

```
If t_1 = 0.0100 s and t_2 = 0.0102 s with f_s = 44100 Hz:
n_1 = round(0.0100 × 44100) = 441
n_2 = round(0.0102 × 44100) = 450  (different sample)

But if t_1 = 0.0100 s and t_2 = 0.0101 s:
n_1 = round(0.0100 × 44100) = 441
n_2 = round(0.0101 × 44100) = 445 or 446 (might be same or different)
```

When multiple images land on the same sample:

```
h[n] = a_1 + a_2 + a_3 + ...
```

#### Step 2: Compute Energy

```python
e[n] = h[n]²
```

If multiple images landed on sample n:

```
e[n] = (a_1 + a_2 + ...)²
     = a_1² + a_2² + ... + 2a_1a_2 + 2a_1a_3 + ...
     ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
     incoherent terms     coherent cross-terms
```

This **violates the approximation** that only k=ℓ terms contribute!

#### Step 3: Backward Cumulative Sum

```python
EDC[n] = Σ_{m=n}^{N-1} e[m]
```

## Why They Differ

### 1. Coherent vs Incoherent Summation

| Aspect | Analytical | Discrete |
|--------|-----------|----------|
| Energy at time τ | Σ a_k² (incoherent) | (Σ a_k)² when coincident (coherent) |
| Cross-terms | Excluded by assumption | Included when images coincide |
| Total energy | Σ a_k² | Σ (Σ_coincident a_k)² |

**Physical Interpretation:**
- **Analytical**: Treats each image as arriving at a distinct, infinitesimal time → energies add
- **Discrete**: Images arriving within the same sample period interfere → amplitudes add, then square

### 2. Time Quantization

| Aspect | Analytical | Discrete |
|--------|-----------|----------|
| Time resolution | Continuous (exact t_k) | Quantized (Δt = 1/f_s) |
| Distinct arrivals | All preserved | May be merged into one sample |
| Echo density | Infinite precision | Limited by sample rate |

### 3. Expected Differences

The discrete EDC will differ from the analytical EDC because:

1. **Cross-terms are positive when amplitudes have the same sign**
   - Since all image amplitudes are positive (1/(4πR_k) > 0)
   - Cross-terms 2a_i·a_j > 0 always
   - Therefore: **(Σ a_k)² ≥ Σ a_k²** (with equality only if one term exists)

2. **Higher echo density → more coincidences → larger differences**
   - Early in RIR: Fewer images, less likely to coincide
   - Late in RIR: Many images, high probability of coincidence
   - Small rooms: Higher echo density at all times

3. **Sample rate matters**
   - Higher f_s → better time resolution → fewer coincidences
   - f_s = 44100 Hz → Δt ≈ 22.7 μs
   - For sound at 343 m/s, this is ~7.8 mm spatial resolution

## Verification

The implementation includes diagnostics to check:

```python
sample_indices = np.round(times * fs).astype(int)
unique_samples, counts = np.unique(sample_indices, return_counts=True)
n_coincident = np.sum(counts > 1)
max_coincident = np.max(counts)
```

This reports:
- How many samples have multiple images
- Maximum number of images per sample
- Mean relative difference between methods

## Theoretical Bounds

For N images landing on the same sample with amplitudes {a_1, ..., a_N}:

```
Incoherent energy: E_incoh = Σ a_i²

Coherent energy: E_coh = (Σ a_i)²

Ratio: E_coh / E_incoh = (Σ a_i)² / (Σ a_i²) ≥ 1
```

By Cauchy-Schwarz inequality:
```
(Σ a_i)² ≤ N × (Σ a_i²)
```

Therefore:
```
1 ≤ E_coh / E_incoh ≤ N
```

The ratio approaches N when all amplitudes are equal (a_i = a for all i).

## Conclusion

**Both methods are correct implementations of different physical scenarios:**

1. **Analytical EDC**: Models the theoretical case where each image source arrives at a distinct time (continuous time, incoherent energy summation)

2. **Discrete EDC**: Models the practical case where the RIR is sampled at finite rate, and images arriving within the same sample period interfere coherently

**For room acoustics modeling:**
- The discrete method is what actually happens in reality when the RIR is sampled
- The analytical method provides a theoretical reference
- Differences quantify the impact of time discretization on energy distribution
- Higher sample rates reduce (but don't eliminate) the discrepancy

**Neither is "wrong" - they answer slightly different questions!**





