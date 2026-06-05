# Monte Carlo Configuration Update

## Changes Made

Updated `research/monte_carlo_proximity.py` with new configuration:

### New Parameters (legacy_pair_single_mic preset):

```python
"NUM_ROOMS": 10                    # ← Changed from 4
"PAIRS_PER_ROOM": 8                # ← Changed from 20
"CORNER_BIAS_PROBABILITY": 0.25    # ← NEW: 25% corner bias (was 50%)
"RIMPY_SEED": 42                   # ← NEW: Fixed seed for reproducibility
```

### Room Ranges (updated):
```python
"WIDTH_RANGE": (4.0, 8.0)          # ← Changed from (3.0, 9.5)
"DEPTH_RANGE": (4.0, 8.0)          # ← Changed from (4.0, 7.5)
"HEIGHT_RANGE": (2.5, 5.0)         # ← Changed from (2.6, 4.2)
```

---

## Total Configurations

- **10 rooms** × **8 pairs** = **80 total configurations** (same as before)
- More rooms, fewer pairs per room
- Better diversity across different acoustic spaces

---

## What is Corner Bias?

**Corner bias** controls how sources are positioned in the room to ensure diverse wall-proximity conditions.

### Two Placement Modes:

1. **Corner-biased placement** (probability = 25%):
   - Source is placed near a randomly chosen corner
   - Distance from corner walls: **0.1m to 1.5m** (randomly chosen)
   - Example: If corner is (0, 0, 0), source might be at (0.8, 1.2, 0.4)
   - This ensures sampling of **high wall-proximity** (low H_src_norm)

2. **Uniform placement** (probability = 75%):
   - Source is uniformly distributed in the room
   - Minimum distance from walls: **0.5m** (padding)
   - This provides **diverse wall-proximity** conditions

### Why Corner Bias?

Corner bias ensures adequate sampling of sources close to boundaries, where:
- H_src_norm is **low** (high wall proximity)
- SDN behavior is most sensitive to C parameter
- Without corner bias, most sources would be mid-room (less interesting geometrically)

### Visual Example:

```
Room (top view):
┌─────────────────────────┐
│ 0.1-1.5m                │  ← Corner-biased zone (25% chance)
│  [C]                    │
│                         │
│         0.5m padding    │  ← Uniform zone (75% chance)
│      ┌──────────┐       │
│      │          │       │
│      │   [U]    │       │
│      │          │       │
│      └──────────┘       │
│                   [C]   │  ← Corner-biased zone
│                1.5m max │
└─────────────────────────┘

[C] = Corner-biased placement (0.1-1.5m from corners)
[U] = Uniform placement (>0.5m from all walls)
```

---

## Receivers

**Receivers remain uniformly distributed** (no bias):
- Minimum distance from walls: **0.5m** (padding)
- Equal probability anywhere in the valid volume

---

## RIMPY Seed = 42

**Fixed seed for RIMPY RIR generation**:
- Makes randomized image source positions **reproducible**
- Anyone can regenerate exact same RIRs with same config
- Important for:
  - Paper submissions (reviewers can verify)
  - Comparing algorithm changes
  - Debugging

**Note**: Room dimensions and source/receiver positions already use `np.random.seed(42)` (line 590), so they are deterministic. This seed applies specifically to RIMPY's internal randomization.

---

## Run the New Configuration

```bash
# Optional: Set experiment ID
export MC_EXPERIMENT_ID=seed_42_10rooms_8pairs

# Generate Monte Carlo data
python research/monte_carlo_proximity.py

# This creates:
# results/monte_carlo_experiments/<ID>/
#   ├── results.json (80 configurations: 10 rooms × 8 pairs)
#   └── rimpy_cache/ (RIMPY RIRs with seed=42)
```

---

## Summary Statistics

**Previous config**:
- 4 rooms × 20 pairs = 80 configs
- Corner bias: 50% (implicit)
- RIMPY seed: unspecified

**New config**:
- 10 rooms × 8 pairs = 80 configs
- Corner bias: **25%** (explicit)
- RIMPY seed: **42** (reproducible)
- Better room diversity (10 vs 4)
- More focused pairs (8 vs 20)

---

## For Journal Paper

**Sampling strategy**:
> Source positions were sampled using a corner-biased distribution (25% probability of placement within 0.1-1.5 m of room corners, 75% uniform distribution with 0.5 m boundary padding) to ensure adequate representation of high wall-proximity conditions. Receiver positions were uniformly distributed with 0.5 m boundary padding.

**Reproducibility**:
> All room dimensions and transducer positions were generated using a fixed random seed (seed=42). Reference RIRs were computed using the randomized image source method (RIMPY) with 10% spatial perturbation, negative phase distribution, and fixed randomization seed (seed=42) to ensure reproducibility.

