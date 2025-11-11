
## 📁 Project Structure

```
SDN-EGE/
├── main.py              # Main simulation runner with interactive RIR/EDC/NED plots
├── sdn_core.py          # Source-weighted SDN implementation (DelayNetwork class)
├── geometry.py          # Room geometry, walls, nodes, image sources
├── rir_calculators.py   # RIR calculation wrappers (SDN, HO-SDN, ISM-pra/rimpy/manual)
├── analysis/            # Utility scripts used by sdn_core.py
├── research/            # Paper figure generation and optimization experiments
├── archive/             # sdn_base.py (orchi's) and sdn_timu (timucins base sdn), and  compare_sdn_implementation.py: initially needed to align timu-orchi-ege
├── gui/                 # Dash/Plotly interactive GUI (no readme yet)
```

## 🎯 Core Files

### `main.py`
Main entry point. Configure flags to run:
- **SDN variants:** `RUN_SDN_Test1-7` (c=1 to c=7 source weighting)
- **HO-SW-SDN:** `RUN_MY_HO_SDN_n2_swc5` (HO SDN with source weighting)
- **ISM-PRA:** `PLOT_ISM_with_pra` (PyRoomAcoustics ISM)
- **ISM-rimPy:** `PLOT_ISM_rimPy_neg` (rimPy ISM implementation)
- **HO-SDN-WASPAA:** `RUN_HO_N2`, `RUN_HO_N3` (reference WASPAA implementation)

### `sdn_core.py`
Core SDN implementation featuring many flags for:
- Source-weighted injection (specular vs. diffuse control)
- HO-SDN support (higher-order reflections)
- Other scattering matrix trials

### `geometry.py`
Room setup: walls, nodes (scattering points), image sources, reflection coefficients.

### `rir_calculators.py`
Wrapper functions for calculating RIRs from different methods.

## 📊 Analysis Tools (`analysis/`)

- **`analysis.py`** - RT60, EDC, ERR, C50/C80 metrics
- **`plot_room.py`** - Interactive RIR/EDC/NED plots, 3D room visualization
- **`path_tracker.py`** - ISM vs SDN path length comparison
- **`frequency.py`** - Frequency response, spectral analysis
- **`EchoDensity.py`** - Normalized echo density calculations
- **`spatial_analysis.py`** - Position sweep experiments

## 🔬 Research Scripts (`research/`)

- **`paper_figures*.py`** - Generate publication figures
- **`optimisation_*.py`** - Optimize source weighting coefficients
- **`pressure_drop_analysis*.py`** - Investigate how pressure drops step by step
- **`tables.py`** - Generate comparison tables

## 📚 Archive (`archive/`)

Legacy reference implementations for validation:
- **`sdn_base.py`** - Orchi's SDN implementation
- **`SDN_timu.py`** - Timucin's base SDN (uncoupled)
- **`compare_sdn_implementations.py`** - Initial alignment validation (Timu-Orchi-Ege)
- **`SDN_algo3/`** - Third-party reference implementation

## 🖥️ GUI (`gui/`)

Experimental Dash/Plotly interface for experiment management. No documentation yet.

## 🔧 Key Configuration

Edit `main.py` to select room and methods:

# Room selection
room_parameters = room_aes  # or room_waspaa, room_journal

# Enable methods
RUN_SDN_Test5 = True           # SDN with sw sinjection c=5
PLOT_ISM_rimPy_neg = True      # ISM rimpy with negative reflection coef (reference)
RUN_MY_HO_SDN_n2_swc5 = True   # SW-HO-SDN order 2 with sw injection c=5

## 📝 Research Focus

Investigating why SDN exhibits energy drop after first-order reflections despite good perceptual performance. Comparing path lengths, energy distribution, and echo density between SDN and ISM methods, different injection and scattering variants

