# SDN-EGE Project Structure

This document explains the organization of the SDN-ISM research codebase.

## 📁 Directory Structure

```
SDN-EGE/
├── Core Files (Root Directory)
│   ├── main.py                    # Main simulation entry point
│   ├── sdn_core.py                # Core SDN implementation (DelayNetwork class)
│   ├── geometry.py                # Room geometry, walls, nodes, image sources
│   ├── rir_calculators.py         # RIR calculation functions (SDN, ISM, HO-SDN)
│   ├── room.py                    # Room class and utilities
│   ├── ISM_with_pra.py            # ISM using PyRoomAcoustics
│   └── run_HO_core.py             # Higher-order SDN runner
│
├── analysis/                   # Analysis & Visualization Tools
│   ├── analysis.py                # RT60, EDC, ERR, C50/C80 metrics
│   ├── frequency.py               # Frequency response analysis
│   ├── EchoDensity.py             # Normalized echo density calculations
│   ├── plot_room.py               # 3D room visualization, RIR plotting
│   ├── plotting_utils.py          # Plotting utilities and helpers
│   ├── path_tracker.py            # Path tracking for ISM/SDN comparison
│   ├── sdn_path_calculator.py     # Path calculation utilities (ISM vs SDN)
│   └── spatial_analysis.py        # Spatial grid analysis, position sweeps
│
├── research/                   # Research & Paper Generation
│   ├── paper_figures*.py          # Generate figures for papers
│   ├── tables.py                  # Generate comparison tables
│   ├── optimisation_*.py          # Optimization experiments
│   ├── pressure_drop_analysis*.py # Energy drop investigation
│   ├── reflection_times.py        # Reflection timing analysis
│   ├── specular_scattering_matrix.py # Specular scattering calculations
│   └── generate_paper_data.py     # Generate datasets for publications
│
├── gui/                       # GUI Applications
│   ├── sdn_experiment_manager.py  # Experiment configuration interface
│   ├── sdn_experiment_visualizer.py # Interactive result visualization
│   └── sdn_manager_load_sims.py   # Load and manage saved simulations
│
├── archive/                    # Legacy Reference Implementations
│   ├── SDN_timu.py                # Original deque-based SDN (Timu's version)
│   ├── sdn_base.py                # Wrapper for SDN_algo3
│   ├── compare_sdn_implementations.py # Compare different SDN versions
│   ├── ISM_core.py                # Early ISM implementation
│   └── SDN_algo3/                 # Third-party SDN implementation
│
├── deprecated/                # Deprecated Code (Not Used)
│   ├── path_logger.py             # Old path logging (replaced by path_tracker)
│   ├── ISM.py, ISM_manual.py      # Old ISM implementations
│   ├── dsp.py, processor.py       # Unused signal processing utilities
│   ├── scratch*.py                # Development scratch files
│   └── test_sdn_ege.py            # Old test file
│
├── docs/                       # Documentation
│   ├── journal.md                 # Research journal and notes
│   ├── SDN_algorithm_explanation.md # Algorithm documentation
│   ├── how_scatteringmatrix.md    # Scattering matrix explanation
│   ├── bug_fixes_summary.md       # Record of bug fixes
│   └── energy_drop_report.pdf     # Research findings
│
├── results/                    # Simulation Results & Data
│   ├── *.pkl                      # Pickled RIR datasets
│   ├── *.npy                      # NumPy array data
│   ├── paper_data/                # Data used in papers
│   └── paper_figures/             # Generated figures
│
├── wavs/                       # Audio Files (RIR outputs)
├── treble/                     # Treble SDK integration scripts
└── SDN-Simplest_Hybrid_HO-SDN/ # External HO-SDN reference implementation

```
