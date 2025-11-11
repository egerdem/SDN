

For Treble Simulations:
Run and Download Treble batch simulations:
- Run multi-location experiments: @treble.py
  - Creates multiple source positions via create_sources()
  - Generates receiver grid via generate_receiver_grid()
  - Downloads to ./results/treble/multi_experiments/{project_name}/

Process and export results in our format:
- @treble_process_data_isolated.py with IS_SINGULAR = False
  - Processes each source-receiver combination
  - Saves to hierarchical structure:
    ./results/rooms/{room_name}/{source_label}/TREBLE/{project_name}_{param_set}/
      ├── config.json  # Contains all receivers for this source
      └── rirs.npy    # All RIRs for this source-receiver set


For SDN and ISM-PRA Simulations:
Run batch simulations via @sdn_experiment_manager.py:
- Generate source & receiver positions:
  receiver_positions = sa.generate_receiver_grid(...)
  source_positions = sa.generate_source_positions(...)

- Run batch experiments:
  batch_manager = get_batch_manager(results_dir)
  batch_manager.run_experiment(
      config={...},
      batch_processing=True,
      source_positions=source_positions,
      receiver_positions=receiver_positions
  )

Saves to:
./results/rooms/{room_name}/{source_label}/{METHOD}/{param_set}/
  ├── config.json  # Contains all receivers for this source
  └── rirs.npy    # All RIRs for this source-receiver set



Directory Structure for Batch Simulations:
./results/
    ├── treble/
    │   └── multi_experiments/           # Raw Treble batch results
    │       └── {project_name}/
    │           ├── simulation_info.json
    │           └── *.h5
    │
    └── rooms/                          # Processed results (both Treble and SDN/ISM)
        └── room_aes/
            ├── room_info.json
            └── {source_label}/         # e.g., "Center_Source", "Lower_Left_Source"
                ├── SDN/
                │   └── {param_set}/    # e.g., "sw5_si_smu0.7"
                │       ├── config.json
                │       └── rirs.npy
                ├── ISM/
                │   └── {param_set}/    # e.g., "order12_rt"
                └── TREBLE/
                    └── {project_name}_{param_set}/  # e.g., "aes_abs20_hybrid"