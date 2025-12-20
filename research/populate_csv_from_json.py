"""
Populate aes_pairwise_c_analysis.csv with optimal_c values from results.json

This script reads the optimal_c values from monte_carlo_proximity.py's results.json
and updates the c_opt_pair and rmse_opt_pair columns in the CSV file.

Usage:
    python3 research/populate_csv_from_json.py
"""

import os
import sys
import json
import csv

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configuration
EXPERIMENT_DIR = os.path.join(
    project_root,
    "results",
    "paper_data",
    "experiments",
    # "aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair",
    "aes_fullgrid_8src_diagonal"
)

JSON_PATH = os.path.join(EXPERIMENT_DIR, "results.json")
CSV_PATH = os.path.join(EXPERIMENT_DIR, "aes_pairwise_c_analysis.csv")
OUTPUT_CSV_PATH = CSV_PATH  # Overwrite in place

# Outlier rejection: if c_opt hits bounds (1 or 7), check if c=1 is better
APPLY_OUTLIER_REJECTION = True
C_BOUNDS = (1.0, 7.0)  # Must match the bounds used in monte_carlo_proximity.py


def main():
    print("=" * 80)
    print("Populate CSV with optimal_c from results.json")
    if APPLY_OUTLIER_REJECTION:
        print("(with outlier rejection enabled)")
    print("=" * 80)
    
    # Load JSON data
    if not os.path.exists(JSON_PATH):
        print(f"\n❌ ERROR: JSON file not found: {JSON_PATH}")
        return
    
    print(f"\nLoading JSON: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
    
    print(f"  Loaded {len(json_data)} entries")
    
    # Create lookup: (src_idx, rx_idx) -> optimal_c, min_rmse
    lookup = {}
    for entry in json_data:
        src_idx = entry.get("src_idx")
        rx_idx = entry.get("rx_idx")
        optimal_c = entry.get("optimal_c")
        min_rmse = entry.get("min_rmse")
        
        if src_idx is not None and rx_idx is not None:
            lookup[(src_idx, rx_idx)] = {
                "optimal_c": optimal_c,
                "min_rmse": min_rmse
            }
    
    print(f"  Created lookup table with {len(lookup)} entries")
    
    # Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: CSV file not found: {CSV_PATH}")
        return
    
    print(f"\nLoading CSV: {CSV_PATH}")
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    print(f"  Loaded {len(rows)} rows")
    
    # Update rows
    updated_count = 0
    missing_count = 0
    outlier_rejected_count = 0
    
    for row in rows:
        # Extract source index from filename
        # e.g., "aes_fullgrid_perpair_srcgrid3x5_corner2.0_per_pair_source_1.npz" -> 0 (0-indexed)
        filename = row["file"]
        if "source_" in filename:
            src_str = filename.split("source_")[1].split(".")[0]
            src_idx = int(src_str) - 1  # Convert to 0-indexed
        else:
            print(f"  ⚠️  Warning: Cannot extract source index from {filename}")
            continue
        
        rx_idx = int(row["rx_idx"])
        
        # Lookup optimal_c
        key = (src_idx, rx_idx)
        if key in lookup:
            c_opt = lookup[key]["optimal_c"]
            rmse_opt = lookup[key]["min_rmse"]
            rmse_orig = float(row["rmse_orig_c1"])
            
            # Apply outlier rejection if enabled
            if APPLY_OUTLIER_REJECTION:
                lower, upper = C_BOUNDS
                eps = 5e-3
                hit_upper = abs(c_opt - upper) <= eps
                hit_lower = abs(c_opt - lower) <= eps
                
                if hit_upper or hit_lower:
                    # Optimizer hit a bound - check if c=1 is better
                    if rmse_orig <= rmse_opt + 1e-6:
                        # c=1 is better (or equal), use it instead
                        c_opt = 1.0
                        rmse_opt = rmse_orig
                        outlier_rejected_count += 1
            
            row["c_opt_pair"] = c_opt
            row["rmse_opt_pair"] = rmse_opt
            
            # Compute improvement
            row["rmse_improvement"] = rmse_orig - rmse_opt
            
            updated_count += 1
        else:
            missing_count += 1
            print(f"  ⚠️  Warning: No data for src_idx={src_idx}, rx_idx={rx_idx}")
    
    print(f"\n  Updated {updated_count} rows")
    if outlier_rejected_count > 0:
        print(f"  🔧 Applied outlier rejection to {outlier_rejected_count} rows (c_opt hit bounds, c=1 was better)")
    if missing_count > 0:
        print(f"  ⚠️  {missing_count} rows missing data")
    
    # Write updated CSV
    print(f"\nWriting updated CSV: {OUTPUT_CSV_PATH}")
    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"  ✅ Saved {len(rows)} rows")
    
    # Show sample
    print("\n" + "=" * 80)
    print("Sample of updated data (first 3 rows):")
    print("=" * 80)
    for i, row in enumerate(rows[:3]):
        print(f"\nRow {i+1}:")
        print(f"  File: {row['file']}")
        print(f"  Source: {row['source_tag']}, Receiver: {row['rx_idx']}")
        print(f"  RMSE(c=1): {row['rmse_orig_c1']}")
        print(f"  c_opt: {row['c_opt_pair']}")
        print(f"  RMSE(c_opt): {row['rmse_opt_pair']}")
        print(f"  Improvement: {row['rmse_improvement']}")
    
    print("\n🎉 CSV successfully updated!")


if __name__ == "__main__":
    main()
