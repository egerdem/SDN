#!/usr/bin/env python3
"""
Compare two results.json files from Monte Carlo experiments.
Check for discrepancies in rmse_c1, c_optimal (optimal_c), and min_rmse values.
"""

import json
import sys

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_results(file1_path, file2_path):
    """Compare two results files."""
    
    # Load both files
    results1 = load_json(file1_path)
    results2 = load_json(file2_path)
    
    print(f"File 1: {file1_path}")
    print(f"  Total entries: {len(results1)}")
    print(f"\nFile 2: {file2_path}")
    print(f"  Total entries: {len(results2)}")
    print("="*80)
    
    # Create dictionaries indexed by run_id
    dict1 = {item['run_id']: item for item in results1}
    dict2 = {item['run_id']: item for item in results2}
    
    # Find common IDs
    ids1 = set(dict1.keys())
    ids2 = set(dict2.keys())
    common_ids = ids1.intersection(ids2)
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1
    
    print(f"\nCommon run_ids: {len(common_ids)}")
    print(f"Only in file 1: {len(only_in_1)}")
    if only_in_1:
        print(f"  IDs: {sorted(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")
    if only_in_2:
        print(f"  IDs: {sorted(only_in_2)}")
    
    print("\n" + "="*80)
    print("COMPARING VALUES FOR COMMON run_ids")
    print("="*80)
    
    # Compare values for common IDs
    discrepancies = []
    fields_to_compare = ['rmse_c1', 'optimal_c', 'min_rmse']
    tolerance = 1e-10  # Numerical tolerance for floating point comparison
    
    for run_id in sorted(common_ids):
        item1 = dict1[run_id]
        item2 = dict2[run_id]
        
        run_key = item1.get('run_key', f'run_{run_id}')
        
        for field in fields_to_compare:
            val1 = item1.get(field)
            val2 = item2.get(field)
            
            if val1 is None and val2 is None:
                continue
            elif val1 is None or val2 is None:
                discrepancies.append({
                    'run_id': run_id,
                    'run_key': run_key,
                    'field': field,
                    'value1': val1,
                    'value2': val2,
                    'diff': 'N/A',
                    'issue': 'Missing value'
                })
            else:
                diff = abs(val1 - val2)
                if diff > tolerance:
                    discrepancies.append({
                        'run_id': run_id,
                        'run_key': run_key,
                        'field': field,
                        'value1': val1,
                        'value2': val2,
                        'diff': diff,
                        'rel_diff_pct': 100 * diff / max(abs(val1), abs(val2)) if max(abs(val1), abs(val2)) > 0 else 0,
                        'issue': 'Value mismatch'
                    })
    
    # Report discrepancies
    if discrepancies:
        print(f"\n⚠️  FOUND {len(discrepancies)} DISCREPANCIES:\n")
        for disc in discrepancies:
            print(f"Run ID: {disc['run_id']} ({disc['run_key']})")
            print(f"  Field: {disc['field']}")
            print(f"  File 1 value: {disc['value1']}")
            print(f"  File 2 value: {disc['value2']}")
            if isinstance(disc.get('diff'), float):
                print(f"  Absolute diff: {disc['diff']:.2e}")
                print(f"  Relative diff: {disc.get('rel_diff_pct', 0):.4f}%")
            print(f"  Issue: {disc['issue']}")
            print()
    else:
        print(f"\n✅ NO DISCREPANCIES FOUND!")
        print(f"All {len(common_ids)} common entries have matching values for:")
        for field in fields_to_compare:
            print(f"  - {field}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR COMMON ENTRIES")
    print("="*80)
    
    for field in fields_to_compare:
        values1 = [dict1[rid].get(field) for rid in common_ids if dict1[rid].get(field) is not None]
        values2 = [dict2[rid].get(field) for rid in common_ids if dict2[rid].get(field) is not None]
        
        if values1 and values2:
            print(f"\n{field}:")
            print(f"  File 1: min={min(values1):.6f}, max={max(values1):.6f}, mean={sum(values1)/len(values1):.6f}")
            print(f"  File 2: min={min(values2):.6f}, max={max(values2):.6f}, mean={sum(values2)/len(values2):.6f}")
            
            # Check if all values are identical
            diffs = [abs(v1 - v2) for v1, v2 in zip(
                [dict1[rid].get(field) for rid in sorted(common_ids) if dict1[rid].get(field) is not None],
                [dict2[rid].get(field) for rid in sorted(common_ids) if dict2[rid].get(field) is not None]
            )]
            max_diff = max(diffs) if diffs else 0
            print(f"  Max absolute difference: {max_diff:.2e}")

if __name__ == "__main__":
    file1 = "/Users/ege/Projects/SDN/SDN-EGE/results/monte_carlo_experiments/10x60_seed42_0cornerplacement_big/results.json"
    file2 = "/Users/ege/Projects/SDN/SDN-EGE/results/monte_carlo_experiments/default/results.json"
    
    compare_results(file1, file2)
