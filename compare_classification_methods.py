#!/usr/bin/env python3
"""
Comparison script for different classification methods.
Runs rule-based, supervised ML, and unsupervised ML classification on the same dataset
and compares the results.
"""

import os
import sys
import pandas as pd
import time
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from logghostbuster.main import run_bot_annotator
from logghostbuster.utils import logger


def compare_methods(input_parquet, sample_size=None, base_output_dir='output/comparison'):
    """
    Compare all three classification methods on the same dataset.
    
    Args:
        input_parquet: Path to input parquet file
        sample_size: Number of records to sample (None = use all data)
        base_output_dir: Base directory for output files
    """
    methods = ['rules', 'ml', 'deep']
    results = {}
    
    logger.info("=" * 80)
    logger.info("CLASSIFICATION METHODS COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Input file: {input_parquet}")
    if sample_size:
        logger.info(f"Sample size: {sample_size:,} records")
    else:
        logger.info("Sample size: ALL DATA (no sampling)")
    logger.info(f"Methods to compare: {', '.join(methods)}")
    logger.info("=" * 80)
    
    for method in methods:
        logger.info("\n" + "=" * 80)
        logger.info(f"Running {method.upper()} classification method")
        logger.info("=" * 80)
        
        output_dir = os.path.join(base_output_dir, method)
        start_time = time.time()
        
        try:
            # Create a temporary output file for each method to avoid conflicts
            output_parquet = os.path.join(output_dir, f'annotated_{method}.parquet')
            
            result = run_bot_annotator(
                input_parquet=input_parquet,
                output_parquet=output_parquet,
                output_dir=output_dir,
                contamination=0.15,
                compute_importances=False,
                sample_size=sample_size,
                classification_method=method,
                annotate=False  # Do not write large annotated parquet files during comparison
            )
            
            elapsed_time = time.time() - start_time
            
            results[method] = {
                'stats': result['stats'],
                'bot_locations': result['bot_locations'],
                'hub_locations': result['hub_locations'],
                'elapsed_time': elapsed_time,
                'output_dir': output_dir
            }
            
            logger.info(f"\n{method.upper()} completed in {elapsed_time:.2f} seconds")
            logger.info(f"  Bot locations: {result['bot_locations']:,}")
            logger.info(f"  Hub locations: {result['hub_locations']:,}")
            logger.info(f"  Bot downloads: {result['stats']['bots']:,} ({result['stats']['bots']/result['stats']['total']*100:.2f}%)")
            logger.info(f"  Hub downloads: {result['stats']['hubs']:,} ({result['stats']['hubs']/result['stats']['total']*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error running {method}: {e}", exc_info=True)
            results[method] = {'error': str(e)}
    
    # Generate comparison report
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    comparison_file = os.path.join(base_output_dir, 'comparison_report.txt')
    with open(comparison_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLASSIFICATION METHODS COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_parquet}\n")
        if sample_size:
            f.write(f"Sample size: {sample_size:,} records\n")
        else:
            f.write("Sample size: ALL DATA (no sampling)\n")

        # Compute total downloads from the first successful method.
        # This uses the algorithm output (stats['total']), ensuring that
        # the reported downloads match the classification results.
        total_downloads = None
        for method in methods:
            if 'error' not in results.get(method, {}):
                total_downloads = results[method]['stats']['total']
                break

        if total_downloads is not None:
            f.write(f"Total downloads (from analysis): {total_downloads:,}\n\n")
        else:
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Create comparison table
        f.write(f"{'Method':<20} {'Bot Locs':<12} {'Hub Locs':<12} {'Bot DLs':<15} {'Hub DLs':<15} {'Time (s)':<12}\n")
        f.write("-" * 80 + "\n")
        
        for method in methods:
            if 'error' not in results[method]:
                r = results[method]
                bot_pct = r['stats']['bots'] / r['stats']['total'] * 100
                hub_pct = r['stats']['hubs'] / r['stats']['total'] * 100
                f.write(f"{method:<20} {r['bot_locations']:<12,} {r['hub_locations']:<12,} "
                       f"{r['stats']['bots']:<12,} ({bot_pct:>5.2f}%) "
                       f"{r['stats']['hubs']:<12,} ({hub_pct:>5.2f}%) "
                       f"{r['elapsed_time']:<12.2f}\n")
            else:
                f.write(f"{method:<20} ERROR: {results[method]['error']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        for method in methods:
            if 'error' not in results[method]:
                r = results[method]
                f.write(f"\n{method.upper()} METHOD:\n")
                f.write(f"  Execution time: {r['elapsed_time']:.2f} seconds\n")
                f.write(f"  Bot locations: {r['bot_locations']:,}\n")
                f.write(f"  Hub locations: {r['hub_locations']:,}\n")
                f.write(f"  Total downloads: {r['stats']['total']:,}\n")
                f.write(f"  Bot downloads: {r['stats']['bots']:,} ({r['stats']['bots']/r['stats']['total']*100:.2f}%)\n")
                f.write(f"  Hub downloads: {r['stats']['hubs']:,} ({r['stats']['hubs']/r['stats']['total']*100:.2f}%)\n")
                f.write(f"  Normal downloads: {r['stats']['normal']:,} ({r['stats']['normal']/r['stats']['total']*100:.2f}%)\n")
                f.write(f"  Output directory: {r['output_dir']}\n")
        
        # Calculate differences
        if all('error' not in results[m] for m in methods):
            f.write("\n" + "=" * 80 + "\n")
            f.write("DIFFERENCES BETWEEN METHODS\n")
            f.write("=" * 80 + "\n\n")
            
            rules_bots = results['rules']['bot_locations']
            rules_hubs = results['rules']['hub_locations']
            
            for method in ['ml', 'deep']:
                if 'error' not in results[method]:
                    ml_bots = results[method]['bot_locations']
                    ml_hubs = results[method]['hub_locations']
                    
                    bot_diff = ml_bots - rules_bots
                    hub_diff = ml_hubs - rules_hubs
                    
                    f.write(f"\n{method.upper()} vs RULES:\n")
                    f.write(f"  Bot locations difference: {bot_diff:+,} ({bot_diff/rules_bots*100:+.2f}%)\n")
                    f.write(f"  Hub locations difference: {hub_diff:+,} ({hub_diff/rules_hubs*100:+.2f}%)\n")
    
    logger.info(f"\nComparison report saved to: {comparison_file}")
    
    # Also create a CSV comparison
    comparison_csv = os.path.join(base_output_dir, 'comparison_results.csv')
    comparison_data = []
    for method in methods:
        if 'error' not in results[method]:
            r = results[method]
            comparison_data.append({
                'method': method,
                'bot_locations': r['bot_locations'],
                'hub_locations': r['hub_locations'],
                'total_downloads': r['stats']['total'],
                'bot_downloads': r['stats']['bots'],
                'hub_downloads': r['stats']['hubs'],
                'normal_downloads': r['stats']['normal'],
                'bot_downloads_pct': r['stats']['bots'] / r['stats']['total'] * 100,
                'hub_downloads_pct': r['stats']['hubs'] / r['stats']['total'] * 100,
                'normal_downloads_pct': r['stats']['normal'] / r['stats']['total'] * 100,
                'elapsed_time_seconds': r['elapsed_time']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info(f"Comparison CSV saved to: {comparison_csv}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare classification methods (rules, supervised ML, unsupervised ML)'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input parquet file')
    parser.add_argument('--sample-size', '-s', type=int, default=None,
                       help='Number of records to sample (default: None = use all data)')
    parser.add_argument('--output-dir', '-o', default='output/comparison',
                       help='Base output directory (default: output/comparison)')
    
    args = parser.parse_args()
    
    compare_methods(
        input_parquet=args.input,
        sample_size=args.sample_size,
        base_output_dir=args.output_dir
    )

