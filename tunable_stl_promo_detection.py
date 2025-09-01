#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tunable STL-based Promotion Detection Script (Clean Progress Version)

This script uses Seasonal-Trend decomposition using LOESS (STL) to detect promotional
periods in retail sales time series data. It identifies unusual spikes in sales that likely
represent promotional activities.

Modified to show only clean progress bars without verbose output.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
import time
import argparse
import gc
import sys
import warnings

# Suppress pandas FutureWarnings and other warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def parse_arguments():
    """Parse command line arguments for tunable promotion detection parameters."""
    parser = argparse.ArgumentParser(description='Tunable STL-based promotion detection')
    
    # Input/output
    parser.add_argument('--input', type=str, default='sales_data_full_history_only.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='tunable_stl_promo_flagged.csv',
                       help='Path to output CSV file')
    
    # Tuning parameters
    parser.add_argument('--z-threshold', type=float, default=1.0,
                       help='Base z-score threshold (lower = more sensitive)')
    parser.add_argument('--quantile-threshold', type=float, default=0.65, 
                       help='Quantile threshold (lower = more sensitive)')
    parser.add_argument('--min-promo-jump', type=float, default=0.4,
                       help='Minimum promotion jump multiplier (lower = more sensitive)')
    parser.add_argument('--cooling-period', type=int, default=1,
                       help='Cooling period between promotions in weeks (lower = more sensitive)')
    parser.add_argument('--max-promo-rate', type=float, default=0.40,
                       help='Maximum promotion rate (higher = more sensitive)')
    
    # Verbosity control
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output (default: only progress bars)')
    
    return parser.parse_args()

def print_if_verbose(message, verbose=False):
    """Print message only if verbose mode is enabled."""
    if verbose:
        print(message)

def main():
    """Main execution function for tunable STL-based promotion detection."""
    args = parse_arguments()
    verbose = args.verbose
    
    # Show parameters only if verbose
    if verbose:
        print(f"Starting Tunable STL Promotion Detection with parameters:")
        print(f"- Z-score threshold: {args.z_threshold}")
        print(f"- Quantile threshold: {args.quantile_threshold}")
        print(f"- Min promo jump: {args.min_promo_jump}")
        print(f"- Cooling period: {args.cooling_period}")
        print(f"- Max promo rate: {args.max_promo_rate}")
    
    start_time = time.time()
    
    # Load weekly sales data with progress bar
    with tqdm(desc="Loading data", unit="MB", disable=False) as pbar:
        df = pd.read_csv(args.input)
        pbar.update(1)
    
    # Ensure proper sorting
    df.sort_values(["Country", "Forecasting Group", "Sales_week"], inplace=True)
    
    # Container for processed groups and metrics
    results = []
    metrics = []
    
    # Group by country and product
    grouped = df.groupby(["Country", "Forecasting Group"])
    total_groups = len(grouped)
    
    # Get list of all group keys for batch processing
    all_groups = list(grouped.groups.keys())
    
    # Process in batches to reduce memory usage
    batch_size = 100
    batch_count = (total_groups + batch_size - 1) // batch_size  # ceiling division
    
    # Overall progress bar for all batches
    overall_pbar = tqdm(total=total_groups, desc="Processing groups", unit="group")
    
    for batch_idx in range(batch_count):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_groups)
        
        batch_groups = all_groups[start_idx:end_idx]
        batch_results = []
        
        # Process each group in the batch
        for country, fg in batch_groups:
            group = grouped.get_group((country, fg)).copy().reset_index(drop=True)
            sales = group["Sales_volume"].values
            
            # Skip short or flat series
            #if len(sales) < 30 or np.std(sales) < 1e-2:
             #   overall_pbar.update(1)
              #  continue
                
            # Calculate long-term moving average as baseline reference
            baseline_window = min(26, len(sales) // 2)
            if baseline_window < 4:
                overall_pbar.update(1)
                continue
                
            sales_series = pd.Series(sales)
            baseline = sales_series.rolling(window=baseline_window, center=True, min_periods=1).mean()
            baseline = baseline.ffill().bfill()
            baseline = baseline.values
                
            # Step change detection (permanent shifts)
            baseline_changes = np.zeros_like(sales, dtype=bool)
            window_size = min(12, len(sales) // 3)
            if window_size >= 4:
                for i in range(window_size, len(sales) - window_size):
                    pre_window = sales[i-window_size:i]
                    post_window = sales[i:i+window_size]
                    
                    pre_mean = np.mean(pre_window)
                    post_mean = np.mean(post_window)
                    
                    if abs(post_mean - pre_mean) > 0.3 * pre_mean:
                        baseline_changes[i] = True
            
            # STL Decomposition
            try:
                stl = STL(sales_series, seasonal=13, period=52, robust=True).fit()
                seasonal = stl.seasonal
                resid = stl.resid
                trend = stl.trend
                    
            except Exception as e:
                if verbose:
                    print(f"STL error for {country}-{fg}: {e}")
                overall_pbar.update(1)
                continue
            
            # Get residuals as Series for easier manipulation
            resid_series = pd.Series(resid)
            
            # Extract tunable parameters
            z_threshold = args.z_threshold
            quantile_threshold = args.quantile_threshold
            min_promo_jump = args.min_promo_jump
            cooling_period = args.cooling_period
            max_promo_rate = args.max_promo_rate
            
            # Calculate robust z-score for each point
            median_resid = np.median(resid)
            mad = np.median(np.abs(resid - median_resid))
            mad_adj = mad * 1.4826
            robust_z = np.zeros_like(resid)
            
            if mad_adj > 1e-6:
                robust_z = (resid - median_resid) / mad_adj
            
            # Calculate sales volatility for adaptive thresholds
            sales_mean = np.mean(sales) if np.mean(sales) > 0 else 0.0001
            sales_volatility = np.std(sales) / sales_mean
            
            # Adjustable z threshold with minimum value
            z_threshold = args.z_threshold + min(1.0, sales_volatility)
            
            # Flag based on robust z-score
            flag_robust = (robust_z > z_threshold).astype(int)
            
            # Calculate rolling quantile
            window = min(26, len(resid) // 4)
            if window < 4:
                window = 4
            
            rolling_q = resid_series.rolling(window=window, center=True).quantile(quantile_threshold)
            rolling_q = rolling_q.ffill().bfill()
            
            if len(resid_series) >= window:
                rolling_q_adaptive = resid_series.rolling(
                    window=window, min_periods=3).quantile(quantile_threshold)
                flag_rolling = (resid_series > rolling_q_adaptive).astype(int)
            else:
                flag_rolling = np.zeros_like(resid_series)
            
            # Combined flagging
            promo_flag = np.zeros_like(sales)
            promo_flag = ((robust_z > z_threshold) | (resid_series > rolling_q_adaptive)).astype(int)
            
            # Use adaptive threshold based on IQR
            q75 = np.percentile(sales, 75)
            q25 = np.percentile(sales, 25)
            iqr = q75 - q25
            p75 = q75
            
            min_promo_jump = p75 + (q75 - q25) * args.min_promo_jump
            
            # Filter out flags that don't represent significant sales jumps
            # significant_promo = np.zeros_like(promo_flag)
            #size_threshold = np.median(sales) + (min_promo_jump * iqr)
            #size_filter = (sales > size_threshold).astype(int)
            #significant_promo = promo_flag & size_filter
            significant_promo = promo_flag
            
            # Apply cooling period
            cooling_period = args.cooling_period
            for i in range(1, len(significant_promo)):
                if significant_promo[i] == 1:
                    if i >= cooling_period:
                        if np.any(significant_promo[i-cooling_period:i] == 1):
                            significant_promo[i] = 0
            
            # Apply maximum promotion rate
            max_promo_rate = args.max_promo_rate
            if sum(significant_promo) / len(sales) > max_promo_rate:
                promo_strength = np.zeros_like(significant_promo, dtype=float)
                promo_indices = np.where(significant_promo == 1)[0]
                
                for idx in promo_indices:
                    promo_strength[idx] = robust_z[idx]
                
                max_promos = int(len(sales) * max_promo_rate)
                sorted_indices = promo_indices[np.argsort(-promo_strength[promo_indices])]
                
                significant_promo = np.zeros_like(significant_promo)
                for idx in sorted_indices[:max_promos]:
                    significant_promo[idx] = 1
            
            # Calculate confidence scores
            confidence = np.zeros_like(sales, dtype=float)
            
            for i in np.where(significant_promo == 1)[0]:
                z_conf = min(100, max(0, (robust_z[i] - z_threshold) / 3 * 100))
                
                if p75 > 0:
                    size_conf = min(100, max(0, (sales[i] - p75) / p75 * 100))
                else:
                    size_conf = 0
                
                recent_window = min(8, i)
                if recent_window > 0:
                    recent_sales = sales[max(0, i-recent_window):i]
                    if len(recent_sales) > 0 and np.mean(recent_sales) > 0:
                        history_conf = min(100, max(0, (sales[i] / np.mean(recent_sales) - 1) * 100))
                    else:
                        history_conf = 0
                else:
                    history_conf = 0
                
                confidence[i] = 0.4*z_conf + 0.4*size_conf + 0.2*history_conf
            
            # Add detection results to group dataframe
            group["LikelyPromo"] = significant_promo
            group["PromoConfidence"] = confidence.round(1)
            
            # Save metrics
            total_promo_weeks = significant_promo.sum()
            promo_rate = total_promo_weeks / len(significant_promo) if len(significant_promo) > 0 else 0
            
            metrics.append({
                'Country': country,
                'FG': fg,
                'DataPoints': len(sales),
                'PromoWeeks': total_promo_weeks,
                'PromoRate': promo_rate,
                'AvgConfidence': confidence[confidence > 0].mean() if np.any(confidence > 0) else 0
            })
            
            batch_results.append(group)
            overall_pbar.update(1)
        
        # Combine batch results and release memory
        if batch_results:
            partial_df = pd.concat(batch_results)
            results.append(partial_df)
            del batch_results
            gc.collect()
    
    overall_pbar.close()
    
    # Combine all batches and save
    if results:
        # Final progress bar for saving
        with tqdm(desc="Saving results", total=2, unit="file") as save_pbar:
            promo_df = pd.concat(results)
            promo_df.sort_values(["Country", "Forecasting Group", "Sales_week"], inplace=True)
            
            # Save main results
            output_file = args.output
            promo_df.to_csv(output_file, index=False)
            save_pbar.update(1)
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics)
            metrics_file = "tunable_promo_detection_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            save_pbar.update(1)
        
        # Summary (always shown, but compact)
        execution_time = time.time() - start_time
        avg_promo_rate = metrics_df['PromoRate'].mean() if len(metrics_df) > 0 else 0
        avg_confidence = metrics_df['AvgConfidence'].mean() if len(metrics_df) > 0 else 0
        
        print(f"\n✓ Completed in {execution_time:.1f}s | "
              f"Groups: {len(metrics_df)} | "
              f"Avg promo rate: {avg_promo_rate:.2%} | "
              f"Avg confidence: {avg_confidence:.1f}/100")
        print(f"Files saved: {output_file}, {metrics_file}")
        
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()