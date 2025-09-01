#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Cluster Evaluation Script - FIXED VERSION

This script evaluates the quality and business interpretability of promotion clusters
identified by the clustering_promos.py script. It computes statistical metrics,
business insights, and generates comprehensive reports.

Key Features:
- Loads clustered promotion data with automatic column detection
- Computes silhouette score and other cluster quality metrics
- Analyzes promotional uplift and business impact by cluster
- Generates markdown reports and CSV exports
- Handles missing data and different column name formats

Input:
- tunable_promo_clustered.csv: Output from clustering_promos.py with cluster assignments
- sales_data_train.csv: Full sales dataset for baseline calculations

Usage:
    python cluster_evaluation.py
"""

import pandas as pd
import numpy as np
import gower
from sklearn.metrics import silhouette_score
from tabulate import tabulate
import warnings
import os.path
from pandas.api.types import is_numeric_dtype


def load_clustered_data():
    """
    Load the clustered promotion data with automatic column detection.
    
    Returns:
        pd.DataFrame: DataFrame containing promotion records with cluster assignments
    """
    input_file = "tunable_promo_clustered.csv"
    print(f"Loading clustered promotion data from {input_file}...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Clustered data file '{input_file}' not found! Run clustering_promos.py first.")
        
    df = pd.read_csv(input_file)
    
    # Auto-detect column names (handle variations)
    column_mapping = {}
    for col in df.columns:
        if 'forecast' in col.lower():
            column_mapping['product_id'] = col
        elif col.lower() in ['country', 'state_id']:
            column_mapping['country'] = col
        elif 'cluster' in col.lower():
            column_mapping['cluster'] = col
        elif 'promo' in col.lower() and 'likely' in col.lower():
            column_mapping['promo_flag'] = col
    
    print(f"Detected columns: {column_mapping}")
    print(f"Loaded {len(df):,} promotion records")
    
    # Basic cluster summary
    if 'cluster' in column_mapping:
        cluster_col = column_mapping['cluster']
        cluster_counts = df[cluster_col].value_counts(dropna=False)
        print(f"Found {len(cluster_counts)} clusters (including noise)")
        print(f"Cluster distribution:\n{cluster_counts}")
    
    return df

def load_full_data():
    """
    Load the full sales dataset for baseline calculations.
    
    Returns:
        pd.DataFrame: Full sales dataset
    """
    input_file = "sales_data_train.csv"
    print(f"Loading full sales data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found. Uplift calculations will be limited.")
        return None
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} total sales records")
    
    return df

def compute_silhouette(df, features):
    """
    Compute silhouette score for the promotion clusters using Gower distance.
    
    Args:
        df (pd.DataFrame): Clustered promotion data
        features (list): Features to use for distance calculation
        
    Returns:
        float: Silhouette score (excluding noise points)
    """
    print("\n--- Computing Silhouette Score ---")
    print(f"Using features for evaluation: {', '.join(features)}")
    
    # Auto-detect cluster column
    cluster_col = None
    for col in df.columns:
        if 'cluster' in col.lower():
            cluster_col = col
            break
    
    if cluster_col is None:
        print("Error: No cluster column found!")
        return None
    
    # Prepare data for Gower distance
    df_copy = df.copy()
    
    # Ensure features exist
    available_features = [f for f in features if f in df_copy.columns]
    if len(available_features) < len(features):
        missing = [f for f in features if f not in df_copy.columns]
        print(f"Warning: Missing features {missing}, using available: {available_features}")
        features = available_features
    
    if len(features) == 0:
        print("Error: No features available for clustering evaluation")
        return None
    
    # Convert data types for Gower distance
    for col in features:
        if df_copy[col].dtype.name.startswith("UInt") or df_copy[col].dtype.name.startswith("Int"):
            df_copy[col] = df_copy[col].astype(float)
        elif df_copy[col].dtype.name == "category":
            df_copy[col] = df_copy[col].astype(object)
    
    try:
        print(f"Computing distance matrix for {len(df_copy):,} records using features: {features}")
        
        # Sample data if too large (for memory efficiency)
        if len(df_copy) > 5000:
            print(f"Sampling 5000 records from {len(df_copy)} for silhouette calculation...")
            df_sample = df_copy.sample(n=5000, random_state=42)
        else:
            df_sample = df_copy
            
        gower_dist = gower.gower_matrix(df_sample[features])
        
        # Get cluster labels and exclude noise (-1)
        labels = df_sample[cluster_col].values
        mask = labels != -1  # Exclude noise points
        
        if mask.sum() < 2:
            print("Error: Too few non-noise points for silhouette calculation")
            return None
            
        # Compute silhouette score
        score = silhouette_score(gower_dist[mask][:, mask], labels[mask], metric="precomputed")
        print(f"✅ Silhouette Score (excluding noise): {round(score, 4)} — {'good' if score > 0.5 else 'fair'} cluster separation")
        return score
        
    except Exception as e:
        print(f"Error computing silhouette score: {e}")
        return None

def compute_cluster_profiles(df, full_df=None):
    """
    Compute detailed cluster profiles with business metrics.
    """

    print("\n--- Computing Cluster Profiles ---")

    # --- Helpers -------------------------------------------------------------
    def _pick_volume_col(_df):
        # Prefer exact/common numeric names first
        preferred = ["Sales_volume", "sales_volume", "Volume", "volume", "Units", "units", "Qty", "qty"]
        for col in preferred:
            if col in _df.columns and is_numeric_dtype(_df[col]):
                return col

        # Fallback: any column that mentions volume/units but is numeric,
        # explicitly skip date/week-like columns
        for col in _df.columns:
            low = col.lower()
            if any(k in low for k in ["volume", "units", "qty", "sales_volume"]):
                if any(bad in low for bad in ["week", "date", "iso", "dt"]):
                    continue
                if is_numeric_dtype(_df[col]):
                    return col
        return None

    def _pick_product_col(_df):
        # Prefer your screenshot column, then common alternatives
        if "Forecasting Group" in _df.columns:
            return "Forecasting Group"
        for cand in ["product_id", "item_id", "Product", "Item", "SKU", "sku", "forecasting_group"]:
            if cand in _df.columns:
                return cand
        return None

    def _pick_country_col(_df):
        for cand in ["Country", "country", "state_id", "State_id", "state"]:
            if cand in _df.columns:
                return cand
        return None

    def _pick_cluster_col(_df):
        # Prefer your screenshot column name
        if "promo_cluster" in _df.columns:
            return "promo_cluster"
        for c in _df.columns:
            if "cluster" in c.lower():
                return c
        return None

    # --- Detect columns on promo DF -----------------------------------------
    cluster_col  = _pick_cluster_col(df)
    product_col  = _pick_product_col(df)
    country_col  = _pick_country_col(df)
    volume_col   = _pick_volume_col(df)

    print(f"Using columns - Cluster: {cluster_col}, Product: {product_col}, Country: {country_col}, Volume: {volume_col}")

    if cluster_col is None:
        print("Error: No cluster column found!")
        return pd.DataFrame()

    # --- Basic cluster statistics -------------------------------------------
    cluster_stats = []

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]

        stats = {
            "Cluster": cluster_id,
            "Count": len(cluster_data),
            "Percentage": 100.0 * len(cluster_data) / len(df)
        }

        # Volume stats (safe numeric cast)
        if volume_col and volume_col in cluster_data.columns:
            v = pd.to_numeric(cluster_data[volume_col], errors="coerce")
            stats["Avg_Volume"] = v.mean()
            stats["Std_Volume"] = v.std()

        # Product diversity
        if product_col and product_col in cluster_data.columns:
            stats["Unique_Products"] = cluster_data[product_col].nunique()

        # Country diversity
        if country_col and country_col in cluster_data.columns:
            stats["Unique_Countries"] = cluster_data[country_col].nunique()

        cluster_stats.append(stats)

    result_df = pd.DataFrame(cluster_stats)

    # --- Uplift (optional; only if we have compatible full_df) --------------
    if (
        full_df is not None
        and product_col in full_df.columns
        and country_col in full_df.columns
        and _pick_volume_col(full_df) is not None
    ):
        try:
            print("Computing promotional uplift estimates...")

            full_volume_col = _pick_volume_col(full_df)

            # find LikelyPromo / promo flag column
            promo_flag_col = None
            for c in full_df.columns:
                if ("promo" in c.lower()) and ("likely" in c.lower()):
                    promo_flag_col = c
                    break

            if promo_flag_col is None:
                print("Warning: promo flag column not found in full_df; skipping uplift.")
            else:
                baseline_df = full_df[full_df[promo_flag_col] == 0].copy()
                # numeric safety
                baseline_df[full_volume_col] = pd.to_numeric(baseline_df[full_volume_col], errors="coerce")

                baseline_avg = (
                    baseline_df
                    .groupby([product_col, country_col], dropna=False)[full_volume_col]
                    .mean()
                    .rename("baseline_sales")
                    .reset_index()
                )

                # Join baseline back to promo rows
                df_join = df.copy()
                df_join = df_join.merge(baseline_avg, on=[product_col, country_col], how="left")

                # numeric safety on promo volume
                df_join[volume_col] = pd.to_numeric(df_join[volume_col], errors="coerce")

                df_join["estimated_uplift"] = df_join[volume_col] - df_join["baseline_sales"]
                df_join["uplift_pct"] = 100.0 * (df_join["estimated_uplift"] / df_join["baseline_sales"])

                uplift_stats = (
                    df_join
                    .groupby(cluster_col, dropna=False)[["estimated_uplift", "uplift_pct"]]
                    .agg(["mean", "std"])
                )

                uplift_stats.columns = ["_".join(col).strip() for col in uplift_stats.columns.to_flat_index()]
                uplift_stats = uplift_stats.reset_index()

                result_df = result_df.merge(
                    uplift_stats, left_on="Cluster", right_on=cluster_col, how="left"
                )

        except Exception as e:
            print(f"Warning: Could not compute uplift statistics: {e}")

    # --- Display -------------------------------------------------------------
    print("\n--- Cluster Profile Summary ---")
    if not result_df.empty:
        print(tabulate(result_df, headers="keys", tablefmt="grid", floatfmt=".2f"))
    else:
        print("No cluster stats to display.")

    return result_df

def save_report(score, stats_df, output_path="reports/cluster_eval_report.md", csv_output="reports/cluster_stats.csv"):
    """
    Save comprehensive evaluation report.
    
    Args:
        score (float): Silhouette score
        stats_df (pd.DataFrame): Cluster statistics
        output_path (str): Path for markdown report
        csv_output (str): Path for CSV export
    """
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    print(f"\nSaving evaluation results to {output_path} and {csv_output}...")
    
    with open(output_path, "w") as f:
        f.write("# Promotion Cluster Evaluation Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if score is not None:
            f.write(f"## Cluster Quality Metrics\n\n")
            f.write(f"**Silhouette Score (excluding noise):** {round(score, 4)}\n")
            f.write(f"**Quality Assessment:** {'Good' if score > 0.5 else 'Fair'} cluster separation\n\n")
        
        if not stats_df.empty:
            f.write("## Cluster Profiles\n\n")
            f.write(tabulate(stats_df, headers='keys', tablefmt='github', floatfmt='.2f'))
            f.write("\n\n")
        
        # Add interpretation guidance
        f.write("## Interpretation Guide\n\n")
        f.write("- **Silhouette Score > 0.5:** Good cluster separation\n")
        f.write("- **Silhouette Score 0.25-0.5:** Fair cluster separation\n")
        f.write("- **Silhouette Score < 0.25:** Poor cluster separation\n\n")
        f.write("- **Noise (Cluster -1):** Outlier promotional periods\n")
        f.write("- **Count:** Number of promotional periods in cluster\n")
        f.write("- **Percentage:** Proportion of total promotional periods\n")
    
    # Save CSV
    if not stats_df.empty:
        stats_df.to_csv(csv_output, index=False)
    
    print("✅ Report and CSV saved successfully.")

def main():
    """
    Main execution function for cluster evaluation.
    """
    print("\n===== Promotion Cluster Evaluation =====\n")
    
    # Suppress warnings for cleaner output
    warnings.simplefilter("ignore")
    
    # Features for evaluation (flexible - will use what's available)
    potential_features = [
        "Category", "Country", "Month", "WeekOfYear", "Sales_volume",
        "Forecasting Group", "state_id", "item_id"  # Alternative column names
    ]
    
    try:
        # Load data
        promo_df = load_clustered_data()
        full_df = load_full_data()
        
        if promo_df is None or len(promo_df) == 0:
            print("Error: No clustered data loaded")
            return
        
        # Filter to available features
        available_features = [f for f in potential_features if f in promo_df.columns]
        print(f"\nUsing available features for evaluation: {available_features}")
        
        # Compute cluster quality
        score = compute_silhouette(promo_df, available_features)
        
        # Compute cluster profiles
        stats_df = compute_cluster_profiles(promo_df, full_df)
        
        # Save comprehensive report
        save_report(score, stats_df)
        
        print("\n===== Evaluation Complete =====")
        
        # Summary for user
        if score is not None:
            print(f"\n🎯 KEY RESULTS:")
            print(f"   Silhouette Score: {score:.4f} ({'Good' if score > 0.5 else 'Fair'} separation)")
        
        if not stats_df.empty:
            total_clusters = len(stats_df[stats_df['Cluster'] != -1])
            noise_pct = stats_df[stats_df['Cluster'] == -1]['Percentage'].sum() if -1 in stats_df['Cluster'].values else 0
            print(f"   Total Clusters: {total_clusters}")
            print(f"   Noise Rate: {noise_pct:.1f}%")
            print(f"   Largest Cluster: {stats_df.loc[stats_df['Percentage'].idxmax(), 'Percentage']:.1f}% of data")
        
    except Exception as e:
        print(f"\nError in cluster evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nTips:")
        print("1. Ensure clustering_promos.py has been run successfully")
        print("2. Check that tunable_promo_clustered.csv exists and has the expected columns")
        print("3. Verify sales_data_train.csv exists for uplift calculations")

if __name__ == "__main__":
    main()