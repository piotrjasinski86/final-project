#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Clustering Script

This script performs clustering on detected promotional periods to identify different promotion
patterns across countries and product categories. It uses Gower distance (which handles mixed
data types) and HDBSCAN clustering to group similar promotions together.

Key Features:
- Processes output from the tunable STL promo detection script
- Handles mixed categorical and numerical features
- Efficient batch processing for Gower distance calculation with progress tracking
- Memory-efficient implementation for large datasets
- Hierarchical density-based clustering via HDBSCAN

Usage:
    python clustering_promos.py
"""

import pandas as pd
import gower
import hdbscan
import time
import psutil
import os
import numpy as np
from tqdm import tqdm

def main():
    """
    Main execution function for promotion clustering.
    
    This function performs the following steps:
    1. Loads promotion-flagged data from the tunable STL detection output
    2. Filters for promotion weeks only
    3. Extracts and transforms temporal features (month, week of year)
    4. Samples data if too large for memory constraints
    5. Computes Gower distance matrix for mixed data types using batched processing
    6. Applies HDBSCAN clustering to identify promotion patterns
    7. Saves the clustered results for further analysis
    
    The function handles memory efficiently with batch processing and progress tracking.
    """
    start = time.time()
    print("Loading flagged M5 data...")
    # Updated to use the tunable STL promo detection output
    df = pd.read_csv("tunable_stl_promo_flagged.csv")

    print("Filtering promo rows using LikelyPromo...")
    promo_df = df[df["LikelyPromo"] == 1].copy()

    print(f"Total detected promo weeks: {len(promo_df)}")

    # --- Add Month & WeekOfYear safely ---
    print("Parsing Sales_week to extract Month and WeekOfYear...")
    # Parsing for YYYY-MM format (year and month number)
    # First split the Sales_week string into year and month components
    promo_df["year"] = promo_df["Sales_week"].str.split("-").str[0].astype(int)
    promo_df["week"] = promo_df["Sales_week"].str.split("-").str[1].astype(int)
    
    # Convert year-week to datetime using proper ISO week format
    # Using the first day of the week (Monday) as the representative date
    # This allows extraction of month and week of year in the next steps
    promo_df["Sales_week_dt"] = promo_df.apply(
        lambda row: pd.to_datetime(f"{row['year']}-W{row['week']:02d}-1", format="%G-W%V-%u"),
        axis=1
    )

    # Extract month (1-12) and ISO week of year (1-53)
    # These are important temporal features for clustering seasonal patterns
    promo_df["Month"] = promo_df["Sales_week_dt"].dt.month
    promo_df["WeekOfYear"] = promo_df["Sales_week_dt"].dt.isocalendar().week

    # --- Optional: sample if too large ---
    # Memory constraints: Gower distance requires O(n²) memory
    # 25,000 points = ~5GB memory for distance matrix alone
    if len(promo_df) > 25000:
        print("Sampling 25,000 promo rows for clustering...")
        promo_df = promo_df.sample(25000, random_state=42)  # Fixed random seed for reproducibility

    # --- Define clustering features ---
    # These features capture both the context and characteristics of promotions
    features = [
        "Category",     # categorical: product category (e.g., FOODS, HOUSEHOLD)
        "Country",      # categorical: country/market context
        "Month",        # numeric: captures seasonality (1-12)
        "WeekOfYear",   # numeric: more granular seasonality (1-53)
        "Sales_volume",  # numeric: size/impact of the promotion
        "PromoConfidence"  # numeric: confidence score from detection algorithm (0-100)
    ]

    print(f"Features used for clustering: {features}")

    X = promo_df[features].copy()

    # --- Force correct dtypes ---
    # Gower distance requires specific data types:
    # - Numeric features must be float64
    # - Categorical features must be object dtype
    for col in ["Month", "WeekOfYear", "Sales_volume", "PromoConfidence"]:
        X[col] = X[col].astype("float64")   # numeric must be float for Gower
    for col in ["Category", "Country"]:
        X[col] = X[col].astype("object")    # categoricals as object

    print(X.dtypes)

    # --- Check memory usage ---
    process = psutil.Process(os.getpid())
    print(f"Memory before Gower: {round(process.memory_info().rss / 1e6)} MB")

    # --- Compute Gower distance matrix with progress tracking ---
    print("Computing Gower distance matrix...")
    # Gower distance is ideal for mixed data types (categorical + numerical)
    # It normalizes all features and computes appropriate distances per type
    
    # Using batch processing for large matrices to reduce peak memory usage
    # and provide progress feedback during the long computation
    n = len(X)
    batch_size = 500  # Adjust based on memory constraints
    num_batches = n // batch_size + (1 if n % batch_size > 0 else 0)
    
    # Initialize empty distance matrix
    gower_dist = np.zeros((n, n))
    
    # Process in batches with progress bar
    with tqdm(total=num_batches, desc="Gower matrix batches") as pbar:
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            # Calculate distances for this batch against all data
            batch_dist = gower.gower_matrix(X.iloc[i:end_i], X).astype("float64")
            # Copy to the full distance matrix
            gower_dist[i:end_i, :] = batch_dist
            pbar.update(1)
    
    print("Gower distance matrix complete.")
    # Ensure the matrix is symmetric by averaging with its transpose
    # This handles any small numerical inconsistencies
    gower_dist = (gower_dist + gower_dist.T) / 2

    # --- Run HDBSCAN clustering ---
    print("Running HDBSCAN clustering...")
    # HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise
    # Benefits over traditional clustering:
    # - No need to specify number of clusters in advance
    # - Can find clusters of varying sizes and densities
    # - Robust to noise (outliers get label -1)
    # - Handles non-globular cluster shapes
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=800,       # minimum cluster size (larger = fewer, more stable clusters)
        min_samples=50,             # points needed to form core of a cluster (higher = more conservative)
        cluster_selection_epsilon=0.2,  # distance threshold for merging borderline points
        metric='precomputed',       # use our pre-computed Gower distance matrix
        cluster_selection_method='eom'  # Excess of Mass: better for varying density clusters
    )

    cluster_labels = clusterer.fit_predict(gower_dist)
    promo_df["promo_cluster"] = cluster_labels

    print("Cluster label distribution:")
    print(promo_df["promo_cluster"].value_counts(dropna=False))

    # --- Save results ---
    print("Saving clustered data to tunable_promo_clustered.csv...")
    promo_df.to_csv("tunable_promo_clustered.csv", index=False)

    print(f"Done in {round(time.time() - start, 2)} seconds.")

if __name__ == "__main__":
    main()
