#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Feature Engineering Script

This script prepares features for promotion prediction models by combining
sales data with promotion clusters and generating time-based features, lags,
rolling statistics, and seasonality indicators.

Input:
- sales_data_train.csv: Historical sales data
- tunable_stl_promo_flagged.csv: Data with promotion flags
- tunable_promo_clustered.csv: Clustered promotion data

Output:
- promo_prediction_features.csv: Feature dataset for promotion prediction models

Usage:
    python promo_feature_engineering.py [--sales FILENAME] [--flags FILENAME] 
                                       [--clusters FILENAME] [--output FILENAME]
                                       [--n-lags INT] [--rolling-windows INT,INT,...]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare features for promotion prediction')
    
    parser.add_argument('--sales', type=str, default='sales_data_train.csv',
                       help='Path to sales data file')
    parser.add_argument('--flags', type=str, default='tunable_stl_promo_flagged.csv',
                       help='Path to promotion-flagged data file')
    parser.add_argument('--clusters', type=str, default='tunable_promo_clustered.csv',
                       help='Path to clustered promotions file')
    parser.add_argument('--output', type=str, default='promo_prediction_features.csv',
                       help='Path to output features file')
    parser.add_argument('--n-lags', type=int, default=8,
                       help='Number of lag weeks to create')
    parser.add_argument('--rolling-windows', type=str, default='4,8,12',
                       help='Comma-separated list of rolling window sizes')
    
    args = parser.parse_args()
    
    # Parse rolling windows to list of integers
    args.rolling_windows = [int(x) for x in args.rolling_windows.split(',')]
    
    return args


def load_data(sales_path, flags_path, clusters_path):
    """Load and validate input data files."""
    # Check all files exist
    for path, name in [(sales_path, "Sales data"), 
                      (flags_path, "Promotion flags"), 
                      (clusters_path, "Promotion clusters")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Load data files
    print(f"Loading sales data from {sales_path}...")
    sales = pd.read_csv(sales_path)

    # Rename columns to match what the script expects
    sales = sales.rename(columns={
        'Sales_week': 'Year_Week',
        'Sales_volume': 'Volume'
    })
    
    print(f"Loading promotion flags from {flags_path}...")
    flags = pd.read_csv(flags_path)
    flags = flags.rename(columns={
        'Sales_week': 'Year_Week',
        'LikelyPromo': 'is_promo'
    })
    
    print(f"Loading promotion clusters from {clusters_path}...")
    clusters = pd.read_csv(clusters_path)
    clusters = clusters.rename(columns={
        'Sales_week': 'Year_Week'
    })
    
    # Validate required columns
    sales_req_cols = ['Year_Week', 'Forecasting Group', 'Country', 'Volume']
    flags_req_cols = ['Year_Week', 'Forecasting Group', 'Country', 'is_promo']
    clusters_req_cols = ['Year_Week', 'Forecasting Group', 'Country', 'promo_cluster']
    
    for df, name, req_cols in [(sales, "Sales", sales_req_cols), 
                              (flags, "Flags", flags_req_cols), 
                              (clusters, "Clusters", clusters_req_cols)]:
        missing_cols = [col for col in req_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name}: {', '.join(missing_cols)}")
    
    # Ensure consistent date format
    for df in [sales, flags, clusters]:
        if 'Sales_week' not in df.columns:
            # Convert Year_Week to datetime
            df['Sales_week'] = df['Year_Week'].apply(lambda yw: 
                                                  datetime.strptime(f"{yw}-1", "%Y%W-%w")
                                                  if isinstance(yw, (int, float)) else
                                                  datetime.strptime(f"{yw}-1", "%Y-%W-%w"))
    
    print(f"Loaded {len(sales)} sales records")
    print(f"Loaded {len(flags)} promotion flags")
    print(f"Loaded {len(clusters)} clustered promotions")
    
    return sales, flags, clusters


def merge_data(sales, flags, clusters):
    """Merge sales data with promotion flags and clusters."""
    print("\nMerging data...")
    
    # Ensure we have common keys
    merge_keys = ['Forecasting Group', 'Country', 'Sales_week']
    
    # Merge sales with flags
    data = pd.merge(sales, flags[merge_keys + ['is_promo']], 
                   on=merge_keys, how='left')
    
    # Fill missing promo flags with 0 (no promotion)
    data['is_promo'] = data['is_promo'].fillna(0)
    
    # Merge with clusters for weeks that have promotions
    # First create a dummy cluster column with 0 for non-promo weeks
    data['promo_cluster'] = 0
    
    # Now update cluster values for promo weeks
    clusters_subset = clusters[merge_keys + ['promo_cluster']]
    clusters_dict = {(row['Forecasting Group'], row['Country'], row['Sales_week']): 
                     row['promo_cluster'] 
                     for _, row in clusters_subset.iterrows()}
    
    # Function to lookup cluster value
    def lookup_cluster(row):
        key = (row['Forecasting Group'], row['Country'], row['Sales_week'])
        if key in clusters_dict and row['is_promo'] == 1:
            return clusters_dict[key]
        return 0
    
    # Apply lookup
    data['promo_cluster'] = data.apply(lookup_cluster, axis=1)
    
    print(f"Merged data shape: {data.shape}")
    
    return data


def create_time_features(data):
    """Create time-based features for seasonality."""
    print("\nCreating time features...")
    
    # Extract time components
    data['Year'] = data['Sales_week'].dt.year
    data['Quarter'] = data['Sales_week'].dt.quarter
    data['Month'] = data['Sales_week'].dt.month
    data['WeekOfYear'] = data['Sales_week'].dt.isocalendar().week
    data['DayOfYear'] = data['Sales_week'].dt.dayofyear
    
    # Create cyclical features for week of year (captures seasonality)
    # Convert to sine/cosine to avoid discontinuity at year boundaries
    data['WeekOfYear_sin'] = np.sin(2 * np.pi * data['WeekOfYear'] / 52)
    data['WeekOfYear_cos'] = np.cos(2 * np.pi * data['WeekOfYear'] / 52)
    
    # Month has similar issue
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    
    # Quarter
    data['Quarter_sin'] = np.sin(2 * np.pi * data['Quarter'] / 4)
    data['Quarter_cos'] = np.cos(2 * np.pi * data['Quarter'] / 4)
    
    # Holiday indicators (approximate major holidays)
    # This is a simplification - would be better to use country-specific holiday calendars
    data['is_christmas_period'] = ((data['Month'] == 12) & (data['WeekOfYear'] >= 50)) | \
                                 ((data['Month'] == 1) & (data['WeekOfYear'] <= 2))
    
    data['is_easter_period'] = ((data['Month'] == 3) | (data['Month'] == 4)) & \
                              (data['WeekOfYear'] >= 13) & (data['WeekOfYear'] <= 17)
    
    data['is_summer_holiday'] = (data['Month'] >= 6) & (data['Month'] <= 8)
    
    return data


def create_lag_features(data, n_lags):
    """Create lagged features for promotion occurrence and types."""
    print(f"\nCreating lag features (lag={n_lags} weeks)...")
    
    # Sort by product, country, and time
    data = data.sort_values(['Forecasting Group', 'Country', 'Sales_week'])
    
    # Group by product and country
    groups = data.groupby(['Forecasting Group', 'Country'])
    
    # For each group, create lags
    result_dfs = []
    
    for name, group in groups:
        group_copy = group.copy()
        
        # Create lag features for promo flag
        for lag in range(1, n_lags + 1):
            group_copy[f'promo_lag_{lag}'] = group_copy['is_promo'].shift(lag)
        
        # Create lag features for promo cluster
        for lag in range(1, n_lags + 1):
            group_copy[f'cluster_lag_{lag}'] = group_copy['promo_cluster'].shift(lag)
        
        result_dfs.append(group_copy)
    
    # Combine results
    result = pd.concat(result_dfs)
    
    # Fill NaN values for lag features (beginning of series)
    lag_columns = [col for col in result.columns if col.startswith('promo_lag_') or 
                                                   col.startswith('cluster_lag_')]
    result[lag_columns] = result[lag_columns].fillna(0)
    
    return result


def create_rolling_features(data, windows):
    """Create rolling window features for promotion density."""
    print(f"\nCreating rolling window features {windows}...")
    
    # Sort by product, country, and time
    data = data.sort_values(['Forecasting Group', 'Country', 'Sales_week'])
    
    # Group by product and country
    groups = data.groupby(['Forecasting Group', 'Country'])
    
    # For each group, create rolling features
    result_dfs = []
    
    for name, group in groups:
        group_copy = group.copy()
        
        # For each window size, calculate rolling stats
        for window in windows:
            # Promotion density (proportion of weeks with promos)
            group_copy[f'promo_density_{window}w'] = (
                group_copy['is_promo'].rolling(window=window, min_periods=1).mean()
            )
            
            # Days since last promotion
            group_copy['promo_flag_for_days'] = (group_copy['is_promo'] == 0).astype(int)
            group_copy[f'days_since_promo_{window}w'] = (
                group_copy['promo_flag_for_days'].rolling(window=window, min_periods=1).sum()
            )
            
            # Most common cluster type in window (excluding 0 which is no promo)
            def most_common_nonzero(x):
                # Filter out zeros
                nonzero = x[x > 0]
                if len(nonzero) > 0:
                    # Return most common value
                    return nonzero.value_counts().index[0]
                return 0
            
            group_copy[f'common_cluster_{window}w'] = (
                group_copy['promo_cluster'].rolling(window=window, min_periods=1)
                .apply(most_common_nonzero)
            )
        
        # Drop helper column
        group_copy.drop('promo_flag_for_days', axis=1, inplace=True)
        
        result_dfs.append(group_copy)
    
    # Combine results
    result = pd.concat(result_dfs)
    
    # Fill NaN values for rolling features at beginning of series
    rolling_columns = [col for col in result.columns if 
                      'density' in col or 'days_since' in col or 'common_cluster' in col]
    result[rolling_columns] = result[rolling_columns].fillna(0)
    
    return result


def create_categorical_encodings(data):
    """Create encodings for categorical features."""
    print("\nEncoding categorical features...")
    
    # One-hot encode Country
    country_dummies = pd.get_dummies(data['Country'], prefix='country')
    data = pd.concat([data, country_dummies], axis=1)
    
    # Label encode Forecasting Group (may have too many values for one-hot)
    product_mapping = {product: i for i, product in 
                      enumerate(data['Forecasting Group'].unique())}
    data['product_id'] = data['Forecasting Group'].map(product_mapping)
    
    # Encode promo_cluster as dummy variables if needed for prediction
    cluster_dummies = pd.get_dummies(data['promo_cluster'], prefix='cluster')
    data = pd.concat([data, cluster_dummies], axis=1)
    
    return data


def visualize_features(data, output_dir="reports"):
    """Create visualizations of engineered features."""
    print("\nGenerating feature visualizations...")
    
    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Lag feature correlation with promotion occurrence
    lag_cols = [col for col in data.columns if col.startswith('promo_lag_')]
    corr_data = data[['is_promo'] + lag_cols].corr()['is_promo'].drop('is_promo')
    
    plt.figure(figsize=(10, 6))
    corr_data.plot(kind='bar', color='skyblue')
    plt.title('Correlation of Lag Features with Promotion Occurrence')
    plt.xlabel('Lag Feature')
    plt.ylabel('Correlation Coefficient')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lag_feature_correlation.png'))
    
    # 2. Promotion density by week of year
    plt.figure(figsize=(12, 6))
    data.groupby('WeekOfYear')['is_promo'].mean().plot(kind='line', marker='o')
    plt.title('Promotion Density by Week of Year')
    plt.xlabel('Week of Year')
    plt.ylabel('Proportion of Products on Promotion')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'promo_density_by_week.png'))
    
    # 3. Average time since last promotion by product
    avg_days_since = data.groupby('Forecasting Group')[
        [col for col in data.columns if col.startswith('days_since_promo_')]
    ].mean().mean(axis=1).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    avg_days_since.head(20).plot(kind='bar')
    plt.title('Average Days Since Last Promotion by Product (Top 20)')
    plt.xlabel('Product')
    plt.ylabel('Avg Days Since Last Promo')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_days_since_promo.png'))
    
    # 4. Feature importance preview for promotion prediction
    from sklearn.ensemble import RandomForestClassifier
    
    # Sample a subset for quick model fitting
    sample = data.sample(min(10000, len(data)))
    
    # Select features
    feature_cols = (
        [col for col in sample.columns if col.startswith('promo_lag_')] +
        [col for col in sample.columns if col.startswith('cluster_lag_')] +
        [col for col in sample.columns if col.startswith('promo_density_')] +
        [col for col in sample.columns if col.startswith('days_since_promo_')] +
        [col for col in sample.columns if col.startswith('common_cluster_')] +
        ['WeekOfYear_sin', 'WeekOfYear_cos', 'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos'] +
        [col for col in sample.columns if col.startswith('country_')] +
        ['is_christmas_period', 'is_easter_period', 'is_summer_holiday']
    )
    
    X = sample[feature_cols]
    y = sample['is_promo']
    
    # Train a simple model
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    importances.head(20).plot(kind='barh')
    plt.title('Feature Importance Preview for Promotion Prediction')
    plt.xlabel('Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_preview.png'))
    
    print(f"Feature visualizations saved to {output_dir}")


def prepare_features(args):
    """Main function to prepare and save features."""
    # Load data
    sales, flags, clusters = load_data(args.sales, args.flags, args.clusters)
    
    # Merge data
    data = merge_data(sales, flags, clusters)
    
    # Create features
    data = create_time_features(data)
    data = create_lag_features(data, args.n_lags)
    data = create_rolling_features(data, args.rolling_windows)
    data = create_categorical_encodings(data)
    
    # Visualize features (optional)
    visualize_features(data)
    
    # Save the final features dataset
    print(f"\nSaving features dataset to {args.output}...")
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} records with {len(data.columns)} features")
    
    # Print feature summary
    print("\nFeature categories:")
    print(f"- Base features: {', '.join(['Forecasting Group', 'Country', 'Sales_week', 'Volume'])}")
    print(f"- Time features: {', '.join([col for col in data.columns if col in ['Year', 'Quarter', 'Month', 'WeekOfYear', 'WeekOfYear_sin', 'WeekOfYear_cos', 'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos']])}")
    print(f"- Lag features: {len([col for col in data.columns if col.startswith('promo_lag_') or col.startswith('cluster_lag_')])} features")
    print(f"- Rolling window features: {len([col for col in data.columns if 'density' in col or 'days_since' in col or 'common_cluster' in col])} features")
    print(f"- Categorical encodings: {len([col for col in data.columns if col.startswith('country_') or col.startswith('cluster_') or col == 'product_id'])} features")
    print(f"- Holiday indicators: {len([col for col in data.columns if col.startswith('is_')])} features")
    
    return data


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Prepare features
        data = prepare_features(args)
        print("\nFeature engineering complete!")
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        raise


if __name__ == "__main__":
    main()
