#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Future Promotion Prediction Script

This script generates predictions of future promotions by:
1. Loading trained promotion occurrence and cluster type models
2. Preparing a future dates dataframe with product-country combinations
3. Iteratively predicting week-by-week, updating features as it goes
4. Saving the predicted promotion calendar for use in sales forecasting

Input:
- promo_prediction_features.csv: Features dataset (used for product-country list and latest data)
- models/promo_occurrence_model.pkl: Trained binary classifier
- models/promo_cluster_model.pkl: Trained multiclass classifier
- models/feature_columns.pkl: List of feature columns used by models

Output:
- future_promotion_calendar.csv: Predicted promotions for the forecast horizon
- reports/future_promotion_visualization.png: Visualization of predicted promotions

Usage:
    python future_promotion_prediction.py [--input FILENAME] [--models-dir DIR]
                                         [--horizon WEEKS] [--output FILENAME]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate future promotion predictions')
    
    parser.add_argument('--input', type=str, default='promo_prediction_features.csv',
                        help='Path to feature-engineered dataset')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--reports-dir', type=str, default='reports',
                        help='Directory for output reports')
    parser.add_argument('--horizon', type=int, default=26,
                        help='Forecast horizon in weeks')
    parser.add_argument('--output', type=str, default='future_promotion_calendar.csv',
                        help='Path to output predicted promotions file')
    
    return parser.parse_args()


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_models_and_data(args):
    """Load trained models and feature data."""
    # Check that all required files exist
    models_files = [
        os.path.join(args.models_dir, 'promo_occurrence_model.pkl'),
        os.path.join(args.models_dir, 'promo_cluster_model.pkl'),
        os.path.join(args.models_dir, 'feature_columns.pkl')
    ]
    
    for file_path in models_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required model file not found: {file_path}")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Feature data file not found: {args.input}")
    
    # Load models and feature columns
    print("Loading trained models...")
    binary_model = joblib.load(models_files[0])
    multiclass_model = joblib.load(models_files[1])
    feature_columns = joblib.load(models_files[2])
    
    # Load feature data
    print(f"Loading feature data from {args.input}...")
    data = pd.read_csv(args.input)
    
    # Ensure date format
    data['Sales_week'] = pd.to_datetime(data['Sales_week'])
    
    # Get last date in data
    last_date = data['Sales_week'].max()
    print(f"Latest date in data: {last_date}")
    
    return binary_model, multiclass_model, feature_columns, data, last_date


def prepare_future_dataframe(data, last_date, horizon):
    """Prepare dataframe with future dates for all product-country combinations."""
    print("\nPreparing future dates dataframe...")
    
    # Get unique product-country combinations
    product_countries = data[['Forecasting Group', 'Country']].drop_duplicates()
    n_combinations = len(product_countries)
    print(f"Found {n_combinations} unique product-country combinations")
    
    # Generate future weeks
    future_weeks = []
    for i in range(1, horizon + 1):
        future_weeks.append(last_date + timedelta(weeks=i))
    
    print(f"Generating predictions for {len(future_weeks)} future weeks")
    
    # Create dataframe with all combinations
    future_records = []
    for _, pc in product_countries.iterrows():
        for week in future_weeks:
            future_records.append({
                'Forecasting Group': pc['Forecasting Group'],
                'Country': pc['Country'],
                'Sales_week': week
            })
    
    future_df = pd.DataFrame(future_records)
    
    # Add derived time features
    future_df['Year'] = future_df['Sales_week'].dt.year
    future_df['Quarter'] = future_df['Sales_week'].dt.quarter
    future_df['Month'] = future_df['Sales_week'].dt.month
    future_df['WeekOfYear'] = future_df['Sales_week'].dt.isocalendar().week
    future_df['DayOfYear'] = future_df['Sales_week'].dt.dayofyear
    
    # Create cyclical features for week of year
    future_df['WeekOfYear_sin'] = np.sin(2 * np.pi * future_df['WeekOfYear'] / 52)
    future_df['WeekOfYear_cos'] = np.cos(2 * np.pi * future_df['WeekOfYear'] / 52)
    
    # Month
    future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
    future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
    
    # Quarter
    future_df['Quarter_sin'] = np.sin(2 * np.pi * future_df['Quarter'] / 4)
    future_df['Quarter_cos'] = np.cos(2 * np.pi * future_df['Quarter'] / 4)
    
    # Holiday indicators (simplified)
    future_df['is_christmas_period'] = ((future_df['Month'] == 12) & (future_df['WeekOfYear'] >= 50)) | \
                                     ((future_df['Month'] == 1) & (future_df['WeekOfYear'] <= 2))
    
    future_df['is_easter_period'] = ((future_df['Month'] == 3) | (future_df['Month'] == 4)) & \
                                  (future_df['WeekOfYear'] >= 13) & (future_df['WeekOfYear'] <= 17)
    
    future_df['is_summer_holiday'] = (future_df['Month'] >= 6) & (future_df['Month'] <= 8)
    
    # Add country dummy variables
    country_dummies = pd.get_dummies(future_df['Country'], prefix='country')
    future_df = pd.concat([future_df, country_dummies], axis=1)
    
    print(f"Prepared future dataframe with {len(future_df)} rows")
    return future_df


def get_historical_patterns(data, product, country):
    """Get historical promotion patterns for a specific product-country."""
    mask = (data['Forecasting Group'] == product) & (data['Country'] == country)
    history = data[mask].sort_values('Sales_week')
    
    # Get last few weeks of promotion data
    last_n_weeks = min(12, len(history))
    recent_history = history.tail(last_n_weeks)
    
    return history, recent_history


from tqdm import tqdm

def initialize_lag_features(future_df, data):
    """Efficiently initialize lag features from historical data using groupby, with a progress bar."""
    print("\nInitializing lag features from historical data (vectorized)...")
    
    # Create empty columns for lag and rolling features (use floats to avoid dtype warnings)
    for i in range(1, 9):
        future_df[f'promo_lag_{i}'] = 0
        future_df[f'cluster_lag_{i}'] = 0
    for window in [4, 8, 12]:
        future_df[f'promo_density_{window}w'] = 0.0
        future_df[f'days_since_promo_{window}w'] = float(window * 7)
        future_df[f'common_cluster_{window}w'] = 0

    group_cols = ['Forecasting Group', 'Country']
    product_countries = list(future_df.groupby(group_cols).groups.items())
    
    # Add tqdm progress bar here!
    for (product, country), group_idx in tqdm(product_countries, desc="Lag init for prod-country pairs"):
        hist = data[(data['Forecasting Group'] == product) & (data['Country'] == country)].sort_values('Sales_week')
        if hist.empty:
            continue
        hist_recent = hist.tail(12)
        promo_lags = [0] * (8 - min(8, len(hist_recent))) + hist_recent['is_promo'].tolist()[-8:]
        cluster_lags = [0] * (8 - min(8, len(hist_recent))) + hist_recent['promo_cluster'].tolist()[-8:]
        first_idx = group_idx.min()
        for lag in range(1, 9):
            future_df.at[first_idx, f'promo_lag_{lag}'] = promo_lags[-lag]
            future_df.at[first_idx, f'cluster_lag_{lag}'] = cluster_lags[-lag]
        for window in [4, 8, 12]:
            win_promos = promo_lags[-window:]
            density = float(sum(win_promos)) / len(win_promos) if win_promos else 0.0
            future_df.at[first_idx, f'promo_density_{window}w'] = density
            if 1 in win_promos:
                days_since = win_promos[::-1].index(1) * 7
            else:
                days_since = window * 7
            future_df.at[first_idx, f'days_since_promo_{window}w'] = float(days_since)
            from collections import Counter
            win_clusters = [c for p, c in zip(win_promos, cluster_lags[-window:]) if p == 1]
            most_common = Counter(win_clusters).most_common(1)[0][0] if win_clusters else 0
            future_df.at[first_idx, f'common_cluster_{window}w'] = most_common

    return future_df

def process_group_chunk(args):
    (group_list, future_df, feature_columns, binary_model, multiclass_model) = args
    import numpy as np
    from collections import Counter
    import warnings
    from tqdm import tqdm  # add tqdm import here

    group_cols = ['Forecasting Group', 'Country']
    results = []
    sub_df = future_df[future_df.apply(lambda row: (row['Forecasting Group'], row['Country']) in [tuple(g) for g in group_list], axis=1)].copy()
    weeks = sorted(sub_df['Sales_week'].unique())

    # tqdm progress bar for weeks
    for week_idx, week in enumerate(tqdm(weeks, desc=f"[PID {os.getpid()}] Weeks for {group_list[:2]}...")):
        print(f"\n====== PREDICTING WEEK: {week} (idx {week_idx}) ======")
        week_mask = sub_df['Sales_week'] == week
        week_data = sub_df[week_mask].copy()
        X_week = week_data[feature_columns].copy()
        X_week = X_week.reindex(columns=feature_columns, fill_value=0)

        promo_proba = binary_model.predict_proba(X_week)[:, 1]
        promo_pred = binary_model.predict(X_week)
        cluster_pred = np.zeros(len(promo_pred))
        promo_idx = np.where(promo_pred == 1)[0]

        if len(promo_idx) > 0:
            X_promo = X_week.iloc[promo_idx]
            cluster_promo = multiclass_model.predict(X_promo)
            if isinstance(cluster_promo, np.ndarray) and cluster_promo.ndim == 2:
                print("INFO: cluster_promo is 2D, applying argmax")
                cluster_promo = np.argmax(cluster_promo, axis=1)
            cluster_promo = np.ravel(cluster_promo)
            if len(cluster_promo) == len(promo_idx):
                cluster_pred[promo_idx] = cluster_promo
            else:
                warnings.warn(f"SHAPE ERROR: cluster_promo len {len(cluster_promo)}, promo_idx len {len(promo_idx)}. Filling with zeros.")
                cluster_pred[promo_idx] = 0

        sub_df.loc[week_mask, 'is_promo'] = promo_pred
        sub_df.loc[week_mask, 'promo_probability'] = promo_proba
        sub_df.loc[week_mask, 'promo_cluster'] = cluster_pred

        # ---- Update lag and rolling features for next week
        if week_idx < len(weeks) - 1:
            next_week = weeks[week_idx + 1]
            next_week_mask = sub_df['Sales_week'] == next_week
            this_week_df = sub_df[week_mask].set_index(group_cols)
            next_week_df = sub_df[next_week_mask].set_index(group_cols)

            for lag in range(8, 1, -1):
                from_col = f'promo_lag_{lag-1}'
                to_col = f'promo_lag_{lag}'
                if from_col in this_week_df.columns and to_col in next_week_df.columns:
                    sub_df.loc[next_week_mask, to_col] = this_week_df[from_col].values
                from_col_c = f'cluster_lag_{lag-1}'
                to_col_c = f'cluster_lag_{lag}'
                if from_col_c in this_week_df.columns and to_col_c in next_week_df.columns:
                    sub_df.loc[next_week_mask, to_col_c] = this_week_df[from_col_c].values

            sub_df.loc[next_week_mask, 'promo_lag_1'] = this_week_df['is_promo'].values
            sub_df.loc[next_week_mask, 'cluster_lag_1'] = this_week_df['promo_cluster'].values

            for window in [4, 8, 12]:
                promo_col = f'promo_density_{window}w'
                days_col = f'days_since_promo_{window}w'
                cluster_col = f'common_cluster_{window}w'
                for group in next_week_df.index:
                    mask = (
                        (sub_df['Forecasting Group'] == group[0]) &
                        (sub_df['Country'] == group[1]) &
                        (sub_df['Sales_week'] <= week)
                    )
                    idx = sub_df[
                        (sub_df['Forecasting Group'] == group[0]) &
                        (sub_df['Country'] == group[1]) &
                        (sub_df['Sales_week'] == next_week)
                    ].index[0]
                    recent_weeks = sub_df[mask].sort_values('Sales_week').tail(window)
                    recent_promos = recent_weeks['is_promo'].tolist()
                    density = sum(recent_promos) / len(recent_promos) if recent_promos else 0
                    days_since = window * 7
                    if 1 in recent_promos:
                        days_since = recent_promos[::-1].index(1) * 7
                    elif recent_promos and recent_promos[-1] == 1:
                        days_since = 0
                    recent_clusters = recent_weeks['promo_cluster'].tolist()
                    window_clusters = [c for p, c in zip(recent_promos, recent_clusters) if p == 1]
                    most_common = Counter(window_clusters).most_common(1)[0][0] if window_clusters else 0
                    sub_df.at[idx, promo_col] = density
                    sub_df.at[idx, days_col] = days_since
                    sub_df.at[idx, cluster_col] = most_common

    return sub_df



def predict_promotions_iteratively(binary_model, multiclass_model, feature_columns, future_df, horizon, n_jobs=4):
    import pandas as pd
    from multiprocessing import Pool

    group_cols = ['Forecasting Group', 'Country']
    unique_groups = future_df[group_cols].drop_duplicates().values.tolist()

    # Helper to split list into n_jobs roughly equal chunks
    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    group_chunks = chunkify(unique_groups, n_jobs)
    # Pass all required arguments as tuple (must be pickleable)
    args_list = [
        (chunk, future_df, feature_columns, binary_model, multiclass_model)
        for chunk in group_chunks
    ]

    print(f"Multiprocessing: Using {n_jobs} worker(s) for {len(unique_groups)} groups.")
    with Pool(n_jobs) as pool:
        results = pool.map(process_group_chunk, args_list)

    # Combine all processed DataFrames back together
    result_df = pd.concat(results, axis=0).sort_values(['Sales_week', 'Forecasting Group', 'Country']).reset_index(drop=True)

    return result_df



def visualize_predictions(future_df, reports_dir):
    """Generate visualizations of the predicted promotions."""
    print("\nGenerating prediction visualizations...")
    
    # 1. Promotion density by week
    weekly_promo_density = future_df.groupby('Sales_week')['is_promo'].mean()
    
    plt.figure(figsize=(12, 6))
    weekly_promo_density.plot(marker='o', linestyle='-')
    plt.title('Predicted Promotion Density Over Time', fontsize=14)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Proportion of Products on Promotion', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(reports_dir, 'future_promo_density.png')
    plt.savefig(output_path, dpi=150)
    
    # 2. Cluster type distribution by week
    weekly_cluster_dist = future_df[future_df['is_promo'] == 1].pivot_table(
        index='Sales_week', 
        columns='promo_cluster', 
        values='Forecasting Group', 
        aggfunc='count',
        fill_value=0
    )
    
    plt.figure(figsize=(12, 8))
    weekly_cluster_dist.plot(kind='area', stacked=True, alpha=0.7, colormap='viridis')
    plt.title('Predicted Promotion Types Over Time', fontsize=14)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Number of Promotions', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title='Cluster Type')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(reports_dir, 'future_cluster_distribution.png')
    plt.savefig(output_path, dpi=150)
    
    # 3. Heatmap of promotions by country
    country_weekly = pd.pivot_table(
        future_df,
        values='is_promo',
        index='Country',
        columns='Sales_week',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(country_weekly, cmap='YlOrRd', linewidths=.5)
    plt.title('Predicted Promotion Density by Country Over Time', fontsize=14)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(reports_dir, 'future_country_heatmap.png')
    plt.savefig(output_path, dpi=150)
    
    # 4. Example promotion calendar for a specific product-country
    # Find product-country with frequent promotions
    pc_promo_count = future_df[future_df['is_promo'] == 1].groupby(
        ['Forecasting Group', 'Country']
    ).size().sort_values(ascending=False)
    
    if len(pc_promo_count) > 0:
        example_product, example_country = pc_promo_count.index[0]
        
        example_df = future_df[
            (future_df['Forecasting Group'] == example_product) & 
            (future_df['Country'] == example_country)
        ].sort_values('Sales_week')
        
        plt.figure(figsize=(14, 6))
        
        # Plot promotion probability
        plt.plot(example_df['Sales_week'], example_df['promo_probability'], 
                 marker='o', linestyle='-', label='Promotion Probability')
        
        # Highlight actual predictions
        for idx, row in example_df.iterrows():
            if row['is_promo'] == 1:
                plt.axvline(x=row['Sales_week'], color='red', linestyle='--', alpha=0.3)
                plt.annotate(f"Cluster {int(row['promo_cluster'])}", 
                             xy=(row['Sales_week'], row['promo_probability']),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center')
        
        plt.title(f'Promotion Forecast for {example_product} in {example_country}', fontsize=14)
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Promotion Probability', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(reports_dir, 'example_promo_forecast.png')
        plt.savefig(output_path, dpi=150)
    
    print(f"Visualizations saved to {reports_dir}")


def generate_future_promotions(args):
    """Main function to generate future promotion predictions."""
    # Load models and data
    binary_model, multiclass_model, feature_columns, data, last_date = load_models_and_data(args)
    
    # Prepare future dataframe
    future_df = prepare_future_dataframe(data, last_date, args.horizon)
    
    # Initialize lag features
    future_df = initialize_lag_features(future_df, data)
    
    # Predict promotions
    future_df = predict_promotions_iteratively(binary_model, multiclass_model, 
                                              feature_columns, future_df, args.horizon)
    
    # Generate visualizations
    visualize_predictions(future_df, args.reports_dir)
    
    # Save predictions
    print(f"\nSaving future promotion calendar to {args.output}...")
    future_df.to_csv(args.output, index=False)
    
    # Print summary stats
    total_promos = future_df['is_promo'].sum()
    total_products = future_df['Forecasting Group'].nunique()
    total_countries = future_df['Country'].nunique()
    total_weeks = future_df['Sales_week'].nunique()
    
    print(f"Generated promotion calendar with {total_promos} predicted promotions")
    print(f"Across {total_products} products, {total_countries} countries, and {total_weeks} weeks")
    
    # Generate a simple calendar markdown for easy viewing
    calendar_md_path = os.path.join(args.reports_dir, 'promotion_calendar_summary.md')
    
    with open(calendar_md_path, 'w') as f:
        f.write('# Future Promotion Calendar Summary\n\n')
        
        # Weekly summary
        f.write('## Weekly Promotion Activity\n\n')
        weekly_summary = future_df.groupby('Sales_week')['is_promo'].agg(['sum', 'mean'])
        weekly_summary.columns = ['Promotion Count', 'Promotion Density']
        
        f.write('| Week | Promotion Count | Promotion Density |\n')
        f.write('|------|----------------|-------------------|\n')
        for week, row in weekly_summary.iterrows():
            f.write(f"| {week.strftime('%Y-%m-%d')} | {int(row['Promotion Count']):,} | {row['Promotion Density']:.2%} |\n")
        
        # Cluster type distribution
        f.write('\n## Promotion Type Distribution\n\n')
        cluster_summary = future_df[future_df['is_promo'] == 1]['promo_cluster'].value_counts().sort_index()
        
        f.write('| Cluster | Count | Percentage |\n')
        f.write('|---------|-------|------------|\n')
        for cluster, count in cluster_summary.items():
            percentage = count / total_promos
            f.write(f"| {int(cluster)} | {count:,} | {percentage:.2%} |\n")
        
        # Country distribution
        f.write('\n## Country Promotion Distribution\n\n')
        country_summary = future_df[future_df['is_promo'] == 1].groupby('Country')['is_promo'].count().sort_values(ascending=False)
        
        f.write('| Country | Promotion Count | Percentage |\n')
        f.write('|---------|----------------|------------|\n')
        for country, count in country_summary.items():
            percentage = count / total_promos
            f.write(f"| {country} | {count:,} | {percentage:.2%} |\n")
    
    print(f"Calendar summary saved to {calendar_md_path}")
    
    return future_df


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    ensure_directory(args.reports_dir)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        ensure_directory(output_dir)

    try:
        # Generate future promotions
        future_df = generate_future_promotions(args)
        print("\nFuture promotion prediction complete!")

    except Exception as e:
        print(f"Error during future promotion prediction: {e}")
        raise



if __name__ == "__main__":
    main()
