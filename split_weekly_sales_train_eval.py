#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly Sales Data Split Script

This script splits the weekly aggregated sales data into training and evaluation datasets
based on the year. It creates separate files for historical training data and more recent
evaluation data to support proper forecasting model development and validation.

Key Features:
- Year-based temporal split of sales data
- Maintains data structure and column formatting
- Sorts output for consistent analysis
- Creates clean training and evaluation datasets

Input:
- sales_data_weeks.csv: Weekly aggregated sales data from daily_sales_to_weekly.py

Output:
- sales_data_train.csv: Training dataset (years <= 2017)
- sales_data_eval.csv: Evaluation dataset (years >= 2018)

Note: There is intentional overlap in the boundary year (2018) to ensure
continuity in the time series for both training and evaluation.

Usage:
    python split_weekly_sales_train_eval.py
"""

import pandas as pd

# Step 1: Load weekly sales data from the weekly aggregation script
input_path = 'sales_data_weeks.csv'
print(f"Loading weekly sales data from {input_path}...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df):,} records from {len(df['Sales_week'].unique())} unique weeks")

# Step 2: Extract year from Sales_week column
# Parse YYYY-WW format to get just the year component
print("Extracting year information from Sales_week...")
df['Year'] = df['Sales_week'].str.split('-').str[0].astype(int)
print(f"Years in dataset: {df['Year'].min()} to {df['Year'].max()}")

# Step 3: Split data into training and evaluation datasets
# Training: Historical data through 2017
# Evaluation: Recent data from 2018 onwards
print("Splitting data into training and evaluation sets...")
train_df = df[df['Year'] <= 2017].copy()
eval_df  = df[df['Year'] >= 2018].copy()

# Report the split statistics
print(f"Training set: {len(train_df):,} records from years {train_df['Year'].min()}-{train_df['Year'].max()}")
print(f"Evaluation set: {len(eval_df):,} records from years {eval_df['Year'].min()}-{eval_df['Year'].max()}")

# Step 4: Drop the 'Year' helper column since it was only used for splitting
# This maintains the same schema as the original dataset
print("Removing temporary Year column from output datasets...")
train_df.drop(columns=['Year'], inplace=True)
eval_df.drop(columns=['Year'], inplace=True)

# Step 5: Sort each DataFrame for consistency and readability
# This hierarchical sort organizes data by location, product type, and chronology
sort_cols = ['Country', 'Category', 'Forecasting Group', 'Sales_week']
print("Sorting output datasets...")
train_df.sort_values(by=sort_cols, inplace=True)
eval_df.sort_values(by=sort_cols, inplace=True)

# Step 6: Save split files to disk
print("Saving split datasets to disk...")
train_output = 'sales_data_train.csv'
eval_output = 'sales_data_eval.csv'
train_df.to_csv(train_output, index=False)
eval_df.to_csv(eval_output, index=False)
print(f"Training data saved to: {train_output}")
print(f"Evaluation data saved to: {eval_output}")

# Step 7: Preview samples from each dataset for verification
print("\n===== DATASET PREVIEWS =====")

# Show training data sample
print("\nFirst 5 rows of TRAINING set:")
print(train_df[['Country', 'Category', 'Sales_week', 'Sales_volume']].head())
print(f"Countries: {', '.join(train_df['Country'].unique())}")
print(f"Time range: {train_df['Sales_week'].min()} to {train_df['Sales_week'].max()}")

# Show evaluation data sample
print("\nFirst 5 rows of EVALUATION set:")
print(eval_df[['Country', 'Category', 'Sales_week', 'Sales_volume']].head())
print(f"Countries: {', '.join(eval_df['Country'].unique())}")
print(f"Time range: {eval_df['Sales_week'].min()} to {eval_df['Sales_week'].max()}")

print("\nSplit complete. Datasets ready for analysis and modeling.")
print(f"Total input records: {len(df):,}")
print(f"Total output records: {len(train_df) + len(eval_df):,}")
print(f"Training: {len(train_df):,} records ({len(train_df)/len(df):.1%})")
print(f"Evaluation: {len(eval_df):,} records ({len(eval_df)/len(df):.1%})")

# Verify data integrity
assert len(df) == len(train_df) + len(eval_df), "Record count mismatch after splitting!"
