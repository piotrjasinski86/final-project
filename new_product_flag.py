#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Product Introduction Flagging and Filtering Script

This script identifies and filters out newly introduced products in the sales dataset.
It detects product-country pairs that don't have a full sales history from the beginning
of the dataset and creates a clean dataset containing only products with complete history.

Key Features:
- Identifies first sales week for each product-country pair
- Flags products introduced after the dataset start date
- Creates clean dataset with only full-history products
- Provides statistics on new product introduction rates

This filtering is important for downstream analysis as:
1. New product launches often show atypical sales patterns
2. Incomplete history complicates time series modeling
3. Promotional detection algorithms may falsely flag product launches as promotions

Input:
- sales_data_train.csv: Training dataset from split_weekly_sales_train_eval.py

Output:
- sales_data_full_history_only.csv: Filtered dataset with only full-history products

Usage:
    python new_product_flag.py
"""

import pandas as pd

# Load the training dataset
print("Loading sales training data...")
input_file = "sales_data_train.csv"
df = pd.read_csv(input_file)
print(f"Loaded {len(df):,} records from {input_file}")
print(f"Found {df['Forecasting Group'].nunique()} unique products across {df['Country'].nunique()} countries")

# Determine the first week in the dataset (our reference starting point)
min_week = df["Sales_week"].min()
print(f"First week in dataset: {min_week}")

# Find first nonzero sale week for each product–country pair
# This identifies when each product first appeared in each country
print("Finding first appearance week for each product-country pair...")
first_nonzero_week = (
    # Only consider weeks with actual sales (volume > 0)
    df[df["Sales_volume"] > 0]
    # Group by product and country
    .groupby(["Forecasting Group", "Country"])["Sales_week"]
    # Find earliest week with sales for each group
    .min()
)

# Create new product introduction flag
# 1 = product launched after dataset start (new introduction)
# 0 = product present from the beginning (full history)
print("Creating new product introduction flags...")
intro_flags = (first_nonzero_week > min_week).astype(int).rename("IsNewIntro")

# Merge flag back into the main dataframe
# This adds the IsNewIntro column to every row based on product-country pair
print("Merging flags with main dataset...")
df = df.merge(intro_flags, left_on=["Forecasting Group", "Country"], right_index=True)

# Calculate statistics on new product introductions
# Count unique product-country pairs and how many are new introductions
print("Calculating new product statistics...")

# Total number of unique product-country combinations
total_pairs = df[["Forecasting Group", "Country"]].drop_duplicates().shape[0]

# Count of pairs flagged as new introductions
new_intro_count = intro_flags.sum()

# Percentage of product-country pairs that are new introductions
new_intro_pct = 100 * new_intro_count / total_pairs

# Display summary statistics
print(f"\n===== NEW PRODUCT INTRODUCTION STATISTICS =====")
print(f"Total Forecasting Group-Country pairs: {total_pairs:,}")
print(f"New product introductions: {new_intro_count:,}")
print(f"Share of new introductions: {new_intro_pct:.2f}%")

# Additional statistics by country
print("\nNew introduction rates by country:")
country_stats = df.groupby('Country')[['IsNewIntro']].mean() * 100
print(country_stats)

# Filter the dataset to keep only full-history product-country pairs
# These are products that were present from the very beginning of the dataset
print("\nFiltering out new product introductions...")
df_filtered = df[df["IsNewIntro"] == 0].copy()
print(f"Removed {len(df) - len(df_filtered):,} records of newly introduced products")
print(f"Retained {len(df_filtered):,} records with full history ({len(df_filtered)/len(df):.1%} of original)")

# Save the filtered dataset
output_file = "sales_data_full_history_only.csv"
print(f"\nSaving filtered dataset to {output_file}...")
df_filtered.to_csv(output_file, index=False)
print(f"Successfully saved {len(df_filtered):,} records")
print(f"Full-history products: {df_filtered['Forecasting Group'].nunique():,}")
print(f"Countries: {', '.join(df_filtered['Country'].unique())}")
print(f"Time range: {df_filtered['Sales_week'].min()} to {df_filtered['Sales_week'].max()}")
print("\nFiltering complete. Dataset ready for downstream analysis.")

# Verify integrity
assert all(df_filtered['IsNewIntro'] == 0), "Filtered dataset should only contain full-history products"
