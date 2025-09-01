#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Sales to Weekly Aggregation Script

This script transforms raw daily M5 competition sales data into weekly aggregated format.
It converts the wide-format daily columns (d_1, d_2, etc.) into a long-format dataset with
weekly sales totals, properly labeled with year-week identifiers.

Key Features:
- Temporal aggregation from daily to weekly data
- Format transformation from wide to long format
- Country code standardization to readable names
- Progress tracking for processing long time series
- Memory-efficient processing of large datasets

Input:
- sales_data_original.csv: Raw M5 competition data with daily sales columns

Output:
- sales_data_weeks.csv: Transformed weekly sales data in long format with columns:
  * Forecasting Group: Product identifier
  * Category: Product category
  * Country: Standardized country name
  * Sales_week: Year-week identifier in YYYY-WW format
  * Sales_volume: Weekly aggregated sales quantity

Usage:
    python daily_sales_to_weekly.py
"""

import pandas as pd
from tqdm import tqdm
import warnings

# Suppress pandas PerformanceWarning related to DataFrame.append() deprecation
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Step 1: Define file paths and read original data
input_path = 'sales_data_original.csv'
output_path = 'sales_data_weeks.csv'

# Read only the header to identify columns (memory-efficient approach)
header = pd.read_csv(input_path, nrows=0)
columns = header.columns.tolist()

# Identify day columns (columns starting with 'd_')
day_columns = [col for col in columns if col.startswith('d_')]
num_days = len(day_columns)
num_weeks = num_days // 7  # Integer division to get complete weeks

# Calculate and print years of data
# Note: The M5 competition data starts from 2014-01-01
start_year = 2014  # Known start year for M5 competition data
end_year = start_year + (num_days - 1) // 365  # Approximate calendar year span
print(f"Total days in dataset: {num_days}")
print(f"Approximate time span: {start_year} to {end_year}")
print(f"Approximate number of years: {end_year - start_year + 1}")

# Read only necessary columns to conserve memory
# First 6 columns contain metadata (item_id, cat_id, state_id, etc.)
usecols = columns[:6] + day_columns
df = pd.read_csv(input_path, usecols=usecols)

# Step 2: Summarize to weekly sales
# Create week identifiers in YYYY-WW format (e.g., 2014-01, 2014-02, etc.)
yearly_weekly_labels = []
weeks_per_year = 52  # Standard number of weeks per year

print("Aggregating daily sales to weekly totals...")
for i in tqdm(range(num_weeks), desc="Weeks processed"):
    # Calculate year and week number
    year = start_year + (i // weeks_per_year)
    week = (i % weeks_per_year) + 1
    
    # Format as YYYY-WW with leading zero for single-digit weeks
    yearweek = f"{year}-{week:02d}"
    yearly_weekly_labels.append(yearweek)
    
    # Get the 7 daily columns for this week and sum them
    week_days = day_columns[i*7:(i+1)*7]
    df[yearweek] = df[week_days].sum(axis=1)  # Sum across the 7 days

# Handle leftover days (if total days isn't divisible by 7)
if num_days % 7 != 0:
    # Get remaining days that don't form a complete week
    week_days = day_columns[num_weeks*7:]
    
    # Create label for the partial week
    last_label = f"{start_year + (num_weeks // weeks_per_year)}-{(num_weeks % weeks_per_year) + 1:02d}"
    
    # Sum the remaining days and add to the dataframe
    df[last_label] = df[week_days].sum(axis=1)
    yearly_weekly_labels.append(last_label)

print("Weekly aggregation complete.")

# Step 3: Group by required columns and rename for better readability
# Map original column names to more descriptive business terms
rename_dict = {
    'item_id': 'Forecasting Group',  # Product identifier
    'cat_id': 'Category',            # Product category
    'state_id': 'Country'            # Market/country identifier
}
group_cols = ['Forecasting Group', 'Category', 'Country']  # Columns to keep as identifiers

# Aggregate data by product, category, and country
# This sums up sales for identical items across different stores/locations
grouped = df.groupby(['item_id', 'cat_id', 'state_id'])[yearly_weekly_labels].sum().reset_index()

# Apply the column renaming for readability
grouped = grouped.rename(columns=rename_dict)

# Map country codes to actual country names for better interpretability
# Note: In the M5 competition, US state codes were used, but we map them to our countries
grouped['Country'] = grouped['Country'].replace({'CA': 'Poland', 'TX': 'Switzerland', 'WI': 'Denmark'})

# Step 4: Transform from wide to long format
# This converts from one row per product with many week columns
# to multiple rows per product with a single sales value per row
long_df = grouped.melt(id_vars=group_cols,              # Identifier columns to keep
                      value_vars=yearly_weekly_labels,  # Week columns to unpivot
                      var_name='Sales_week',           # Name for the week column
                      value_name='Sales_volume')       # Name for the sales value column

# Step 5: Save the final output
long_df.to_csv(output_path, index=False)
print(f"Pipeline complete. Final file saved to {output_path}")
print(f"Total weekly records: {len(long_df)}")
print(f"Date range: {min(yearly_weekly_labels)} to {max(yearly_weekly_labels)}")
print(f"Countries: {', '.join(long_df['Country'].unique())}")
print(f"Categories: {', '.join(long_df['Category'].unique())}")
