#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Visualization Script

This script creates visualizations of detected promotional periods in retail sales data.
It shows weekly sales time series with highlighted promotional points, allowing for visual
validation of the promotion detection algorithm's performance.

Key Features:
- Loads promotion-flagged data from the tunable STL detection script
- Creates time series plots with highlighted promotion points
- Filters data by country and year for focused analysis
- Visualizes products with the most detected promotions
- Supports visual validation of promotion detection quality

Input:
- tunable_stl_promo_flagged.csv: Output from tunable STL promo detection script

Usage:
    python visualize_promo_flags.py
    
    Optional arguments can be added directly in the script:
    - year_to_plot: Year to visualize (default: 2014)
    - top_n_products: Number of products to plot per country (default: 2)
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Define function to parse command line arguments
def parse_arguments():
    """
    Parse command line arguments for visualization parameters.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Visualize detected promotions in sales data')
    parser.add_argument('--input', type=str, default='tunable_stl_promo_flagged.csv',
                        help='Path to promotion-flagged input CSV file')
    parser.add_argument('--year', type=int, default=2014,
                        help='Year to visualize (e.g., 2014)')
    parser.add_argument('--top-products', type=int, default=2,
                        help='Number of top products to visualize per country')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of displaying')
    parser.add_argument('--output-dir', type=str, default='promo_plots',
                        help='Directory to save plots if --save-plots is used')
    return parser.parse_args()

# Main execution function
def main():
    """
    Main execution function for promotion visualization.
    
    This function loads the promotion-flagged data, processes it, and creates
    visualizations of time series with highlighted promotion points.
    """
    # Get command line arguments
    args = parse_arguments()
    
    # Create output directory if saving plots
    if args.save_plots:
        Path(args.output_dir).mkdir(exist_ok=True)
    
    print(f"Loading promotion-flagged dataset from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} records")

    # Convert Sales_week (YYYY-WW) to datetime
    # Adding "-1" to represent Monday as the first day of the ISO week
    print("Converting year-week format to datetime...")
    try:
        df["Date"] = pd.to_datetime(df["Sales_week"] + "-1", format="%Y-%W-%w")
        print("Date conversion successful.")
    except Exception as e:
        print(f"Warning: Date conversion error - {e}")
        print("Attempting alternative date parsing...")
        # Alternative approach: extract year and week, then convert
        df["year"] = df["Sales_week"].str.split("-").str[0].astype(int)
        df["week"] = df["Sales_week"].str.split("-").str[1].astype(int)
        df["Date"] = df.apply(
            lambda row: pd.to_datetime(f"{row['year']}-W{row['week']:02d}-1", format="%G-W%V-%u"),
            axis=1
        )
        print("Alternative date conversion successful.")

    # Set the year to plot from command line argument
    year_to_plot = args.year
    top_n_products = args.top_products
    
    print(f"Visualizing promotions for year {year_to_plot}")
    print(f"Will show top {top_n_products} products per country with most promotions")

    # Loop through each country in the dataset
    print("\nGenerating visualizations by country:")
    countries = df["Country"].unique()
    print(f"Found {len(countries)} countries: {', '.join(countries)}")
    
    for country in countries:
        print(f"\nProcessing {country}...")
        country_df = df[df["Country"] == country]

        # Pick top N Forecasting Groups per country with most promo spikes
        # These are the products with the most detected promotional periods
        print(f"Finding top {top_n_products} products with most promotions in {country}...")
        top_fg = (
            country_df[country_df["LikelyPromo"] == 1]["Forecasting Group"]
            .value_counts()
            .head(top_n_products)
            .index
            .tolist()
        )
        
        if not top_fg:
            print(f"No products with promotions found in {country}, skipping.")
            continue
            
        print(f"Selected products: {', '.join(top_fg)}")

        # Create visualizations for each selected product
        for fg in top_fg:
            print(f"  Creating visualization for {fg}...")
            
            # Get data for this product and sort chronologically
            fg_df = country_df[country_df["Forecasting Group"] == fg].sort_values("Date")
            
            # Filter to the selected year
            fg_year_df = fg_df[fg_df["Date"].dt.year == year_to_plot]
            
            if len(fg_year_df) == 0:
                print(f"  No data for {fg} in year {year_to_plot}, skipping.")
                continue
                
            # Count promotions in this view
            promo_count = fg_year_df["LikelyPromo"].sum()
            print(f"  Found {promo_count} promotions for this product in {year_to_plot}")

            # Create figure with appropriate size
            plt.figure(figsize=(14, 5))
            
            # Plot the full sales time series
            plt.plot(fg_year_df["Date"], fg_year_df["Sales_volume"], 
                    marker='o', markersize=4, label="Weekly Sales", color="blue")

            # Highlight the detected promotion points
            promo_points = fg_year_df[fg_year_df["LikelyPromo"] == 1]
            
            # Include confidence information if available
            if 'PromoConfidence' in promo_points.columns:
                # Scale marker size based on confidence
                sizes = promo_points['PromoConfidence'] / 5 + 40  # Scale for visibility
                plt.scatter(
                    promo_points["Date"],
                    promo_points["Sales_volume"],
                    s=sizes,
                    color="red",
                    alpha=0.7,
                    label="Detected Promotion",
                    zorder=5
                )
                
                # Add confidence annotations
                for _, row in promo_points.iterrows():
                    plt.annotate(
                        f"{int(row['PromoConfidence'])}",
                        (row['Date'], row['Sales_volume']),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8
                    )
            else:
                # Simple highlighting without confidence info
                plt.scatter(
                    promo_points["Date"],
                    promo_points["Sales_volume"],
                    color="red",
                    s=80,
                    label="Detected Promotion",
                    zorder=5
                )

            # Add plot labels and formatting
            plt.title(f"{fg} ({country}) – Weekly Sales with Detected Promotions ({year_to_plot})")
            plt.xlabel("Week")
            plt.ylabel("Sales Volume")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save or display the plot
            if args.save_plots:
                filename = f"{args.output_dir}/{country}_{fg}_{year_to_plot}.png"
                filename = filename.replace(" ", "_").replace("/", "_").lower()
                plt.savefig(filename, dpi=150)
                print(f"  Saved plot to {filename}")
                plt.close()
            else:
                plt.show()

# Execute main function if run as script
if __name__ == "__main__":
    main()
