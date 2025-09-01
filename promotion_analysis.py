#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Pattern Analysis Script

This script analyzes historical promotion patterns from the clustered promotion data.
It identifies seasonal trends, promotion frequencies, and periodicity patterns
that can be used to inform the promotion prediction models.

Input:
- tunable_promo_clustered.csv: Clustered promotion data

Output:
- Reports directory with various analysis plots and CSV files:
  - promo_seasonality.png: Seasonality of promotions by week of year
  - cluster_frequency.png: Frequency of different promotion types
  - promotion_periodicity.csv: Analysis of time between promotions
  - cluster_calendar_heatmap.png: Heatmap of promotion types by week of year
  - country_cluster_distribution.csv: Promotion types by country

Usage:
    python promotion_analysis.py [--input FILENAME] [--reports DIRECTORY]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import timedelta


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze promotion patterns')
    parser.add_argument('--input', type=str, default='tunable_promo_clustered.csv',
                       help='Path to clustered promotions file')
    parser.add_argument('--reports', type=str, default='reports',
                       help='Directory to save analysis reports')
    return parser.parse_args()


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_promo_data(file_path):
    """Load promotion data and prepare for analysis."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Promotion data file not found: {file_path}")
    
    print(f"Loading promotion data from {file_path}...")
    promos = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = ['Forecasting Group', 'Country', 'Sales_week', 'promo_cluster']
    missing_cols = [col for col in required_cols if col not in promos.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Ensure date format
    promos['Sales_week'] = promos['Sales_week'].apply(
    lambda x: pd.to_datetime(f"{x}-1", format="%Y-%W-%w")
    )
    
    # Add derived time features
    promos['Year'] = promos['Sales_week'].dt.year
    promos['Quarter'] = promos['Sales_week'].dt.quarter
    promos['Month'] = promos['Sales_week'].dt.month
    
    # If WeekOfYear is not already present, add it
    if 'WeekOfYear' not in promos.columns:
        promos['WeekOfYear'] = promos['Sales_week'].dt.isocalendar().week
    
    print(f"Loaded {len(promos)} promotion records")
    print(f"Promotion periods span from {promos['Sales_week'].min()} to {promos['Sales_week'].max()}")
    print(f"Found {promos['promo_cluster'].nunique()} unique cluster types")
    
    return promos


def analyze_weekly_seasonality(promos, report_dir):
    """Analyze promotion frequency by week of year."""
    print("\nAnalyzing weekly seasonality...")
    
    # By week of year (seasonality)
    woy_freq = promos.groupby('WeekOfYear')['promo_cluster'].count()
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(woy_freq.index, woy_freq.values, color='skyblue', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(woy_freq.index, woy_freq.values, 2)
    p = np.poly1d(z)
    plt.plot(woy_freq.index, p(woy_freq.index), "r--", linewidth=2)
    
    # Highlight top weeks
    top_weeks = woy_freq.nlargest(5)
    for week, count in top_weeks.items():
        plt.annotate(f'Week {week}\n({count})', 
                     xy=(week, count),
                     xytext=(0, 15),
                     textcoords='offset points',
                     ha='center',
                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    
    plt.title('Promotion Frequency by Week of Year', fontsize=14)
    plt.xlabel('Week of Year', fontsize=12)
    plt.ylabel('Number of Promotions', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 53)
    plt.xticks(np.arange(0, 53, 4))
    
    # Add season markers
    plt.axvspan(0, 9, alpha=0.2, color='lightblue', label='Winter')
    plt.axvspan(9, 22, alpha=0.2, color='lightgreen', label='Spring')
    plt.axvspan(22, 35, alpha=0.2, color='yellow', label='Summer')
    plt.axvspan(35, 48, alpha=0.2, color='orange', label='Fall')
    plt.axvspan(48, 53, alpha=0.2, color='lightblue')
    
    plt.legend()
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(report_dir, 'promo_seasonality.png')
    plt.savefig(output_path, dpi=150)
    print(f"Seasonality plot saved to {output_path}")
    
    # Return key statistics
    return {
        'weekly_seasonality': woy_freq.to_dict(),
        'top_promo_weeks': top_weeks.to_dict()
    }


def analyze_cluster_frequency(promos, report_dir):
    """Analyze frequency of different promotion types (clusters)."""
    print("\nAnalyzing cluster frequency...")
    
    # By cluster type
    cluster_freq = promos['promo_cluster'].value_counts().sort_index()
    total_promos = len(promos)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_freq.index.astype(str), cluster_freq.values, color='lightcoral', alpha=0.7)
    
    # Add percentage labels
    for i, (cluster, count) in enumerate(cluster_freq.items()):
        percentage = 100 * count / total_promos
        plt.annotate(f'{percentage:.1f}%', 
                     xy=(i, count),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center')
    
    plt.title('Frequency of Different Promotion Types', fontsize=14)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(report_dir, 'cluster_frequency.png')
    plt.savefig(output_path, dpi=150)
    print(f"Cluster frequency plot saved to {output_path}")
    
    # Return key statistics
    return {
        'cluster_distribution': cluster_freq.to_dict()
    }


def analyze_product_country_patterns(promos, report_dir):
    """Analyze promotion patterns by product-country combinations."""
    print("\nAnalyzing product-country patterns...")
    
    # By product-country combination
    product_country_freq = promos.groupby(['Forecasting Group', 'Country']).size().reset_index()
    product_country_freq.columns = ['Forecasting Group', 'Country', 'Promo_Count']
    product_country_freq = product_country_freq.sort_values('Promo_Count', ascending=False)
    
    # Calculate average promotions per product-country
    avg_promos_per_product_country = product_country_freq['Promo_Count'].mean()
    
    print(f"Average promotions per product-country: {avg_promos_per_product_country:.2f}")
    print(f"Top 10 product-country combinations by promotion frequency:")
    print(product_country_freq.head(10))
    
    # By country
    country_freq = promos.groupby('Country').size().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(country_freq.index, country_freq.values, color='lightgreen', alpha=0.7)
    plt.title('Promotion Frequency by Country', fontsize=14)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Number of Promotions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(report_dir, 'country_frequency.png')
    plt.savefig(output_path, dpi=150)
    
    # Analyze cluster distribution by country
    country_cluster = promos.groupby(['Country', 'promo_cluster']).size().unstack().fillna(0)
    
    # Save this data
    country_cluster_file = os.path.join(report_dir, 'country_cluster_distribution.csv')
    country_cluster.to_csv(country_cluster_file)
    print(f"Country-cluster distribution saved to {country_cluster_file}")
    
    # Return key statistics
    return {
        'product_country_frequency': product_country_freq.head(20).to_dict(),
        'country_frequency': country_freq.to_dict()
    }


def analyze_promotion_periodicity(promos, report_dir):
    """Analyze time patterns between consecutive promotions."""
    print("\nAnalyzing promotion periodicity...")
    
    product_groups = promos.groupby(['Forecasting Group', 'Country'])
    periodicities = []
    
    for name, group in product_groups:
        if len(group) >= 3:  # Need at least 3 promotions to analyze periodicity
            # Sort by date
            group = group.sort_values('Sales_week')
            
            # Calculate days between consecutive promotions
            days_between = group['Sales_week'].diff().dt.days
            
            # Get average, min, max periodicity
            periodicities.append({
                'Forecasting Group': name[0],
                'Country': name[1],
                'Avg_Days_Between_Promos': days_between.mean(),
                'Min_Days_Between_Promos': days_between.min(),
                'Max_Days_Between_Promos': days_between.max(),
                'Std_Days_Between_Promos': days_between.std(),
                'Promo_Count': len(group),
                'Most_Common_Cluster': group['promo_cluster'].value_counts().index[0]
            })
    
    periodicity_df = pd.DataFrame(periodicities)
    
    # Calculate overall statistics
    avg_days_between = periodicity_df['Avg_Days_Between_Promos'].mean()
    print(f"Average days between promotions across products: {avg_days_between:.2f}")
    
    # Create histogram of average days between promotions
    plt.figure(figsize=(12, 6))
    plt.hist(periodicity_df['Avg_Days_Between_Promos'], bins=20, color='lightblue', alpha=0.7)
    plt.axvline(avg_days_between, color='red', linestyle='--', 
                label=f'Mean: {avg_days_between:.1f} days')
    plt.title('Distribution of Average Days Between Promotions', fontsize=14)
    plt.xlabel('Days Between Promotions', fontsize=12)
    plt.ylabel('Number of Product-Country Pairs', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save
    periodicity_file = os.path.join(report_dir, 'promotion_periodicity.csv')
    periodicity_df.to_csv(periodicity_file, index=False)
    print(f"Periodicity analysis saved to {periodicity_file}")
    
    hist_file = os.path.join(report_dir, 'days_between_promos_histogram.png')
    plt.savefig(hist_file, dpi=150)
    
    # Return key statistics
    return {
        'periodicity_patterns': {
            'avg_days_between': avg_days_between,
            'min_days_between': periodicity_df['Min_Days_Between_Promos'].min(),
            'max_days_between': periodicity_df['Max_Days_Between_Promos'].max()
        }
    }


def create_cluster_calendar(promos, report_dir):
    """Create a calendar heatmap of promotion clusters by week of year."""
    print("\nCreating cluster calendar heatmap...")
    
    # Count promos by week and cluster
    cluster_calendar = promos.groupby(['WeekOfYear', 'promo_cluster']).size().unstack().fillna(0)
    
    # Normalize by week
    calendar_pct = cluster_calendar.div(cluster_calendar.sum(axis=1), axis=0)
    
    plt.figure(figsize=(14, 8))
    
    # Create heatmap
    ax = sns.heatmap(calendar_pct.T, cmap='viridis', annot=False, 
                     cbar_kws={'label': 'Proportion of Promotions'})
    
    plt.title('Promotion Cluster Types by Week of Year', fontsize=14)
    plt.xlabel('Week of Year', fontsize=12)
    plt.ylabel('Cluster ID', fontsize=12)
    
    # Add season markers
    ax.axvspan(0, 9, alpha=0.3, color='blue', label='Winter')
    ax.axvspan(9, 22, alpha=0.3, color='green', label='Spring')
    ax.axvspan(22, 35, alpha=0.3, color='yellow', label='Summer')
    ax.axvspan(35, 48, alpha=0.3, color='orange', label='Fall')
    ax.axvspan(48, 53, alpha=0.3, color='blue')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(report_dir, 'cluster_calendar_heatmap.png')
    plt.savefig(output_path, dpi=150)
    print(f"Cluster calendar heatmap saved to {output_path}")
    
    # Save the data
    calendar_file = os.path.join(report_dir, 'cluster_calendar.csv')
    cluster_calendar.to_csv(calendar_file)
    
    # Return key statistics
    return {
        'cluster_calendar': cluster_calendar.to_dict()
    }


def analyze_promotion_patterns(promos, report_dir):
    """Run comprehensive analysis on promotion patterns."""
    stats = {}
    
    # Weekly seasonality
    stats.update(analyze_weekly_seasonality(promos, report_dir))
    
    # Cluster frequency
    stats.update(analyze_cluster_frequency(promos, report_dir))
    
    # Product-country patterns
    stats.update(analyze_product_country_patterns(promos, report_dir))
    
    # Promotion periodicity
    stats.update(analyze_promotion_periodicity(promos, report_dir))
    
    # Cluster calendar
    stats.update(create_cluster_calendar(promos, report_dir))
    
    # Generate summary report
    generate_summary_report(stats, promos, report_dir)
    
    return stats


def generate_summary_report(stats, promos, report_dir):
    """Generate a text summary report of key findings."""
    print("\nGenerating summary report...")
    
    report_path = os.path.join(report_dir, 'promotion_analysis_summary.md')
    
    with open(report_path, 'w') as f:
        f.write('# Promotion Pattern Analysis Summary\n\n')
        
        f.write('## Dataset Overview\n')
        f.write(f'- Total promotion records: {len(promos):,}\n')
        f.write(f'- Date range: {promos["Sales_week"].min().date()} to {promos["Sales_week"].max().date()}\n')
        f.write(f'- Unique products: {promos["Forecasting Group"].nunique():,}\n')
        f.write(f'- Countries: {promos["Country"].nunique()}\n')
        f.write(f'- Promotion cluster types: {promos["promo_cluster"].nunique()}\n\n')
        
        f.write('## Seasonal Patterns\n')
        top_weeks = sorted(stats['top_promo_weeks'].items())
        f.write('- Top 5 weeks for promotions:\n')
        for week, count in top_weeks:
            f.write(f'  - Week {week}: {count} promotions\n')
        
        f.write('\n## Promotion Clusters\n')
        total_promos = len(promos)
        for cluster, count in sorted(stats['cluster_distribution'].items()):
            f.write(f'- Cluster {cluster}: {count} promotions ({100*count/total_promos:.1f}%)\n')
        
        f.write('\n## Promotion Periodicity\n')
        periodicity = stats['periodicity_patterns']
        f.write(f'- Average days between promotions: {periodicity["avg_days_between"]:.1f}\n')
        f.write(f'- Minimum days between promotions: {periodicity["min_days_between"]:.1f}\n')
        f.write(f'- Maximum days between promotions: {periodicity["max_days_between"]:.1f}\n')
        
        f.write('\n## Key Insights for Promotion Prediction\n')
        f.write('1. There is clear seasonality in promotion frequency, with peaks around weeks ')
        f.write(', '.join([str(week) for week, _ in top_weeks[:3]]))
        f.write('\n2. Promotion clusters are not evenly distributed, with some types being much more common\n')
        f.write(f'3. The average time between promotions is {periodicity["avg_days_between"]:.1f} days\n')
        f.write('4. Different countries show different preferences for promotion types\n')
        f.write('5. The distribution of cluster types varies throughout the year, suggesting seasonal patterns in promotion types\n')
    
    print(f"Summary report saved to {report_path}")


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure reports directory exists
    ensure_directory(args.reports)
    
    try:
        # Load data
        promos = load_promo_data(args.input)
        
        # Analyze patterns
        stats = analyze_promotion_patterns(promos, args.reports)
        
        print("\nPromotion pattern analysis complete!")
        print(f"All reports saved to {args.reports} directory")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
