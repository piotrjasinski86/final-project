# Promotion Cluster Evaluation Report

Generated on: 2025-08-31 17:16

## Clustering Quality Metrics

**Silhouette Score (excluding noise):** 0.1013 (poor cluster separation)

**Total Records:** 25,000

**Noise Points:** 64 (0.3%)

**Clusters Found:** 2

## Cluster Profiles with Estimated Uplift

This table shows key characteristics of each detected promotion cluster, including:

- **Count:** Number of promotions in each cluster
- **Sales Statistics:** Average sales and standard deviation
- **Seasonality:** Number of unique months when these promotions occur
- **Geography:** Number of countries where these promotions appear
- **Uplift:** Estimated sales increase compared to baseline

|   Cluster | Type   |   Count |   Avg Sales |   Avg Uplift % |   Countries |   Months |
|-----------|--------|---------|-------------|----------------|-------------|----------|
|        -1 | Noise  |      64 |    187.047  |       1.10149  |           3 |        2 |
|         0 | Large  |   19548 |     43.8187 |       0.345797 |           3 |       12 |
|         1 | Large  |    5388 |     50.0876 |       0.366146 |           2 |       12 |

## Interpretation Guidelines

- **High Uplift Clusters:** Promotions with >50% average sales increase
- **Large Clusters:** Common promotion patterns (>10% of all promotions)
- **Noise:** Outlier promotions that don't fit well in any cluster
- **Regular:** Standard promotion patterns

Higher silhouette scores indicate better-defined, more distinct clusters.
