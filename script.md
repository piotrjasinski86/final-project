# Data Processing Pipeline - Script Workflow

This document describes the sequential execution order and purpose of each script in the promotion detection and clustering pipeline.

## 0. sales_data_original.csv
**Input file:** Raw M5 competition dataset with daily sales data in wide format (d_1, d_2, ... d_1913 columns).

## 1. python daily_sales_to_weekly.py
**Purpose:** Data transformation and aggregation  
**Description:** This script transforms raw daily sales data into weekly sales totals for each product, category, and country. It reshapes the data from wide format (daily columns) into a long, analysis-friendly format and standardizes region codes to readable country names. The result is a single, clean dataset ready for time series analysis or modeling.  
**Input:** `sales_data_original.csv`  
**Output:** `sales_data_weeks.csv`

## 2. python split_weekly_sales_train_eval.py
**Purpose:** Train/test data splitting  
**Description:** This script takes the weekly sales data and splits it into separate training and evaluation datasets based on the year. It extracts the year from each weekly label and creates two files: one for training (up to 2017) and one for evaluation (2018 and later). This ensures your modeling pipeline uses proper historical data for training and reserves the most recent data for out-of-sample testing.  
**Input:** `sales_data_weeks.csv`  
**Output:** `sales_data_train.csv`, `sales_data_eval.csv`

## 3. python new_product_flag.py
**Purpose:** Data quality control and filtering  
**Description:** This script flags and removes newly introduced products for each product–country pair in the sales data. It identifies when each product first appears in every country and retains only those with a complete sales history from the start of the dataset. The result is a cleaned dataset containing only full-history product–country combinations, ready for reliable downstream analysis.  
**Input:** `sales_data_train.csv`  
**Output:** `sales_data_full_history_only.csv`

## 4. python tunable_stl_promo_detection.py
**Purpose:** Promotional period detection using advanced statistical methods  
**Description:** This script automatically detects promotional periods in retail sales data using STL decomposition and robust statistical methods. It decomposes weekly sales time series into trend, seasonal, and residual components, then applies configurable z-score and quantile thresholds to identify sales spikes that likely represent promotions. The script includes multiple tunable parameters (sensitivity thresholds, cooling periods, maximum promotion rates) to reduce false positives, filters out newly introduced products to avoid mislabeling launch spikes, and outputs promotion flags with confidence scores for downstream clustering and forecasting analysis.  
**Input:** `sales_data_full_history_only.csv`  
**Output:** `tunable_stl_promo_flagged.csv`, `tunable_promo_detection_metrics.csv`

## 5. python visualize_promo_flags.py
**Purpose:** Quality assurance and visual validation  
**Description:** This script creates interactive visualizations to validate the promotion detection results. It generates plots showing the original sales time series overlaid with detected promotion points, confidence-based marker sizing and annotations, and flexible filtering by country, year, and top products. These visualizations allow manual inspection of detection quality and help identify potential false positives or missed promotions for algorithm tuning. The script supports both interactive display and automated plot saving for documentation purposes.  
**Input:** `tunable_stl_promo_flagged.csv`  
**Output:** Interactive matplotlib plots and saved visualization files (optional)

## 6. python clustering_promos.py
**Purpose:** Unsupervised promotion type discovery  
**Description:** This script clusters detected promotional periods into meaningful types using Gower distance and HDBSCAN algorithms. It handles mixed categorical and numerical features (temporal context, product categories, sales volumes, confidence scores), computes distance matrices using batch processing for memory efficiency, and applies density-based clustering to discover natural promotion archetypes. The result is promotion data labeled with cluster IDs representing different promotional patterns, ready for business interpretation and forecasting integration.  
**Input:** `tunable_stl_promo_flagged.csv`  
**Output:** `tunable_promo_clustered.csv`

## 7. python cluster_evaluation.py
**Purpose:** Cluster quality assessment and business interpretation  
**Description:** This script evaluates the quality and business interpretability of the discovered promotion clusters through comprehensive statistical analysis. It calculates clustering validation metrics (silhouette scores using Gower distance, noise ratios), generates detailed cluster profiles showing average characteristics (uplift percentages, seasonality patterns, geographic distribution), estimates promotional uplift relative to baseline sales, and produces structured summary reports. The output helps interpret what each cluster represents in business terms and validates the meaningfulness of the discovered promotional archetypes.  
**Input:** `tunable_promo_clustered.csv`, `m5_promo_flagged_stl_poland.csv` (for baseline calculation)  
**Output:** `cluster_eval_report.md`, `cluster_stats.csv`

## 8. python cluster_visualization.py
**Purpose:** Cluster visualization and pattern exploration  
**Description:** This script creates comprehensive visual representations of the promotion clusters using advanced dimensionality reduction and visualization techniques. It applies UMAP to project the high-dimensional promotional feature space into interpretable 2D plots colored by cluster assignments, generates cluster distribution summaries with noise analysis, and provides sample examples from top clusters for qualitative validation. These visualizations support both technical validation of clustering quality and business presentation of discovered promotional patterns.  
**Input:** `tunable_promo_clustered.csv`  
**Output:** UMAP cluster plots, cluster distribution summaries, sample examples display

## 9. python evaluate_forecast_performance.py
**Purpose:** Research Question 2 evaluation - Forecasting comparison with promotion clusters
**Description:** This script directly addresses the core research question by comparing forecasting performance with and without promotion cluster information. It implements a controlled comparison using XGBoost models with three feature sets: baseline (lagged sales only), binary promotion flags, and cluster-based promotion features. The script attempts to merge real promotional data from the clustering pipeline, with fallback to simulated promotional uplift effects if needed. It trains models on historical data, evaluates performance using RMSE and MAE metrics, and generates comprehensive academic results. The output provides direct evidence for whether promotion clustering improves forecasting accuracy and quantifies the magnitude of improvement.
**Input:** sales_data_eval.csv, tunable_promo_clustered.csv
**Output:** forecast_performance_results.csv, rq2_academic_summary.csv

---

## Pipeline Summary

**Data Flow:**
```
sales_data_original.csv
    ↓ (daily_sales_to_weekly.py)
sales_data_weeks.csv
    ↓ (split_weekly_sales_train_eval.py)
sales_data_train.csv + sales_data_eval.csv
    ↓ (new_product_flag.py)
sales_data_full_history_only.csv
    ↓ (tunable_stl_promo_detection.py)
tunable_stl_promo_flagged.csv + tunable_promo_detection_metrics.csv
    ↓ (visualize_promo_flags.py - optional validation)
tunable_stl_promo_flagged.csv
    ↓ (clustering_promos.py)
tunable_promo_clustered.csv
    ↓ (cluster_evaluation.py + cluster_visualization.py)
cluster_eval_report.md + cluster_stats.csv + UMAP visualizations
    ↓ (promotion_analysis.py)
promotion_patterns_report.md + pattern visualizations
    ↓ (promo_feature_engineering.py)
promo_prediction_features.csv + feature engineering reports
    ↓ (promo_prediction_models.py)
trained models (binary + multiclass) + model evaluation reports
    ↓ (future_promotion_prediction.py)
future_promotion_calendar.csv + forecast visualizations
```

**Validation and Quality Assurance:**
- **Step 5**: Visual validation of promotion detection quality
- **Step 7**: Statistical validation of cluster quality and business interpretation
- **Step 8**: Visual validation of cluster separation and pattern discovery
- **Step 9**: Analysis of historical promotion patterns and seasonality
- **Step 10**: Feature correlation analysis and importance validation
- **Step 11**: Model performance evaluation with train/validation/test splits
- **Step 12**: Visual validation of future promotion predictions

**Key Design Principles:**
- **Reproducibility:** Each script reads from clearly defined inputs and produces standardized outputs with consistent naming conventions
- **Modularity:** Pipeline can be run step-by-step or individual components can be re-executed for iterative development
- **Quality Control:** Multiple validation and visualization steps ensure data quality at each stage
- **Tunability:** Key parameters can be adjusted through command-line arguments without modifying core logic
- **Scalability:** Batch processing and memory management techniques enable application to large retail datasets
- **Documentation:** Comprehensive logging and reporting support academic reproducibility requirements

**Output File Structure:**
```
├── Intermediate Data Files
│   ├── sales_data_weeks.csv
│   ├── sales_data_train.csv, sales_data_eval.csv
│   ├── sales_data_full_history_only.csv
│   ├── tunable_stl_promo_flagged.csv
│   ├── tunable_promo_clustered.csv
│   └── promo_prediction_features.csv
├── Analysis Results
│   ├── tunable_promo_detection_metrics.csv
│   ├── cluster_stats.csv
│   └── future_promotion_calendar.csv
├── Models
│   ├── promo_occurrence_model.pkl
│   ├── promo_cluster_model.pkl
│   └── feature_columns.pkl
├── Reports and Documentation
│   ├── cluster_eval_report.md
│   ├── promotion_patterns_report.md
│   ├── feature_engineering_report.md
│   ├── model_evaluation_report.md
│   └── promotion_calendar_summary.md
└── Visualizations
    ├── Promotion validation plots (optional save)
    ├── UMAP cluster visualizations
    ├── Promotion pattern plots
    ├── Feature correlation and importance plots
    ├── Model performance plots
    └── Future promotion forecast visualizations
```

**Total Processing Time:** Approximately 20-35 minutes for full M5 dataset on standard hardware, including validation steps

**Memory Requirements:** Peak usage during Gower distance calculation (~4-6GB for 25,000 promotional periods), with batch processing to manage memory constraints

---

# Promotion Prediction Pipeline

This section describes the scripts and workflow for analyzing historical promotion patterns and predicting future promotions.

## 10. python promotion_analysis.py
**Purpose:** Historical promotion pattern analysis  
**Description:** This script analyzes historical promotion patterns from clustered promotional data to identify seasonal trends, promotional frequencies, and periodicity patterns. It computes weekly seasonality with bar plots and trend lines, analyzes cluster frequency distributions, examines promotion patterns by product-country combinations, calculates promotion periodicity metrics (time between promotions), and creates calendar heatmaps of cluster distribution. All findings are synthesized into a comprehensive markdown report with supporting visualizations and CSV summaries for data-driven promotion planning.  
**Input:** `tunable_promo_clustered.csv`  
**Output:** `reports/promotion_patterns_report.md`, `reports/promotion_patterns_*.png`, `reports/promotion_patterns_*.csv`

## 11. python promo_feature_engineering.py
**Purpose:** Feature preparation for promotion prediction models  
**Description:** This script prepares a feature-rich dataset for training promotion prediction models by merging sales data, promotion flags, and clustered promotions. It creates time-based cyclical features (sin/cos encodings of week, month, quarter), lag features for promotion occurrence and cluster type, rolling window features (promotion density, days since last promotion, common cluster types), categorical encodings (one-hot for country, label encoding for product), and holiday indicators. The script also visualizes feature correlations and importances, generates feature engineering reports, and outputs a prepared dataset ready for model training.  
**Input:** `tunable_stl_promo_flagged.csv`, `tunable_promo_clustered.csv`  
**Output:** `promo_prediction_features.csv`, `reports/feature_correlation.png`, `reports/feature_engineering_report.md`

## 12. python promo_prediction_models.py
**Purpose:** Train and evaluate promotion prediction models  
**Description:** This script trains and evaluates two XGBoost models: a binary classifier for promotion occurrence and a multiclass classifier for promotion cluster type. It implements time-based train/validation/test splits to avoid data leakage, performs hyperparameter tuning for both models, generates comprehensive evaluation metrics (accuracy, precision, recall, F1-score) and confusion matrices, plots feature importances, and produces detailed markdown reports on model performance. The trained models and feature columns are saved for later use in future promotion prediction.  
**Input:** `promo_prediction_features.csv`  
**Output:** `models/promo_occurrence_model.pkl`, `models/promo_cluster_model.pkl`, `models/feature_columns.pkl`, `reports/model_evaluation_report.md`

## 13. python future_promotion_prediction.py
**Purpose:** Generate predictions of future promotions  
**Description:** This script generates predictions of future promotions by loading trained promotion occurrence and cluster type models, preparing a future dates dataframe with product-country combinations, iteratively predicting week-by-week while updating features as it goes (incorporating feedback from each prediction into the next), and producing a comprehensive future promotion calendar. It creates visualizations including promotion density over time, cluster type distribution, country-specific heatmaps, and example product-country forecasts, along with a markdown summary report detailing the predicted promotions for use in sales forecasting.  
**Inpsut:** `promo_prediction_features.csv`, `models/promo_occurrence_model.pkl`, `models/promo_cluster_model.pkl`, `models/feature_columns.pkl`  
**Output:** `future_promotion_calendar.csv`, `reports/future_promotion_*.png`, `reports/promotion_calendar_summary.md`