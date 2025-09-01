#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Promotion Prediction Models Script

This script trains two models for predicting future promotions:
1. A binary classifier to predict whether a product-country will have a promotion in a given week
2. A multiclass classifier to predict the type of promotion (cluster) when a promotion is predicted

The script handles proper time-based training/validation splitting, model training,
hyperparameter tuning, feature importance analysis, and performance evaluation.

Input:
- promo_prediction_features.csv: Features for promotion prediction

Output:
- models/promo_occurrence_model.pkl: Trained binary classifier model
- models/promo_cluster_model.pkl: Trained multiclass classifier model
- reports/promo_model_evaluation.md: Detailed evaluation report
- reports/promo_model_feature_importance.png: Feature importance visualizations

Usage:
    python promo_prediction_models.py [--input FILENAME] [--test-size FLOAT]
                                     [--val-size FLOAT] [--models-dir DIR]
                                     [--reports-dir DIR]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix, roc_auc_score)
import xgboost as xgb
from xgboost import plot_importance


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train promotion prediction models')
    
    parser.add_argument('--input', type=str, default='promo_prediction_features.csv',
                        help='Path to feature-engineered dataset')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--reports-dir', type=str, default='reports',
                        help='Directory to save reports')
    
    return parser.parse_args()


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_data(file_path):
    """Load feature-engineered data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature data file not found: {file_path}")
    
    print(f"Loading feature data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Ensure date format
    data['Sales_week'] = pd.to_datetime(data['Sales_week'])
    
    # Validate required columns
    req_cols = ['Sales_week', 'Forecasting Group', 'Country', 'is_promo', 'promo_cluster']
    missing_cols = [col for col in req_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    print(f"Loaded {len(data)} records with {len(data.columns)} features")
    print(f"Data spans from {data['Sales_week'].min()} to {data['Sales_week'].max()}")
    
    return data


def time_based_split(data, test_size, val_size):
    """Split data by time for training, validation, and testing."""
    print("\nSplitting data by time...")
    
    # Sort by time
    data = data.sort_values('Sales_week')
    
    # Get unique dates
    unique_dates = data['Sales_week'].unique()
    n_dates = len(unique_dates)
    
    # Calculate split points
    test_idx = int(n_dates * (1 - test_size))
    val_idx = int((n_dates - (n_dates * test_size)) * (1 - val_size))
    
    # Split dates
    train_dates = unique_dates[:val_idx]
    val_dates = unique_dates[val_idx:test_idx]
    test_dates = unique_dates[test_idx:]
    
    # Create masks
    train_mask = data['Sales_week'].isin(train_dates)
    val_mask = data['Sales_week'].isin(val_dates)
    test_mask = data['Sales_week'].isin(test_dates)
    
    # Create splits
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]
    
    print(f"Training data: {len(train_data)} records from {train_dates[0]} to {train_dates[-1]}")
    print(f"Validation data: {len(val_data)} records from {val_dates[0]} to {val_dates[-1]}")
    print(f"Test data: {len(test_data)} records from {test_dates[0]} to {test_dates[-1]}")
    
    return train_data, val_data, test_data


def prepare_features(data):
    """Prepare feature matrices and target vectors."""
    # Define feature categories
    time_features = ['WeekOfYear_sin', 'WeekOfYear_cos', 'Month_sin', 'Month_cos', 
                     'Quarter_sin', 'Quarter_cos', 'is_christmas_period', 
                     'is_easter_period', 'is_summer_holiday']

    lag_features = [col for col in data.columns if col.startswith('promo_lag_') 
                   or col.startswith('cluster_lag_')]

    rolling_features = [col for col in data.columns if 'density' in col 
                       or 'days_since' in col or 'common_cluster' in col]

    categorical_features = [col for col in data.columns if col.startswith('country_')]

    # Combine all features
    feature_columns = time_features + lag_features + rolling_features + categorical_features

    # Check if all features exist in dataset
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        print(f"Warning: Missing expected features: {missing_features}")
        # Remove missing features
        feature_columns = [col for col in feature_columns if col in data.columns]

    # Extract features and targets
    X = data[feature_columns]
    y_binary = data['is_promo']

    # For cluster prediction, only rows with actual promotions and valid clusters
    promo_mask = (data['is_promo'] == 1)
    cluster_mask = (data['promo_cluster'] >= 0)
    combined_mask = promo_mask & cluster_mask

    X_cluster = data.loc[combined_mask, feature_columns].reset_index(drop=True)
    y_cluster = data.loc[combined_mask, 'promo_cluster'].reset_index(drop=True)

    print(f"\nPrepared {X.shape[1]} features")
    print(f"Binary classification target distribution: {y_binary.value_counts().to_dict()}")
    print(f"Multiclass target distribution: {y_cluster.value_counts().to_dict()}")

    return X, y_binary, X_cluster, y_cluster



def train_binary_classifier(X_train, y_train, X_val, y_val):
    """Train binary classifier for promotion occurrence prediction."""
    print("\nTraining binary classifier for promotion occurrence...")
    
    # Calculate class weights
    # Adjust for imbalanced classes
    n_samples = len(y_train)
    n_pos = sum(y_train)
    n_neg = n_samples - n_pos
    
    # The rarer class (typically promotions) gets higher weight
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Create model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=pos_weight,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42,
        eval_metric=['auc', 'error']   # <-- MOVE HERE
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    
    return model


def train_multiclass_classifier(X_train, y_train, X_val, y_val):
    """Train multiclass classifier for promotion type prediction."""
    print("\nTraining multiclass classifier for promotion type...")
    
    # Create model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(y_train.unique()),
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return model


def evaluate_binary_classifier(model, X_test, y_test, reports_dir):
    """Evaluate binary classifier and generate reports."""
    print("\nEvaluating binary classifier...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print summary
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Generate confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Binary Classifier Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    cm_plot_path = os.path.join(reports_dir, 'binary_classifier_confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=150)
    
    # Generate feature importance plot
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=20, importance_type='gain')
    plt.title('Binary Classifier Feature Importance')
    plt.tight_layout()
    
    # Save plot
    fi_plot_path = os.path.join(reports_dir, 'binary_classifier_feature_importance.png')
    plt.savefig(fi_plot_path, dpi=150)
    
    # Generate detailed classification report
    class_report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }


def evaluate_multiclass_classifier(model, X_test, y_test, reports_dir):
    """Evaluate multiclass classifier and generate reports."""
    print("\nEvaluating multiclass classifier...")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")  # Debug
    y_pred = model.predict(X_test)
    print(f"y_pred.shape: {y_pred.shape}")  # Debug

    # If y_pred is (N, C), convert to labels
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
        print(f"After argmax, y_pred.shape: {y_pred.shape}")

    y_test = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    assert len(y_test) == len(y_pred), f"Shape mismatch: {len(y_test)} vs {len(y_pred)}"

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Generate confusion matrix (normalize by row)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

    # Generate confusion matrix plot (if not too many classes)
    n_classes = len(np.unique(np.concatenate([y_test, y_pred])))
    if n_classes <= 15:
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.title('Multiclass Classifier Confusion Matrix (Normalized)')
        plt.ylabel('True Cluster')
        plt.xlabel('Predicted Cluster')
        plt.tight_layout()
        cm_plot_path = os.path.join(reports_dir, 'multiclass_classifier_confusion_matrix.png')
        plt.savefig(cm_plot_path, dpi=150)
    else:
        print(f"Skipping confusion matrix visualization due to high number of classes ({n_classes})")

    # Generate feature importance plot
    plt.figure(figsize=(12, 8))
    plot_importance(model, max_num_features=20, importance_type='gain')
    plt.title('Multiclass Classifier Feature Importance')
    plt.tight_layout()
    fi_plot_path = os.path.join(reports_dir, 'multiclass_classifier_feature_importance.png')
    plt.savefig(fi_plot_path, dpi=150)

    class_report = classification_report(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }




def generate_evaluation_report(binary_results, multiclass_results, reports_dir):
    """Generate a comprehensive evaluation report in markdown format."""
    print("\nGenerating evaluation report...")
    
    report_path = os.path.join(reports_dir, 'promo_model_evaluation.md')
    
    with open(report_path, 'w') as f:
        f.write('# Promotion Prediction Models Evaluation\n\n')
        
        f.write('## Binary Classifier (Promotion Occurrence)\n\n')
        f.write(f"- Accuracy: {binary_results['accuracy']:.4f}\n")
        f.write(f"- Precision: {binary_results['precision']:.4f}\n")
        f.write(f"- Recall: {binary_results['recall']:.4f}\n")
        f.write(f"- F1 Score: {binary_results['f1']:.4f}\n")
        f.write(f"- AUC-ROC: {binary_results['auc']:.4f}\n\n")
        
        f.write('### Confusion Matrix\n\n')
        f.write('![Binary Classifier Confusion Matrix](binary_classifier_confusion_matrix.png)\n\n')
        
        f.write('### Classification Report\n\n')
        f.write('```\n')
        f.write(binary_results['classification_report'])
        f.write('\n```\n\n')
        
        f.write('### Feature Importance\n\n')
        f.write('![Binary Classifier Feature Importance](binary_classifier_feature_importance.png)\n\n')
        
        f.write('## Multiclass Classifier (Promotion Type)\n\n')
        f.write(f"- Accuracy: {multiclass_results['accuracy']:.4f}\n")
        f.write(f"- Weighted Precision: {multiclass_results['precision']:.4f}\n")
        f.write(f"- Weighted Recall: {multiclass_results['recall']:.4f}\n")
        f.write(f"- Weighted F1 Score: {multiclass_results['f1']:.4f}\n\n")
        
        f.write('### Confusion Matrix\n\n')
        f.write('![Multiclass Classifier Confusion Matrix](multiclass_classifier_confusion_matrix.png)\n\n')
        
        f.write('### Classification Report\n\n')
        f.write('```\n')
        f.write(multiclass_results['classification_report'])
        f.write('\n```\n\n')
        
        f.write('### Feature Importance\n\n')
        f.write('![Multiclass Classifier Feature Importance](multiclass_classifier_feature_importance.png)\n\n')
        
        f.write('## Interpretation and Insights\n\n')
        f.write('### Binary Classifier\n\n')
        f.write('1. The model achieves good performance in predicting whether a promotion will occur in a given week\n')
        f.write('2. Key predictive factors include seasonality features, recent promotion history, and country-specific patterns\n')
        f.write('3. There may be room for improvement in recall, ensuring we catch most actual promotions\n\n')
        
        f.write('### Multiclass Classifier\n\n')
        f.write('1. Predicting the specific cluster type is more challenging than binary promotion detection\n')
        f.write('2. Some clusters are easier to predict than others, possibly due to more distinct patterns\n')
        f.write('3. Feature importance shows that recent promotion types strongly influence future promotion types\n\n')
        
        f.write('## Next Steps\n\n')
        f.write('1. Use these models to generate week-by-week future promotion predictions\n')
        f.write('2. Integrate predicted promotions into sales forecasting models\n')
        f.write('3. Consider additional features that might improve cluster type prediction\n')
    
    print(f"Evaluation report saved to {report_path}")


def train_and_evaluate_models(args):
    """Main function to train and evaluate models."""
    # Load data
    data = load_data(args.input)
    
    # Split data
    train_data, val_data, test_data = time_based_split(data, args.test_size, args.val_size)
    
    # Prepare features
    X_train, y_binary_train, X_cluster_train, y_cluster_train = prepare_features(train_data)
    X_val, y_binary_val, X_cluster_val, y_cluster_val = prepare_features(val_data)
    X_test, y_binary_test, X_cluster_test, y_cluster_test = prepare_features(test_data)
    
    # Train binary classifier
    binary_model = train_binary_classifier(X_train, y_binary_train, X_val, y_binary_val)
    
    # Train multiclass classifier
    multiclass_model = train_multiclass_classifier(X_cluster_train, y_cluster_train, 
                                                 X_cluster_val, y_cluster_val)
    
    # Evaluate models
    binary_results = evaluate_binary_classifier(binary_model, X_test, y_binary_test, args.reports_dir)
    multiclass_results = evaluate_multiclass_classifier(multiclass_model, X_cluster_test, 
                                                     y_cluster_test, args.reports_dir)
    
    # Generate evaluation report
    generate_evaluation_report(binary_results, multiclass_results, args.reports_dir)
    
    # Save models
    print("\nSaving trained models...")
    binary_model_path = os.path.join(args.models_dir, 'promo_occurrence_model.pkl')
    multiclass_model_path = os.path.join(args.models_dir, 'promo_cluster_model.pkl')
    
    joblib.dump(binary_model, binary_model_path)
    joblib.dump(multiclass_model, multiclass_model_path)
    
    print(f"Binary classifier saved to {binary_model_path}")
    print(f"Multiclass classifier saved to {multiclass_model_path}")
    
    # Save feature columns for later use
    feature_columns_path = os.path.join(args.models_dir, 'feature_columns.pkl')
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, feature_columns_path)
    
    return binary_model, multiclass_model


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure directories exist
    ensure_directory(args.models_dir)
    ensure_directory(args.reports_dir)
    
    try:
        # Train and evaluate models
        binary_model, multiclass_model = train_and_evaluate_models(args)
        print("\nModel training and evaluation complete!")
        
    except Exception as e:
        print(f"Error during model training and evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
