"""
Configuration settings for the Olist Review Score Prediction project.
"""
from pathlib import Path
from typing import Dict, List, Any
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" if (PROJECT_ROOT / "data").exists() else PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data file paths
DATA_FILES = {
    'customers': DATA_DIR / 'olist_customers_dataset.csv',
    'geolocation': DATA_DIR / 'olist_geolocation_dataset.csv', 
    'order_items': DATA_DIR / 'olist_order_items_dataset.csv',
    'order_payments': DATA_DIR / 'olist_order_payments_dataset.csv',
    'order_reviews': DATA_DIR / 'olist_order_reviews_dataset.csv',
    'orders': DATA_DIR / 'olist_orders_dataset.csv',
    'products': DATA_DIR / 'olist_products_dataset.csv',
    'sellers': DATA_DIR / 'olist_sellers_dataset.csv',
    'product_translation': DATA_DIR / 'product_category_name_translation.csv'
}

# Model configuration
MODEL_CONFIG = {
    'target_column': 'review_score',
    'high_score_threshold': 4,  # Scores >= 4 are considered "high"
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'exclude_columns': [
        'review_id', 'review_comment_title', 'review_comment_message', 
        'review_creation_date', 'review_answer_timestamp'
    ],
    'datetime_columns': [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'shipping_limit_date'
    ],
    'categorical_columns': [
        'order_status', 'payment_type', 'customer_state', 
        'seller_state', 'product_category_name_english'
    ]
}

# Data quality thresholds
QUALITY_CONFIG = {
    'max_missing_ratio': 0.95,  # Drop columns with >95% missing values
    'outlier_threshold': 3,     # Z-score threshold for outlier detection
    'min_category_frequency': 10  # Minimum frequency for categorical values
}

# Output configuration
OUTPUT_CONFIG = {
    'html_report': OUTPUT_DIR / 'review_prediction_report.html',
    'model_artifacts': OUTPUT_DIR / 'model_artifacts',
    'data_artifacts': OUTPUT_DIR / 'processed_data'
}

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_CONFIG['model_artifacts'].mkdir(exist_ok=True, parents=True)
OUTPUT_CONFIG['data_artifacts'].mkdir(exist_ok=True, parents=True)