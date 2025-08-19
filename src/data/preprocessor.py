"""
Data preprocessing pipeline for the Olist ML project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from ..utils.logger import get_project_logger

logger = get_project_logger("preprocessor")

class OlistDataPreprocessor:
    """
    Handles data preprocessing including joining, cleaning, and feature preparation.
    Uses exclusion-based approach for missing values with detailed tracking.
    """
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the preprocessor.
        
        Args:
            datasets: Dictionary of loaded datasets
        """
        self.datasets = datasets
        self.missing_value_report = {}
        self.exclusion_stats = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_master_dataset(self) -> pd.DataFrame:
        """
        Create a master dataset by joining all relevant tables.
        Excludes review-dependent features to prevent data leakage.
        
        Returns:
            Master dataset ready for feature engineering
        """
        logger.info("Creating master dataset through strategic joins...")
        
        # Start with orders as the base table
        master_df = self.datasets['orders'].copy()
        logger.info(f"Base orders dataset: {master_df.shape}")
        
        # Join with reviews to get target variable (review_score only)
        reviews_df = self.datasets['order_reviews'][['order_id', 'review_score']].copy()
        master_df = master_df.merge(reviews_df, on='order_id', how='inner')
        logger.info(f"After joining reviews (target): {master_df.shape}")
        
        # Join with customers
        customers_df = self.datasets['customers'].copy()
        master_df = master_df.merge(customers_df, on='customer_id', how='left')
        logger.info(f"After joining customers: {master_df.shape}")
        
        # Join with order items (aggregate to order level)
        items_agg = self._aggregate_order_items()
        master_df = master_df.merge(items_agg, on='order_id', how='left')
        logger.info(f"After joining aggregated items: {master_df.shape}")
        
        # Join with payments (aggregate to order level)
        payments_agg = self._aggregate_order_payments()
        master_df = master_df.merge(payments_agg, on='order_id', how='left')
        logger.info(f"After joining aggregated payments: {master_df.shape}")
        
        # Join with sellers (through items)
        seller_features = self._create_seller_features()
        master_df = master_df.merge(seller_features, on='order_id', how='left')
        logger.info(f"After joining seller features: {master_df.shape}")
        
        # Join with product features
        product_features = self._create_product_features()
        master_df = master_df.merge(product_features, on='order_id', how='left')
        logger.info(f"After joining product features: {master_df.shape}")
        
        # Add geolocation features
        master_df = self._add_geolocation_features(master_df)
        logger.info(f"Final master dataset: {master_df.shape}")
        
        return master_df
    
    def _aggregate_order_items(self) -> pd.DataFrame:
        """Aggregate order items to order level."""
        items_df = self.datasets['order_items'].copy()
        
        # Calculate aggregations
        agg_dict = {
            'order_item_id': 'count',  # Number of items
            'price': ['sum', 'mean', 'std', 'min', 'max'],
            'freight_value': ['sum', 'mean', 'std'],
            'product_id': 'nunique',  # Number of unique products
            'seller_id': 'nunique'    # Number of unique sellers
        }
        
        items_agg = items_df.groupby('order_id').agg(agg_dict)
        
        # Flatten column names
        items_agg.columns = [
            f'items_{col[1]}_{col[0]}' if col[1] != '' else f'items_{col[0]}'
            for col in items_agg.columns
        ]
        
        # Rename for clarity
        items_agg = items_agg.rename(columns={
            'items_count_order_item_id': 'total_items',
            'items_sum_price': 'total_price',
            'items_mean_price': 'avg_item_price',
            'items_std_price': 'price_variation',
            'items_min_price': 'min_item_price',
            'items_max_price': 'max_item_price',
            'items_sum_freight_value': 'total_freight',
            'items_mean_freight_value': 'avg_freight',
            'items_std_freight_value': 'freight_variation',
            'items_nunique_product_id': 'unique_products',
            'items_nunique_seller_id': 'unique_sellers'
        })
        
        # Fill NaN std values with 0 (single item orders)
        items_agg['price_variation'] = items_agg['price_variation'].fillna(0)
        items_agg['freight_variation'] = items_agg['freight_variation'].fillna(0)
        
        return items_agg.reset_index()
    
    def _aggregate_order_payments(self) -> pd.DataFrame:
        """Aggregate order payments to order level."""
        payments_df = self.datasets['order_payments'].copy()
        
        agg_dict = {
            'payment_sequential': 'max',  # Number of payment methods
            'payment_installments': ['max', 'mean'],
            'payment_value': ['sum', 'mean', 'std'],
            'payment_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }
        
        payments_agg = payments_df.groupby('order_id').agg(agg_dict)
        
        # Flatten columns
        payments_agg.columns = [
            f'payment_{col[1]}_{col[0]}' if col[1] != '' else f'payment_{col[0]}'
            for col in payments_agg.columns
        ]
        
        # Rename for clarity
        payments_agg = payments_agg.rename(columns={
            'payment_max_payment_sequential': 'payment_methods_count',
            'payment_max_payment_installments': 'max_installments',
            'payment_mean_payment_installments': 'avg_installments',
            'payment_sum_payment_value': 'total_payment_value',
            'payment_mean_payment_value': 'avg_payment_value',
            'payment_std_payment_value': 'payment_variation',
            'payment_<lambda>_payment_type': 'primary_payment_type'
        })
        
        # Fill NaN std values
        payments_agg['payment_variation'] = payments_agg['payment_variation'].fillna(0)
        
        return payments_agg.reset_index()
    
    def _create_seller_features(self) -> pd.DataFrame:
        """Create seller-related features."""
        items_df = self.datasets['order_items'].copy()
        sellers_df = self.datasets['sellers'].copy()
        
        # Join items with seller info
        items_sellers = items_df.merge(sellers_df, on='seller_id', how='left')
        
        # Create seller features per order
        seller_features = items_sellers.groupby('order_id').agg({
            'seller_state': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'seller_zip_code_prefix': 'nunique'
        }).rename(columns={
            'seller_state': 'primary_seller_state',
            'seller_zip_code_prefix': 'seller_locations_count'
        })
        
        return seller_features.reset_index()
    
    def _create_product_features(self) -> pd.DataFrame:
        """Create product-related features."""
        items_df = self.datasets['order_items'].copy()
        products_df = self.datasets['products'].copy()
        translation_df = self.datasets['product_translation'].copy()
        
        # Add English category names
        products_enhanced = products_df.merge(
            translation_df, on='product_category_name', how='left'
        )
        
        # Join items with product info
        items_products = items_df.merge(products_enhanced, on='product_id', how='left')
        
        # Aggregate product features per order
        product_agg = items_products.groupby('order_id').agg({
            'product_category_name_english': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'product_weight_g': ['sum', 'mean', 'max'],
            'product_length_cm': 'mean',
            'product_height_cm': 'mean', 
            'product_width_cm': 'mean',
            'product_photos_qty': 'mean'
        })
        
        # Flatten columns
        product_agg.columns = [
            f'product_{col[1]}_{col[0]}' if col[1] != '' else f'product_{col[0]}'
            for col in product_agg.columns
        ]
        
        # Rename for clarity
        product_agg = product_agg.rename(columns={
            'product_<lambda>_product_category_name_english': 'primary_product_category',
            'product_sum_product_weight_g': 'total_weight_g',
            'product_mean_product_weight_g': 'avg_weight_g',
            'product_max_product_weight_g': 'max_weight_g',
            'product_mean_product_length_cm': 'avg_length_cm',
            'product_mean_product_height_cm': 'avg_height_cm',
            'product_mean_product_width_cm': 'avg_width_cm',
            'product_mean_product_photos_qty': 'avg_photos_qty'
        })
        
        return product_agg.reset_index()
    
    def _add_geolocation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geolocation-based features."""
        geo_df = self.datasets['geolocation'].copy()
        
        # Get customer geolocation
        customer_geo = geo_df.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean',
            'geolocation_city': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        # Merge with master dataset
        df_geo = df.merge(
            customer_geo, 
            left_on='customer_zip_code_prefix', 
            right_on='geolocation_zip_code_prefix',
            how='left'
        )
        
        # Drop duplicate zip code column
        df_geo = df_geo.drop('geolocation_zip_code_prefix', axis=1)
        
        # Rename geolocation columns for clarity
        df_geo = df_geo.rename(columns={
            'geolocation_lat': 'customer_lat',
            'geolocation_lng': 'customer_lng',
            'geolocation_city': 'customer_geo_city'
        })
        
        return df_geo
    
    def preprocess_for_ml(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess the master dataset for machine learning.
        
        Args:
            df: Master dataset
            
        Returns:
            Tuple of (processed_df, preprocessing_report)
        """
        logger.info("Starting ML preprocessing...")
        
        # Create binary target variable
        df_processed = df.copy()
        df_processed['target'] = (df_processed['review_score'] >= 4).astype(int)
        
        # Track original dataset size
        original_size = len(df_processed)
        
        # Handle missing values with exclusion approach
        df_processed, missing_report = self._handle_missing_values_exclusion(df_processed)
        
        # Remove target leakage columns
        leakage_columns = [
            'review_score', 'order_id', 'customer_id', 'review_id'
        ]
        
        features_to_drop = [col for col in leakage_columns if col in df_processed.columns]
        df_processed = df_processed.drop(features_to_drop, axis=1)
        
        # Handle datetime features
        df_processed = self._engineer_datetime_features(df_processed)
        
        # Handle categorical features
        df_processed = self._encode_categorical_features(df_processed)
        
        # Create preprocessing report
        preprocessing_report = {
            'original_size': original_size,
            'final_size': len(df_processed),
            'rows_excluded': original_size - len(df_processed),
            'exclusion_percentage': ((original_size - len(df_processed)) / original_size) * 100,
            'missing_value_handling': missing_report,
            'features_dropped': features_to_drop,
            'final_features': list(df_processed.columns),
            'target_distribution': df_processed['target'].value_counts().to_dict()
        }
        
        logger.info(f"Preprocessing complete: {original_size} -> {len(df_processed)} rows ({preprocessing_report['exclusion_percentage']:.1f}% excluded)")
        
        return df_processed, preprocessing_report
    
    def _handle_missing_values_exclusion(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values using exclusion approach with detailed tracking.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, missing_value_report)
        """
        logger.info("Handling missing values using exclusion approach...")
        
        original_shape = df.shape
        
        # Analyze missing values before exclusion
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_analysis[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'total_values': len(df),
                'non_missing_count': len(df) - missing_count
            }
        
        # Strategy 1: Drop columns with >95% missing values
        high_missing_cols = [
            col for col, stats in missing_analysis.items() 
            if stats['missing_percentage'] > 95
        ]
        
        if high_missing_cols:
            logger.info(f"Dropping {len(high_missing_cols)} columns with >95% missing values: {high_missing_cols}")
            df = df.drop(high_missing_cols, axis=1)
        
        # Strategy 2: Drop rows with any missing values in critical columns
        critical_columns = ['target', 'customer_state', 'total_items', 'total_price']
        critical_columns = [col for col in critical_columns if col in df.columns]
        
        if critical_columns:
            rows_before = len(df)
            df = df.dropna(subset=critical_columns)
            rows_after = len(df)
            logger.info(f"Dropped {rows_before - rows_after} rows with missing critical values")
        
        # Strategy 3: Drop remaining rows with any missing values
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        
        logger.info(f"Final exclusion: {rows_before - rows_after} rows with any missing values")
        
        # Create detailed report
        missing_report = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'columns_dropped': high_missing_cols,
            'missing_analysis_original': missing_analysis,
            'exclusion_strategy': {
                'high_missing_columns_threshold': '95%',
                'critical_columns_checked': critical_columns,
                'final_approach': 'Complete case analysis (exclude any missing)'
            },
            'exclusion_summary': {
                'rows_excluded_total': original_shape[0] - df.shape[0],
                'columns_excluded_total': len(high_missing_cols),
                'data_retention_rate': (df.shape[0] / original_shape[0]) * 100
            }
        }
        
        return df, missing_report
    
    def _engineer_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from datetime columns."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in datetime_cols:
            if col in df.columns:
                # Extract time components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour
                
                # Drop original datetime column
                df = df.drop(col, axis=1)
        
        # Calculate delivery time features if possible
        if 'order_purchase_timestamp' in datetime_cols and 'order_delivered_customer_date' in datetime_cols:
            purchase_col = 'order_purchase_timestamp'
            delivery_col = 'order_delivered_customer_date'
            
            if purchase_col in df.columns and delivery_col in df.columns:
                # This was handled before dropping, so we'll create it differently
                pass
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using appropriate methods."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col != 'target':  # Don't encode target
                # Use label encoding for now (could be enhanced with one-hot for low cardinality)
                le = LabelEncoder()
                
                # Handle unknown values by filling with most frequent
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown'
                df[col] = df[col].fillna(mode_value)
                
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df