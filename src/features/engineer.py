"""
Advanced feature engineering for the Olist ML project.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import warnings

from ..utils.logger import get_project_logger

logger = get_project_logger("feature_engineer")

class FeatureEngineer:
    """
    Advanced feature engineering class that creates meaningful features
    from the e-commerce data while avoiding target leakage.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.created_features = []
        self.feature_descriptions = {}
        
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering techniques.
        
        Args:
            df: Preprocessed master dataset
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        df_engineered = df.copy()
        
        # Order-level features
        df_engineered = self._create_order_complexity_features(df_engineered)
        df_engineered = self._create_price_features(df_engineered)
        df_engineered = self._create_logistics_features(df_engineered)
        
        # Customer behavior features (without review leakage)
        df_engineered = self._create_customer_features(df_engineered)
        
        # Product portfolio features
        df_engineered = self._create_product_portfolio_features(df_engineered)
        
        # Geographic features
        df_engineered = self._create_geographic_features(df_engineered)
        
        # Temporal features
        df_engineered = self._create_temporal_features(df_engineered)
        
        # Interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # Risk and quality indicators
        df_engineered = self._create_risk_indicators(df_engineered)
        
        logger.info(f"Feature engineering complete. Added {len(self.created_features)} new features.")
        logger.info(f"Final feature count: {df_engineered.shape[1]}")
        
        return df_engineered
    
    def _create_order_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to order complexity."""
        logger.info("Creating order complexity features...")
        
        # Order size indicators
        if 'total_items' in df.columns:
            df['is_bulk_order'] = (df['total_items'] > df['total_items'].quantile(0.8)).astype(int)
            order_cats = pd.cut(
                df['total_items'].fillna(1), 
                bins=[0, 1, 3, 5, float('inf')], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['order_size_category'] = order_cats.cat.codes
            self._add_feature('is_bulk_order', 'Binary indicator for large orders (top 20%)')
            self._add_feature('order_size_category', 'Categorical order size grouping')
        
        # Seller diversity
        if 'unique_sellers' in df.columns:
            df['is_multi_seller'] = (df['unique_sellers'] > 1).astype(int)
            df['seller_concentration'] = df['unique_sellers'] / df.get('total_items', 1)
            self._add_feature('is_multi_seller', 'Binary indicator for orders from multiple sellers')
            self._add_feature('seller_concentration', 'Ratio of unique sellers to total items')
        
        # Product diversity
        if 'unique_products' in df.columns and 'total_items' in df.columns:
            df['product_variety'] = df['unique_products'] / df['total_items']
            df['has_duplicate_products'] = (df['total_items'] > df['unique_products']).astype(int)
            self._add_feature('product_variety', 'Ratio of unique products to total items')
            self._add_feature('has_duplicate_products', 'Binary indicator for repeated products')
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features."""
        logger.info("Creating price-related features...")
        
        # Price distribution features
        if 'total_price' in df.columns and 'avg_item_price' in df.columns:
            # Price range and variation
            if 'max_item_price' in df.columns and 'min_item_price' in df.columns:
                df['price_range'] = df['max_item_price'] - df['min_item_price']
                df['price_ratio'] = df['max_item_price'] / (df['min_item_price'] + 0.01)  # Avoid division by zero
                self._add_feature('price_range', 'Difference between highest and lowest item prices')
                self._add_feature('price_ratio', 'Ratio of max to min item price')
            
            # Price categories
            price_cats = pd.cut(
                df['total_price'].fillna(0),
                bins=[0, 50, 150, 500, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['price_category'] = price_cats.cat.codes
            self._add_feature('price_category', 'Categorical price segment')
        
        # Freight analysis
        if 'total_freight' in df.columns and 'total_price' in df.columns:
            df['freight_to_price_ratio'] = df['total_freight'] / (df['total_price'] + 0.01)
            df['high_freight_indicator'] = (df['freight_to_price_ratio'] > 0.2).astype(int)
            self._add_feature('freight_to_price_ratio', 'Ratio of freight cost to product price')
            self._add_feature('high_freight_indicator', 'Binary indicator for high freight relative to price')
        
        # Payment features
        if 'max_installments' in df.columns:
            # Fill NaN values first
            df['max_installments'] = df['max_installments'].fillna(1)
            df['uses_installments'] = (df['max_installments'] > 1).astype(int)
            
            # Create categorical feature with proper NaN handling
            installment_cats = pd.cut(
                df['max_installments'],
                bins=[0, 1, 6, 12, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['installment_category'] = installment_cats.cat.codes
            
            self._add_feature('uses_installments', 'Binary indicator for installment payments')
            self._add_feature('installment_category', 'Categorical installment term length')
        
        return df
    
    def _create_logistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create logistics and delivery-related features."""
        logger.info("Creating logistics features...")
        
        # Weight and dimension features
        weight_cols = ['total_weight_g', 'avg_weight_g', 'max_weight_g']
        dimension_cols = ['avg_length_cm', 'avg_height_cm', 'avg_width_cm']
        
        if all(col in df.columns for col in weight_cols[:2]):
            weight_cats = pd.cut(
                df['total_weight_g'].fillna(0),
                bins=[0, 500, 2000, 10000, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['weight_category'] = weight_cats.cat.codes
            self._add_feature('weight_category', 'Categorical weight classification')
        
        if all(col in df.columns for col in dimension_cols):
            # Calculate volume
            df['avg_volume_cm3'] = (
                df['avg_length_cm'] * df['avg_height_cm'] * df['avg_width_cm']
            ).fillna(0)
            
            # Size categories
            size_cats = pd.cut(
                df['avg_volume_cm3'].fillna(0),
                bins=[0, 1000, 10000, 100000, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['size_category'] = size_cats.cat.codes
            
            self._add_feature('avg_volume_cm3', 'Average product volume')
            self._add_feature('size_category', 'Categorical size classification')
        
        return df
    
    def _create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-related features without review leakage."""
        logger.info("Creating customer features...")
        
        # Geographic features
        if 'customer_state' in df.columns:
            # State-based features (these would be enhanced with external data in production)
            major_states = ['SP', 'RJ', 'MG', 'RS', 'SC']  # Major Brazilian states
            df['is_major_state'] = df['customer_state'].isin(major_states).astype(int)
            self._add_feature('is_major_state', 'Binary indicator for major Brazilian states')
        
        # Location risk indicators
        if 'customer_zip_code_prefix' in df.columns:
            zip_frequency = df['customer_zip_code_prefix'].value_counts()
            df['zip_order_frequency'] = df['customer_zip_code_prefix'].map(zip_frequency)
            df['is_rare_location'] = (df['zip_order_frequency'] <= 5).astype(int)
            self._add_feature('zip_order_frequency', 'Number of orders from same zip code')
            self._add_feature('is_rare_location', 'Binary indicator for infrequent delivery locations')
        
        return df
    
    def _create_product_portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product portfolio features."""
        logger.info("Creating product portfolio features...")
        
        if 'primary_product_category' in df.columns:
            # Category popularity (based on frequency in dataset)
            category_frequency = df['primary_product_category'].value_counts()
            df['category_popularity'] = df['primary_product_category'].map(category_frequency)
            
            # Popular categories indicator
            top_categories = category_frequency.head(10).index
            df['is_popular_category'] = df['primary_product_category'].isin(top_categories).astype(int)
            
            self._add_feature('category_popularity', 'Frequency of primary product category')
            self._add_feature('is_popular_category', 'Binary indicator for top 10 popular categories')
        
        # Photo quality indicator
        if 'avg_photos_qty' in df.columns:
            df['has_good_photos'] = (df['avg_photos_qty'] >= 3).astype(int)
            photo_cats = pd.cut(
                df['avg_photos_qty'].fillna(1),
                bins=[0, 1, 3, 6, float('inf')],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['photo_category'] = photo_cats.cat.codes
            self._add_feature('has_good_photos', 'Binary indicator for good photo coverage')
            self._add_feature('photo_category', 'Categorical photo quality level')
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features."""
        logger.info("Creating geographic features...")
        
        # Distance features would require seller coordinates
        # For now, create state-based features
        
        if 'customer_state' in df.columns and 'primary_seller_state' in df.columns:
            df['same_state_delivery'] = (
                df['customer_state'] == df['primary_seller_state']
            ).astype(int)
            self._add_feature('same_state_delivery', 'Binary indicator for same-state delivery')
        
        # Coordinate-based features if available
        if 'customer_lat' in df.columns and 'customer_lng' in df.columns:
            # Distance from major cities (example: São Paulo)
            sao_paulo_lat, sao_paulo_lng = -23.5505, -46.6333
            
            df['distance_from_sao_paulo'] = np.sqrt(
                (df['customer_lat'] - sao_paulo_lat)**2 + 
                (df['customer_lng'] - sao_paulo_lng)**2
            )
            
            df['is_near_major_city'] = (df['distance_from_sao_paulo'] < 1.0).astype(int)
            
            self._add_feature('distance_from_sao_paulo', 'Distance from São Paulo coordinates')
            self._add_feature('is_near_major_city', 'Binary indicator for proximity to major city')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from datetime components."""
        logger.info("Creating temporal features...")
        
        # Purchase timing features
        purchase_cols = [col for col in df.columns if 'order_purchase_timestamp' in col]
        
        for col in purchase_cols:
            if 'month' in col:
                # Seasonal indicators
                df['is_holiday_season'] = df[col].isin([11, 12]).astype(int)  # Nov, Dec
                df['is_summer_brazil'] = df[col].isin([12, 1, 2]).astype(int)  # Summer in Brazil
                self._add_feature('is_holiday_season', 'Binary indicator for holiday shopping season')
                self._add_feature('is_summer_brazil', 'Binary indicator for Brazilian summer')
            
            if 'day_of_week' in col:
                # Weekend shopping
                df['is_weekend_purchase'] = df[col].isin([5, 6]).astype(int)  # Sat, Sun
                df['is_weekday_purchase'] = (~df[col].isin([5, 6])).astype(int)
                self._add_feature('is_weekend_purchase', 'Binary indicator for weekend purchases')
                self._add_feature('is_weekday_purchase', 'Binary indicator for weekday purchases')
            
            if 'hour' in col:
                # Business hours
                df['is_business_hours'] = df[col].between(9, 17).astype(int)
                df['is_evening_purchase'] = df[col].between(18, 22).astype(int)
                self._add_feature('is_business_hours', 'Binary indicator for business hours purchases')
                self._add_feature('is_evening_purchase', 'Binary indicator for evening purchases')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different dimensions."""
        logger.info("Creating interaction features...")
        
        # Price-size interactions
        if 'total_price' in df.columns and 'total_items' in df.columns:
            df['price_per_item'] = df['total_price'] / df['total_items']
            self._add_feature('price_per_item', 'Average price per item in order')
        
        # Weight-price efficiency
        if 'total_weight_g' in df.columns and 'total_price' in df.columns:
            df['price_per_gram'] = df['total_price'] / (df['total_weight_g'] + 0.01)
            self._add_feature('price_per_gram', 'Price efficiency per gram')
        
        # Category-price interaction
        if 'primary_product_category' in df.columns and 'total_price' in df.columns:
            category_avg_price = df.groupby('primary_product_category')['total_price'].transform('mean')
            df['price_vs_category_avg'] = df['total_price'] / (category_avg_price + 0.01)
            self._add_feature('price_vs_category_avg', 'Price relative to category average')
        
        return df
    
    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk and quality indicators."""
        logger.info("Creating risk indicators...")
        
        # Payment risk indicators
        if 'max_installments' in df.columns and 'total_price' in df.columns:
            df['high_installment_risk'] = (
                (df['max_installments'] > 10) & (df['total_price'] > 200)
            ).astype(int)
            self._add_feature('high_installment_risk', 'Binary indicator for high installment payment risk')
        
        # Logistics complexity
        complexity_score = 0
        if 'unique_sellers' in df.columns:
            complexity_score += (df['unique_sellers'] - 1) * 2
        if 'total_items' in df.columns:
            complexity_score += (df['total_items'] > 5).astype(int)
        if 'total_weight_g' in df.columns:
            complexity_score += (df['total_weight_g'] > 5000).astype(int)
        
        df['logistics_complexity_score'] = complexity_score
        df['high_complexity_order'] = (complexity_score >= 3).astype(int)
        
        self._add_feature('logistics_complexity_score', 'Composite logistics complexity score')
        self._add_feature('high_complexity_order', 'Binary indicator for high complexity orders')
        
        return df
    
    def _add_feature(self, feature_name: str, description: str) -> None:
        """Track created features and their descriptions."""
        self.created_features.append(feature_name)
        self.feature_descriptions[feature_name] = description
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all created features."""
        return self.feature_descriptions.copy()
    
    def get_created_features(self) -> List[str]:
        """Get list of all created feature names."""
        return self.created_features.copy()