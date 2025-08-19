"""
Data quality analysis and anomaly detection utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import warnings

from ..utils.logger import get_project_logger

logger = get_project_logger("data_quality")

class DataQualityAnalyzer:
    """Analyzes data quality and detects anomalies in the dataset."""
    
    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initialize the analyzer.
        
        Args:
            datasets: Dictionary of loaded datasets
        """
        self.datasets = datasets
        self.quality_report = {}
        self.anomalies = {}
        
    def analyze_all_datasets(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality analysis.
        
        Returns:
            Complete quality analysis report
        """
        logger.info("Starting comprehensive data quality analysis...")
        
        for name, df in self.datasets.items():
            logger.info(f"Analyzing dataset: {name}")
            self.quality_report[name] = self._analyze_single_dataset(df, name)
            self.anomalies[name] = self._detect_anomalies(df, name)
            
        self._analyze_cross_dataset_quality()
        
        return {
            'quality_report': self.quality_report,
            'anomalies': self.anomalies,
            'summary': self._generate_summary()
        }
    
    def _analyze_single_dataset(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """
        Analyze quality of a single dataset.
        
        Args:
            df: DataFrame to analyze
            name: Dataset name
            
        Returns:
            Quality analysis results
        """
        analysis = {
            'basic_stats': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'duplicate_rows': df.duplicated().sum()
            },
            'missing_values': self._analyze_missing_values(df),
            'data_types': self._analyze_data_types(df),
            'numeric_quality': self._analyze_numeric_quality(df),
            'categorical_quality': self._analyze_categorical_quality(df),
            'datetime_quality': self._analyze_datetime_quality(df)
        }
        
        return analysis
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing_cells': missing_counts.sum(),
            'missing_percentage_overall': (missing_counts.sum() / df.size) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'completely_missing_columns': missing_percentages[missing_percentages == 100].index.tolist()
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data type consistency and issues."""
        type_counts = df.dtypes.value_counts().to_dict()
        
        # Check for potential type issues
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        numeric_in_object = []
        
        for col in object_columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric to see if it should be numeric
                try:
                    pd.to_numeric(df[col].dropna().head(1000), errors='raise')
                    numeric_in_object.append(col)
                except (ValueError, TypeError):
                    pass
        
        return {
            'type_distribution': type_counts,
            'object_columns': object_columns,
            'potential_numeric_in_object': numeric_in_object
        }
    
    def _analyze_numeric_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric column quality."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        quality = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                quality[col] = {
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'zeros': int((series == 0).sum()),
                    'negatives': int((series < 0).sum()),
                    'infinite_values': int(np.isinf(series).sum())
                }
        
        return quality
    
    def _analyze_categorical_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical column quality."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        quality = {}
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                quality[col] = {
                    'unique_values': int(series.nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'cardinality_ratio': series.nunique() / len(series),
                    'empty_strings': int((series == '').sum()),
                    'very_rare_categories': int((value_counts < 5).sum())
                }
        
        return quality
    
    def _analyze_datetime_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze datetime column quality."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        quality = {}
        for col in datetime_cols:
            series = df[col].dropna()
            if len(series) > 0:
                quality[col] = {
                    'min_date': str(series.min()),
                    'max_date': str(series.max()),
                    'date_range_days': (series.max() - series.min()).days,
                    'future_dates': int((series > pd.Timestamp.now()).sum()),
                    'very_old_dates': int((series < pd.Timestamp('1900-01-01')).sum())
                }
        
        return quality
    
    def _detect_anomalies(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """
        Detect various types of anomalies in the dataset.
        
        Args:
            df: DataFrame to analyze
            name: Dataset name
            
        Returns:
            Anomaly detection results
        """
        anomalies = {
            'statistical_outliers': self._detect_statistical_outliers(df),
            'business_logic_violations': self._detect_business_violations(df, name),
            'data_consistency_issues': self._detect_consistency_issues(df),
            'suspicious_patterns': self._detect_suspicious_patterns(df)
        }
        
        return anomalies
    
    def _detect_statistical_outliers(self, df: pd.DataFrame, z_threshold: float = 3) -> Dict[str, Any]:
        """Detect statistical outliers using Z-score method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 10:  # Need sufficient data points
                z_scores = np.abs(stats.zscore(series))
                outlier_indices = np.where(z_scores > z_threshold)[0]
                
                outliers[col] = {
                    'count': len(outlier_indices),
                    'percentage': (len(outlier_indices) / len(series)) * 100,
                    'extreme_values': series.iloc[outlier_indices].tolist()[:10]  # Top 10
                }
        
        return outliers
    
    def _detect_business_violations(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Detect business logic violations specific to e-commerce data."""
        violations = []
        
        # Price/value violations
        if 'price' in df.columns:
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                violations.append(f"Found {negative_prices} negative prices")
                
            zero_prices = (df['price'] == 0).sum()
            if zero_prices > 0:
                violations.append(f"Found {zero_prices} zero prices")
        
        # Payment violations
        if 'payment_value' in df.columns:
            negative_payments = (df['payment_value'] < 0).sum()
            if negative_payments > 0:
                violations.append(f"Found {negative_payments} negative payment values")
        
        # Date logic violations
        if 'order_purchase_timestamp' in df.columns and 'order_delivered_customer_date' in df.columns:
            invalid_delivery = (df['order_delivered_customer_date'] < df['order_purchase_timestamp']).sum()
            if invalid_delivery > 0:
                violations.append(f"Found {invalid_delivery} orders delivered before purchase")
        
        # Review score violations
        if 'review_score' in df.columns:
            invalid_scores = (~df['review_score'].isin([1, 2, 3, 4, 5])).sum()
            if invalid_scores > 0:
                violations.append(f"Found {invalid_scores} invalid review scores (not 1-5)")
        
        return violations
    
    def _detect_consistency_issues(self, df: pd.DataFrame) -> List[str]:
        """Detect data consistency issues."""
        issues = []
        
        # Check for ID consistency
        id_columns = [col for col in df.columns if col.endswith('_id')]
        for col in id_columns:
            if col in df.columns:
                # Check for empty IDs
                empty_ids = df[col].isnull().sum()
                if empty_ids > 0:
                    issues.append(f"Found {empty_ids} empty {col} values")
                
                # Check for duplicate IDs where uniqueness expected
                if col in ['order_id', 'customer_id', 'product_id', 'seller_id']:
                    duplicates = df[col].duplicated().sum()
                    if duplicates > 0:
                        issues.append(f"Found {duplicates} duplicate {col} values")
        
        return issues
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect suspicious data patterns."""
        patterns = []
        
        # Check for repeated values that might indicate data issues
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 0:
                most_common_count = series.value_counts().iloc[0]
                if most_common_count > len(series) * 0.8:  # More than 80% same value
                    patterns.append(f"Column {col} has {most_common_count} repeated values ({most_common_count/len(series)*100:.1f}%)")
        
        return patterns
    
    def _analyze_cross_dataset_quality(self) -> None:
        """Analyze quality across datasets."""
        logger.info("Analyzing cross-dataset relationships...")
        
        # Check referential integrity
        if 'orders' in self.datasets and 'order_reviews' in self.datasets:
            orders_ids = set(self.datasets['orders']['order_id'])
            reviews_ids = set(self.datasets['order_reviews']['order_id'])
            
            orphaned_reviews = len(reviews_ids - orders_ids)
            if orphaned_reviews > 0:
                logger.warning(f"Found {orphaned_reviews} reviews for non-existent orders")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the quality analysis."""
        total_datasets = len(self.datasets)
        total_rows = sum(df.shape[0] for df in self.datasets.values())
        total_columns = sum(df.shape[1] for df in self.datasets.values())
        
        # Count issues
        total_anomalies = 0
        critical_issues = []
        
        for dataset_name, anomalies in self.anomalies.items():
            for anomaly_type, anomaly_data in anomalies.items():
                if isinstance(anomaly_data, dict):
                    total_anomalies += sum(
                        item.get('count', 0) if isinstance(item, dict) else 0 
                        for item in anomaly_data.values()
                    )
                elif isinstance(anomaly_data, list):
                    total_anomalies += len(anomaly_data)
                    if len(anomaly_data) > 0:
                        critical_issues.extend(anomaly_data)
        
        return {
            'total_datasets': total_datasets,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'total_anomalies_detected': total_anomalies,
            'critical_issues': critical_issues[:10],  # Top 10 most critical
            'datasets_with_issues': [
                name for name, anomalies in self.anomalies.items() 
                if any(anomalies.values())
            ]
        }