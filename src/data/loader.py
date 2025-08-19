"""
Data loading utilities for the Olist dataset.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import warnings

from ..utils.logger import get_project_logger

logger = get_project_logger("data_loader")

class OlistDataLoader:
    """Handles loading and basic validation of Olist datasets."""
    
    def __init__(self, data_files: Dict[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_files: Dictionary mapping dataset names to file paths
        """
        self.data_files = data_files
        self._datasets = {}
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets and perform basic validation.
        
        Returns:
            Dictionary of loaded datasets
        """
        logger.info("Loading all datasets...")
        
        for name, file_path in self.data_files.items():
            try:
                logger.info(f"Loading {name} from {file_path}")
                self._datasets[name] = self._load_single_dataset(file_path, name)
                logger.info(f"Loaded {name}: {self._datasets[name].shape}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {str(e)}")
                raise
                
        self._validate_datasets()
        return self._datasets.copy()
    
    def _load_single_dataset(self, file_path: Path, dataset_name: str) -> pd.DataFrame:
        """
        Load a single CSV dataset with appropriate data types.
        
        Args:
            file_path: Path to the CSV file
            dataset_name: Name of the dataset for logging
            
        Returns:
            Loaded DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load with pandas, inferring data types
            df = pd.read_csv(file_path, low_memory=False)
            
            # Convert datetime columns if present
            datetime_columns = [
                'order_purchase_timestamp', 'order_approved_at',
                'order_delivered_carrier_date', 'order_delivered_customer_date',
                'order_estimated_delivery_date', 'shipping_limit_date',
                'review_creation_date', 'review_answer_timestamp'
            ]
            
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
        logger.info(f"Dataset {dataset_name} loaded successfully with shape {df.shape}")
        return df
    
    def _validate_datasets(self) -> None:
        """Perform basic validation on loaded datasets."""
        logger.info("Validating dataset relationships...")
        
        required_datasets = ['orders', 'order_reviews', 'order_items']
        for dataset in required_datasets:
            if dataset not in self._datasets:
                raise ValueError(f"Required dataset '{dataset}' not loaded")
                
        # Validate key relationships
        orders_df = self._datasets['orders']
        reviews_df = self._datasets['order_reviews']
        items_df = self._datasets['order_items']
        
        # Check order_id relationships
        orders_ids = set(orders_df['order_id'])
        review_ids = set(reviews_df['order_id'])
        item_ids = set(items_df['order_id'])
        
        logger.info(f"Orders with reviews: {len(review_ids & orders_ids)}/{len(orders_ids)}")
        logger.info(f"Orders with items: {len(item_ids & orders_ids)}/{len(orders_ids)}")
        
        if len(review_ids - orders_ids) > 0:
            logger.warning(f"Found {len(review_ids - orders_ids)} reviews for non-existent orders")
            
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset DataFrame or None if not found
        """
        return self._datasets.get(name)
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get summary information about all loaded datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {}
        for name, df in self._datasets.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        return info