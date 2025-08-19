"""
ML model training module for the Olist review prediction project.
Includes comprehensive class imbalance handling techniques.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import warnings

# Optional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Class imbalance handling imports
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

from ..utils.logger import get_project_logger

logger = get_project_logger("model_trainer")

class ModelTrainer:
    """
    Handles training and comparison of multiple ML models for review score prediction.
    """
    
    def __init__(self, random_state: int = 42, imbalance_technique: str = 'smote'):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random state for reproducibility
            imbalance_technique: Class imbalance handling technique
                Options: 'none', 'smote', 'adasyn', 'random_over', 'random_under', 
                        'tomek', 'smoteenn', 'borderline_smote'
        """
        self.random_state = random_state
        self.imbalance_technique = imbalance_technique
        self.models = {}
        self.trained_models = {}
        self.model_performance = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.imbalance_handler = None
        
    def setup_models(self) -> None:
        """Setup different ML models with optimal hyperparameters."""
        logger.info("Setting up ML models...")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
        }
        
        # Add LightGBM only if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
        
        logger.info(f"Setup {len(self.models)} models: {list(self.models.keys())}")
    
    def _setup_imbalance_handler(self) -> Any:
        """
        Setup class imbalance handling technique.
        
        Returns:
            Configured imbalance handler or None
        """
        if not IMBALANCED_LEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available. Using class weights only.")
            return None
        
        if self.imbalance_technique == 'none':
            return None
        elif self.imbalance_technique == 'smote':
            return SMOTE(random_state=self.random_state, k_neighbors=5)
        elif self.imbalance_technique == 'adasyn':
            return ADASYN(random_state=self.random_state, n_neighbors=5)
        elif self.imbalance_technique == 'random_over':
            return RandomOverSampler(random_state=self.random_state)
        elif self.imbalance_technique == 'random_under':
            return RandomUnderSampler(random_state=self.random_state)
        elif self.imbalance_technique == 'tomek':
            return TomekLinks()
        elif self.imbalance_technique == 'smoteenn':
            return SMOTEENN(random_state=self.random_state)
        elif self.imbalance_technique == 'borderline_smote':
            return BorderlineSMOTE(random_state=self.random_state)
        else:
            logger.warning(f"Unknown imbalance technique: {self.imbalance_technique}")
            return None
    
    def _apply_imbalance_handling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply class imbalance handling to training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (resampled_X_train, resampled_y_train)
        """
        if self.imbalance_technique == 'none' or not IMBALANCED_LEARN_AVAILABLE:
            return X_train, y_train
        
        if self.imbalance_handler is None:
            self.imbalance_handler = self._setup_imbalance_handler()
        
        if self.imbalance_handler is None:
            return X_train, y_train
        
        try:
            logger.info(f"Applying {self.imbalance_technique} class imbalance handling...")
            
            # Check original distribution
            original_dist = y_train.value_counts()
            logger.info(f"Original distribution: {original_dist.to_dict()}")
            
            # Apply resampling
            X_resampled, y_resampled = self.imbalance_handler.fit_resample(X_train, y_train)
            
            # Check new distribution
            new_dist = pd.Series(y_resampled).value_counts()
            logger.info(f"New distribution: {new_dist.to_dict()}")
            logger.info(f"Samples change: {len(X_train)} -> {len(X_resampled)} ({len(X_resampled) - len(X_train):+})")
            
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
            
        except Exception as e:
            logger.error(f"Failed to apply {self.imbalance_technique}: {str(e)}")
            logger.info("Falling back to original data...")
            return X_train, y_train

    def prepare_data(self, df: pd.DataFrame, target_column: str = 'target', 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: Input dataset
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for model training...")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        logger.info(f"Feature count: {len(self.feature_names)}")
        
        # Check target distribution
        target_distribution = y.value_counts(normalize=True)
        logger.info(f"Target distribution: {target_distribution.to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Scale only numeric features
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns
        
        logger.info(f"Numeric features: {len(numeric_columns)}, Categorical features: {len(categorical_columns)}")
        
        # Scale numeric features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if len(numeric_columns) > 0:
            X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
            X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train all models and evaluate performance with class imbalance handling.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training all models with {self.imbalance_technique} class imbalance handling...")
        
        if not self.models:
            self.setup_models()
        
        # Apply class imbalance handling
        X_train_resampled, y_train_resampled = self._apply_imbalance_handling(X_train, y_train)
        
        training_results = {
            'imbalance_technique': self.imbalance_technique,
            'original_samples': len(X_train),
            'resampled_samples': len(X_train_resampled),
            'samples_change': len(X_train_resampled) - len(X_train)
        }
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train model on resampled data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_resampled, y_train_resampled)
                
                self.trained_models[model_name] = model
                
                # Evaluate model (test on original test set)
                performance = self._evaluate_model(
                    model, model_name, X_train_resampled, X_test, y_train_resampled, y_test
                )
                
                self.model_performance[model_name] = performance
                training_results[model_name] = performance
                
                logger.info(f"{model_name} - Test Accuracy: {performance['test_accuracy']:.4f}, AUC: {performance['test_auc']:.4f}, F1: {performance['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        # Find best model
        best_model_name = self._find_best_model()
        training_results['best_model'] = best_model_name
        
        logger.info(f"Training complete. Best model: {best_model_name}")
        logger.info(f"Class imbalance technique: {self.imbalance_technique}")
        
        return training_results
    
    def _evaluate_model(self, model: Any, model_name: str, X_train: pd.DataFrame, 
                       X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Prediction probabilities
        try:
            train_pred_proba = model.predict_proba(X_train)[:, 1]
            test_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            train_pred_proba = train_pred
            test_pred_proba = test_pred
        
        # Basic metrics
        train_accuracy = (train_pred == y_train).mean()
        test_accuracy = (test_pred == y_test).mean()
        
        # AUC scores
        try:
            train_auc = roc_auc_score(y_train, train_pred_proba)
            test_auc = roc_auc_score(y_test, test_pred_proba)
        except:
            train_auc = train_accuracy
            test_auc = test_accuracy
        
        # Classification report
        test_report = classification_report(y_test, test_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, test_pred)
        
        # Cross-validation score
        cv_scores = self._cross_validate_model(model, X_train, y_train)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model, model_name)
        
        evaluation = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'classification_report': test_report,
            'confusion_matrix': conf_matrix.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'feature_importance': feature_importance,
            'overfitting_score': train_accuracy - test_accuracy,
            'precision': test_report['1']['precision'],
            'recall': test_report['1']['recall'],
            'f1_score': test_report['1']['f1-score']
        }
        
        return evaluation
    
    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                             cv_folds: int = 5) -> np.ndarray:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv_folds: Number of CV folds
            
        Returns:
            Array of CV scores
        """
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            return cv_scores
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return np.array([0])
    
    def _get_feature_importance(self, model: Any, model_name: str) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary of feature importance or None
        """
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importance = np.abs(model.coef_[0])
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return sorted_importance
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {model_name}: {str(e)}")
            return None
    
    def _find_best_model(self) -> str:
        """
        Find the best performing model based on F1-score (better for imbalanced data).
        
        Returns:
            Name of best model
        """
        if not self.model_performance:
            return None
        
        best_model = max(
            self.model_performance.items(),
            key=lambda x: x[1].get('f1_score', 0)
        )
        
        return best_model[0]
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Create a comparison table of all models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, performance in self.model_performance.items():
            if 'error' not in performance:
                comparison_data.append({
                    'Model': model_name,
                    'Test_Accuracy': performance['test_accuracy'],
                    'Test_AUC': performance['test_auc'],
                    'Precision': performance['precision'],
                    'Recall': performance['recall'],
                    'F1_Score': performance['f1_score'],
                    'CV_Mean': performance['cv_mean'],
                    'CV_Std': performance['cv_std'],
                    'Overfitting': performance['overfitting_score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
        
        return comparison_df
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models and artifacts.
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trained models
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = os.path.join(output_dir, 'feature_names.joblib')
        joblib.dump(self.feature_names, features_path)
        
        # Save performance metrics
        performance_path = os.path.join(output_dir, 'model_performance.joblib')
        joblib.dump(self.model_performance, performance_path)
        
        logger.info(f"All artifacts saved to {output_dir}")
    
    def get_best_model(self) -> Tuple[Any, str]:
        """
        Get the best trained model.
        
        Returns:
            Tuple of (best_model, model_name)
        """
        best_model_name = self._find_best_model()
        if best_model_name and best_model_name in self.trained_models:
            return self.trained_models[best_model_name], best_model_name
        return None, None