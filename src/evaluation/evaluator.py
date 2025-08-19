"""
Model evaluation and analysis module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings

from ..utils.logger import get_project_logger

logger = get_project_logger("evaluator")

class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def comprehensive_evaluation(self, trained_models: Dict[str, Any], 
                                X_test: pd.DataFrame, y_test: pd.Series,
                                model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of all models.
        
        Args:
            trained_models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            model_performance: Performance metrics from training
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        evaluation = {
            'model_comparison': self._create_model_comparison_table(model_performance),
            'confusion_matrices': self._analyze_confusion_matrices(trained_models, X_test, y_test),
            'roc_analysis': self._analyze_roc_curves(trained_models, X_test, y_test),
            'feature_importance_analysis': self._analyze_feature_importance(model_performance),
            'business_impact_analysis': self._analyze_business_impact(trained_models, X_test, y_test),
            'model_limitations': self._analyze_model_limitations(model_performance),
            'recommendations': self._generate_recommendations(model_performance)
        }
        
        self.evaluation_results = evaluation
        logger.info("Comprehensive evaluation completed")
        
        return evaluation
    
    def _create_model_comparison_table(self, model_performance: Dict[str, Any]) -> pd.DataFrame:
        """Create a comprehensive model comparison table."""
        comparison_data = []
        
        for model_name, metrics in model_performance.items():
            if 'error' not in metrics:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['test_accuracy']:.4f}",
                    'AUC-ROC': f"{metrics['test_auc']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'CV Mean': f"{metrics['cv_mean']:.4f}",
                    'CV Std': f"{metrics['cv_std']:.4f}",
                    'Overfitting': f"{metrics['overfitting_score']:.4f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Sort by AUC-ROC
            comparison_df['AUC_numeric'] = comparison_df['AUC-ROC'].astype(float)
            comparison_df = comparison_df.sort_values('AUC_numeric', ascending=False)
            comparison_df = comparison_df.drop('AUC_numeric', axis=1)
        
        return comparison_df
    
    def _analyze_confusion_matrices(self, trained_models: Dict[str, Any], 
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze confusion matrices for all models."""
        confusion_analysis = {}
        
        for model_name, model in trained_models.items():
            try:
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                
                # Calculate metrics from confusion matrix
                tn, fp, fn, tp = cm.ravel()
                
                confusion_analysis[model_name] = {
                    'confusion_matrix': cm.tolist(),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Error analyzing confusion matrix for {model_name}: {str(e)}")
                confusion_analysis[model_name] = {'error': str(e)}
        
        return confusion_analysis
    
    def _analyze_roc_curves(self, trained_models: Dict[str, Any], 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze ROC curves for all models."""
        roc_analysis = {}
        
        for model_name, model in trained_models.items():
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = model.predict(X_test)
                
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Calculate Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
                avg_precision = average_precision_score(y_test, y_pred_proba)
                
                roc_analysis[model_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'roc_auc': roc_auc,
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'pr_thresholds': pr_thresholds.tolist(),
                    'avg_precision': avg_precision
                }
                
            except Exception as e:
                logger.error(f"Error analyzing ROC for {model_name}: {str(e)}")
                roc_analysis[model_name] = {'error': str(e)}
        
        return roc_analysis
    
    def _analyze_feature_importance(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        importance_analysis = {}
        
        # Collect all feature importances
        all_importances = {}
        
        for model_name, metrics in model_performance.items():
            if 'feature_importance' in metrics and metrics['feature_importance']:
                importance_analysis[model_name] = metrics['feature_importance']
                
                # Aggregate importances
                for feature, importance in metrics['feature_importance'].items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)
        
        # Calculate consensus importance
        if all_importances:
            consensus_importance = {}
            for feature, importances in all_importances.items():
                consensus_importance[feature] = {
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'models_count': len(importances)
                }
            
            # Sort by mean importance
            consensus_importance = dict(
                sorted(consensus_importance.items(), 
                      key=lambda x: x[1]['mean_importance'], reverse=True)
            )
            
            importance_analysis['consensus'] = consensus_importance
        
        return importance_analysis
    
    def _analyze_business_impact(self, trained_models: Dict[str, Any], 
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Analyze business impact of model predictions."""
        business_analysis = {}
        
        for model_name, model in trained_models.items():
            try:
                y_pred = model.predict(X_test)
                
                # Calculate business metrics
                total_reviews = len(y_test)
                actual_high_reviews = y_test.sum()
                predicted_high_reviews = y_pred.sum()
                
                # Correctly identified cases
                correctly_identified_high = ((y_test == 1) & (y_pred == 1)).sum()
                correctly_identified_low = ((y_test == 0) & (y_pred == 0)).sum()
                
                # Missed opportunities and false alarms
                missed_opportunities = ((y_test == 1) & (y_pred == 0)).sum()
                false_alarms = ((y_test == 0) & (y_pred == 1)).sum()
                
                business_analysis[model_name] = {
                    'total_reviews': int(total_reviews),
                    'actual_high_reviews': int(actual_high_reviews),
                    'predicted_high_reviews': int(predicted_high_reviews),
                    'correctly_identified_high': int(correctly_identified_high),
                    'correctly_identified_low': int(correctly_identified_low),
                    'missed_opportunities': int(missed_opportunities),
                    'false_alarms': int(false_alarms),
                    'high_review_capture_rate': correctly_identified_high / actual_high_reviews if actual_high_reviews > 0 else 0,
                    'precision_in_targeting': correctly_identified_high / predicted_high_reviews if predicted_high_reviews > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Error in business analysis for {model_name}: {str(e)}")
                business_analysis[model_name] = {'error': str(e)}
        
        return business_analysis
    
    def _analyze_model_limitations(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model limitations and potential issues."""
        limitations = {
            'general_limitations': [
                'Binary classification may oversimplify customer satisfaction spectrum',
                'Missing values excluded rather than imputed may reduce model generalizability',
                'Temporal patterns not fully captured due to cross-sectional approach',
                'Limited external validation on different time periods or regions'
            ],
            'model_specific_limitations': {},
            'data_limitations': [
                'Review scores are subjective and may vary by customer expectations',
                'Seasonal and promotional effects not fully captured',
                'Seller and product characteristics may change over time',
                'Geographic features limited to state-level granularity'
            ],
            'potential_biases': [
                'Selection bias: Only customers who completed purchases and left reviews',
                'Temporal bias: Data may not represent current customer behavior',
                'Geographic bias: May be skewed toward certain Brazilian regions',
                'Category bias: Performance may vary significantly across product categories'
            ]
        }
        
        # Analyze specific model limitations
        for model_name, metrics in model_performance.items():
            if 'error' not in metrics:
                model_limitations = []
                
                # Check for overfitting
                if metrics['overfitting_score'] > 0.05:
                    model_limitations.append(f"Potential overfitting (train-test gap: {metrics['overfitting_score']:.3f})")
                
                # Check for low performance
                if metrics['test_auc'] < 0.7:
                    model_limitations.append("Low discriminative ability (AUC < 0.7)")
                
                # Check for imbalanced performance
                if abs(metrics['precision'] - metrics['recall']) > 0.2:
                    model_limitations.append("Imbalanced precision-recall trade-off")
                
                # Check cross-validation stability
                if metrics['cv_std'] > 0.05:
                    model_limitations.append(f"High cross-validation variance (std: {metrics['cv_std']:.3f})")
                
                limitations['model_specific_limitations'][model_name] = model_limitations
        
        return limitations
    
    def _generate_recommendations(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for model improvement and business implementation."""
        # Find best model
        best_model = max(
            model_performance.items(),
            key=lambda x: x[1].get('test_auc', 0) if 'error' not in x[1] else 0
        )
        
        recommendations = {
            'model_selection': {
                'recommended_model': best_model[0],
                'reasoning': f"Highest AUC-ROC score ({best_model[1].get('test_auc', 0):.4f}) with balanced performance",
                'deployment_readiness': 'Ready for pilot testing' if best_model[1].get('test_auc', 0) > 0.7 else 'Needs improvement before deployment'
            },
            'model_improvements': [
                'Implement advanced hyperparameter tuning (GridSearch/RandomSearch)',
                'Consider ensemble methods combining multiple models',
                'Engineer additional features from external data sources',
                'Implement time-series cross-validation for temporal robustness',
                'Add feature selection techniques to reduce overfitting'
            ],
            'feature_engineering': [
                'Incorporate customer purchase history and lifecycle stage',
                'Add seller reputation and performance metrics',
                'Include seasonal and promotional event indicators',
                'Engineer product similarity and substitution features',
                'Add delivery distance and logistics complexity metrics'
            ],
            'business_implementation': [
                'Start with pilot program on small subset of orders',
                'Implement A/B testing to measure business impact',
                'Create real-time monitoring dashboard for model performance',
                'Establish model retraining schedule (monthly/quarterly)',
                'Develop intervention strategies for predicted low-satisfaction orders'
            ],
            'scaling_considerations': [
                'Implement feature store for consistent feature computation',
                'Set up automated model training and validation pipelines',
                'Establish model performance monitoring and alerting',
                'Plan for handling increased data volume and velocity',
                'Consider distributed computing for large-scale predictions'
            ]
        }
        
        return recommendations
    
    def generate_summary_insights(self) -> Dict[str, Any]:
        """Generate high-level summary insights from evaluation."""
        if not self.evaluation_results:
            return {}
        
        # Get best model info
        comparison_df = self.evaluation_results.get('model_comparison', pd.DataFrame())
        
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            
            summary = {
                'best_model_performance': {
                    'model_name': best_model['Model'],
                    'accuracy': best_model['Accuracy'],
                    'auc_roc': best_model['AUC-ROC'],
                    'f1_score': best_model['F1-Score']
                },
                'model_comparison_summary': {
                    'total_models_tested': len(comparison_df),
                    'performance_range': {
                        'accuracy_range': f"{comparison_df['Accuracy'].min()} - {comparison_df['Accuracy'].max()}",
                        'auc_range': f"{comparison_df['AUC-ROC'].min()} - {comparison_df['AUC-ROC'].max()}"
                    }
                },
                'key_insights': [
                    f"Best performing model: {best_model['Model']} with {best_model['AUC-ROC']} AUC-ROC",
                    f"Model achieves {best_model['Accuracy']} accuracy on unseen test data",
                    f"Cross-validation shows {best_model['CV Mean']} ± {best_model['CV Std']} stability"
                ]
            }
            
            return summary
        
        return {'error': 'No valid evaluation results available'}