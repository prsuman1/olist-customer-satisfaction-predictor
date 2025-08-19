#!/usr/bin/env python3
"""
Execute each notebook cell and compare with HTML report results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from pathlib import Path

# Configure warnings and display
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("🧪 RUNNING NOTEBOOK CELLS SYSTEMATICALLY")
print("=" * 70)

def compare_with_expected(actual, expected, description):
    """Compare actual vs expected results"""
    if actual == expected:
        print(f"✅ {description}: {actual} (MATCHES)")
    else:
        print(f"❌ {description}: Got {actual}, Expected {expected}")

# Expected results from HTML report
EXPECTED_FINAL_RECORDS = 94750
EXPECTED_BEST_ACCURACY = 0.804  # 80.4%
EXPECTED_BEST_MODEL = "xgboost"

try:
    # Cell 1: Setup and Imports
    print("\n📦 CELL 1: Setup and Imports")
    print("-" * 40)
    
    from config.config import DATA_FILES, MODEL_CONFIG
    from src.data.loader import OlistDataLoader
    from src.data.quality import DataQualityAnalyzer
    from src.data.preprocessor import OlistDataPreprocessor
    from src.features.engineer import FeatureEngineer
    from src.models.trainer import ModelTrainer
    from src.evaluation.evaluator import ModelEvaluator
    
    print("✅ All imports successful")
    print(f"   Working directory: {os.getcwd()}")

    # Cell 2: Data Loading
    print("\n📂 CELL 2: Data Loading")
    print("-" * 40)
    
    loader = OlistDataLoader(DATA_FILES)
    datasets = loader.load_all_datasets()
    
    print(f"✅ Loaded {len(datasets)} datasets")
    total_rows = sum(df.shape[0] for df in datasets.values())
    print(f"   Total rows across all datasets: {total_rows:,}")
    
    # Check key datasets
    key_datasets = ['orders', 'order_reviews', 'customers', 'order_items']
    for name in key_datasets:
        if name in datasets:
            print(f"   • {name}: {datasets[name].shape}")

    # Cell 3: Target Variable Analysis
    print("\n🎯 CELL 3: Target Variable Analysis")
    print("-" * 40)
    
    reviews_df = datasets['order_reviews'].copy()
    
    # Review score distribution
    score_counts = reviews_df['review_score'].value_counts().sort_index()
    print("   Review Score Distribution:")
    for score, count in score_counts.items():
        percentage = (count / len(reviews_df)) * 100
        print(f"     {score} stars: {count:,} ({percentage:.1f}%)")
    
    # Create binary target
    reviews_df['target'] = (reviews_df['review_score'] >= 4).astype(int)
    target_counts = reviews_df['target'].value_counts().sort_index()
    
    print("   Binary Target Distribution:")
    for target, count in target_counts.items():
        percentage = (count / len(reviews_df)) * 100
        label = 'High Satisfaction (4-5 stars)' if target == 1 else 'Low Satisfaction (1-3 stars)'
        print(f"     Target {target} ({label}): {count:,} ({percentage:.1f}%)")
    
    class_ratio = target_counts.max() / target_counts.min()
    print(f"   Class Imbalance Ratio: {class_ratio:.2f}:1")
    
    # Update the dataset
    datasets['order_reviews'] = reviews_df

    # Cell 4: Data Quality Analysis
    print("\n🔍 CELL 4: Data Quality Analysis")
    print("-" * 40)
    
    quality_summary = {}
    
    for name, df in datasets.items():
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        quality_summary[name] = {
            'shape': df.shape,
            'total_missing_values': int(total_missing),
            'missing_percentage': (total_missing / df.size) * 100,
            'duplicate_rows': int(df.duplicated().sum()),
            'columns_with_missing': [col for col in df.columns if missing_counts[col] > 0]
        }
    
    print("   Data Quality Summary:")
    for dataset_name in ['orders', 'order_reviews', 'customers', 'order_items']:
        if dataset_name in quality_summary:
            report = quality_summary[dataset_name]
            print(f"     {dataset_name.upper()}:")
            print(f"       Shape: {report['shape']}")
            print(f"       Missing values: {report['total_missing_values']} ({report['missing_percentage']:.1f}%)")
            print(f"       Duplicate rows: {report['duplicate_rows']}")

    # Cell 5: Data Preprocessing & Master Dataset Creation
    print("\n🔧 CELL 5: Data Preprocessing")
    print("-" * 40)
    
    preprocessor = OlistDataPreprocessor(datasets)
    master_df = preprocessor.create_master_dataset()
    
    print(f"✅ Master dataset created: {master_df.shape}")
    print(f"   Rows: {master_df.shape[0]:,}")
    print(f"   Columns: {master_df.shape[1]}")

    # Cell 6: Missing Value Handling & Data Exclusion
    print("\n🧹 CELL 6: Missing Value Handling")
    print("-" * 40)
    
    processed_df, preprocessing_report = preprocessor.preprocess_for_ml(master_df)
    
    print("   Data Exclusion Summary:")
    print(f"     Original size: {preprocessing_report['original_size']:,} records")
    print(f"     Final size: {preprocessing_report['final_size']:,} records")
    print(f"     Rows excluded: {preprocessing_report['rows_excluded']:,}")
    print(f"     Retention rate: {100 - preprocessing_report['exclusion_percentage']:.1f}%")
    
    # CRITICAL CHECK: Compare with expected 94,750
    compare_with_expected(
        preprocessing_report['final_size'], 
        EXPECTED_FINAL_RECORDS, 
        "Final dataset size"
    )
    
    # Target distribution
    target_dist = preprocessing_report['target_distribution']
    print("   Target Distribution in Final Dataset:")
    for value, count in target_dist.items():
        percentage = (count / preprocessing_report['final_size']) * 100
        label = "High Satisfaction (4-5 stars)" if value == 1 else "Low Satisfaction (1-3 stars)"
        print(f"     {label}: {count:,} ({percentage:.1f}%)")

    # Cell 7: Feature Engineering
    print("\n⚙️ CELL 7: Feature Engineering")
    print("-" * 40)
    
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_all_features(processed_df)
    
    print(f"✅ Feature engineering complete: {engineered_df.shape}")
    print(f"   New features created: {len(feature_engineer.created_features)}")
    print(f"   Total features available: {engineered_df.shape[1]}")
    
    # Check for categorical features
    feature_cols = [col for col in engineered_df.columns if col != 'target']
    categorical_features = engineered_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    if categorical_features:
        print(f"   ⚠️ Remaining categorical features: {categorical_features}")
    else:
        print("   ✅ All features are numeric")

    # Cell 8: Model Training Preparation
    print("\n🤖 CELL 8: Model Training Preparation")
    print("-" * 40)
    
    trainer = ModelTrainer(random_state=MODEL_CONFIG['random_state'])
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        engineered_df, 
        target_column='target',
        test_size=MODEL_CONFIG['test_size']
    )
    
    print(f"✅ Data prepared for training")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Target distribution: {y_train.value_counts().to_dict()}")

    # Cell 9: Model Training
    print("\n🚀 CELL 9: Model Training")
    print("-" * 40)
    
    training_results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    print(f"✅ Model training complete")
    print(f"   Models trained: {len(trainer.trained_models)}")
    
    # Display model performance
    print("   Model Performance:")
    for model_name in trainer.trained_models.keys():
        performance = trainer.model_performance[model_name]
        print(f"     {model_name}: {performance['test_accuracy']:.4f} accuracy")
    
    # Find best model and compare with expected
    best_model_name = trainer._find_best_model()
    if best_model_name:
        best_performance = trainer.model_performance[best_model_name]
        print(f"   🏆 Best model: {best_model_name}")
        
        # Compare with expected results
        compare_with_expected(
            best_model_name.lower(), 
            EXPECTED_BEST_MODEL, 
            "Best model name"
        )
        compare_with_expected(
            round(best_performance['test_accuracy'], 3), 
            EXPECTED_BEST_ACCURACY, 
            "Best model accuracy"
        )

    # Cell 10: Model Evaluation
    print("\n📊 CELL 10: Model Evaluation")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.comprehensive_evaluation(
        trainer.trained_models, X_test, y_test, trainer.model_performance
    )
    
    print("✅ Model evaluation complete")
    print("   Comprehensive evaluation results available")

    # Cell 11: Feature Importance
    print("\n🔍 CELL 11: Feature Importance")
    print("-" * 40)
    
    best_model, best_model_name = trainer.get_best_model()
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        print(f"✅ Feature importance analyzed ({best_model_name})")
        print("   Top 10 features:")
        for idx, row in importance_df.iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    else:
        print(f"   ℹ️ Feature importance not available for {best_model_name}")

    # Final Summary
    print("\n📈 FINAL SUMMARY")
    print("=" * 70)
    
    print("✅ All notebook cells executed successfully!")
    print(f"✅ Final dataset: {len(engineered_df):,} records")
    print(f"✅ Features: {X_train.shape[1]} features")
    print(f"✅ Best model: {best_model_name}")
    if best_model_name:
        print(f"✅ Best accuracy: {best_performance['test_accuracy']:.4f}")
    
    # CRITICAL VALIDATION
    print("\n🎯 CRITICAL VALIDATION AGAINST HTML REPORT:")
    print("-" * 50)
    
    validation_passed = True
    
    # Check 1: Record count
    if preprocessing_report['final_size'] == EXPECTED_FINAL_RECORDS:
        print(f"✅ Record count: {preprocessing_report['final_size']:,} (MATCHES)")
    else:
        print(f"❌ Record count: Got {preprocessing_report['final_size']:,}, Expected {EXPECTED_FINAL_RECORDS:,}")
        validation_passed = False
    
    # Check 2: Best model
    if best_model_name and best_model_name.lower() == EXPECTED_BEST_MODEL:
        print(f"✅ Best model: {best_model_name} (MATCHES)")
    else:
        print(f"❌ Best model: Got {best_model_name}, Expected {EXPECTED_BEST_MODEL}")
        validation_passed = False
    
    # Check 3: Accuracy (with tolerance)
    if best_model_name:
        accuracy_diff = abs(best_performance['test_accuracy'] - EXPECTED_BEST_ACCURACY)
        if accuracy_diff < 0.01:  # 1% tolerance
            print(f"✅ Best accuracy: {best_performance['test_accuracy']:.4f} (MATCHES within tolerance)")
        else:
            print(f"❌ Best accuracy: Got {best_performance['test_accuracy']:.4f}, Expected {EXPECTED_BEST_ACCURACY:.4f}")
            validation_passed = False
    
    if validation_passed:
        print("\n🎉 SUCCESS: All notebook results match HTML report expectations!")
    else:
        print("\n⚠️ WARNING: Some results differ from HTML report")
    
    print("\n" + "=" * 70)
    print("🎯 NOTEBOOK CELL EXECUTION AND VALIDATION COMPLETE")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error during cell execution: {e}")
    import traceback
    traceback.print_exc()