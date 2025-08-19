"""
Main execution script for the Olist Review Score Prediction project.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config.config import DATA_FILES, MODEL_CONFIG, OUTPUT_CONFIG
from src.data.loader import OlistDataLoader
from src.data.quality import DataQualityAnalyzer
from src.data.preprocessor import OlistDataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import get_project_logger

# Setup logging
logger = get_project_logger("main")

def main():
    """Main execution function."""
    logger.info("Starting Olist Review Score Prediction Pipeline...")
    
    # Step 1: Load Data
    logger.info("=" * 50)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 50)
    
    data_loader = OlistDataLoader(DATA_FILES)
    datasets = data_loader.load_all_datasets()
    
    # Step 2: Data Quality Analysis
    logger.info("=" * 50)
    logger.info("STEP 2: DATA QUALITY ANALYSIS")
    logger.info("=" * 50)
    
    quality_analyzer = DataQualityAnalyzer(datasets)
    quality_analysis = quality_analyzer.analyze_all_datasets()
    
    # Save quality analysis
    quality_report_path = OUTPUT_CONFIG['data_artifacts'] / 'data_quality_report.json'
    import json
    
    def json_serializer(obj):
        """Custom JSON serializer for pandas and numpy objects."""
        if hasattr(obj, 'dtype'):
            return str(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return str(obj)
    
    with open(quality_report_path, 'w') as f:
        json.dump(quality_analysis, f, indent=2, default=json_serializer)
    logger.info(f"Data quality report saved to {quality_report_path}")
    
    # Step 3: Data Preprocessing
    logger.info("=" * 50)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("=" * 50)
    
    preprocessor = OlistDataPreprocessor(datasets)
    
    # Create master dataset
    master_df = preprocessor.create_master_dataset()
    logger.info(f"Master dataset created: {master_df.shape}")
    
    # Preprocess for ML
    processed_df, preprocessing_report = preprocessor.preprocess_for_ml(master_df)
    logger.info(f"Processed dataset ready: {processed_df.shape}")
    
    # Save preprocessing report
    preprocessing_report_path = OUTPUT_CONFIG['data_artifacts'] / 'preprocessing_report.json'
    with open(preprocessing_report_path, 'w') as f:
        json.dump(preprocessing_report, f, indent=2, default=json_serializer)
    
    # Step 4: Feature Engineering
    logger.info("=" * 50)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 50)
    
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_all_features(processed_df)
    logger.info(f"Feature engineering complete: {engineered_df.shape}")
    
    # Save feature descriptions
    feature_descriptions = feature_engineer.get_feature_descriptions()
    feature_desc_path = OUTPUT_CONFIG['data_artifacts'] / 'feature_descriptions.json'
    with open(feature_desc_path, 'w') as f:
        json.dump(feature_descriptions, f, indent=2, default=json_serializer)
    
    # Step 5: Model Training
    logger.info("=" * 50)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("=" * 50)
    
    # Use SMOTE as the best class imbalance technique based on comprehensive analysis
    trainer = ModelTrainer(
        random_state=MODEL_CONFIG['random_state'],
        imbalance_technique='smote'  # Best performing technique from analysis
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        engineered_df, 
        target_column=MODEL_CONFIG['target_column'], 
        test_size=MODEL_CONFIG['test_size']
    )
    
    # Train all models
    training_results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    trainer.save_models(str(OUTPUT_CONFIG['model_artifacts']))
    
    # Step 6: Model Evaluation
    logger.info("=" * 50)
    logger.info("STEP 6: MODEL EVALUATION")
    logger.info("=" * 50)
    
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.comprehensive_evaluation(
        trainer.trained_models, X_test, y_test, trainer.model_performance
    )
    
    # Save evaluation results
    evaluation_path = OUTPUT_CONFIG['data_artifacts'] / 'evaluation_results.json'
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=json_serializer)
    
    # Step 7: Generate Summary Report
    logger.info("=" * 50)
    logger.info("STEP 7: GENERATING REPORTS")
    logger.info("=" * 50)
    
    # Create comprehensive results dictionary
    final_results = {
        'project_summary': {
            'total_orders_analyzed': len(master_df),
            'final_dataset_size': engineered_df.shape,
            'features_engineered': len(feature_engineer.get_created_features()),
            'models_trained': len(trainer.trained_models),
            'best_model': trainer._find_best_model()
        },
        'data_quality_analysis': quality_analysis,
        'preprocessing_report': preprocessing_report,
        'feature_engineering': {
            'created_features': feature_engineer.get_created_features(),
            'feature_descriptions': feature_descriptions
        },
        'model_training_results': training_results,
        'evaluation_results': evaluation_results,
        'summary_insights': evaluator.generate_summary_insights()
    }
    
    # Save final results
    final_results_path = OUTPUT_CONFIG['data_artifacts'] / 'final_results.json'
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=json_serializer)
    
    logger.info(f"Final results saved to {final_results_path}")
    
    # Generate HTML Report
    from src.visualization.report_generator import HTMLReportGenerator
    
    report_generator = HTMLReportGenerator(final_results)
    html_report = report_generator.generate_report()
    
    with open(OUTPUT_CONFIG['html_report'], 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    logger.info(f"HTML report generated: {OUTPUT_CONFIG['html_report']}")
    
    # Print summary
    logger.info("=" * 50)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 50)
    
    best_model_name = trainer._find_best_model()
    if best_model_name:
        best_performance = trainer.model_performance[best_model_name]
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Test Accuracy: {best_performance['test_accuracy']:.4f}")
        logger.info(f"Test AUC-ROC: {best_performance['test_auc']:.4f}")
        logger.info(f"F1-Score: {best_performance['f1_score']:.4f}")
    
    logger.info(f"Data Quality Issues Found: {len(quality_analysis['summary']['critical_issues'])}")
    logger.info(f"Data Retention Rate: {preprocessing_report['exclusion_summary']['data_retention_rate']:.1f}%")
    logger.info(f"Features Engineered: {len(feature_engineer.get_created_features())}")
    
    logger.info("Pipeline execution completed successfully!")
    
    return final_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*60)
        print("OLIST REVIEW PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Check the output directory for detailed results and HTML report:")
        print(f"  - HTML Report: {OUTPUT_CONFIG['html_report']}")
        print(f"  - Data Artifacts: {OUTPUT_CONFIG['data_artifacts']}")
        print(f"  - Model Artifacts: {OUTPUT_CONFIG['model_artifacts']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise