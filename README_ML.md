# 🤖 Olist ML Pipeline - Technical Documentation

This file contains the technical documentation for the machine learning pipeline. For the Streamlit dashboard, see the main README.md.

## 🔧 Core ML Pipeline

### Project Structure
```
src/                          # Core ML pipeline
├── data/                     # Data processing modules
├── features/                 # Feature engineering
├── models/                   # Model training
├── evaluation/               # Model evaluation
├── visualization/            # Report generation
└── utils/                    # Utilities
```

### Running the ML Pipeline
```bash
# Run the full ML pipeline
python main.py

# Run enhanced pipeline with reports
python run_enhanced_pipeline.py
```

### Model Training
```bash
# Train individual models
python -m src.models.trainer

# Evaluate models
python -m src.evaluation.evaluator
```

### Generated Outputs
- `outputs/model_artifacts/` - Trained models
- `outputs/processed_data/` - Processed datasets
- `outputs/comprehensive_ml_report.html` - Full analysis report

## 📊 Technical Results

### Model Performance
- **Best Model**: XGBoost with 80.4% accuracy
- **AUC-ROC**: 0.665
- **Features**: 38+ engineered features
- **Data Retention**: 95.3% after cleaning

### Business Impact
- **Annual Revenue Impact**: R$ 2.3M potential
- **First-Year ROI**: 340%
- **Orders at Risk**: 21% identified
- **Intervention Success Rate**: 67%

## 🧪 Development Commands

### Testing
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

This documentation covers the backend ML pipeline. The Streamlit dashboard provides an interactive interface to explore these results.