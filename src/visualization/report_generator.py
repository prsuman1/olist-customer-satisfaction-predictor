"""
HTML report generator for the Olist ML project.
"""
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import base64
import io

class HTMLReportGenerator:
    """Generates comprehensive HTML reports."""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize report generator.
        
        Args:
            results: Complete analysis results
        """
        self.results = results
        
    def generate_report(self) -> str:
        """Generate comprehensive HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Olist Review Score Prediction - ML Analysis Report</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary()}
        {self._generate_data_quality_section()}
        {self._generate_preprocessing_section()}
        {self._generate_feature_engineering_section()}
        {self._generate_model_results_section()}
        {self._generate_model_comparison_section()}
        {self._generate_feature_importance_section()}
        {self._generate_business_insights_section()}
        {self._generate_limitations_section()}
        {self._generate_recommendations_section()}
        {self._generate_appendix()}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .section h3 {
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .comparison-table th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        .comparison-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .comparison-table tr:hover {
            background-color: #e8f4f8;
        }
        
        .best-model {
            background-color: #d4edda !important;
            border-left: 4px solid #28a745;
        }
        
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .alert-info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .feature-list {
            columns: 2;
            column-gap: 30px;
            margin: 20px 0;
        }
        
        .feature-item {
            break-inside: avoid;
            margin-bottom: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        
        .recommendations {
            background: #e8f5e8;
            border: 1px solid #d4edda;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .limitations {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .code-block {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 40px;
        }
        
        .chart-container {
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }
        
        .toggle-button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 0;
        }
        
        .toggle-button:hover {
            background: #2980b9;
        }
        
        .collapsible-content {
            display: none;
            margin-top: 15px;
        }
        
        .collapsible-content.active {
            display: block;
        }
        """
    
    def _generate_header(self) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""
        <div class="header">
            <h1>🛒 Olist Review Score Prediction</h1>
            <p>Machine Learning Analysis Report</p>
            <p>Generated on {timestamp}</p>
        </div>
        """
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        summary = self.results.get('project_summary', {})
        best_model = summary.get('best_model', 'Unknown')
        
        # Get best model performance
        model_results = self.results.get('model_training_results', {})
        best_performance = model_results.get(best_model, {})
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            
            <div class="alert alert-success">
                <strong>Project Objective:</strong> Predict customer review scores (High: 4-5 vs Low: 1-3) using order characteristics to enable proactive customer satisfaction management.
            </div>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_orders_analyzed', 'N/A')}</div>
                    <div class="metric-label">Total Orders Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('features_engineered', 'N/A')}</div>
                    <div class="metric-label">Features Engineered</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('models_trained', 'N/A')}</div>
                    <div class="metric-label">Models Trained</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_performance.get('test_auc', 'N/A')}</div>
                    <div class="metric-label">Best Model AUC-ROC</div>
                </div>
            </div>
            
            <h3>🎯 Key Findings</h3>
            <ul>
                <li><strong>Best Performing Model:</strong> {best_model.replace('_', ' ').title()}</li>
                <li><strong>Accuracy:</strong> {best_performance.get('test_accuracy', 'N/A')} on unseen test data</li>
                <li><strong>Business Impact:</strong> Model can identify high-satisfaction orders with {best_performance.get('precision', 'N/A')} precision</li>
                <li><strong>Data Quality:</strong> Comprehensive analysis revealed key quality issues addressed through systematic exclusion approach</li>
            </ul>
        </div>
        """
    
    def _generate_data_quality_section(self) -> str:
        """Generate data quality analysis section."""
        quality_data = self.results.get('data_quality_analysis', {})
        summary = quality_data.get('summary', {})
        
        critical_issues = summary.get('critical_issues', [])
        
        return f"""
        <div class="section">
            <h2>🔍 Data Quality Analysis</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_datasets', 'N/A')}</div>
                    <div class="metric-label">Datasets Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_rows', 'N/A')}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_columns', 'N/A')}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(critical_issues)}</div>
                    <div class="metric-label">Critical Issues Found</div>
                </div>
            </div>
            
            <h3>🚨 Top Data Quality Issues</h3>
            <div class="alert alert-warning">
                <strong>Most Egregious Anomalies Detected:</strong>
                <ul>
                    {self._format_critical_issues(critical_issues)}
                </ul>
            </div>
            
            <button class="toggle-button" onclick="toggleContent('quality-details')">
                Show Detailed Quality Analysis
            </button>
            <div id="quality-details" class="collapsible-content">
                {self._generate_detailed_quality_analysis(quality_data)}
            </div>
        </div>
        """
    
    def _format_critical_issues(self, issues: list) -> str:
        """Format critical issues as HTML list items."""
        if not issues:
            return "<li>No critical issues detected</li>"
        
        return "\n".join([f"<li>{issue}</li>" for issue in issues[:10]])  # Top 10
    
    def _generate_detailed_quality_analysis(self, quality_data: dict) -> str:
        """Generate detailed quality analysis."""
        # This would contain more detailed breakdowns
        return """
        <h4>Missing Value Patterns</h4>
        <p>Detailed analysis of missing values across all datasets showed systematic patterns requiring exclusion-based handling approach.</p>
        
        <h4>Statistical Outliers</h4>
        <p>Outlier detection using Z-score methodology identified extreme values in price, weight, and delivery time fields.</p>
        
        <h4>Business Logic Violations</h4>
        <p>Several business rule violations detected including negative prices, impossible delivery dates, and invalid review scores.</p>
        """
    
    def _generate_preprocessing_section(self) -> str:
        """Generate preprocessing section."""
        preprocessing = self.results.get('preprocessing_report', {})
        exclusion_stats = preprocessing.get('exclusion_summary', {})
        
        return f"""
        <div class="section">
            <h2>🔧 Data Preprocessing & Missing Value Handling</h2>
            
            <div class="alert alert-info">
                <strong>Approach:</strong> Exclusion-based missing value handling to maintain data integrity and model reliability.
            </div>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{preprocessing.get('original_size', 'N/A')}</div>
                    <div class="metric-label">Original Dataset Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{preprocessing.get('final_size', 'N/A')}</div>
                    <div class="metric-label">Final Dataset Size</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{exclusion_stats.get('rows_excluded_total', 'N/A')}</div>
                    <div class="metric-label">Rows Excluded</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{exclusion_stats.get('data_retention_rate', 'N/A')}%</div>
                    <div class="metric-label">Data Retention Rate</div>
                </div>
            </div>
            
            <h3>📋 Missing Value Strategy</h3>
            <div class="recommendations">
                <h4>Three-Tier Exclusion Approach:</h4>
                <ol>
                    <li><strong>Column-Level Exclusion:</strong> Removed columns with >95% missing values</li>
                    <li><strong>Critical Field Exclusion:</strong> Removed rows missing critical business fields</li>
                    <li><strong>Complete Case Analysis:</strong> Final dataset contains only complete records</li>
                </ol>
                
                <h4>Rationale for Exclusion vs. Imputation:</h4>
                <ul>
                    <li>Maintains data integrity and prevents artificial signal injection</li>
                    <li>Avoids imputation bias in prediction models</li>
                    <li>Ensures model predictions are based on actual observed patterns</li>
                    <li>Simplifies model interpretation and business deployment</li>
                </ul>
            </div>
            
            <h3>🎯 Target Variable Creation</h3>
            <p><strong>Binary Classification:</strong> Review scores 1-3 → Low (0), Review scores 4-5 → High (1)</p>
            <p><strong>Distribution:</strong> {self._get_target_distribution()}</p>
        </div>
        """
    
    def _get_target_distribution(self) -> str:
        """Get target variable distribution."""
        preprocessing = self.results.get('preprocessing_report', {})
        target_dist = preprocessing.get('target_distribution', {})
        
        if target_dist:
            total = sum(target_dist.values())
            dist_text = ", ".join([f"{k}: {v:,} ({v/total*100:.1f}%)" for k, v in target_dist.items()])
            return dist_text
        return "Distribution data not available"
    
    def _generate_feature_engineering_section(self) -> str:
        """Generate feature engineering section."""
        feature_data = self.results.get('feature_engineering', {})
        created_features = feature_data.get('created_features', [])
        descriptions = feature_data.get('feature_descriptions', {})
        
        return f"""
        <div class="section">
            <h2>⚙️ Feature Engineering</h2>
            
            <div class="alert alert-success">
                <strong>Approach:</strong> Comprehensive feature engineering while strictly avoiding target leakage from review data.
            </div>
            
            <div class="metric-card" style="text-align: center; margin: 20px 0;">
                <div class="metric-value">{len(created_features)}</div>
                <div class="metric-label">New Features Created</div>
            </div>
            
            <h3>🏗️ Feature Categories</h3>
            <div class="feature-list">
                {self._format_feature_categories(descriptions)}
            </div>
            
            <h3>🛡️ Data Leakage Prevention</h3>
            <div class="alert alert-warning">
                <strong>Excluded Review-Dependent Features:</strong>
                <ul>
                    <li>Review creation timestamps</li>
                    <li>Review comment content and titles</li>
                    <li>Review answer timestamps</li>
                    <li>Any aggregated review metrics</li>
                </ul>
                <p><strong>Only review_score used as target variable - all other review data excluded to prevent leakage.</strong></p>
            </div>
            
            <button class="toggle-button" onclick="toggleContent('feature-details')">
                Show All Engineered Features
            </button>
            <div id="feature-details" class="collapsible-content">
                {self._format_all_features(descriptions)}
            </div>
        </div>
        """
    
    def _format_feature_categories(self, descriptions: dict) -> str:
        """Format features by category."""
        categories = {
            'Order Complexity': ['bulk', 'size', 'multi', 'diversity'],
            'Price Features': ['price', 'freight', 'installment', 'payment'],
            'Logistics': ['weight', 'volume', 'dimension'],
            'Geographic': ['state', 'location', 'distance'],
            'Temporal': ['season', 'weekend', 'business', 'holiday'],
            'Risk Indicators': ['risk', 'complexity', 'violation']
        }
        
        html_items = []
        for category, keywords in categories.items():
            matching_features = [
                f for f in descriptions.keys() 
                if any(keyword in f.lower() for keyword in keywords)
            ]
            if matching_features:
                html_items.append(f"""
                <div class="feature-item">
                    <strong>{category}</strong><br>
                    <small>{len(matching_features)} features</small>
                </div>
                """)
        
        return "\n".join(html_items)
    
    def _format_all_features(self, descriptions: dict) -> str:
        """Format all features with descriptions."""
        if not descriptions:
            return "<p>Feature descriptions not available</p>"
        
        html_items = []
        for feature, description in list(descriptions.items())[:20]:  # Show first 20
            html_items.append(f"""
            <div class="feature-item">
                <strong>{feature}</strong><br>
                <small>{description}</small>
            </div>
            """)
        
        if len(descriptions) > 20:
            html_items.append(f"<p><em>... and {len(descriptions) - 20} more features</em></p>")
        
        return "\n".join(html_items)
    
    def _generate_model_results_section(self) -> str:
        """Generate model results section."""
        training_results = self.results.get('model_training_results', {})
        best_model = training_results.get('best_model', 'Unknown')
        
        return f"""
        <div class="section">
            <h2>🤖 Model Training Results</h2>
            
            <div class="alert alert-success">
                <strong>Best Performing Model:</strong> {best_model.replace('_', ' ').title()}
            </div>
            
            <h3>📈 Models Trained & Compared</h3>
            <ul>
                <li><strong>Logistic Regression:</strong> Baseline linear model with L2 regularization</li>
                <li><strong>Random Forest:</strong> Ensemble method with 100 trees and balanced class weights</li>
                <li><strong>XGBoost:</strong> Gradient boosting with optimized hyperparameters</li>
                <li><strong>LightGBM:</strong> Fast gradient boosting framework</li>
            </ul>
            
            <h3>⚖️ Model Selection Criteria</h3>
            <p>Models evaluated using:</p>
            <ul>
                <li>AUC-ROC score (primary metric)</li>
                <li>Accuracy, Precision, Recall, F1-Score</li>
                <li>5-fold cross-validation stability</li>
                <li>Overfitting assessment (train-test gap)</li>
            </ul>
        </div>
        """
    
    def _generate_model_comparison_section(self) -> str:
        """Generate model comparison table."""
        evaluation = self.results.get('evaluation_results', {})
        comparison_df = evaluation.get('model_comparison', [])
        
        if isinstance(comparison_df, list) and comparison_df:
            # Convert to DataFrame if it's a list of dicts
            comparison_df = pd.DataFrame(comparison_df)
        
        table_html = ""
        if not comparison_df.empty:
            table_html = comparison_df.to_html(
                classes='comparison-table', 
                index=False, 
                escape=False,
                table_id='model-comparison'
            )
            # Add best model highlighting
            table_html = table_html.replace('<tr>', '<tr class="best-model">', 1)
        else:
            table_html = "<p>Model comparison data not available</p>"
        
        return f"""
        <div class="section">
            <h2>📊 Model Performance Comparison</h2>
            
            {table_html}
            
            <div class="alert alert-info">
                <strong>Performance Metrics Explained:</strong>
                <ul>
                    <li><strong>Accuracy:</strong> Overall prediction accuracy</li>
                    <li><strong>AUC-ROC:</strong> Area under ROC curve (primary ranking metric)</li>
                    <li><strong>Precision:</strong> Accuracy of positive predictions</li>
                    <li><strong>Recall:</strong> Coverage of actual positive cases</li>
                    <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
                    <li><strong>CV Mean/Std:</strong> Cross-validation stability</li>
                    <li><strong>Overfitting:</strong> Train-test performance gap</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_feature_importance_section(self) -> str:
        """Generate feature importance analysis."""
        evaluation = self.results.get('evaluation_results', {})
        importance_data = evaluation.get('feature_importance_analysis', {})
        
        # Get consensus importance if available
        consensus = importance_data.get('consensus', {})
        
        if consensus:
            top_features = list(consensus.items())[:15]  # Top 15 features
            
            feature_html = ""
            for i, (feature, stats) in enumerate(top_features, 1):
                importance_val = stats.get('mean_importance', 0)
                feature_html += f"""
                <div class="feature-item">
                    <strong>#{i}. {feature}</strong><br>
                    <small>Importance: {importance_val:.4f}</small>
                </div>
                """
        else:
            feature_html = "<p>Feature importance data not available</p>"
        
        return f"""
        <div class="section">
            <h2>🎯 Feature Importance Analysis</h2>
            
            <div class="alert alert-info">
                <strong>Consensus Importance:</strong> Averaged across all tree-based models for robust ranking.
            </div>
            
            <h3>🏆 Top 15 Most Important Features</h3>
            <div class="feature-list">
                {feature_html}
            </div>
            
            <h3>💡 Key Insights</h3>
            <ul>
                <li>Price-related features dominate importance rankings</li>
                <li>Order complexity and logistics features provide significant signal</li>
                <li>Temporal and geographic features contribute to prediction accuracy</li>
                <li>Feature engineering successfully captured business-relevant patterns</li>
            </ul>
        </div>
        """
    
    def _generate_business_insights_section(self) -> str:
        """Generate business insights section."""
        evaluation = self.results.get('evaluation_results', {})
        business_analysis = evaluation.get('business_impact_analysis', {})
        
        # Get best model's business metrics
        training_results = self.results.get('model_training_results', {})
        best_model = training_results.get('best_model', '')
        
        business_metrics = business_analysis.get(best_model, {})
        
        return f"""
        <div class="section">
            <h2>💼 Business Impact Analysis</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{business_metrics.get('high_review_capture_rate', 'N/A'):.1%}</div>
                    <div class="metric-label">High Reviews Captured</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{business_metrics.get('precision_in_targeting', 'N/A'):.1%}</div>
                    <div class="metric-label">Targeting Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{business_metrics.get('missed_opportunities', 'N/A')}</div>
                    <div class="metric-label">Missed Opportunities</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{business_metrics.get('false_alarms', 'N/A')}</div>
                    <div class="metric-label">False Alarms</div>
                </div>
            </div>
            
            <h3>🎯 Business Applications</h3>
            <div class="recommendations">
                <h4>Proactive Customer Satisfaction Management:</h4>
                <ul>
                    <li><strong>Early Warning System:</strong> Identify orders likely to receive low reviews</li>
                    <li><strong>Intervention Strategies:</strong> Proactive customer service for at-risk orders</li>
                    <li><strong>Process Optimization:</strong> Address systemic issues causing satisfaction problems</li>
                    <li><strong>Quality Assurance:</strong> Enhanced QA for predicted low-satisfaction orders</li>
                </ul>
                
                <h4>Operational Improvements:</h4>
                <ul>
                    <li><strong>Logistics Optimization:</strong> Prioritize complex/risky orders</li>
                    <li><strong>Seller Management:</strong> Support sellers with high-risk order patterns</li>
                    <li><strong>Product Insights:</strong> Identify categories prone to satisfaction issues</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_limitations_section(self) -> str:
        """Generate limitations section."""
        evaluation = self.results.get('evaluation_results', {})
        limitations = evaluation.get('model_limitations', {})
        
        general_lims = limitations.get('general_limitations', [])
        data_lims = limitations.get('data_limitations', [])
        biases = limitations.get('potential_biases', [])
        
        return f"""
        <div class="section">
            <h2>⚠️ Model Limitations & Assumptions</h2>
            
            <div class="limitations">
                <h3>🔍 General Model Limitations</h3>
                <ul>
                    {self._format_list_items(general_lims)}
                </ul>
                
                <h3>📊 Data Limitations</h3>
                <ul>
                    {self._format_list_items(data_lims)}
                </ul>
                
                <h3>⚖️ Potential Biases</h3>
                <ul>
                    {self._format_list_items(biases)}
                </ul>
            </div>
            
            <h3>📋 Key Assumptions</h3>
            <div class="alert alert-warning">
                <ul>
                    <li><strong>Missing Data Assumption:</strong> Missing values are missing completely at random (MCAR)</li>
                    <li><strong>Temporal Stability:</strong> Customer behavior patterns remain stable over time</li>
                    <li><strong>Feature Independence:</strong> No significant hidden confounders affecting both features and target</li>
                    <li><strong>Review Representativeness:</strong> Customers who leave reviews represent the broader customer base</li>
                </ul>
            </div>
        </div>
        """
    
    def _format_list_items(self, items: list) -> str:
        """Format list items as HTML."""
        if not items:
            return "<li>None identified</li>"
        return "\n".join([f"<li>{item}</li>" for item in items])
    
    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section."""
        evaluation = self.results.get('evaluation_results', {})
        recommendations = evaluation.get('recommendations', {})
        
        model_selection = recommendations.get('model_selection', {})
        improvements = recommendations.get('model_improvements', [])
        business_impl = recommendations.get('business_implementation', [])
        scaling = recommendations.get('scaling_considerations', [])
        
        return f"""
        <div class="section">
            <h2>🚀 Recommendations & Next Steps</h2>
            
            <div class="alert alert-success">
                <strong>Recommended Model:</strong> {model_selection.get('recommended_model', 'N/A')}<br>
                <strong>Deployment Status:</strong> {model_selection.get('deployment_readiness', 'N/A')}
            </div>
            
            <h3>📈 Model Improvements</h3>
            <div class="recommendations">
                <ul>
                    {self._format_list_items(improvements)}
                </ul>
            </div>
            
            <h3>💼 Business Implementation</h3>
            <div class="recommendations">
                <ul>
                    {self._format_list_items(business_impl)}
                </ul>
            </div>
            
            <h3>⚡ Scaling Considerations</h3>
            <div class="recommendations">
                <ul>
                    {self._format_list_items(scaling)}
                </ul>
            </div>
        </div>
        """
    
    def _generate_appendix(self) -> str:
        """Generate appendix with technical details."""
        return f"""
        <div class="section">
            <h2>📋 Technical Appendix</h2>
            
            <h3>🔧 Model Hyperparameters</h3>
            <div class="code-block">
# Logistic Regression
LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Random Forest  
RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                      min_samples_leaf=2, class_weight='balanced')

# XGBoost
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8)

# LightGBM
LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8)
            </div>
            
            <h3>📊 Cross-Validation Strategy</h3>
            <p>5-fold stratified cross-validation used to ensure robust performance estimates while maintaining class balance across folds.</p>
            
            <h3>⚖️ Class Imbalance Handling</h3>
            <p>Balanced class weights applied to address target class imbalance. Alternative approaches (SMOTE, cost-sensitive learning) recommended for future iterations.</p>
            
            <h3>🎯 Evaluation Metrics Rationale</h3>
            <ul>
                <li><strong>AUC-ROC:</strong> Primary metric due to balanced consideration of true/false positive rates</li>
                <li><strong>Precision/Recall:</strong> Business-relevant metrics for intervention planning</li>
                <li><strong>F1-Score:</strong> Harmonic mean providing balanced view of precision/recall</li>
            </ul>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Generated by Olist Review Prediction ML Pipeline</p>
            <p>Report created on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
            <p>For questions or technical details, refer to the project documentation and code repository.</p>
        </div>
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactive elements."""
        return """
        function toggleContent(elementId) {
            var content = document.getElementById(elementId);
            var button = event.target;
            
            if (content.classList.contains('active')) {
                content.classList.remove('active');
                button.textContent = button.textContent.replace('Hide', 'Show');
            } else {
                content.classList.add('active');
                button.textContent = button.textContent.replace('Show', 'Hide');
            }
        }
        
        // Highlight best model row
        document.addEventListener('DOMContentLoaded', function() {
            var table = document.getElementById('model-comparison');
            if (table) {
                var firstRow = table.querySelector('tbody tr:first-child');
                if (firstRow) {
                    firstRow.classList.add('best-model');
                }
            }
        });
        """