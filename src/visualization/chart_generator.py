"""
Advanced chart generation for the enhanced HTML report.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, Any, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_project_logger

logger = get_project_logger("chart_generator")

class AdvancedChartGenerator:
    """Generates comprehensive interactive charts for the ML report."""
    
    def __init__(self):
        """Initialize the chart generator."""
        self.charts = {}
        
    def generate_all_charts(self, datasets: Dict[str, pd.DataFrame], 
                           engineered_df: pd.DataFrame,
                           model_results: Dict[str, Any],
                           evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate all charts for the enhanced report.
        
        Args:
            datasets: Original datasets
            engineered_df: Feature engineered dataset
            model_results: Model training results
            evaluation_results: Model evaluation results
            
        Returns:
            Dictionary of chart HTML strings
        """
        logger.info("Generating comprehensive chart suite...")
        
        charts = {}
        
        # 1. Data Quality Charts
        charts.update(self._create_data_quality_charts(datasets))
        
        # 2. EDA Charts
        charts.update(self._create_eda_charts(datasets, engineered_df))
        
        # 3. Target Analysis Charts
        charts.update(self._create_target_analysis_charts(engineered_df))
        
        # 4. Feature Importance Charts
        charts.update(self._create_feature_importance_charts(model_results))
        
        # 5. Model Performance Charts
        charts.update(self._create_model_performance_charts(evaluation_results))
        
        # 6. Geographic Analysis Charts
        charts.update(self._create_geographic_charts(datasets))
        
        # 7. Temporal Analysis Charts
        charts.update(self._create_temporal_charts(datasets))
        
        # 8. Business Impact Charts
        charts.update(self._create_business_impact_charts(evaluation_results))
        
        logger.info(f"Generated {len(charts)} interactive charts")
        return charts
    
    def _create_data_quality_charts(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create data quality visualization charts."""
        charts = {}
        
        # Missing Values Heatmap
        missing_data = []
        for name, df in datasets.items():
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                missing_data.append({
                    'Dataset': name,
                    'Column': col,
                    'Missing_Percentage': missing_pct,
                    'Missing_Count': df[col].isnull().sum(),
                    'Total_Rows': len(df)
                })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df[missing_df['Missing_Percentage'] > 0]  # Only show columns with missing data
        
        if not missing_df.empty:
            fig = px.bar(
                missing_df.head(20), 
                x='Missing_Percentage', 
                y='Column',
                color='Dataset',
                title='📊 Missing Values Analysis - Top 20 Columns',
                labels={'Missing_Percentage': 'Missing Percentage (%)', 'Column': 'Column Name'},
                text='Missing_Count'
            )
            fig.update_layout(height=600, showlegend=True)
            charts['missing_values_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Dataset Size Comparison
        size_data = [{'Dataset': name, 'Rows': df.shape[0], 'Columns': df.shape[1]} 
                     for name, df in datasets.items()]
        size_df = pd.DataFrame(size_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Dataset Rows', 'Dataset Columns'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=size_df['Dataset'], y=size_df['Rows'], name='Rows', marker_color='skyblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=size_df['Dataset'], y=size_df['Columns'], name='Columns', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(title_text="📈 Dataset Size Analysis", height=500)
        charts['dataset_size_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_eda_charts(self, datasets: Dict[str, pd.DataFrame], 
                          engineered_df: pd.DataFrame) -> Dict[str, str]:
        """Create comprehensive EDA charts."""
        charts = {}
        
        # Price Distribution Analysis
        if 'order_items' in datasets:
            items_df = datasets['order_items']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price Distribution', 'Freight Distribution', 
                               'Price vs Freight', 'Log Price Distribution'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # Price histogram
            fig.add_trace(
                go.Histogram(x=items_df['price'], nbinsx=50, name='Price', marker_color='blue'),
                row=1, col=1
            )
            
            # Freight histogram
            fig.add_trace(
                go.Histogram(x=items_df['freight_value'], nbinsx=50, name='Freight', marker_color='green'),
                row=1, col=2
            )
            
            # Price vs Freight scatter
            sample_data = items_df.sample(n=min(5000, len(items_df)))  # Sample for performance
            fig.add_trace(
                go.Scatter(x=sample_data['price'], y=sample_data['freight_value'], 
                          mode='markers', name='Price vs Freight', marker_color='red', opacity=0.5),
                row=2, col=1
            )
            
            # Log price distribution
            log_prices = np.log1p(items_df['price'])
            fig.add_trace(
                go.Histogram(x=log_prices, nbinsx=50, name='Log Price', marker_color='purple'),
                row=2, col=2
            )
            
            fig.update_layout(title_text="💰 Price & Freight Analysis", height=800, showlegend=True)
            charts['price_analysis_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Review Score Distribution
        if 'order_reviews' in datasets:
            reviews_df = datasets['order_reviews']
            score_counts = reviews_df['review_score'].value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=score_counts.index, y=score_counts.values, 
                       marker_color=['red', 'orange', 'yellow', 'lightgreen', 'green'],
                       text=score_counts.values, textposition='auto')
            ])
            fig.update_layout(
                title="⭐ Review Score Distribution",
                xaxis_title="Review Score",
                yaxis_title="Count",
                height=400
            )
            charts['review_distribution_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_target_analysis_charts(self, engineered_df: pd.DataFrame) -> Dict[str, str]:
        """Create target variable analysis charts."""
        charts = {}
        
        if 'target' in engineered_df.columns:
            # Target distribution
            target_counts = engineered_df['target'].value_counts()
            
            # Ensure correct order for pie chart
            target_0_count = target_counts.get(0, 0)  # Low reviews (1-3 stars)
            target_1_count = target_counts.get(1, 0)  # High reviews (4-5 stars)
            
            fig = go.Figure(data=[
                go.Pie(labels=['Low Reviews (1-3 stars)', 'High Reviews (4-5 stars)'], 
                       values=[target_0_count, target_1_count],
                       marker_colors=['#ff7f7f', '#7fbf7f'],
                       textinfo='label+percent+value')
            ])
            fig.update_layout(title="🎯 Target Distribution (High vs Low Satisfaction)", height=500)
            charts['target_distribution_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            # Class imbalance analysis
            imbalance_ratio = target_counts.max() / target_counts.min()
            
            # Ensure correct order: target 0 (Low) first, then target 1 (High)
            target_0_count = target_counts.get(0, 0)  # Low reviews (1-3 stars)
            target_1_count = target_counts.get(1, 0)  # High reviews (4-5 stars)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Low Reviews (1-3 stars) → Target=0', 'High Reviews (4-5 stars) → Target=1'],
                y=[target_0_count, target_1_count],
                marker_color=['red', 'green'],
                text=[f'{target_0_count:,} ({target_0_count/(target_0_count+target_1_count)*100:.1f}%)', 
                      f'{target_1_count:,} ({target_1_count/(target_0_count+target_1_count)*100:.1f}%)'],
                textposition='auto'
            ))
            fig.update_layout(
                title=f"⚖️ Class Imbalance Analysis (Ratio: {imbalance_ratio:.2f}:1)",
                yaxis_title="Count",
                height=400
            )
            charts['class_imbalance_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_feature_importance_charts(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Create feature importance visualization charts."""
        charts = {}
        
        # Extract feature importance from all models
        importance_data = []
        for model_name, results in model_results.items():
            if model_name != 'best_model' and 'feature_importance' in results:
                feature_imp = results['feature_importance']
                if feature_imp:
                    for feature, importance in list(feature_imp.items())[:20]:  # Top 20
                        importance_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Feature': feature,
                            'Importance': importance
                        })
        
        if importance_data:
            imp_df = pd.DataFrame(importance_data)
            
            # Feature importance comparison across models
            fig = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature',
                color='Model',
                orientation='h',
                title='🎯 Feature Importance Comparison Across Models',
                height=800
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            charts['feature_importance_comparison'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            # Top 10 features radar chart
            top_features = imp_df.groupby('Feature')['Importance'].mean().nlargest(10)
            
            fig = go.Figure()
            
            for model in imp_df['Model'].unique():
                model_data = imp_df[imp_df['Model'] == model]
                model_features = model_data.set_index('Feature')['Importance']
                
                values = [model_features.get(feature, 0) for feature in top_features.index]
                values.append(values[0])  # Close the radar chart
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=list(top_features.index) + [top_features.index[0]],
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(top_features.values)])
                ),
                title="🎯 Top 10 Features - Model Comparison (Radar Chart)",
                height=600
            )
            charts['feature_radar_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_model_performance_charts(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive model performance charts."""
        charts = {}
        
        # ROC Curves for all models
        roc_data = evaluation_results.get('roc_analysis', {})
        if roc_data:
            fig = go.Figure()
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random Classifier'
            ))
            
            colors = ['blue', 'red', 'green', 'purple']
            for i, (model_name, roc_info) in enumerate(roc_data.items()):
                if 'fpr' in roc_info and 'tpr' in roc_info:
                    fig.add_trace(go.Scatter(
                        x=roc_info['fpr'],
                        y=roc_info['tpr'],
                        mode='lines',
                        name=f'{model_name.replace("_", " ").title()} (AUC: {roc_info.get("roc_auc", 0):.3f})',
                        line=dict(color=colors[i % len(colors)])
                    ))
            
            fig.update_layout(
                title='📈 ROC Curves - Model Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                width=600
            )
            charts['roc_curves_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Confusion Matrix Heatmaps
        confusion_data = evaluation_results.get('confusion_matrices', {})
        if confusion_data:
            n_models = len([k for k in confusion_data.keys() if 'confusion_matrix' in confusion_data[k]])
            if n_models > 0:
                fig = make_subplots(
                    rows=1, cols=min(n_models, 3),
                    subplot_titles=[model.replace('_', ' ').title() for model in list(confusion_data.keys())[:3]],
                    specs=[[{"type": "heatmap"}] * min(n_models, 3)]
                )
                
                for i, (model_name, conf_data) in enumerate(list(confusion_data.items())[:3]):
                    if 'confusion_matrix' in conf_data:
                        cm = conf_data['confusion_matrix']
                        fig.add_trace(
                            go.Heatmap(
                                z=cm,
                                x=['Predicted Low', 'Predicted High'],
                                y=['Actual Low', 'Actual High'],
                                colorscale='Blues',
                                text=cm,
                                texttemplate="%{text}",
                                textfont={"size": 16},
                                showscale=(i == 0)
                            ),
                            row=1, col=i+1
                        )
                
                fig.update_layout(title="🔥 Confusion Matrix Heatmaps", height=400)
                charts['confusion_matrix_heatmaps'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Model Metrics Comparison
        model_comparison = evaluation_results.get('model_comparison', [])
        if model_comparison is not None and (isinstance(model_comparison, list) and len(model_comparison) > 0) or (hasattr(model_comparison, 'empty') and not model_comparison.empty):
            if isinstance(model_comparison, list):
                comp_df = pd.DataFrame(model_comparison)
            else:
                comp_df = model_comparison
            
            if not comp_df.empty:
                # Convert string percentages to float
                numeric_cols = ['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1-Score']
                for col in numeric_cols:
                    if col in comp_df.columns:
                        comp_df[col] = pd.to_numeric(comp_df[col], errors='coerce')
                
                fig = go.Figure()
                
                metrics = ['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1-Score']
                colors = ['blue', 'red', 'green', 'purple', 'orange']
                
                for i, metric in enumerate(metrics):
                    if metric in comp_df.columns:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=comp_df['Model'],
                            y=comp_df[metric],
                            marker_color=colors[i % len(colors)]
                        ))
                
                fig.update_layout(
                    title="📊 Model Performance Metrics Comparison",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    barmode='group',
                    height=500
                )
                charts['model_metrics_comparison'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_geographic_charts(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create geographic analysis charts."""
        charts = {}
        
        # Customer state distribution
        if 'customers' in datasets:
            customers_df = datasets['customers']
            state_counts = customers_df['customer_state'].value_counts().head(15)
            
            fig = go.Figure(data=[
                go.Bar(x=state_counts.index, y=state_counts.values,
                       marker_color='lightblue',
                       text=state_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                title="🗺️ Customer Distribution by Brazilian State (Top 15)",
                xaxis_title="State",
                yaxis_title="Number of Customers",
                height=500
            )
            charts['customer_state_distribution'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Seller vs Customer state analysis
        if 'customers' in datasets and 'sellers' in datasets:
            customer_states = datasets['customers']['customer_state'].value_counts().head(10)
            seller_states = datasets['sellers']['seller_state'].value_counts().head(10)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top Customer States', 'Top Seller States'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=customer_states.index, y=customer_states.values, 
                       name='Customers', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=seller_states.index, y=seller_states.values, 
                       name='Sellers', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(title_text="🏪 Customer vs Seller Geographic Distribution", height=500)
            charts['customer_seller_geography'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_temporal_charts(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Create temporal analysis charts."""
        charts = {}
        
        if 'orders' in datasets:
            orders_df = datasets['orders'].copy()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(orders_df['order_purchase_timestamp']):
                orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
            
            # Monthly order trends
            orders_df['year_month'] = orders_df['order_purchase_timestamp'].dt.to_period('M')
            monthly_orders = orders_df['year_month'].value_counts().sort_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[str(period) for period in monthly_orders.index],
                y=monthly_orders.values,
                mode='lines+markers',
                name='Orders per Month',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="📅 Order Volume Trends Over Time",
                xaxis_title="Month",
                yaxis_title="Number of Orders",
                height=500
            )
            charts['monthly_trends_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
            
            # Day of week analysis
            orders_df['day_of_week'] = orders_df['order_purchase_timestamp'].dt.day_name()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_counts = orders_df['day_of_week'].value_counts().reindex(dow_order)
            
            fig = go.Figure(data=[
                go.Bar(x=dow_counts.index, y=dow_counts.values,
                       marker_color=['lightblue' if day in ['Saturday', 'Sunday'] else 'lightcoral' 
                                   for day in dow_counts.index],
                       text=dow_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                title="📊 Orders by Day of Week",
                xaxis_title="Day of Week",
                yaxis_title="Number of Orders",
                height=400
            )
            charts['day_of_week_chart'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def _create_business_impact_charts(self, evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """Create business impact analysis charts."""
        charts = {}
        
        # Business metrics comparison
        business_data = evaluation_results.get('business_impact_analysis', {})
        if business_data:
            models = []
            capture_rates = []
            precision_rates = []
            missed_opps = []
            false_alarms = []
            
            for model_name, metrics in business_data.items():
                if isinstance(metrics, dict) and 'high_review_capture_rate' in metrics:
                    models.append(model_name.replace('_', ' ').title())
                    capture_rates.append(metrics.get('high_review_capture_rate', 0) * 100)
                    precision_rates.append(metrics.get('precision_in_targeting', 0) * 100)
                    missed_opps.append(metrics.get('missed_opportunities', 0))
                    false_alarms.append(metrics.get('false_alarms', 0))
            
            if models:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Capture Rate (%)', 'Targeting Precision (%)', 
                                   'Missed Opportunities', 'False Alarms'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                           [{"type": "bar"}, {"type": "bar"}]]
                )
                
                fig.add_trace(go.Bar(x=models, y=capture_rates, name='Capture Rate', 
                                   marker_color='green'), row=1, col=1)
                fig.add_trace(go.Bar(x=models, y=precision_rates, name='Precision', 
                                   marker_color='blue'), row=1, col=2)
                fig.add_trace(go.Bar(x=models, y=missed_opps, name='Missed Opportunities', 
                                   marker_color='red'), row=2, col=1)
                fig.add_trace(go.Bar(x=models, y=false_alarms, name='False Alarms', 
                                   marker_color='orange'), row=2, col=2)
                
                fig.update_layout(title_text="💼 Business Impact Analysis", height=700, showlegend=False)
                charts['business_impact_comparison'] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        return charts
    
    def to_html_string(self, chart_dict: Dict[str, str]) -> str:
        """Convert chart dictionary to HTML string with Plotly.js inclusion."""
        html_parts = [
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            '<div class="charts-container">'
        ]
        
        for chart_name, chart_html in chart_dict.items():
            html_parts.append(f'<div class="chart-section" id="{chart_name}">')
            html_parts.append(chart_html)
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)