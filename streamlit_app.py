"""
🛒 Olist Review Score Prediction - Interactive Streamlit Dashboard
================================================================

A comprehensive machine learning dashboard for predicting customer review scores
using the Olist Brazilian e-commerce dataset.

Author: AI ML Pipeline
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import page functions
from streamlit_pages.data_overview import show_data_overview
from streamlit_pages.data_quality import show_data_quality
from streamlit_pages.eda import show_eda
from streamlit_pages.feature_engineering import show_feature_engineering
from streamlit_pages.model_performance import show_model_performance
from streamlit_pages.business_insights import show_business_insights
from streamlit_pages.prediction import show_prediction
from streamlit_pages.technical import show_technical_details

# Configure Streamlit page
st.set_page_config(
    page_title="🛒 Olist Review Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light mode theme
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Sidebar light theme */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff;
        color: #262730;
    }
    
    /* Override any dark backgrounds */
    .stApp > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 1;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .insights-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .insights-box h4 {
        color: #28a745;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .insights-box p {
        color: #495057;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    .warning-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .warning-box h4 {
        color: #ffc107;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .warning-box p {
        color: #495057;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Additional styling for content boxes */
    .success-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .success-box h4 {
        color: #28a745;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .success-box ul, .success-box li {
        color: #495057;
        line-height: 1.6;
    }
    
    .info-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .info-box h4 {
        color: #17a2b8;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .info-box ul, .info-box li {
        color: #495057;
        line-height: 1.6;
    }
    
    .danger-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .danger-box h4 {
        color: #dc3545;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .danger-box ul, .danger-box li {
        color: #495057;
        line-height: 1.6;
    }
    
    /* Ensure content boxes work well in light mode */
    div[style*="background"] {
        color: #262730 !important;
    }
    
    div[style*="background"] h4 {
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    div[style*="background"] ul,
    div[style*="background"] li,
    div[style*="background"] p,
    div[style*="background"] span {
        color: #262730 !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# App configuration
@st.cache_data
def load_app_config():
    """Load application configuration and metadata."""
    return {
        'app_title': '🛒 Olist Review Score Prediction Dashboard',
        'version': '2.0',
        'description': 'Interactive ML Dashboard for Customer Satisfaction Prediction',
        'github_url': 'https://github.com/your-username/olist-review-prediction',
        'contact': 'your-email@example.com'
    }

def main():
    """Main Streamlit application."""
    
    # Load configuration
    config = load_app_config()
    
    # Main header
    st.markdown(f'<div class="main-header">{config["app_title"]}</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    st.sidebar.markdown("Select a section to explore:")
    
    # Navigation menu
    pages = {
        "🏠 Home": "home",
        "📊 Data Overview": "data_overview", 
        "🔍 Data Quality": "data_quality",
        "📈 Exploratory Analysis": "eda",
        "⚙️ Feature Engineering": "feature_engineering",
        "🤖 Model Performance": "model_performance",
        "💼 Business Insights": "business_insights",
        "🎯 Make Predictions": "prediction",
        "📋 Technical Details": "technical"
    }
    
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    page_key = pages[selected_page]
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(f"""
    **Version:** {config['version']}
    
    This dashboard presents a comprehensive machine learning analysis for predicting customer review scores using the Olist Brazilian e-commerce dataset.
    
    **Key Features:**
    - 📊 Interactive data visualizations
    - 🤖 4 ML models comparison
    - 💼 Business impact analysis
    - 🎯 Real-time predictions
    """)
    
    # Route to selected page
    try:
        if page_key == "home":
            show_home_page()
        elif page_key == "data_overview":
            show_data_overview()
        elif page_key == "data_quality":
            show_data_quality()
        elif page_key == "eda":
            show_eda()
        elif page_key == "feature_engineering":
            show_feature_engineering()
        elif page_key == "model_performance":
            show_model_performance()
        elif page_key == "business_insights":
            show_business_insights()
        elif page_key == "prediction":
            show_prediction()
        elif page_key == "technical":
            show_technical_details()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        If you're seeing this error, please try:
        1. Refreshing the page
        2. Selecting a different page from the sidebar
        3. Checking your internet connection
        
        **Error details:** `{}`
        """.format(str(e)))

def show_home_page():
    """Display the home page with project overview."""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Project Overview")
        
        st.markdown("""
        Welcome to the **Olist Review Score Prediction Dashboard**! This comprehensive machine learning 
        application analyzes Brazilian e-commerce data to predict customer satisfaction levels.
        
        #### 🚀 What This Dashboard Offers:
        
        **📊 Data Analysis**
        - Quality assessment of 9 datasets
        - Comprehensive exploratory data analysis
        - Missing value and anomaly detection
        
        **⚙️ Feature Engineering**
        - 38+ engineered features
        - Anti-leakage design principles
        - Business-relevant feature categories
        
        **🤖 Machine Learning**
        - 4 different algorithms compared
        - Class imbalance handling techniques
        - Cross-validation and robust evaluation
        
        **💼 Business Intelligence**
        - ROI and intervention analysis
        - Actionable recommendations
        - Real-time prediction interface
        """)
    
    # Key metrics overview
    st.markdown("---")
    st.markdown("### 📈 Project Highlights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94,750</div>
            <div class="metric-label">Orders Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">38+</div>
            <div class="metric-label">Features Engineered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">ML Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">80.4%</div>
            <div class="metric-label">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started section
    st.markdown("---")
    st.markdown("### 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insights-box">
            <h4>📊 Explore the Data</h4>
            <p>Start with <strong>Data Overview</strong> to understand the Brazilian e-commerce dataset structure and key statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insights-box">
            <h4>🤖 Review Models</h4>
            <p>Check <strong>Model Performance</strong> to see how different algorithms compare in predicting customer satisfaction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insights-box">
            <h4>💼 Business Value</h4>
            <p>Visit <strong>Business Insights</strong> to understand the practical applications and ROI potential.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insights-box">
            <h4>🎯 Try Predictions</h4>
            <p>Use <strong>Make Predictions</strong> to test the model with custom order characteristics.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()