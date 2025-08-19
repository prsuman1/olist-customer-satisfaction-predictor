"""
Robust Streamlit App with Error Handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🛒 Olist Review Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import page functions with error handling
page_functions = {}

try:
    from streamlit_pages.data_overview import show_data_overview
    page_functions["data_overview"] = show_data_overview
except Exception as e:
    st.error(f"Failed to import data_overview: {e}")
    page_functions["data_overview"] = None

try:
    from streamlit_pages.data_quality import show_data_quality
    page_functions["data_quality"] = show_data_quality
except Exception as e:
    st.error(f"Failed to import data_quality: {e}")
    page_functions["data_quality"] = None

try:
    from streamlit_pages.eda import show_eda
    page_functions["eda"] = show_eda
except Exception as e:
    st.error(f"Failed to import eda: {e}")
    page_functions["eda"] = None

try:
    from streamlit_pages.feature_engineering import show_feature_engineering
    page_functions["feature_engineering"] = show_feature_engineering
except Exception as e:
    st.error(f"Failed to import feature_engineering: {e}")
    page_functions["feature_engineering"] = None

try:
    from streamlit_pages.model_performance import show_model_performance
    page_functions["model_performance"] = show_model_performance
except Exception as e:
    st.error(f"Failed to import model_performance: {e}")
    page_functions["model_performance"] = None

try:
    from streamlit_pages.business_insights import show_business_insights
    page_functions["business_insights"] = show_business_insights
except Exception as e:
    st.error(f"Failed to import business_insights: {e}")
    page_functions["business_insights"] = None

try:
    from streamlit_pages.prediction import show_prediction
    page_functions["prediction"] = show_prediction
except Exception as e:
    st.error(f"Failed to import prediction: {e}")
    page_functions["prediction"] = None

try:
    from streamlit_pages.technical import show_technical_details
    page_functions["technical"] = show_technical_details
except Exception as e:
    st.error(f"Failed to import technical: {e}")
    page_functions["technical"] = None

def main():
    """Main Streamlit application."""
    
    # Main header
    st.markdown('<div style="font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold;">🛒 Olist Review Prediction Dashboard</div>', unsafe_allow_html=True)
    
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
    st.sidebar.info("""
    **Version:** 2.0
    
    This dashboard presents a comprehensive machine learning analysis for predicting customer review scores using the Olist Brazilian e-commerce dataset.
    
    **Key Features:**
    - 📊 Interactive data visualizations
    - 🤖 4 ML models comparison
    - 💼 Business impact analysis
    - 🎯 Real-time predictions
    """)
    
    # Route to selected page with robust error handling
    try:
        if page_key == "home":
            show_home_page()
        else:
            page_function = page_functions.get(page_key)
            if page_function:
                page_function()
            else:
                st.error(f"Page '{selected_page}' is currently unavailable due to an import error.")
                st.info("Please try selecting a different page from the sidebar.")
    except Exception as e:
        st.error(f"Error loading page '{selected_page}': {str(e)}")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown(f"""
        If you're seeing this error, please try:
        1. Refreshing the page
        2. Selecting a different page from the sidebar
        3. Checking your internet connection
        
        **Error details:** `{str(e)}`
        """)

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
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;">
            <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">94,750</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Orders Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;">
            <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">38+</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Features Engineered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;">
            <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">4</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">ML Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 0.5rem 0;">
            <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">80.4%</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Best Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()