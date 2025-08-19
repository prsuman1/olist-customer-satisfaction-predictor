#!/usr/bin/env python3
"""
Test script to validate all imports for Streamlit app
"""

try:
    print("Testing streamlit_app.py imports...")
    
    # Test basic imports
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json
    from pathlib import Path
    import warnings
    print("✅ Basic imports successful")
    
    # Test page imports
    from streamlit_pages.data_overview import show_data_overview
    print("✅ data_overview import successful")
    
    from streamlit_pages.data_quality import show_data_quality
    print("✅ data_quality import successful")
    
    from streamlit_pages.eda import show_eda
    print("✅ eda import successful")
    
    from streamlit_pages.feature_engineering import show_feature_engineering
    print("✅ feature_engineering import successful")
    
    from streamlit_pages.model_performance import show_model_performance
    print("✅ model_performance import successful")
    
    from streamlit_pages.business_insights import show_business_insights
    print("✅ business_insights import successful")
    
    from streamlit_pages.prediction import show_prediction
    print("✅ prediction import successful")
    
    from streamlit_pages.technical import show_technical_details
    print("✅ technical import successful")
    
    print("\n🎉 All imports successful!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()