"""
Minimal version to test basic Streamlit functionality
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="🛒 Olist Review Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    st.title("🛒 Olist Review Prediction Dashboard")
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    
    # Simple navigation without complex imports
    pages = {
        "🏠 Home": "home",
        "📊 Test Page": "test"
    }
    
    selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
    page_key = pages[selected_page]
    
    # Route to selected page
    if page_key == "home":
        show_home_page()
    elif page_key == "test":
        show_test_page()

def show_home_page():
    """Display the home page."""
    st.markdown("### 🎯 Project Overview")
    st.write("This is a minimal test of the Streamlit app.")
    st.success("✅ Home page loaded successfully!")

def show_test_page():
    """Display test page."""
    st.markdown("### 🧪 Test Page")
    st.write("Testing basic functionality...")
    
    # Test basic operations
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    st.dataframe(df)
    st.success("✅ Test page loaded successfully!")

if __name__ == "__main__":
    main()