#!/usr/bin/env python3
"""
Minimal Streamlit app to test basic functionality
"""

import streamlit as st

st.title("🧪 Minimal Test App")
st.write("If you can see this, basic Streamlit is working!")

try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except Exception as e:
    st.error(f"❌ Pandas import failed: {e}")

try:
    import plotly.express as px
    st.success("✅ Plotly imported successfully") 
except Exception as e:
    st.error(f"❌ Plotly import failed: {e}")

try:
    from streamlit_pages.data_overview import show_data_overview
    st.success("✅ streamlit_pages.data_overview imported successfully")
except Exception as e:
    st.error(f"❌ streamlit_pages.data_overview import failed: {e}")

st.write("Test completed!")