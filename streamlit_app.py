import streamlit as st

st.title("🛒 Olist Dashboard - Testing")

# Test basic imports one by one
st.write("Testing imports...")

try:
    import pandas as pd
    st.success("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas failed: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported")
except Exception as e:
    st.error(f"❌ numpy failed: {e}")

try:
    import plotly.express as px
    st.success("✅ plotly imported")
except Exception as e:
    st.error(f"❌ plotly failed: {e}")

st.write("Basic test complete!")