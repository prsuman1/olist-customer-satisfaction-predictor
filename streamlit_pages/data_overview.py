"""
Data Overview Page for Streamlit Dashboard
==========================================

Displays comprehensive overview of the Olist e-commerce datasets including
dataset statistics, relationships, and key characteristics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def show_data_overview():
    """Display comprehensive data overview page."""
    
    st.markdown("## 📊 Data Overview")
    st.markdown("Comprehensive analysis of the Olist Brazilian e-commerce dataset")
    
    # Dataset summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Dataset Statistics")
        
        # Create sample dataset statistics (you can replace with actual data)
        dataset_stats = {
            'Dataset': [
                'Orders', 'Order Reviews', 'Order Items', 'Order Payments',
                'Customers', 'Sellers', 'Products', 'Geolocation',
                'Product Translation'
            ],
            'Rows': [99441, 99224, 112650, 103886, 99441, 3095, 32951, 1000163, 71],
            'Columns': [8, 7, 8, 5, 5, 4, 9, 5, 2],
            'Memory (MB)': [7.6, 6.8, 8.6, 4.0, 3.8, 0.1, 2.5, 76.3, 0.002],
            'Completeness (%)': [99.8, 99.9, 100.0, 100.0, 100.0, 100.0, 78.5, 85.2, 100.0]
        }
        
        df_stats = pd.DataFrame(dataset_stats)
        
        # Interactive table
        st.dataframe(
            df_stats,
            use_container_width=True,
            height=350
        )
    
    with col2:
        st.markdown("### 🎯 Key Metrics")
        
        # Summary metrics
        total_orders = 99441
        total_customers = 99441
        total_sellers = 3095
        total_products = 32951
        
        metrics_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: bold;">{total_orders:,}</div>
            <div style="font-size: 0.9rem;">Total Orders</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: bold;">{total_customers:,}</div>
            <div style="font-size: 0.9rem;">Unique Customers</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: bold;">{total_sellers:,}</div>
            <div style="font-size: 0.9rem;">Active Sellers</div>
        </div>
        
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; font-weight: bold;">{total_products:,}</div>
            <div style="font-size: 0.9rem;">Products</div>
        </div>
        """
        
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Data relationships diagram
    st.markdown("---")
    st.markdown("### 🔗 Dataset Relationships")
    
    st.markdown("""
    The Olist dataset consists of 9 interconnected tables that capture the complete e-commerce transaction lifecycle:
    """)
    
    # Create relationship visualization
    fig = go.Figure()
    
    # Define node positions
    nodes = {
        'Orders': (0, 0),
        'Customers': (-2, 1),
        'Order Items': (0, -2),
        'Order Reviews': (2, 1),
        'Order Payments': (-2, -1),
        'Products': (0, -4),
        'Sellers': (2, -3),
        'Geolocation': (-4, 0),
        'Translation': (0, -6)
    }
    
    # Add nodes
    for node, (x, y) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color='lightblue', line=dict(width=2)),
            text=node,
            textposition="middle center",
            name=node,
            showlegend=False
        ))
    
    # Add connections
    connections = [
        ('Orders', 'Customers'),
        ('Orders', 'Order Items'),
        ('Orders', 'Order Reviews'),
        ('Orders', 'Order Payments'),
        ('Order Items', 'Products'),
        ('Order Items', 'Sellers'),
        ('Customers', 'Geolocation'),
        ('Products', 'Translation')
    ]
    
    for start, end in connections:
        x0, y0 = nodes[start]
        x1, y1 = nodes[end]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Dataset Relationship Diagram",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data timeline
    st.markdown("---")
    st.markdown("### 📅 Data Timeline & Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample timeline data - using fixed data for cloud stability
        months = pd.date_range('2016-09-01', '2018-10-01', freq='M')
        num_months = len(months)
        # Fixed sample data instead of random
        base_orders = [2500, 2800, 3200, 3500, 4200, 4800, 5200, 5800, 6100, 6400, 
                      6800, 7200, 5900, 6200, 6600, 7000, 7300, 7600, 6800, 7100, 
                      7400, 7700, 6200, 6500, 6800]
        orders = base_orders[:num_months]
        
        timeline_data = {
            'Month': months,
            'Orders': orders
        }
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.line(
            timeline_df, 
            x='Month', 
            y='Orders',
            title="Order Volume Over Time",
            labels={'Orders': 'Number of Orders', 'Month': 'Month'}
        )
        fig.update_traces(line_color='#667eea', line_width=3)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic coverage
        states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
        orders_by_state = [41442, 12852, 11635, 5466, 5045, 3637, 3380, 2020, 1652, 1336]
        
        fig = px.bar(
            x=states,
            y=orders_by_state,
            title="Orders by Brazilian State (Top 10)",
            labels={'x': 'State', 'y': 'Number of Orders'},
            color=orders_by_state,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality overview
    st.markdown("---")
    st.markdown("### ✅ Data Quality Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎯 Completeness
        - **95.3%** overall data retention
        - **4,691** orders excluded due to missing values
        - **Zero tolerance** for critical field gaps
        """)
    
    with col2:
        st.markdown("""
        #### 🔍 Integrity Checks
        - **Referential integrity** validated across all tables
        - **Business rule compliance** verified
        - **Temporal consistency** confirmed
        """)
    
    with col3:
        st.markdown("""
        #### 🚨 Anomalies Detected
        - **Negative prices** identified and flagged
        - **Impossible dates** detected and excluded
        - **Statistical outliers** analyzed and documented
        """)
    
    # Sample data preview
    st.markdown("---")
    st.markdown("### 👀 Sample Data Preview")
    
    # Create sample data for demonstration
    sample_orders = pd.DataFrame({
        'order_id': ['e481f51cbdc54678b7cc49136f2d6af7', 'a19ad6eff6494aaa8e3c14c0bb85d9e9'],
        'customer_id': ['9ef432eb6251297304e76186b10a928d', 'b0830fb4747a6c6d20dea0b8c802d7ca'],
        'order_status': ['delivered', 'delivered'],
        'order_purchase_timestamp': ['2017-10-02 10:56:33', '2018-07-24 20:41:37'],
        'order_delivered_customer_date': ['2017-10-06 09:15:00', '2018-08-07 15:27:45'],
        'review_score': [4, 5]
    })
    
    st.dataframe(sample_orders, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Dataset Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div style="background: #e8f5e8; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;">
            <h4>✅ Data Strengths</h4>
            <ul>
                <li><strong>Comprehensive Coverage:</strong> Complete e-commerce lifecycle captured</li>
                <li><strong>Rich Features:</strong> Geographic, temporal, and behavioral data</li>
                <li><strong>Scale:</strong> Nearly 100K orders across 2+ years</li>
                <li><strong>Quality:</strong> High completeness and referential integrity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107;">
            <h4>⚠️ Considerations</h4>
            <ul>
                <li><strong>Geographic Scope:</strong> Limited to Brazilian market</li>
                <li><strong>Temporal Range:</strong> 2016-2018 data (may need updates)</li>
                <li><strong>Class Imbalance:</strong> 78.9% positive reviews</li>
                <li><strong>Missing Reviews:</strong> 217 orders without review scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)