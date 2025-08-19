"""
Interactive Prediction Interface for Streamlit Dashboard
======================================================

Real-time prediction interface allowing users to input order characteristics
and receive satisfaction predictions with explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_prediction():
    """Display interactive prediction interface."""
    
    st.markdown("## 🎯 Interactive Prediction Interface")
    st.markdown("Enter order characteristics to predict customer satisfaction probability")
    
    # Prediction form
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Order Characteristics")
        
        # Order basics
        st.markdown("#### 🛒 Order Information")
        total_price = st.number_input("Total Order Value (R$)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
        total_items = st.selectbox("Number of Items", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)
        freight_value = st.number_input("Freight Cost (R$)", min_value=0.0, max_value=500.0, value=15.0, step=1.0)
        
        # Customer information
        st.markdown("#### 👤 Customer Information")
        customer_state = st.selectbox("Customer State", options=[
            'SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE', 'DF', 'ES', 'Other'
        ], index=0)
        
        # Seller information
        st.markdown("#### 🏪 Seller Information")
        unique_sellers = st.selectbox("Number of Different Sellers", options=[1, 2, 3, 4, 5], index=0)
        seller_state = st.selectbox("Primary Seller State", options=[
            'SP', 'MG', 'RJ', 'RS', 'SC', 'PR', 'BA', 'GO', 'PE', 'DF', 'Other'
        ], index=0)
        
        # Product characteristics
        st.markdown("#### 📦 Product Characteristics")
        product_weight = st.number_input("Total Weight (grams)", min_value=0, max_value=50000, value=1000, step=100)
        product_photos = st.selectbox("Average Product Photos", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
        
        # Temporal factors
        st.markdown("#### 📅 Timing Factors")
        is_weekend = st.checkbox("Weekend Purchase")
        is_holiday_season = st.checkbox("Holiday Season (Nov-Dec)")
        
        # Payment information
        st.markdown("#### 💳 Payment Information")
        payment_installments = st.selectbox("Number of Installments", options=[1, 2, 3, 4, 5, 6, 10, 12, 15, 18, 24], index=0)
        payment_type = st.selectbox("Payment Method", options=['credit_card', 'boleto', 'voucher', 'debit_card'], index=0)
        
        # Predict button
        predict_button = st.button("🔮 Predict Customer Satisfaction", type="primary")
    
    with col2:
        if predict_button:
            # Calculate prediction (simulated)
            prediction_result = calculate_prediction(
                total_price, total_items, freight_value, customer_state, unique_sellers,
                seller_state, product_weight, product_photos, is_weekend, 
                is_holiday_season, payment_installments, payment_type
            )
            
            show_prediction_results(prediction_result)
        else:
            # Show example prediction
            st.markdown("### 💡 Example Prediction")
            st.info("👆 Fill in the order characteristics on the left and click 'Predict' to see the results!")
            
            # Show sample prediction for demonstration
            sample_result = {
                'satisfaction_probability': 0.73,
                'risk_level': 'Medium',
                'risk_color': 'orange',
                'confidence': 0.85,
                'key_factors': [
                    ('Order Value', 0.15, 'Positive'),
                    ('Freight Ratio', -0.12, 'Negative'), 
                    ('Multi-seller', -0.08, 'Negative'),
                    ('Major State', 0.06, 'Positive'),
                    ('Weekend Purchase', -0.03, 'Negative')
                ],
                'recommendations': [
                    'Consider expedited shipping option',
                    'Proactive customer communication recommended',
                    'Monitor delivery timeline closely'
                ]
            }
            
            st.markdown("#### 📊 Sample Prediction Result")
            display_prediction_summary(sample_result)

def calculate_prediction(total_price, total_items, freight_value, customer_state, 
                        unique_sellers, seller_state, product_weight, product_photos,
                        is_weekend, is_holiday_season, payment_installments, payment_type):
    """Calculate prediction based on input features (simulated)."""
    
    # Simulate feature engineering
    freight_ratio = freight_value / total_price if total_price > 0 else 0
    avg_item_price = total_price / total_items
    is_major_state = 1 if customer_state in ['SP', 'RJ', 'MG'] else 0
    same_state_delivery = 1 if customer_state == seller_state else 0
    is_multi_seller = 1 if unique_sellers > 1 else 0
    is_bulk_order = 1 if total_items > 5 else 0
    weight_per_item = product_weight / total_items
    
    # Simulated XGBoost prediction logic (simplified)
    base_probability = 0.78  # Base positive class probability
    
    # Price factors
    price_factor = 0
    if total_price < 50:
        price_factor = -0.15
    elif total_price > 200:
        price_factor = 0.10
    
    # Freight factors
    freight_factor = 0
    if freight_ratio > 0.2:
        freight_factor = -0.12
    elif freight_ratio < 0.05:
        freight_factor = 0.08
    
    # Logistics factors
    logistics_factor = 0
    if is_multi_seller:
        logistics_factor -= 0.08
    if is_bulk_order:
        logistics_factor -= 0.05
    if same_state_delivery:
        logistics_factor += 0.06
    
    # Geographic factors
    geo_factor = 0.06 if is_major_state else -0.03
    
    # Temporal factors
    temporal_factor = 0
    if is_weekend:
        temporal_factor -= 0.03
    if is_holiday_season:
        temporal_factor -= 0.04
    
    # Product factors
    product_factor = 0
    if product_photos >= 4:
        product_factor += 0.04
    if weight_per_item > 2000:  # Heavy items
        product_factor -= 0.02
    
    # Payment factors
    payment_factor = 0
    if payment_installments > 6:
        payment_factor -= 0.06
    if payment_type == 'credit_card':
        payment_factor += 0.02
    
    # Calculate final probability
    total_adjustment = (price_factor + freight_factor + logistics_factor + 
                       geo_factor + temporal_factor + product_factor + payment_factor)
    
    satisfaction_probability = max(0.1, min(0.9, base_probability + total_adjustment))
    
    # Determine risk level
    if satisfaction_probability >= 0.7:
        risk_level = 'Low'
        risk_color = 'green'
    elif satisfaction_probability >= 0.5:
        risk_level = 'Medium'
        risk_color = 'orange'
    else:
        risk_level = 'High'
        risk_color = 'red'
    
    # Calculate confidence (simulated)
    confidence = min(0.95, 0.75 + abs(satisfaction_probability - 0.5) * 0.4)
    
    # Key contributing factors
    key_factors = [
        ('Order Value', price_factor, 'Positive' if price_factor > 0 else 'Negative' if price_factor < 0 else 'Neutral'),
        ('Freight Ratio', freight_factor, 'Positive' if freight_factor > 0 else 'Negative' if freight_factor < 0 else 'Neutral'),
        ('Multi-seller Order', logistics_factor, 'Positive' if logistics_factor > 0 else 'Negative' if logistics_factor < 0 else 'Neutral'),
        ('Geographic Location', geo_factor, 'Positive' if geo_factor > 0 else 'Negative'),
        ('Timing Factors', temporal_factor, 'Positive' if temporal_factor > 0 else 'Negative' if temporal_factor < 0 else 'Neutral'),
        ('Product Characteristics', product_factor, 'Positive' if product_factor > 0 else 'Negative' if product_factor < 0 else 'Neutral'),
        ('Payment Method', payment_factor, 'Positive' if payment_factor > 0 else 'Negative' if payment_factor < 0 else 'Neutral')
    ]
    
    # Filter significant factors
    key_factors = [(name, impact, direction) for name, impact, direction in key_factors 
                   if abs(impact) > 0.01]
    
    # Sort by absolute impact
    key_factors.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Generate recommendations
    recommendations = []
    
    if satisfaction_probability < 0.6:
        recommendations.append("🚨 High-risk order - immediate intervention recommended")
        recommendations.append("📞 Proactive customer service contact suggested")
    elif satisfaction_probability < 0.7:
        recommendations.append("⚠️ Medium-risk order - enhanced monitoring recommended")
    
    if freight_ratio > 0.15:
        recommendations.append("📦 Consider freight cost optimization or free shipping promotion")
    
    if is_multi_seller:
        recommendations.append("🏪 Coordinate between multiple sellers for seamless experience")
    
    if not same_state_delivery:
        recommendations.append("🚚 Monitor cross-state delivery for potential delays")
    
    if is_weekend or is_holiday_season:
        recommendations.append("📅 Account for seasonal/weekend delivery variations")
    
    if not recommendations:
        recommendations.append("✅ Order appears low-risk - standard processing recommended")
    
    return {
        'satisfaction_probability': satisfaction_probability,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'confidence': confidence,
        'key_factors': key_factors,
        'recommendations': recommendations,
        'feature_values': {
            'total_price': total_price,
            'freight_ratio': freight_ratio,
            'avg_item_price': avg_item_price,
            'is_major_state': is_major_state,
            'same_state_delivery': same_state_delivery,
            'is_multi_seller': is_multi_seller,
            'weight_per_item': weight_per_item
        }
    }

def show_prediction_results(result):
    """Display prediction results with visualizations."""
    
    st.markdown("### 🔮 Prediction Results")
    
    # Main prediction display
    display_prediction_summary(result)
    
    # Detailed analysis
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Factor Analysis", "📈 Risk Assessment", "💡 Recommendations"])
    
    with tab1:
        show_factor_analysis(result)
    
    with tab2:
        show_risk_assessment(result)
    
    with tab3:
        show_recommendations_analysis(result)

def display_prediction_summary(result):
    """Display the main prediction summary."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['satisfaction_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Satisfaction Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk level indicator
        risk_color = result['risk_color']
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h3>Risk Level</h3>
            <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 10px; font-size: 1.5em; font-weight: bold; margin: 1rem 0;">
                {result['risk_level']} Risk
            </div>
            <p>Confidence: {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Quick stats
        st.markdown("#### 📊 Quick Stats")
        
        prob_pct = result['satisfaction_probability'] * 100
        if prob_pct >= 70:
            icon = "😊"
            message = "Likely satisfied"
        elif prob_pct >= 50:
            icon = "😐"
            message = "Uncertain outcome"
        else:
            icon = "😟"
            message = "At risk"
        
        st.markdown(f"""
        - {icon} **Prediction:** {message}
        - 🎯 **Probability:** {prob_pct:.1f}%
        - 🎪 **Confidence:** {result['confidence']:.1%}
        - 🚨 **Risk Level:** {result['risk_level']}
        - 📋 **Factors:** {len(result['key_factors'])} analyzed
        """)

def show_factor_analysis(result):
    """Show detailed factor analysis."""
    
    st.markdown("#### 🔍 Contributing Factors Analysis")
    
    if result['key_factors']:
        # Factor impact chart
        factors_df = pd.DataFrame(result['key_factors'], columns=['Factor', 'Impact', 'Direction'])
        factors_df['Abs_Impact'] = factors_df['Impact'].abs()
        factors_df = factors_df.sort_values('Abs_Impact', ascending=True)
        
        # Color code by direction
        colors = ['red' if direction == 'Negative' else 'green' if direction == 'Positive' else 'gray' 
                 for direction in factors_df['Direction']]
        
        fig = go.Figure(go.Bar(
            y=factors_df['Factor'],
            x=factors_df['Impact'],
            orientation='h',
            marker_color=colors,
            text=[f"{impact:+.3f}" for impact in factors_df['Impact']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Factor Impact on Satisfaction Probability",
            xaxis_title="Impact Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor details table
        st.markdown("#### 📋 Factor Details")
        display_df = factors_df[['Factor', 'Impact', 'Direction']].copy()
        display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x:+.3f}")
        
        # Color code the dataframe
        def color_direction(val):
            if val == 'Positive':
                return 'background-color: #d4edda'
            elif val == 'Negative':
                return 'background-color: #f8d7da'
            return 'background-color: #f8f9fa'
        
        styled_df = display_df.style.applymap(color_direction, subset=['Direction'])
        st.dataframe(styled_df, use_container_width=True)
    
    else:
        st.info("No significant contributing factors identified.")

def show_risk_assessment(result):
    """Show risk assessment details."""
    
    st.markdown("#### ⚠️ Risk Assessment Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level explanation
        risk_level = result['risk_level']
        probability = result['satisfaction_probability']
        
        if risk_level == 'Low':
            st.markdown("""
            <div style="background: #d4edda; padding: 1rem; border-radius: 10px; border-left: 5px solid #28a745;">
                <h4>✅ Low Risk Assessment</h4>
                <p><strong>Interpretation:</strong> This order has a high probability of resulting in customer satisfaction.</p>
                <ul>
                    <li>Expected satisfaction probability: {:.1%}</li>
                    <li>Minimal intervention required</li>
                    <li>Standard processing recommended</li>
                    <li>Monitor for any delivery delays</li>
                </ul>
            </div>
            """.format(probability), unsafe_allow_html=True)
        
        elif risk_level == 'Medium':
            st.markdown("""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 5px solid #ffc107;">
                <h4>⚠️ Medium Risk Assessment</h4>
                <p><strong>Interpretation:</strong> This order shows mixed signals and requires careful monitoring.</p>
                <ul>
                    <li>Expected satisfaction probability: {:.1%}</li>
                    <li>Enhanced monitoring recommended</li>
                    <li>Proactive communication suggested</li>
                    <li>Consider quality assurance checks</li>
                </ul>
            </div>
            """.format(probability), unsafe_allow_html=True)
        
        else:  # High risk
            st.markdown("""
            <div style="background: #f8d7da; padding: 1rem; border-radius: 10px; border-left: 5px solid #dc3545;">
                <h4>🚨 High Risk Assessment</h4>
                <p><strong>Interpretation:</strong> This order has significant risk factors that could lead to customer dissatisfaction.</p>
                <ul>
                    <li>Expected satisfaction probability: {:.1%}</li>
                    <li>Immediate intervention required</li>
                    <li>Proactive customer service contact</li>
                    <li>Enhanced quality assurance</li>
                    <li>Consider expedited processing</li>
                </ul>
            </div>
            """.format(probability), unsafe_allow_html=True)
    
    with col2:
        # Risk distribution comparison
        st.markdown("#### 📊 Risk Distribution Context")
        
        # Simulated risk distribution from historical data
        risk_dist = {
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Historical %': [65, 25, 10],
            'Current Order': [0, 0, 0]
        }
        
        # Mark current prediction
        if risk_level == 'Low':
            risk_dist['Current Order'][0] = 100
        elif risk_level == 'Medium':
            risk_dist['Current Order'][1] = 100
        else:
            risk_dist['Current Order'][2] = 100
        
        risk_df = pd.DataFrame(risk_dist)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Historical Distribution',
            x=risk_df['Risk Level'],
            y=risk_df['Historical %'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Current Order',
            x=risk_df['Risk Level'],
            y=risk_df['Current Order'],
            mode='markers',
            marker=dict(color='red', size=15, symbol='diamond'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Risk Level Comparison",
            yaxis=dict(title="Historical Distribution (%)", side="left"),
            yaxis2=dict(title="Current Order", side="right", overlaying="y", showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence explanation
        confidence = result['confidence']
        st.markdown(f"""
        #### 🎯 Prediction Confidence
        
        **Confidence Level: {confidence:.1%}**
        
        {'🟢 High Confidence' if confidence > 0.8 else '🟡 Medium Confidence' if confidence > 0.6 else '🔴 Low Confidence'}
        
        The model is {confidence:.1%} confident in this prediction based on the feature values and historical patterns.
        """)

def show_recommendations_analysis(result):
    """Show detailed recommendations analysis."""
    
    st.markdown("#### 💡 Actionable Recommendations")
    
    recommendations = result['recommendations']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Detailed recommendations
        for i, rec in enumerate(recommendations, 1):
            priority = "High" if "🚨" in rec else "Medium" if "⚠️" in rec else "Low"
            color = "#dc3545" if priority == "High" else "#ffc107" if priority == "Medium" else "#28a745"
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span><strong>#{i}</strong> {rec}</span>
                    <span style="background: {color}; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8em;">{priority}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Action timeline
        st.markdown("#### ⏰ Suggested Timeline")
        
        timeline_items = []
        for rec in recommendations:
            if "immediate" in rec.lower() or "🚨" in rec:
                timeline_items.append(("Immediate", "0-4 hours"))
            elif "proactive" in rec.lower() or "⚠️" in rec:
                timeline_items.append(("Short-term", "4-24 hours"))
            else:
                timeline_items.append(("Standard", "1-3 days"))
        
        # Group by timeline
        timeline_counts = {}
        for timeline, duration in timeline_items:
            if timeline not in timeline_counts:
                timeline_counts[timeline] = 0
            timeline_counts[timeline] += 1
        
        if timeline_counts:
            fig = px.pie(
                values=list(timeline_counts.values()),
                names=list(timeline_counts.keys()),
                title="Action Priority Distribution",
                color_discrete_sequence=['#dc3545', '#ffc107', '#28a745']
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    # Implementation checklist
    st.markdown("---")
    st.markdown("#### ✅ Implementation Checklist")
    
    checklist_items = [
        "Review prediction results with team",
        "Assess intervention capacity and resources", 
        "Prioritize actions based on risk level",
        "Implement immediate interventions (if any)",
        "Schedule follow-up monitoring",
        "Document outcomes for model improvement"
    ]
    
    checklist_df = pd.DataFrame({
        'Task': checklist_items,
        'Status': ['⬜'] * len(checklist_items),
        'Owner': ['Customer Service', 'Operations', 'Management', 'Customer Service', 'Quality Assurance', 'Data Team'],
        'Timeline': ['Immediate', 'Short-term', 'Immediate', 'Immediate', 'Ongoing', 'Long-term']
    })
    
    st.dataframe(checklist_df, use_container_width=True)
    
    # Cost-benefit analysis
    st.markdown("---")
    st.markdown("#### 💰 Intervention Cost-Benefit Analysis")
    
    prob = result['satisfaction_probability']
    intervention_cost = 35  # Average cost per intervention
    missed_opportunity_cost = 180  # Cost of unsatisfied customer
    
    expected_value_no_action = prob * 0 + (1 - prob) * (-missed_opportunity_cost)
    expected_value_with_action = prob * (-intervention_cost * 0.3) + (1 - prob) * (-intervention_cost)  # Assume 70% intervention success
    
    net_benefit = expected_value_with_action - expected_value_no_action
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💸 Intervention Cost",
            value=f"R$ {intervention_cost}",
            help="Average cost per proactive intervention"
        )
    
    with col2:
        st.metric(
            label="💔 Missed Opportunity Cost", 
            value=f"R$ {missed_opportunity_cost}",
            help="Cost of not addressing dissatisfied customer"
        )
    
    with col3:
        st.metric(
            label="💰 Expected Net Benefit",
            value=f"R$ {net_benefit:.0f}",
            delta="Per order intervention",
            help="Expected value of intervention vs no action"
        )
    
    if net_benefit > 0:
        st.success(f"💡 **Recommendation:** Intervention is cost-effective with expected benefit of R$ {net_benefit:.0f}")
    else:
        st.warning(f"⚠️ **Note:** Intervention may not be cost-effective (expected cost: R$ {abs(net_benefit):.0f})")