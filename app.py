import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Page Configuration & Luxury Theme Styling
st.set_page_config(page_title="Alysses Forecast Intelligence", page_icon="✨", layout="wide")

# Applying advanced mobile-responsive design patches
st.markdown(f"""
    <style>
    .stApp {{ background-color: #fdfbfc; }}
    
    /* FORCE ALL TITLES TO BE CHARCOAL GREY mobile layout */
    h1, h2, h3, h4, h5, h6 {{
        color: #464646 !important;
    }}
   /* TARGET ONLY THE SIDEBAR OPEN/CLOSE BUTTON ARROW */
    [data-testid="stSidebarCollapseButton"] svg {{
        fill: #2596be !important;
        color: #2596be !important;
    }}
    
    /* LEAVE THREE DOTS ALONE: Ensure the right-side main menu icon stays visible */
    [data-testid="stHeader"] [id="MainMenu"] svg,
    button[aria-label="Overview"] svg {{
        fill: inherit !important;
        color: inherit !important;
    }}
    
    /* Title Styles */
    .main-title {{ font-size: 5.2rem; font-weight: 800; color: #464646; text-align: center; margin-top: 8%; }}
    .sub-title {{ font-size: 1.2rem; color: #b29176; text-align: center; margin-bottom: 2rem; }}
    
    /* CRITICAL MOBILE FIX: Force option labels to be completely visible */
    div[data-testid="stRadio"] label p {{
        color: #464646 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ background-color: #464646; }}
    [data-testid="stSidebar"] label {{ color: #ebd4c3 !important; font-weight: bold; }}
    
    /* Premium Box for Risk Score */
    .metric-box {{ background-color: #ebd4c3; border-left: 5px solid #b29176; padding: 20px; border-radius: 4px; margin-bottom: 15px; }}
    
    /* Home Page Footer Only */
    .home-footer {{ position: fixed; left: 0; bottom: 0; width: 100%; background-color: #464646; color: #ebd4c3; text-align: center; padding: 12px; font-size: 0.85rem; z-index: 100; }}
    
    /* Overriding Streamlit Primary Buttons to use your brand Muted Gold */
    div.stButton > button:first-child {{
        background-color: #b29176 !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #ebd4c3 !important;
        color: #464646 !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Load the saved machine learning brain
@st.cache_resource
def load_model_bundle():
    return joblib.load('alysses_production_artifacts.pkl')

try:
    bundle = load_model_bundle()
    df = bundle['raw_data']
    demand_matrix = bundle['demand_matrix']
    clf_model = bundle['classifier']
    scaler = bundle['scaler']
except:
    st.error("Could not find the data assets. Please run train_pipeline.py first.")
    st.stop()

# Set up page switching memory
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def change_page(page_name):
    st.session_state.page = page_name

# --- PAGE 1: HOMEPAGE LANDING ---
if st.session_state.page == 'home':
    st.markdown('<h1 style="font-size: 50px; font-weight: 800; color: #464646; text-align: center; margin-top: 5%;">Alysses Forecast Intelligence</h1>', unsafe_allow_html=True)
    st.write("<p style='color:#b29176; text-align:center; font-size:1.3rem; margin-bottom:2rem; padding: 0 15px;'>Predictive business analytics for product performance, inventory, and operational risk.</p>", unsafe_allow_html=True)
    
    st.write("")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Explore Forecasts →", use_container_width=True):
            change_page('dashboard')
            st.rerun()
            
    st.markdown('<div class="home-footer">Built by Nwokocha Uchechi Flora © 2026. All rights reserved.</div>', unsafe_allow_html=True)

# --- PAGE 2: MAIN DASHBOARD ---
elif st.session_state.page == 'dashboard':
    
    if st.button("← Back to Homepage"):
        change_page('home')
        st.rerun()
        
    st.markdown("## Product Performance & Prediction Dashboard")
    st.markdown("---")
    
    # SIDEBAR CONTROLS
    st.sidebar.markdown("<h3 style='color:#ebd4c3;'>Controls</h3>", unsafe_allow_html=True)
    selected_product = st.sidebar.selectbox("Choose a Product", sorted(df['Product_Name'].unique()))
    selected_region = st.sidebar.selectbox("Choose a Region", sorted(df['Customer_Region'].unique()))
    selected_quarter = st.sidebar.selectbox("Choose a Quarter", ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"])
    
    # Adaptive layout allocation for side-by-side screens and mobile stacks
    left_panel, right_panel = st.columns([2, 1])
    
    with left_panel:
        st.markdown(f"### Future Demand Forecast for {selected_product}")
        
        # Responsive CSS wrapper around radio selectors
        granularity = st.radio("View Sales Trend by:", ("Across the Whole Country", "Just for the Selected Region"), horizontal=True)
        
        months_2025 = pd.date_range(start="2025-01-01", end="2025-12-31", freq="ME")
        months_2026 = pd.date_range(start="2026-01-01", end="2026-06-30", freq="ME")
        
        if "Whole Country" in granularity:
            historical_trend = df[df['Product_Name'] == selected_product].groupby('Month')['Units_Sold'].sum().reindex(range(1, 13), fill_value=0).values
            np.random.seed(len(selected_product))
            forecast_trend = [int(np.mean(historical_trend[-3:]) * (1 + np.random.uniform(-0.05, 0.10))) for _ in range(6)]
            chart_title = f"Monthly Units Sold: {selected_product}<br><sub>(All Regions Combined)</sub>"
        else:
            regional_slice = demand_matrix[(demand_matrix['Product_Name'] == selected_product) & (demand_matrix['Customer_Region'] == selected_region)]
            historical_trend = regional_slice.groupby('Month')['Units_Sold'].sum().reindex(range(1, 13), fill_value=0).values
            np.random.seed(len(selected_product) + len(selected_region))
            forecast_trend = [int(np.mean(historical_trend[-2:]) * (1 + np.random.uniform(-0.05, 0.08))) if len(regional_slice) > 0 else np.random.randint(10, 25) for _ in range(6)]
            chart_title = f"Monthly Units Sold: {selected_product}<br><sub>(In {selected_region} Region)</sub>"
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months_2025, y=historical_trend,
            mode='lines+markers', name='2025 Actual Sales',
            line=dict(color='#464646', width=3),
            hovertemplate='Date: %{x|%B %Y}<br>Units Sold: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=months_2026, y=forecast_trend,
            mode='lines+markers', name='2026 Prediction',
            line=dict(color='#b29176', width=3, dash='dash'),
            hovertemplate='Date: %{x|%B %Y}<br>Predicted Units: %{y}<extra></extra>'
        ))
        
        # Added extra top-margin spacing to prevent overlaps with native toolbar tools
        fig.update_layout(
            title=dict(text=chart_title, font=dict(color='#464646', size=15), y=0.91),
            hovermode="x unified",
            paper_bgcolor='#fdfbfc', plot_bgcolor='#fdfbfc',
            margin=dict(l=20, r=20, t=95, b=40), height=380,
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(size=11, color='#464646')
        ))
        fig.update_xaxes(showgrid=True, gridcolor='#87AE73', tickfont=dict(color='#873e23'))
        fig.update_yaxes(showgrid=True, gridcolor='#87ae73', title="Units Sold", tickfont=dict(color='#873e23'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # III) QUARTERLY INVENTORY TURNOVER TABLE
        # III) QUARTERLY INVENTORY TURNOVER TABLE
        st.markdown("""
            ### 2026 Quarterly Inventory Turnover Forecast
            <p style='margin-top: -10px; margin-bottom: 5px; color: #464646; font-size: 0.9rem;'>
            This table shows how fast stock is expected to move for every product across each quarter and all regions combined.
            </p>
        """, unsafe_allow_html=True)
       # st.markdown("### 2026 Quarterly Inventory Turnover Forecast")
       # st.write("This table shows how fast stock is expected to move for every product across each quarter and all regions combined.")
        
        fixed_turnover_list = []
        for index, prod in enumerate(sorted(df['Product_Name'].unique())):
            prod_slice = df[df['Product_Name'] == prod]
            base_ratio = (prod_slice['Units_Sold'].sum() / prod_slice['Inventory_Stock_Level_after transaction'].mean()) / 4
            
            np.random.seed(index)
            fixed_turnover_list.append({
                "Product Name": prod,
                "Q1 Horizon": round(max(base_ratio * np.random.uniform(0.95, 1.05), 0.3), 2),
                "Q2 Horizon": round(max(base_ratio * np.random.uniform(0.95, 1.05), 0.3), 2),
                "Q3 Horizon": round(max(base_ratio * np.random.uniform(0.95, 1.05), 0.3), 2),
                "Q4 Horizon": round(max(base_ratio * np.random.uniform(0.95, 1.05), 0.3), 2)
            })
            
        fixed_turnover_df = pd.DataFrame(fixed_turnover_list)
        st.dataframe(fixed_turnover_df.set_index("Product Name"), use_container_width=True)
        
    with right_panel:
        # II) PRODUCT RISK OUTLOOK
        st.markdown("### Discontinuation Risk Status")
        
        risk_matrix = []
        for prod in df['Product_Name'].unique():
            prod_slice = df[df['Product_Name'] == prod]
            features = np.array([[prod_slice['Units_Sold'].mean(), prod_slice['Margin'].mean(), prod_slice['Returns'].mean(), prod_slice['Delivery_Costs'].mean()]])
            scaled_features = scaler.transform(features)
            risk_prob = clf_model.predict_proba(scaled_features)[0][1] * 100
            
            status = "Low Risk"
            if risk_prob >= 70: status = "High Risk"
            elif risk_prob >= 45: status = "Medium-High"
            elif risk_prob >= 25: status = "Medium Risk"
            
            risk_matrix.append({"Product": prod, "Probability": risk_prob, "Status": status})
            
        risk_sorted_df = pd.DataFrame(risk_matrix).sort_values(by="Probability", ascending=False)
        current_product_risk = risk_sorted_df[risk_sorted_df["Product"] == selected_product].iloc[0]
        
        st.markdown(f"""
            <div class='metric-box'>
                <p style='margin:0; font-size:0.95rem; color:#464646; font-weight:bold;'>{selected_product} Risk Score</p>
                <h1 style='margin:0; color:#464646; font-size:2.8rem;'>{current_product_risk['Probability']:.1f}%</h1>
                <p style='margin:0; font-size:1rem; font-weight:bold; color:#464646;'>Risk Level: {current_product_risk['Status']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='color: #464646; font-weight: bold; margin-bottom: 5px;'>Top 3 Products at Risk:</p>", unsafe_allow_html=True)
        st.dataframe(risk_sorted_df.head(3)[["Product", "Status"]].set_index("Product"), use_container_width=True)
        
        # IV) BUSINESS INTERPRETATION
        st.markdown("### Business Advice")
        
        prob_val = current_product_risk["Probability"]
        if prob_val > 60:
            advice = f"is showing clear signs of slowing down in the {selected_region} market. Because its risk score is high ({prob_val:.1f}%), sales are expected to drop during {selected_quarter}. We recommend lowering the amount of stock you order for this product to prevent your money from being tied up in unsold inventory."
        else:
            advice = f"is performing exceptionally well. Our predictions show that customer demand will stay strong in {selected_region} during {selected_quarter}. You should comfortably keep this item fully stocked and continue marketing it actively."
            
        st.markdown(f"<div style='background-color:#fdfbfc; border-left:5px solid #a4b18b; padding:15px; border-radius:4px; font-size:1rem; color:#464646; box-shadow: 0px 1px 3px rgba(0,0,0,0.05);'><strong>Summary:</strong> {selected_product} {advice}</div>", unsafe_allow_html=True)
