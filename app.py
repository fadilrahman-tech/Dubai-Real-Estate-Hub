import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Dubai Luxe Real Estate 2026", 
    page_icon="🏙️", 
    layout="wide"
)

# --- MODERN UI STYLING (CSS) ---
st.markdown("""
    <style>
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphism Cards for Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Custom Header Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(#60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FILE PATHS (Relative for GitHub) ---
CSV_FILE = "dubai_properties.csv"
MODEL_FILE = 'rf_model.pkl'
SCALER_FILE = 'scaler.pkl'
COLUMNS_FILE = 'model_columns.pkl'

# --- DATA LOADING ---
@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        return df
    return None

# --- ML ARTIFACT LOADING ---
@st.cache_resource
def load_models():
    if all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, COLUMNS_FILE]):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE), joblib.load(COLUMNS_FILE)
    return None, None, None

df = load_data()
model, scaler, model_columns = load_models()

# --- HEADER ---
st.markdown('<h1 class="main-title">🏙️ DUBAI REAL ESTATE HUB</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7; font-size: 1.1rem;'>Advanced Market Dashboard & AI Predictive Valuation</p>", unsafe_allow_html=True)
st.write("##")

tab1, tab2, tab3 = st.tabs(["📊 Market Analytics", "🔮 AI Rent Predictor", "🤖 Data Intelligence"])

# --- TAB 1: ANALYTICS ---
with tab1:
    if df is not None:
        # Metric Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Listings", f"{len(df):,}")
        m2.metric("Avg Rent", f"{df['Rent'].mean():,.0f} AED")
        m3.metric("Avg Area", f"{df['Area_in_sqft'].mean():,.0f} sqft")
        m4.metric("Active Areas", f"{df['Location'].nunique()}")

        st.write("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Rent Distribution")
            # Use Plotly for interactive modern charts
            fig_rent = px.histogram(df[df['Rent'] < df['Rent'].quantile(0.98)], x="Rent", 
                                   color_discrete_sequence=['#60a5fa'], template="plotly_dark", nbins=40)
            fig_rent.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rent, use_container_width=True)

        with c2:
            st.markdown("#### Listings by Property Type")
            fig_type = px.pie(df, names='Type', hole=0.5, 
                             color_discrete_sequence=px.colors.sequential.Blues_r, template="plotly_dark")
            fig_type.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("#### Top 10 High-Listing Locations")
        top_locs = df['Location'].value_counts().head(10).reset_index()
        fig_loc = px.bar(top_locs, x='count', y='Location', orientation='h',
                        color='count', color_continuous_scale='Blues', template="plotly_dark")
        fig_loc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_loc, use_container_width=True)
    else:
        st.error(f"⚠️ '{CSV_FILE}' not found. Please upload it to your GitHub repository.")

# --- TAB 2: PREDICTOR ---
with tab2:
    if model and df is not None:
        st.markdown("### 🔮 Predict Property Market Value")
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                beds = st.slider("Number of Beds", 0, 12, 1)
                baths = st.slider("Number of Baths", 1, 12, 1)
                area = st.number_input("Total Area (sqft)", value=1000, step=50)
            with col2:
                prop_type = st.selectbox("Property Type", options=df['Type'].unique())
                furnishing = st.radio("Furnishing Status", options=df['Furnishing'].unique(), horizontal=True)
                location = st.selectbox("Location", options=sorted(df['Location'].unique()))
            
            if st.button("Generate Valuation 🏠", use_container_width=True, type="primary"):
                # Prediction logic
                input_data = pd.DataFrame([{
                    'Beds': beds, 'Baths': baths, 'Area_in_sqft': area,
                    'Age_of_listing_in_days': 30, 'Type': prop_type,
                    'Furnishing': furnishing, 'Location': location
                }])
                input_encoded = pd.get_dummies(input_data).reindex(columns=model_columns, fill_value=0)
                input_scaled = scaler.transform(input_encoded)
                prediction = model.predict(input_scaled)[0]
                
                st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.1); padding: 30px; border-radius: 15px; border: 1px solid #3b82f6; text-align: center; margin-top: 20px;">
                        <h4 style="color: #f8fafc; margin-bottom: 0;">Predicted Annual Rent</h4>
                        <h1 style="color: #60a5fa; font-size: 3.5rem; margin-top: 10px;">{prediction:,.2f} AED</h1>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Prediction artifacts (.pkl) not found. Ensure they are in the GitHub root.")

# --- TAB 3: ASSISTANT ---
with tab3:
    st.markdown("### 🤖 Data Intelligence Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me about market stats, average prices, or listing counts in specific areas."}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the Dubai market..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        p_low = prompt.lower()
        res = "I can currently provide market stats and area averages. Try asking 'What is the average rent?' or 'Top locations?'"
        
        if df is not None:
            if "average rent" in p_low:
                res = f"Across our dataset, the average annual rent is **{df['Rent'].mean():,.2f} AED**."
            elif "total" in p_low:
                res = f"There are currently **{len(df):,}** verified listings available for analysis."
            elif "popular" in p_low or "top" in p_low:
                res = f"The area with the highest listing volume is **{df['Location'].value_counts().index[0]}**."
        
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.chat_message("assistant").write(res)
