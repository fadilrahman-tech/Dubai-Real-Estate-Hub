import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


CSV_FILE = "dubai_properties.csv"
MODEL_FILE = 'rf_model.pkl'
SCALER_FILE = 'scaler.pkl'
COLUMNS_FILE = 'model_columns.pkl'

st.set_page_config(page_title="Dubai Real Estate App", page_icon="🏙️", layout="wide")

# 2. LOAD DATA
@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        return df
    return None

df = load_data()

# 3. LOAD ML ARTIFACTS
@st.cache_resource
def load_models():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(COLUMNS_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        model_columns = joblib.load(COLUMNS_FILE)
        return model, scaler, model_columns
    return None, None, None

model, scaler, model_columns = load_models()

# --- UI START ---
st.title("🏙️ Dubai Real Estate Hub 2026")
st.markdown("Welcome to the Dubai Real Estate Dashboard, Predictor, and Assistant.")
st.set_page_config(
    page_icon="https://raw.githubusercontent.com/fadilrahman-tech/Dubai-Real-Estate-Hub/main/dubai.png", # Make sure the filename matches your GitHub
    layout="wide"
)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Rent Predictor", "🤖 Data Assistant"])

with tab1:
    st.header("📊 Market Dashboard")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", f"{len(df):,}")
        col2.metric("Average Rent (AED)", f"{df['Rent'].mean():,.2f}")
        col3.metric("Avg Area (sqft)", f"{df['Area_in_sqft'].mean():,.2f}")
        col4.metric("Locations", f"{df['Location'].nunique()}")

        st.subheader("Distribution of Property Features")
        num_cols = ['Rent', 'Beds', 'Baths', 'Area_in_sqft']
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        for i, col in enumerate(num_cols):
            if col in df.columns:
                sns.histplot(df[col], bins=20, ax=axes[i], color='skyblue', kde=True)
                axes[i].set_title(col)
        st.pyplot(fig)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Top Locations")
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, y='Location', order=df['Location'].value_counts().head(10).index, palette='magma', ax=ax2)
            st.pyplot(fig2)
        with colB:
            st.subheader("Property Types")
            fig3, ax3 = plt.subplots()
            sns.countplot(data=df, y='Type', order=df['Type'].value_counts().index, palette='viridis', ax=ax3)
            st.pyplot(fig3)
    else:
        st.error(f"⚠️ **Error:** '{CSV_FILE}' not found in GitHub repository.")

with tab2:
    st.header("🔮 Rent Predictor")
    if model and scaler and model_columns and df is not None:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                beds = st.number_input("Beds", min_value=0, max_value=20, value=1)
                baths = st.number_input("Baths", min_value=1, max_value=20, value=1)
                area = st.number_input("Area (sqft)", min_value=100, max_value=100000, value=800)
            with col2:
                prop_type = st.selectbox("Type", options=df['Type'].unique())
                furnishing = st.selectbox("Furnishing", options=df['Furnishing'].unique())
                location = st.selectbox("Location", options=sorted(df['Location'].unique()))
            
            submit = st.form_submit_button("Predict Rent 🏠")
            
            if submit:
                input_data = pd.DataFrame([{
                    'Beds': beds, 'Baths': baths, 'Area_in_sqft': area,
                    'Age_of_listing_in_days': 30, 'Type': prop_type,
                    'Furnishing': furnishing, 'Location': location
                }])
                input_encoded = pd.get_dummies(input_data)
                input_reindexed = input_encoded.reindex(columns=model_columns, fill_value=0)
                input_scaled = scaler.transform(input_reindexed)
                prediction = model.predict(input_scaled)[0]
                st.success(f"### Estimated Rent: {prediction:,.2f} AED")
    else:
        st.error("⚠️ Model files (pkl) not found. Check your GitHub file names.")

with tab3:
    st.header("🤖 Data Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about 'average rent', 'cheapest location', or 'total listings'!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        p_low = prompt.lower()
        res = "I'm not sure. Try asking for 'average rent' or 'top locations'."
        
        if df is not None:
            if "average rent" in p_low:
                res = f"The average rent is **{df['Rent'].mean():,.2f} AED**."
            elif "total" in p_low:
                res = f"There are **{len(df):,}** properties listed."
            elif "location" in p_low:
                res = f"The most popular area is **{df['Location'].value_counts().index[0]}**."

        st.session_state.messages.append({"role": "assistant", "content": res})
        with st.chat_message("assistant"): st.markdown(res)
