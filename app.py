import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Dubai Real Estate App", page_icon="🏙️", layout="wide")
import joblib
import os

# This looks for the files in the same folder as app.py
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Load artifacts
@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        model_columns = joblib.load(COLS_PATH)
        return model, scaler, model_columns
    return None, None, None

model, scaler, model_columns = load_models()

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Basic cleaning as per script
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        return df
    return None

df = load_data()

st.title("🏙️ Dubai Real Estate Hub")
st.markdown("Welcome to the Dubai Real Estate Dashboard, Predictor, and Assistant.")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Rent Predictor", "🤖 Data Assistant"])

with tab1:
    st.header("📊 Market Dashboard")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Listings", f"{len(df):,}")
        col2.metric("Average Rent (AED)", f"{df['Rent'].mean():,.2f}")
        col3.metric("Avg Area (sqft)", f"{df['Area_in_sqft'].mean():,.2f}")
        col4.metric("Locations", f"{df['Location'].nunique()}")

        st.subheader("Distribution of Property Numerical Features")
        num_cols = ['Rent', 'Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft', 'Age_of_listing_in_days']
        
        # Configure styling for dark background
        plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.edgecolor': 'white'})

        # We will plot using matplotlib
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor('none')
        available_num_cols = [c for c in num_cols if c in df.columns]
        for i, col in enumerate(available_num_cols):
            ax = axes[i//3, i%3]
            ax.set_facecolor('none')
            df[col].hist(bins=20, ax=ax, color='skyblue', edgecolor='white')
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Top Locations by Listings")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            fig2.patch.set_facecolor('none')
            ax2.set_facecolor('none')
            sns.countplot(data=df, y='Location', order=df['Location'].value_counts().head(10).index, palette='magma', ax=ax2)
            st.pyplot(fig2)
        
        with colB:
            st.subheader("Property Types")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            fig3.patch.set_facecolor('none')
            ax3.set_facecolor('none')
            sns.countplot(data=df, y='Type', order=df['Type'].value_counts().index, palette='viridis', ax=ax3)
            st.pyplot(fig3)
    else:
        st.warning("Data file not found. Please ensure 'dubai_properties.csv' is in the 'D:/pers/dubai/' folder.")

with tab2:
    st.header("🔮 Rent Predictor")
    st.write("Enter the property details below to get an estimated rent price.")
    
    if model and scaler and model_columns and df is not None:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                beds = st.number_input("Number of Beds", min_value=0, max_value=20, value=1)
                baths = st.number_input("Number of Baths", min_value=1, max_value=20, value=1)
                area = st.number_input("Area in sqft", min_value=100, max_value=100000, value=800)
                age = st.number_input("Age of listing (days)", min_value=0, max_value=3600, value=30)
            
            with col2:
                prop_type = st.selectbox("Property Type", options=df['Type'].unique())
                furnishing = st.selectbox("Furnishing Status", options=df['Furnishing'].unique())
                location = st.selectbox("Location", options=sorted(df['Location'].unique()))
            
            submit = st.form_submit_button("Predict Rent 🏠")
            
            if submit:
                # Create input dataframe
                input_data = {
                    'Beds': [beds],
                    'Baths': [baths],
                    'Area_in_sqft': [area],
                    'Age_of_listing_in_days': [age],
                    'Type': [prop_type],
                    'Furnishing': [furnishing],
                    'Location': [location]
                }
                
                input_df = pd.DataFrame(input_data)
                
                # We need to one-hot encode matching the training
                categorical_cols = ['Type', 'Furnishing', 'Location']
                input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
                
                # Reindex to match the columns the model was trained on
                # It will fill missing columns (e.g. locations not selected) with 0
                input_reindexed = input_encoded.reindex(columns=model_columns, fill_value=0)
                
                # Scale
                input_scaled = scaler.transform(input_reindexed)
                
                # Predict
                prediction = model.predict(input_scaled)[0]
                
                st.success(f"### Estimated Rent: {prediction:,.2f} AED")
    else:
        st.warning("Models not found. Please ensure the models are saved in the 'D:/pers/dubai/' folder.")

with tab3:
    st.header("🤖 Data Assistant")
    st.write("Ask simple questions about the data! (e.g. 'What is the average rent?', 'How many listings in Dubai Marina?', 'Highest rent?')")
    
    # Simple rule-based chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I can answer basic statistical questions about the dataset. Try asking 'average rent', 'total listings', or 'highest rent'."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Simple logic for responses
        prompt_lower = prompt.lower()
        response = "I'm sorry, I don't understand that question yet. Try asking about 'average rent', 'highest rent', 'total listings', or 'most popular location'."
        
        if df is not None:
            if "average rent" in prompt_lower:
                avg_r = df['Rent'].mean()
                response = f"The average rent across all properties is **{avg_r:,.2f} AED**."
            elif "highest rent" in prompt_lower or "max rent" in prompt_lower:
                max_r = df['Rent'].max()
                response = f"The highest rent in the dataset is **{max_r:,.2f} AED**."
            elif "lowest list" in prompt_lower or "cheapest" in prompt_lower:
                min_r = df['Rent'].min()
                response = f"The lowest rent in the dataset is **{min_r:,.2f} AED**."
            elif "total" in prompt_lower and "listing" in prompt_lower:
                response = f"There are a total of **{len(df):,}** listings in the dataset."
            elif "popular location" in prompt_lower or "top location" in prompt_lower:
                top_loc = df['Location'].value_counts().index[0]
                response = f"The most popular location by number of listings is **{top_loc}**."
            elif "average area" in prompt_lower or "average size" in prompt_lower:
                avg_a = df['Area_in_sqft'].mean()
                response = f"The average area of the properties is **{avg_a:,.2f} sqft**."
            elif any(loc.lower() in prompt_lower for loc in df['Location'].unique()):
                # Identify which location they asked about
                for loc in df['Location'].unique():
                    if loc.lower() in prompt_lower:
                        loc_df = df[df['Location'] == loc]
                        avg_loc_rent = loc_df['Rent'].mean()
                        response = f"There are **{len(loc_df)}** listings in **{loc}**. The average rent there is **{avg_loc_rent:,.2f} AED**."
                        break
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# Run instructions for the user (Will not be displayed in UI, just for standard streamlit workflow)
