import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# Just the filename!
df = pd.read_csv("dubai_properties.csv")
print(df.info())
print(df.head())
df.columns.tolist()# Display the column names
print("Shape of the DataFrame:", df.shape) # Display the shapes of dataframe
print("Data types of each column:\n", df.dtypes) # Display the data types of each column
print("Summary statistics of the DataFrame:\n", df.describe()) # Display summary statistics of the DataFrame
print("Missing values per column:\n", df.isnull().sum()) # Display the number of missing values in each column
print("\nNumber of duplicate rows:", df.duplicated().sum()) # Display the number of duplicate rows


import matplotlib.pyplot as plt
import seaborn as sns
# 3. Exploratory Data Analysis (EDA)
# Numerical columns for distribution and correlation
# We'll focus on Rent, Size, Room counts, and Listing Age
num_cols = ['Rent', 'Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft', 'Age_of_listing_in_days']
# Distributions of numerical features
df[num_cols].hist(bins=20, figsize=(15, 10), layout=(2, 3), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Property Numerical Features")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Rent and Property Features")
plt.show()
# 4. categorical Distributions
# Distribution of Property Types (Mapping from 'Travel Type')
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Type', order=df['Type'].value_counts().index, palette='viridis')
plt.title("Distribution of Property Types")
plt.xlabel("Count")
plt.show()
# Distribution of Furnishing Status (Mapping from 'Travel Class')
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Furnishing', order=df['Furnishing'].value_counts().index, palette='Set2')
plt.title("Distribution of Furnishing Status")
plt.xlabel("Furnishing Type")
plt.ylabel("Count")
plt.show()
# Top 10 Locations (Mapping from 'Aircraft Models')
plt.figure(figsize=(10, 6))
# Using 'Location' or 'City' depending on which gives more granular insight
sns.countplot(data=df, y='Location', order=df['Location'].value_counts().head(10).index, palette='magma')
plt.title("Top 10 Locations by Number of Listings")
plt.xlabel("Count")
plt.show()
# Extra: Rent Category Distribution (since it's a specific column in your data)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Rent_category', order=df['Rent_category'].value_counts().index, palette='husl')
plt.title("Market Segmentation by Rent Category")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
# 3. Exploratory Data Analysis (EDA)
# Numerical columns for distribution and correlation
# We'll focus on Rent, Size, Room counts, and Listing Age
num_cols = ['Rent', 'Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft', 'Age_of_listing_in_days']
# Distributions of numerical features
df[num_cols].hist(bins=20, figsize=(15, 10), layout=(2, 3), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Property Numerical Features")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Rent and Property Features")
plt.show()
# 4. categorical Distributions
# Distribution of Property Types (Mapping from 'Travel Type')
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Type', order=df['Type'].value_counts().index, palette='viridis')
plt.title("Distribution of Property Types")
plt.xlabel("Count")
plt.show()
# Distribution of Furnishing Status (Mapping from 'Travel Class')
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='Furnishing', order=df['Furnishing'].value_counts().index, palette='Set2')
plt.title("Distribution of Furnishing Status")
plt.xlabel("Furnishing Type")
plt.ylabel("Count")
plt.show()
# Top 10 Locations (Mapping from 'Aircraft Models')
plt.figure(figsize=(10, 6))
# Using 'Location' or 'City' depending on which gives more granular insight
sns.countplot(data=df, y='Location', order=df['Location'].value_counts().head(10).index, palette='magma')
plt.title("Top 10 Locations by Number of Listings")
plt.xlabel("Count")
plt.show()
# Extra: Rent Category Distribution (since it's a specific column in your data)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Rent_category', order=df['Rent_category'].value_counts().index, palette='husl')
plt.title("Market Segmentation by Rent Category")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 5. Data Cleaning
# ==========================================
print("\n--- Starting Data Cleaning ---")
print("Shape before cleaning:", df.shape)

# Remove any duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows with missing values (you could also use df.fillna() if you prefer imputing)
df.dropna(inplace=True)

# Drop features derived from the target 'Rent' to prevent data leakage
cols_to_drop = ['Rent_category', 'Rent_per_sqft'] 
df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
print("Shape after cleaning:", df_cleaned.shape)

# ==========================================
# 6. Feature Engineering (Categorical Encoding)
# ==========================================
# We have text columns that need to be converted to numbers for machine learning
categorical_cols = ['Type', 'Furnishing', 'Location'] 

# One-Hot Encoding creates True/False (1 or 0) columns for every category
# drop_first=True prevents the "dummy variable trap" (multicollinearity)
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# ==========================================
# 7. Splitting the Data (Train/Test Split)
# ==========================================
# Define our Features (X) and our Target (y)
# We add .select_dtypes(exclude=['object']) to automatically drop any remaining text columns!
X = df_encoded.drop(columns=['Rent']).select_dtypes(exclude=['object'])
y = df_encoded['Rent']

# Split 80% of data for training the model, and 20% for testing the model's performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 8. Feature Scaling
# ==========================================
# Scaling puts all numerical values on a similar scale so large numbers (Area_in_sqft) 
# don't completely overpower small numbers (Beds, Baths)
scaler = StandardScaler()

# Fit the scaler ONLY on the training data, then transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Features Shape:", X_train_scaled.shape)
print("Testing Features Shape:", X_test_scaled.shape)
print("Data Preparation Complete! The X_train_scaled and y_train are ready for modeling.")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 9. Model Training (Linear Regression Baseline)
# ==========================================
print("\n--- Training Linear Regression Model ---")
lr_model = LinearRegression()

# Let the model learn from X_train and y_train
lr_model.fit(X_train_scaled, y_train)

# Make predictions on the test set that the model hasn't seen yet!
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluate the model
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression - Mean Absolute Error (MAE): {lr_mae:,.2f} AED")
print(f"Linear Regression - R-squared (R²): {lr_r2:.4f}")

# ==========================================
# 10. Model Training (Random Forest Regressor)
# ==========================================
print("\n--- Training Random Forest Model (This might take a minute...) ---")
# n_jobs=-1 tells the computer to use all CPU cores to train faster
rf_model = RandomForestRegressor(
    n_estimators=100,     # Don't go over 100-150 trees
    max_depth=12,         # Limits how "tall" the file grows
    min_samples_leaf=10,  # Prevents the model from memorizing every single row 
    random_state=42
)
# Let the Random Forest learn
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)
# Evaluate
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest - Mean Absolute Error (MAE): {rf_mae:,.2f} AED")
print(f"Random Forest - R-squared (R²): {rf_r2:.4f}")

# ==========================================
# 11. Feature Importance Insight
# ==========================================
# The Random Forest can actually tell us which features were the most important when guessing rent!
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort from most important to least important
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


import joblib

# ==========================================
# 12. Save Models for Streamlit App
# ==========================================
print("\n--- Saving Models for Streamlit ---")
# Save the trained Random Forest model
joblib.dump(rf_model, 'D:/pers/dubai/rf_model.pkl')
# Save the Scaler (so user input is scaled correctly)
joblib.dump(scaler, 'D:/pers/dubai/scaler.pkl')
# Save the exact list of 373 columns the model expects
joblib.dump(list(X.columns), 'D:/pers/dubai/model_columns.pkl')

print("✅ rf_model.pkl, scaler.pkl, and model_columns.pkl saved successfully!")
import joblib



