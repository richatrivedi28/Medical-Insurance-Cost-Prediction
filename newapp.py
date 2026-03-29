import streamlit as st

import pandas as pd

import numpy as np

import joblib
 
# Load the trained model and preprocessor

medical_model = joblib.load('best_medicalCost_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💊", layout="centered")
st.title('💊 Medical Insurance Cost Prediction')
st.markdown("Enter your details below to estimate insurance costs using different models.")

# User input widgets
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)

with col2:
    sex = st.selectbox('Sex', ['male', 'female'])
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])
 
if st.button("Predict Charges"):
    # Prepare input data as a DataFrame with original features
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Step 1: Apply one-hot encoding to categorical variables (matching training data)
    #
    input_encoded = pd.get_dummies(input_df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # Step 2: Manually add missing one-hot encoded columns that weren't created
    # (happens when user selects only one category value for a feature)
    expected_categorical_cols = ['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
    for col in expected_categorical_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0  # Add missing column with all zeros
    
    # Step 3: Scale only numerical features using the fitted scaler
    numerical_cols = ['age', 'bmi', 'children']
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    
    # Step 4: Ensure correct column order matching training (8 features total: 3 numerical + 5 categorical)
    expected_cols = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
    
    # Reorder to match expected columns
    input_encoded = input_encoded[expected_cols]
    
    # Step 5: Make prediction
    prediction = medical_model.predict(input_encoded)[0]
    
    st.success(f"✓ Predicted Healthcare Charges: ${prediction:,.2f}")
