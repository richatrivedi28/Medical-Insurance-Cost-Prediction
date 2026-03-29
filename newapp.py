import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# Load the trained model and preprocessor

medical_model = joblib.load('best_medicalCost_model.pkl')

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💊", layout="centered")
st.title('💊 Medical Insurance Cost Prediction')
st.markdown("Enter your details below to estimate insurance costs using different models.")

# take User inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=1, max_value=70, value=25)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)

with col2:
    sex = st.selectbox('sex', ['male', 'female'])
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])
 
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

input_data = np.array([[

    age,
    sex_val,
    bmi,
    children,
    smoker_val,
    region_northwest,
    region_southeast,
    region_southwest

]])

# Click button to predict
if st.button("Predict Insurance Costs"):
    
    #Make prediction
    prediction = medical_model.predict(input_data)[0]

    st.subheader("Prediction Cost")
    col1, col2 = st.columns(2)

    col1.metric("Estimated Cost", f"${prediction:,.2f}")
    col2.metric("Model", "XGBoost")
    st.success(f"✓ Predicted Healthcare Charges: ${prediction:,.2f}")
