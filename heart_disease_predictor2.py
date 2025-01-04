import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# Load the models
RF = joblib.load('RF.pkl')
XGBoost = joblib.load('XGBoost.pkl')
CatBoost = joblib.load('CatBoost.pkl')

# Model dictionary
models = {
    'Random Forest (RF)': RF,
    'XGBoost': XGBoost,
    'CatBoost': CatBoost
}

# Title
st.title("Heart Disease Prediction App")

# Description
st.write("""
This app predicts the likelihood of heart disease based on input features.
Select one or more models, input feature values, and get predictions and probability estimates.
""")

# Sidebar for model selection with multi-select option
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()), default=list(models.keys()))

# Input fields for the features
st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex (1 = male, 0 = female)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.sidebar.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox("Thal (thal)", [0, 1, 2, 3])

# Convert inputs to DataFrame for model prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Add a predict button
if st.sidebar.button("Predict"):
    # Display predictions and probabilities for selected models
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Display the prediction and probabilities for each selected model
        st.write(f"## Model: {model_name}")
        st.write(f"**Prediction**: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        st.write("**Prediction Probabilities**")
        st.write(f"Probability of No Heart Disease: {probabilities[0]:.4f}")
        st.write(f"Probability of Heart Disease: {probabilities[1]:.4f}")

# Display PNG images
st.subheader("1. Information of the Surveyed Medical Experts")
image1 = Image.open("Basic_Information.png")
st.image(image1, caption="Information of the surveyed medical experts", use_column_width=True)

st.subheader("2. Evaluation of the Website-based Tool by the Medical Experts")
image2 = Image.open("accuracy.png")
st.image(image2, caption="Evaluation of the website-based tool by the medical experts", use_column_width=True)