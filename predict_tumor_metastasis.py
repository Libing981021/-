import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

# Allow loading high resolution images
Image.MAX_IMAGE_PIXELS = None

# Load the models
GradientBoosting = joblib.load('best_gb_model.pk3')  # Replace with the correct model path

# Encoding dictionary for features
encoded_features = {
    'Primary Site': {0: 'Ascending colon', 1: 'Cecum', 2: 'Colon, NOS', 3: 'Descending colon', 4: 'Hepatic flexure of colon', 
                     5: 'Overlapping lesion of colon', 6: 'Rectosigmoid junction', 7: 'Rectum, NOS', 8: 'Sigmoid colon', 
                     9: 'Splenic flexure of colon', 10: 'Transverse colon'},
    'Grade': {0: 'Unknown', 1: 'Grade I', 2: 'Grade II', 3: 'Grade III', 4: 'Grade IV'},
    'T Stage': {0: 'Unknown', 1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4'},
    'N Stage': {0: 'Unknown', 1: 'N0', 2: 'N1', 3: 'N2'},
    'Surgery': {0: 'No/Unknown', 1: 'Surgery performed'},
    'Chemotherapy': {0: 'No/Unknown', 1: 'Yes'},
    'CEA': {0: 'Unknown', 1: 'Negative', 2: 'Positive'},
    'Tumor Deposits': {0: 'Unknown', 1: 'No', 2: 'Yes'},
}

# Title and description
st.title("Predicting Distant Metastasis in Colorectal Cancer Using a Machine Learning Model")
st.write("""
This app predicts the likelihood of distant metastasis in colorectal cancer based on input features:
- **Primary Site**: Where the tumor is located.
- **Grade**: The tumor grade (I to IV).
- **T Stage**: The size and extent of the primary tumor.
- **N Stage**: The extent of regional lymph node involvement.
- **Surgery**: Whether surgery has been performed.
- **Chemotherapy**: Whether chemotherapy has been administered.
- **CEA**: Carcinoembryonic antigen (a tumor marker).
- **Tumor Deposits**: Presence of tumor deposits outside the primary tumor site.

Input the relevant feature values to obtain predictions and probability estimates regarding the risk of distant metastasis.
""")

# User input for features
primary_site = st.selectbox("Primary Site", list(encoded_features['Primary Site'].values()))
grade = st.selectbox("Grade", list(encoded_features['Grade'].values()))
t_stage = st.selectbox("T Stage", list(encoded_features['T Stage'].values()))
n_stage = st.selectbox("N Stage", list(encoded_features['N Stage'].values()))
surgery = st.selectbox("Surgery", list(encoded_features['Surgery'].values()))
chemotherapy = st.selectbox("Chemotherapy", list(encoded_features['Chemotherapy'].values()))
cea = st.selectbox("CEA", list(encoded_features['CEA'].values()))
tumor_deposits = st.selectbox("Tumor Deposits", list(encoded_features['Tumor Deposits'].values()))

# Prepare input data for prediction
input_data = {
    'Primary Site': list(encoded_features['Primary Site'].keys())[list(encoded_features['Primary Site'].values()).index(primary_site)],
    'Grade': list(encoded_features['Grade'].keys())[list(encoded_features['Grade'].values()).index(grade)],
    'T Stage': list(encoded_features['T Stage'].keys())[list(encoded_features['T Stage'].values()).index(t_stage)],
    'N Stage': list(encoded_features['N Stage'].keys())[list(encoded_features['N Stage'].values()).index(n_stage)],
    'Surgery': list(encoded_features['Surgery'].keys())[list(encoded_features['Surgery'].values()).index(surgery)],
    'Chemotherapy': list(encoded_features['Chemotherapy'].keys())[list(encoded_features['Chemotherapy'].values()).index(chemotherapy)],
    'CEA': list(encoded_features['CEA'].keys())[list(encoded_features['CEA'].values()).index(cea)],
    'Tumor Deposits': list(encoded_features['Tumor Deposits'].keys())[list(encoded_features['Tumor Deposits'].values()).index(tumor_deposits)],
}

# Convert to a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Make prediction using the loaded model
prediction_proba = GradientBoosting.predict_proba(input_df)  # Get probabilities for both classes
predicted_class = GradientBoosting.predict(input_df)  # Get the predicted class

# Display results
if st.button("Predict"):
    st.write(f"Predicted Class: {predicted_class[0]}")
    st.write(f"Probability of No Metastasis: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Distant Metastasis: {prediction_proba[0][1]:.2f}")

