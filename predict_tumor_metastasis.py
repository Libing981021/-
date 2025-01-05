import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

# Allow loading high resolution images
Image.MAX_IMAGE_PIXELS = None

# Load the model
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

# Streamlit UI
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

# Adjusting layout to have symmetrical columns
col1, col2 = st.columns([1, 1])

with col1:
    # Sidebar for user input on the left side
    st.sidebar.header("Input Features")

    # Get user inputs
    primary_site = st.sidebar.selectbox("Primary Site", list(encoded_features['Primary Site'].keys()))
    grade = st.sidebar.selectbox("Grade", list(encoded_features['Grade'].keys()))
    t_stage = st.sidebar.selectbox("T Stage", list(encoded_features['T Stage'].keys()))
    n_stage = st.sidebar.selectbox("N Stage", list(encoded_features['N Stage'].keys()))
    surgery = st.sidebar.selectbox("Surgery", list(encoded_features['Surgery'].keys()))
    chemotherapy = st.sidebar.selectbox("Chemotherapy", list(encoded_features['Chemotherapy'].keys()))
    cea = st.sidebar.selectbox("CEA", list(encoded_features['CEA'].keys()))
    tumor_deposits = st.sidebar.selectbox("Tumor Deposits", list(encoded_features['Tumor Deposits'].keys()))

with col2:
    # Title and button
    st.subheader("Input Data to Predict")
    st.write("""
    Enter the relevant features in the sidebar to predict the likelihood of distant metastasis.
    Click the "Predict" button to get the result.
    """)

# When user clicks on "Predict", the model will process the inputs
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = np.array([[primary_site, grade, t_stage, n_stage, surgery, chemotherapy, cea, tumor_deposits]])
    
    # Ensure correct shape and encode the input data according to the encoding dictionary
    input_data_encoded = []
    for feature_name, feature_values in encoded_features.items():
        feature_value = locals()[feature_name.lower()]
        input_data_encoded.append(feature_values[feature_value])
    
    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

    # Make prediction using the loaded model
    probability = GradientBoosting.predict_proba(input_data_encoded)[0][1]  # Probability of distant metastasis (class 1)

    # Show the probability
    st.write(f"### Probability of Distant Metastasis: {probability:.4f}")


