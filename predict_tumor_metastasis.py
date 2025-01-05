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
st.set_page_config(page_title="Distant Metastasis Prediction", page_icon=":guardsman:", layout="wide")
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

# Create layout with two columns (one for input and one for the result)
col1, col2 = st.columns([2, 1])

# Sidebar for user input
with st.sidebar:
    st.header("Input Feature Values")

    # Input fields for each feature
    primary_site = st.selectbox('Primary Site', list(encoded_features['Primary Site'].keys()))
    grade = st.selectbox('Grade', list(encoded_features['Grade'].keys()))
    t_stage = st.selectbox('T Stage', list(encoded_features['T Stage'].keys()))
    n_stage = st.selectbox('N Stage', list(encoded_features['N Stage'].keys()))
    surgery = st.selectbox('Surgery', list(encoded_features['Surgery'].keys()))
    chemotherapy = st.selectbox('Chemotherapy', list(encoded_features['Chemotherapy'].keys()))
    cea = st.selectbox('CEA', list(encoded_features['CEA'].keys()))
    tumor_deposits = st.selectbox('Tumor Deposits', list(encoded_features['Tumor Deposits'].keys()))

# Define input features as a DataFrame
input_data = pd.DataFrame([[primary_site, grade, t_stage, n_stage, surgery, chemotherapy, cea, tumor_deposits]],
                          columns=['Primary Site', 'Grade', 'T Stage', 'N Stage', 'Surgery', 'Chemotherapy', 'CEA', 'Tumor Deposits'])

# Mapping categorical inputs to numerical values
input_data['Primary Site'] = input_data['Primary Site'].map(encoded_features['Primary Site'])
input_data['Grade'] = input_data['Grade'].map(encoded_features['Grade'])
input_data['T Stage'] = input_data['T Stage'].map(encoded_features['T Stage'])
input_data['N Stage'] = input_data['N Stage'].map(encoded_features['N Stage'])
input_data['Surgery'] = input_data['Surgery'].map(encoded_features['Surgery'])
input_data['Chemotherapy'] = input_data['Chemotherapy'].map(encoded_features['Chemotherapy'])
input_data['CEA'] = input_data['CEA'].map(encoded_features['CEA'])
input_data['Tumor Deposits'] = input_data['Tumor Deposits'].map(encoded_features['Tumor Deposits'])

# Prediction button
with col1:
    st.subheader("Enter Feature Values and Predict")
    predict_button = st.button("Predict")

# Prediction logic
if predict_button:
    # Use the model to predict the probability of distant metastasis
    probability = GradientBoosting.predict_proba(input_data)[:, 1]  # Get probability for the positive class (metastasis)

    # Display the result in the second column
    with col2:
        st.subheader("Probability of Distant Metastasis:")
        st.write(f"The probability of distant metastasis is **{probability[0]:.4f}**")


