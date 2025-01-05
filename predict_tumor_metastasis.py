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
    'Primary Site': {
        0: 'Ascending colon', 1: 'Cecum', 2: 'Colon, NOS', 3: 'Descending colon', 4: 'Hepatic flexure of colon',
        5: 'Overlapping lesion of colon', 6: 'Rectosigmoid junction', 7: 'Rectum, NOS', 8: 'Sigmoid colon',
        9: 'Splenic flexure of colon', 10: 'Transverse colon'
    },
    'Grade': {0: 'Unknown', 1: 'Grade I', 2: 'Grade II', 3: 'Grade III', 4: 'Grade IV'},
    'T Stage': {0: 'Unknown', 1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4'},
    'N Stage': {0: 'Unknown', 1: 'N0', 2: 'N1', 3: 'N2'},
    'Surgery': {0: 'No/Unknown', 1: 'Surgery performed'},
    'Chemotherapy': {0: 'No/Unknown', 1: 'Yes'},
    'CEA': {0: 'Unknown', 1: 'Negative', 2: 'Positive'},
    'Tumor Deposits': {0: 'Unknown', 1: 'No', 2: 'Yes'}
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


# Sidebar for user input
with st.sidebar:
    st.header("Input Feature Values")
    primary_site = st.selectbox("Primary Site", options=list(encoded_features['Primary Site'].values()))
    grade = st.selectbox("Grade", options=list(encoded_features['Grade'].values()))
    t_stage = st.selectbox("T Stage", options=list(encoded_features['T Stage'].values()))
    n_stage = st.selectbox("N Stage", options=list(encoded_features['N Stage'].values()))
    surgery = st.selectbox("Surgery", options=list(encoded_features['Surgery'].values()))
    chemotherapy = st.selectbox("Chemotherapy", options=list(encoded_features['Chemotherapy'].values()))
    cea = st.selectbox("CEA", options=list(encoded_features['CEA'].values()))
    tumor_deposits = st.selectbox("Tumor Deposits", options=list(encoded_features['Tumor Deposits'].values()))

# Convert user inputs from values back to keys for model input
features = {
    'Primary Site': {v: k for k, v in encoded_features['Primary Site'].items()}[primary_site],
    'Grade': {v: k for k, v in encoded_features['Grade'].items()}[grade],
    'T Stage': {v: k for k, v in encoded_features['T Stage'].items()}[t_stage],
    'N Stage': {v: k for k, v in encoded_features['N Stage'].items()}[n_stage],
    'Surgery': {v: k for k, v in encoded_features['Surgery'].items()}[surgery],
    'Chemotherapy': {v: k for k, v in encoded_features['Chemotherapy'].items()}[chemotherapy],
    'CEA': {v: k for k, v in encoded_features['CEA'].items()}[cea],
    'Tumor Deposits': {v: k for k, v in encoded_features['Tumor Deposits'].items()}[tumor_deposits]
}

input_df = pd.DataFrame([features])

if st.button('Predict'):
    prob = GradientBoosting.predict_proba(input_df)[:, 1]
    st.subheader("Probability of Distant Metastasis")
    st.write(f"The probability of distant metastasis is: {prob[0]:.2f}")

