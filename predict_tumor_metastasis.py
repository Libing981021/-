import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

# Allow loading high resolution images
Image.MAX_IMAGE_PIXELS = None

# Load the models
GradientBoosting = joblib.load('best_gb_model.pk3')  # Replace with correct model path

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

# Model dictionary
models = {
    'GradientBoosting': GradientBoosting,
}

# Title and description
st.title("Colorectal Cancer Distant Metastasis Prediction App")
st.write("""
This app predicts the likelihood of distant metastasis in colorectal cancer based on input features.
Choose one model, input the relevant feature values, and obtain predictions and probability estimates regarding the risk of distant metastasis.
""")

# Sidebar for model selection
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()), default=list(models.keys()))

# Input fields for the features
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex (1 = male, 0 = female)", [0, 1])
primary_site = st.sidebar.selectbox("Primary Site", list(encoded_features['Primary Site'].keys()))
grade = st.sidebar.selectbox("Tumor Grade", list(encoded_features['Grade'].keys()))
t_stage = st.sidebar.selectbox("T Stage", list(encoded_features['T Stage'].keys()))
n_stage = st.sidebar.selectbox("N Stage", list(encoded_features['N Stage'].keys()))
surgery = st.sidebar.selectbox("Surgery Performed", list(encoded_features['Surgery'].keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", list(encoded_features['Chemotherapy'].keys()))
cea = st.sidebar.selectbox("CEA (Carcinoembryonic Antigen)", list(encoded_features['CEA'].keys()))
tumor_deposits = st.sidebar.selectbox("Tumor Deposits", list(encoded_features['Tumor Deposits'].keys()))

# Function to encode feature values
def encode_features(data, encoding_dict):
    for feature, encoding in encoding_dict.items():
        if feature in data:
            data[feature] = data[feature].map(encoding)
    return data

# Create a DataFrame from the input features
data = {
    'Primary Site': primary_site,
    'Grade': grade,
    'T Stage': t_stage,
    'N Stage': n_stage,
    'Surgery': surgery,
    'Chemotherapy': chemotherapy,
    'CEA': cea,
    'Tumor Deposits': tumor_deposits
}

# Convert input data to a DataFrame
input_data = pd.DataFrame([data])

# Encode the feature values
encoded_data = encode_features(input_data, encoded_features)

# Add a prediction button
if st.sidebar.button("Predict"):
    # Display predictions and probabilities for selected models
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(encoded_data)[0]
        probabilities = model.predict_proba(encoded_data)[0]

        # Display the prediction and probabilities for each selected model
        st.write(f"## Model: {model_name}")
        st.write(f"**Prediction**: {'Distant metastasis' if prediction == 1 else 'No Distant metastasis'}")
        st.write("**Prediction Probabilities**")
        st.write(f"Probability of No Distant metastasis: {probabilities[0]:.4f}")
        st.write(f"Probability of Distant metastasis: {probabilities[1]:.4f}")

# Display images related to the research
st.subheader("1. Information of the Surveyed Medical Experts")
image1 = Image.open("Basic_Information.png")  # Replace with actual path to your image
st.image(image1, caption="Information of the surveyed medical experts", use_column_width=True)

st.subheader("2. Evaluation of the Website-based Tool by the Medical Experts")
image2 = Image.open("accuracy.png")  # Replace with actual path to your image
st.image(image2, caption="Evaluation of the website-based tool by the medical experts", use_column_width=True)
