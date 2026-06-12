import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "heart_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")  # Make sure you saved this

threshold_path = BASE_DIR / "threshold.txt"
default_threshold = 0.75
if threshold_path.exists():
    try:
        default_threshold = float(threshold_path.read_text().strip())
    except Exception:
        default_threshold = 0.75

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("❤️ About Project")

st.sidebar.info("""
This AI-powered Heart Disease Prediction System
uses Machine Learning to estimate the likelihood
of heart disease based on patient medical data.

Model Used:
- Random Forest Classifier

Features:
- Disease Prediction
- Risk Percentage
- Feature Importance Visualization
""")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=default_threshold,
    step=0.01
)
st.sidebar.caption(
    "Lower threshold increases sensitivity; higher threshold reduces false positives."
)

# -----------------------------
# Main Title
# -----------------------------
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
Enter the patient's medical details below to predict
the likelihood of heart disease.
""")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

cp = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
)

st.caption(
    "Typical Angina- Chest pain caused by physical activity or stress; relieved by rest. Atypical Angina-Chest pain that doesn't show the usual angina symptoms. Non-Anginal Pain-Chest pain not related to heart disease. Asymptomatic-No chest pain symptoms, but heart disease may still be present."
)

trestbps = st.number_input(
    "Blood Pressure (mmHg)",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)


st.caption(
    "Blood pressure is a key indicator of cardiovascular health and risk."
)

chol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=50,
    max_value=700,
    value=200,
    step=1
)

st.caption(
    "Cholesterol levels affect arterial plaque buildup and heart disease risk."
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    ["FBS ≤ 120 mg/dl", "FBS > 120 mg/dl"]
)

st.caption(
    "Fasting Blood Sugar indicates whether a patient's blood sugar level after fasting is greater than 120 mg/dl and is used as a risk factor in heart disease prediction. In this dataset, 0 means ≤ 120 mg/dl and 1 means > 120 mg/dl."
)

restecg = st.selectbox(
    "Rest ECG",
    [
        "Normal ECG",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ]
)

st.caption(
    "An ECG measures the electrical activity of the heart while the patient is at rest."
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)

st.caption(
    "Maximum heart rate achieved during exercise reflects heart fitness and cardiovascular response."
)

exang = st.selectbox(
    "Exercise Angina",
    ["Yes", "No"]
)

st.caption(
    "Exercise Angina indicates whether a patient experiences chest pain during exercise, which can be a sign of heart disease. In this dataset, 'Yes' corresponds to the higher-risk encoded value."
)

oldpeak = st.number_input(
    "Old Peak",
    format="%.1f"
)

st.caption(
    "Old Peak measures ST depression during exercise, indicating how the heart responds to stress. Higher values generally indicate a higher risk of heart-related problems."
)

slope = st.selectbox(
    "Slope",
    ["Downsloping", "Flat", "Upsloping"],
    index=2
)

st.caption(
    "Slope represents the ST segment shape during peak exercise ECG. while Flat and Downsloping means you have risk of heart disease"
)

ca = st.selectbox(
    "Number of Major Coronary Arteries",
    [0, 1, 2, 3, 4]
)

st.caption(
    "Number of major coronary arteries blocked shows how much blood flow may be restricted."
)

thal = st.selectbox(
    "Thalassemia Test Result",
    [
        "Unknown",
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)

st.caption(
    "Thalassemia refers to the result of a heart stress/blood flow test used to detect abnormalities."
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict"):

    sex_val = 1 if sex == "Male" else 0

    # Dataset encodings (confirmed against the heart.csv labels):
    #   cp: 0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic
    #   fbs: 0=FBS ≤ 120 mg/dl, 1=FBS > 120 mg/dl
    #   restecg: 0=Normal ECG, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy
    #   exang: 0=Yes, 1=No
    #   slope: 0=Downsloping, 1=Flat, 2=Upsloping
    #   thal: 0=Unknown, 1=Normal, 2=Fixed Defect, 3=Reversible Defect
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    cp_val = cp_map[cp]

    fbs_val = 0 if fbs == "FBS ≤ 120 mg/dl" else 1

    restecg_map = {
        "Normal ECG": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }

    restecg_val = restecg_map[restecg]

    # In this dataset, Exercise Angina is encoded so that 'Yes' maps to the higher-risk value.
    exang_val = 0 if exang == "Yes" else 1

    slope_map = {
        "Downsloping": 2,
        "Flat": 1,
        "Upsloping": 0
    }

    slope_val = slope_map[slope]

    thal_map = {
        "Unknown": 0,
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }

    thal_val = thal_map[thal]

    data = np.array([[
        age,
        sex_val,
        cp_val,
        trestbps,
        chol,
        fbs_val,
        restecg_val,
        thalach,
        exang_val,
        oldpeak,
        slope_val,
        ca,
        thal_val
    ]])

    # Scale Input Data
    data_scaled = scaler.transform(data)

    # Use predicted probability and a tuned threshold for sensitivity
    probability = model.predict_proba(data_scaled)
    risk_score = probability[0][1] * 100

    # Use the threshold selected from the sidebar
    THRESHOLD = threshold

    st.subheader("Prediction Result")

    if probability[0][1] >= THRESHOLD:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.metric(
        "Heart Disease Risk",
        f"{risk_score:.2f}%"
    )

# -----------------------------
# Feature Importance
# -----------------------------
st.markdown("---")

st.subheader("📊 Feature Importance")

feature_names = [
    "Age",
    "Sex",
    "Chest Pain",
    "Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Rest ECG",
    "Max Heart Rate",
    "Exercise Angina",
    "Old Peak",
    "Slope",
    "CA",
    "Thal"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.estimator.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "This system is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis."
)