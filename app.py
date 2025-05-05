import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
model_data = joblib.load("stroke_prediction_pipeline.pkl")
model = model_data["model"]
threshold = model_data["threshold"]
feature_names = model_data["feature_names"]

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Prediction App")
st.write(
    """
This app predicts the **risk of stroke**.
Fill in the patient's health data and click Predict.
"""
)

# Input form
data = {}

data["gender"] = st.selectbox("Gender", ["Male", "Female"], key="gender_key")
data["gender"] = 1 if data["gender"] == "Male" else 0

data["age"] = st.slider("Age", 0, 100, 45, key="age_key")

data["hypertension"] = st.selectbox("Hypertension", ["Yes", "No"], key="htn_key")
data["hypertension"] = 1 if data["hypertension"] == "Yes" else 0

data["heart_disease"] = st.selectbox("Heart Disease", ["Yes", "No"], key="hd_key")
data["heart_disease"] = 1 if data["heart_disease"] == "Yes" else 0

data["ever_married"] = st.selectbox("Ever Married", ["Yes", "No"], key="married_key")
data["ever_married"] = 1 if data["ever_married"] == "Yes" else 0

data["Residence_type"] = st.selectbox(
    "Residence Type", ["Urban", "Rural"], key="residence_key"
)
data["Residence_type"] = 1 if data["Residence_type"] == "Urban" else 0

data["avg_glucose_level"] = st.slider(
    "Avg Glucose Level", 50.0, 250.0, 100.0, key="glucose_key"
)
data["bmi"] = st.slider("BMI", 10.0, 60.0, 25.0, key="bmi_key")

work_type_options = ["Private", "Self-employed", "children", "Govt_job", "Never_worked"]
selected_work_type = st.selectbox(
    "Work Type", work_type_options, key="work_type_select"
)
for w in work_type_options:
    data[f"work_type_{w}"] = 1 if selected_work_type == w else 0

smoke_options = ["formerly smoked", "never smoked", "smokes", "Unknown"]
selected_smoke = st.selectbox("Smoking Status", smoke_options, key="smoke_select")
for s in smoke_options:
    data[f"smoking_status_{s}"] = 1 if selected_smoke == s else 0

input_df = pd.DataFrame([data])

for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# Prediction
if st.button("🔍 Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= threshold)

    st.write(f"**Predicted Stroke Risk Probability:** `{prob:.2%}`")
    if pred == 1:
        st.error("⚠️ High Stroke Risk Detected!")
    else:
        st.success("✅ Low Stroke Risk")
