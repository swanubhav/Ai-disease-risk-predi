import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

from src.ensemble import ensemble_risk
from src.cdss_engine import cdss

st.set_page_config(
    page_title="AI Clinical CDSS",
    layout="wide"
)

st.title("🏥 AI-Based Multi-Disease Clinical Decision Support System")

disease = st.selectbox(
    "Select Disease",
    ["heart", "diabetes", "stroke"]
)

# Load models
ml_model = joblib.load(f"models/{disease}_ml.pkl")
scaler = joblib.load("models/scaler.pkl")
dl_model = tf.keras.models.load_model("models/dl_multidisease.h5")

st.subheader("Enter Patient Clinical Data")

age = st.slider("Age", 1, 100, 45)
bp = st.slider("Blood Pressure", 80, 200, 130)
chol = st.slider("Cholesterol", 100, 300, 210)
bmi = st.slider("BMI", 10, 45, 25)
glucose = st.slider("Glucose Level", 60, 300, 110)
smoking = st.selectbox("Smoking", [0, 1])

if st.button("Predict Disease Risk"):
    input_data = np.array([[age, bp, chol, bmi, glucose, smoking]])
    input_scaled = scaler.transform(input_data)

    ml_prob = ml_model.predict_proba(input_scaled)[0][1]
    dl_prob = dl_model.predict(input_scaled)[0][0]

    final_risk = ensemble_risk(ml_prob, dl_prob)
    decision = cdss(final_risk)

    st.subheader("Prediction Result")
    st.metric("Risk Probability", final_risk)
    st.success(decision)
