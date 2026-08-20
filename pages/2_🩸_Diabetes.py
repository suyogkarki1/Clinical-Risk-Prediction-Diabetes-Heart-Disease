import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩸", layout="centered")

# ---------- Load model + config ----------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes/diabetes_pipeline.joblib")
    with open("diabetes/diabetes_column_config.json") as f:
        config = json.load(f)
    return model, config

model, config = load_model()

st.title("🩸 Diabetes Risk Predictor (Women)")
st.markdown(
    "Enter health details below. Trained on the **Pima Indians Diabetes dataset** "
    "(female patients of Pima Indian heritage, age 21+)."
)
st.warning("⚠️ Educational/portfolio use only — not a medical diagnostic tool.")
st.divider()

with st.form("diabetes_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Plasma Glucose Concentration (mg/dL)", min_value=0, max_value=250, value=120)
        blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=28.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                               help="A score reflecting diabetes likelihood based on family history")
        age = st.number_input("Age", min_value=18, max_value=100, value=33)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
    }
    input_df = pd.DataFrame([input_dict])

    # Reorder/select columns to exactly match training feature order
    # NOTE: if your notebook has feature-engineered columns beyond these 8
    # (e.g. glucose_category, bmi_category), add them here before this line,
    # replicating exactly how they were built in your diabetes.ipynb
    missing_cols = [c for c in config['feature_cols'] if c not in input_df.columns]
    if missing_cols:
        st.error(f"App is missing engineered columns your model expects: {missing_cols}. "
                  f"These need to be added to the app to match training features exactly.")
        st.stop()

    input_df = input_df[config['feature_cols']]

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ **Prediction: Higher likelihood of Diabetes**")
    else:
        st.success("✅ **Prediction: Lower likelihood of Diabetes**")

    try:
        proba = model.predict_proba(input_df)[0]
        st.write(f"Confidence — No Diabetes: **{proba[0]*100:.1f}%**, Diabetes: **{proba[1]*100:.1f}%**")
    except AttributeError:
        st.info("Probability scores aren't available for this model configuration.")

    with st.expander("See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))