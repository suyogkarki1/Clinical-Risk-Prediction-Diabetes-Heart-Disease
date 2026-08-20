import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------- Load model + config ----------
@st.cache_resource
def load_model():
    model = joblib.load("heart/heart_disease_model.joblib")
    with open("heart/column_config.json") as f:
        config = json.load(f)
    return model, config

model, config = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "Enter patient clinical details below. Model: **Hard Voting Classifier** "
    "(Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVC, Decision Tree), "
    "trained on the UCI Cleveland Heart Disease dataset (302 records)."
)
st.warning("⚠️ Educational/portfolio use only — not a medical diagnostic tool.")
st.divider()

with st.form("heart_form"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=54)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox(
            "Chest Pain Type",
            options=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)],
            format_func=lambda x: x[0]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=245)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox(
            "Resting ECG Result",
            options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)],
            format_func=lambda x: x[0]
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise-Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0]
        )
        thal = st.selectbox(
            "Thalassemia",
            options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
            format_func=lambda x: x[0]
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_dict = {
        'age': age, 'sex': sex[1], 'cp': cp[1], 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs[1], 'restecg': restecg[1], 'thalach': thalach, 'exang': exang[1],
        'oldpeak': oldpeak, 'slope': slope[1], 'thal': thal[1],
    }
    input_df = pd.DataFrame([input_dict])

    input_df['age_group'] = pd.cut(input_df['age'], bins=[0, 40, 50, 60, 70, 100],
                                    labels=['<40', '40-50', '50-60', '60-70', '70+'])
    input_df['chol_group'] = pd.cut(input_df['chol'], bins=[0, 200, 240, 600],
                                     labels=['normal', 'borderline', 'high'])
    input_df['bp_group'] = pd.cut(input_df['trestbps'], bins=[0, 120, 140, 250],
                                   labels=['normal', 'elevated', 'high'])
    input_df['high_risk_age'] = (input_df['age'] >= 55).astype(int)
    input_df['thalach_age_ratio'] = input_df['thalach'] / input_df['age']
    input_df['pressure_rate_product'] = input_df['trestbps'] * input_df['thalach']

    final_cols = config['num_col'] + config['cat_col'] + config['bin_col']
    input_df = input_df[final_cols]

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ **Prediction: Higher likelihood of Heart Disease**")
    else:
        st.success("✅ **Prediction: Lower likelihood of Heart Disease**")

    try:
        proba = model.predict_proba(input_df)[0]
        st.write(f"Confidence — No Disease: **{proba[0]*100:.1f}%**, Disease: **{proba[1]*100:.1f}%**")
    except AttributeError:
        st.info("Hard voting model — only class prediction is available, not confidence scores.")

    with st.expander("See input summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))