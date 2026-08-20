import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Pulse · Heart Disease Risk", page_icon="🫀", layout="wide")

# ============================================================
#  STYLES — "Cardio Monitor" theme
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0e1526; --panel:#151f36; --panel-2:#1b2740; --line:#26324f;
  --ink:#eaf0fb; --muted:#8ea0c4; --crimson:#ff4d6d; --crimson-dim:#c33652; --ecg:#3ddc97;
}
.stApp{ background:radial-gradient(1200px 500px at 80% -10%, rgba(255,77,109,.10), transparent 60%), var(--bg); }
html, body, [class*="css"]{ font-family:'Inter',sans-serif; color:var(--ink); }
.block-container{ padding-top:2.2rem; padding-bottom:3rem; max-width:1080px; }
#MainMenu, footer, header{ visibility:hidden; }
.monitor{ background:linear-gradient(180deg,var(--panel) 0%,#111a30 100%); border:1px solid var(--line);
  border-radius:20px; padding:2.2rem 2.4rem; position:relative; overflow:hidden; }
.monitor .eyebrow{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--crimson); margin-bottom:.7rem; font-weight:600; }
.monitor h1{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:2.4rem; line-height:1.1;
  margin:0 0 .55rem 0; color:var(--ink); letter-spacing:-.01em; }
.monitor p{ color:var(--muted); font-size:1rem; line-height:1.55; max-width:62ch; margin:0; }
.ecg-wrap{ margin-top:1.4rem; height:46px; border-top:1px solid var(--line); padding-top:.8rem; }
.ecg-line{ width:100%; height:34px; display:block; }
.ecg-line path{ fill:none; stroke:var(--ecg); stroke-width:2; stroke-dasharray:1400; stroke-dashoffset:1400;
  animation:trace 3.2s linear infinite; filter:drop-shadow(0 0 6px rgba(61,220,151,.6)); }
@keyframes trace{ to{ stroke-dashoffset:0; } }
.chips{ margin-top:1.2rem; display:flex; gap:.55rem; flex-wrap:wrap; }
.chip{ font-family:'JetBrains Mono',monospace; font-size:.74rem; font-weight:600; background:var(--panel-2);
  border:1px solid var(--line); padding:.32rem .8rem; border-radius:8px; color:var(--muted); }
.notice{ margin-top:1rem; background:rgba(255,77,109,.08); border:1px solid rgba(255,77,109,.25);
  border-left:4px solid var(--crimson); border-radius:12px; padding:.85rem 1.1rem; font-size:.9rem; color:#ffc2ce; }
.sec{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.2rem; color:var(--ink);
  margin:1.9rem 0 .1rem 0; display:flex; align-items:center; gap:.55rem; }
.sec .num{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:.72rem; color:var(--crimson);
  background:rgba(255,77,109,.12); border:1px solid rgba(255,77,109,.25); border-radius:6px; padding:.18rem .5rem; }
.sec-help{ color:var(--muted); font-size:.9rem; margin:.15rem 0 .3rem 0; }
.stForm{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.4rem 1.6rem .5rem 1.6rem; }
label, .stNumberInput label, .stSelectbox label{ font-weight:600 !important; font-size:.85rem !important; color:var(--ink) !important; }
.stNumberInput input, .stSelectbox div[data-baseweb="select"]>div{ background:var(--panel-2) !important;
  border:1px solid var(--line) !important; border-radius:10px !important; color:var(--ink) !important; }
.stFormSubmitButton>button{ background:linear-gradient(135deg,var(--crimson) 0%,var(--crimson-dim) 100%);
  color:#fff; border:none; border-radius:12px; font-weight:600; font-size:1rem; padding:.75rem 1.2rem;
  box-shadow:0 12px 30px -12px rgba(255,77,109,.6); transition:transform .05s ease; }
.stFormSubmitButton>button:hover{ transform:translateY(-1px); filter:brightness(1.05); }
.result{ border-radius:18px; padding:1.6rem 1.8rem; margin-top:.3rem; background:var(--panel); border:1px solid var(--line); }
.result.higher{ border-top:6px solid var(--crimson); box-shadow:0 0 40px -18px rgba(255,77,109,.7); }
.result.lower{ border-top:6px solid var(--ecg); box-shadow:0 0 40px -18px rgba(61,220,151,.5); }
.result .verdict{ font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1.6rem; margin:0 0 .3rem 0; }
.result.higher .verdict{ color:var(--crimson); }
.result.lower .verdict{ color:var(--ecg); }
.result .sub{ color:var(--muted); font-size:.95rem; margin:0; }
.meter{ margin-top:1.2rem; }
.meter .row{ display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
  font-size:.8rem; font-weight:600; margin-bottom:.35rem; color:var(--muted); }
.meter .track{ height:14px; background:var(--panel-2); border-radius:999px; overflow:hidden; border:1px solid var(--line); }
.meter .fill{ height:100%; border-radius:999px; }
.tiles{ display:flex; gap:.7rem; flex-wrap:wrap; margin-top:1.2rem; }
.tile{ flex:1 1 120px; background:var(--panel-2); border:1px solid var(--line); border-radius:12px; padding:.7rem .9rem; }
.tile .k{ font-family:'JetBrains Mono',monospace; font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.tile .v{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:600; color:var(--ink); }
.foot{ text-align:center; color:var(--muted); font-size:.8rem; margin-top:2.4rem; font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("heart/heart_disease_model.joblib")
    with open("heart/column_config.json") as f:
        config = json.load(f)
    return model, config

model, config = load_model()

# ============================================================
#  MONITOR HEADER
# ============================================================
st.markdown("""
<div class="monitor">
  <div class="eyebrow">Cardiac Risk Screening · Random Forest</div>
  <h1>Read the signs before the heart does.</h1>
  <p>Enter clinical measurements from a standard cardiac work-up. A tuned Random Forest
  model estimates the likelihood of heart disease, trained on the UCI Cleveland dataset.</p>
  <div class="ecg-wrap">
    <svg class="ecg-line" viewBox="0 0 1400 34" preserveAspectRatio="none">
      <path d="M0,17 L260,17 L285,17 L300,6 L315,28 L330,17 L560,17 L585,17 L600,3 L618,31 L636,17
               L900,17 L925,17 L940,8 L955,26 L970,17 L1240,17 L1265,17 L1280,6 L1298,28 L1316,17 L1400,17"/>
    </svg>
  </div>
  <div class="chips">
    <span class="chip">12 CLINICAL INPUTS</span>
    <span class="chip">RANDOM FOREST</span>
    <span class="chip">INSTANT READOUT</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
  <strong>Not a diagnosis.</strong> Educational tool trained on a small historical dataset (302 records).
  It cannot replace an ECG, blood work, or a cardiologist. If you have symptoms, seek care.
</div>
""", unsafe_allow_html=True)

# ============================================================
#  INPUT FORM
# ============================================================
st.markdown('<div class="sec"><span class="num">01</span>Patient information</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-help">Demographics, chest-pain presentation, and stress-test results.</div>', unsafe_allow_html=True)

with st.form("heart_form"):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        age = st.number_input("Age", 18, 100, 54)
        sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest pain type",
            [("Typical Angina", 0), ("Atypical Angina", 1), ("Non-anginal Pain", 2), ("Asymptomatic", 3)],
            format_func=lambda x: x[0],
            help="Asymptomatic chest pain can paradoxically carry higher risk.")
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Serum cholesterol (mg/dl)", 100, 600, 245)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("Resting ECG result",
            [("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)],
            format_func=lambda x: x[0])
    with col2:
        thalach = st.number_input("Max heart rate achieved", 60, 220, 150,
            help="Peak heart rate during exercise testing.")
        exang = st.selectbox("Exercise-induced angina?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1,
            help="ST-segment depression induced by exercise vs rest.")
        slope = st.selectbox("Slope of peak exercise ST segment",
            [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], format_func=lambda x: x[0])
        thal = st.selectbox("Thalassemia",
            [("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)], format_func=lambda x: x[0])

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run risk assessment", use_container_width=True)

# ============================================================
#  PREDICTION + RESULT
# ============================================================
if submitted:
    input_dict = {
        'age': age, 'sex': sex[1], 'cp': cp[1], 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs[1], 'restecg': restecg[1], 'thalach': thalach, 'exang': exang[1],
        'oldpeak': oldpeak, 'slope': slope[1], 'thal': thal[1],
    }
    input_df = pd.DataFrame([input_dict])

    # ---- feature engineering (mirrors the training notebook exactly) ----
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

    st.markdown('<div class="sec"><span class="num">02</span>Assessment</div>', unsafe_allow_html=True)

    proba = None
    try:
        proba = model.predict_proba(input_df)[0]
    except AttributeError:
        pass

    if prediction == 1:
        state_class, verdict = "higher", "Higher likelihood of heart disease"
        sub = ("These measurements resemble patterns the model links to heart disease. "
               "Treat this as a prompt for clinical follow-up — not a diagnosis.")
    else:
        state_class, verdict = "lower", "Lower likelihood of heart disease"
        sub = ("These measurements resemble patterns the model links to no heart disease. "
               "Regular check-ups remain the best safeguard.")

    if proba is not None:
        pct = proba[1] * 100
        bar_color = "var(--crimson)" if prediction == 1 else "var(--ecg)"
        meter_html = f"""
          <div class="meter">
            <div class="row"><span>ESTIMATED RISK</span><span>{pct:.0f}%</span></div>
            <div class="track"><div class="fill" style="width:{pct:.0f}%;background:{bar_color};"></div></div>
          </div>"""
    else:
        meter_html = ""

    st.markdown(f"""
    <div class="result {state_class}">
      <p class="verdict">{verdict}</p>
      <p class="sub">{sub}</p>
      {meter_html}
      <div class="tiles">
        <div class="tile"><div class="k">Age</div><div class="v">{age}</div></div>
        <div class="tile"><div class="k">Resting BP</div><div class="v">{trestbps}</div></div>
        <div class="tile"><div class="k">Cholesterol</div><div class="v">{chol}</div></div>
        <div class="tile"><div class="k">Max HR</div><div class="v">{thalach}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See everything you entered"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

st.markdown("""
<div class="foot">
  UCI Cleveland Heart Disease dataset · Education & portfolio use only · Consult a cardiologist
</div>
""", unsafe_allow_html=True)