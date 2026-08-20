import streamlit as st

st.set_page_config(page_title="Health Risk Predictor", page_icon="🏥", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8A8A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
        max-width: 650px;
        margin: 0 auto;
    }
    .disclaimer-box {
        background-color: #FFF7ED;
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 2.5rem 0;
        font-size: 0.92rem;
        color: #92400E;
    }
    .tool-card {
        border-radius: 18px;
        padding: 2rem 1.8rem;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .tool-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.08);
    }
    .heart-card {
        background: linear-gradient(160deg, #FFF1F2 0%, #FFE4E6 100%);
    }
    .diabetes-card {
        background: linear-gradient(160deg, #EFF6FF 0%, #DBEAFE 100%);
    }
    .card-icon {
        font-size: 3.2rem;
        margin-bottom: 0.6rem;
        display: block;
    }
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    .card-desc {
        color: #4b5563;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1.4rem;
        min-height: 95px;
    }
    .tag-row {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .tag {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        color: #374151;
    }
    .footer-note {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1>🏥 Health Risk Prediction Suite</h1>
    <p>Machine learning tools that estimate risk for common health conditions,
    trained on clinical datasets. Choose a tool below to get started.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Educational use only.</strong> These tools are portfolio/learning projects and are
    <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical concerns.
</div>
""", unsafe_allow_html=True)

# ---------- Tool cards ----------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="tool-card heart-card">
        <span class="card-icon">❤️</span>
        <div class="card-title">Heart Disease Risk</div>
        <div class="card-desc">
            Predicts the likelihood of heart disease from clinical measurements —
            blood pressure, cholesterol, chest pain type, ECG results, and more.
        </div>
        <div class="tag-row">
            <span class="tag">🫀 Cardiology</span>
            <span class="tag">UCI Cleveland Dataset</span>
            <span class="tag">Voting Classifier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_❤️_Heart_Disease.py", label="Go to Heart Disease Predictor →", icon="❤️", use_container_width=True)

with col2:
    st.markdown("""
    <div class="tool-card diabetes-card">
        <span class="card-icon">🩸</span>
        <div class="card-title">Diabetes Risk (Women)</div>
        <div class="card-desc">
            Predicts the likelihood of diabetes in women using health indicators —
            glucose level, BMI, insulin, pregnancies, and family history.
        </div>
        <div class="tag-row">
            <span class="tag">🧬 Endocrinology</span>
            <span class="tag">Pima Indians Dataset</span>
            <span class="tag">ML Pipeline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_🩸_Diabetes.py", label="Go to Diabetes Predictor →", icon="🩸", use_container_width=True)

# ---------- Footer ----------
st.markdown("""
<div class="footer-note">
    Built with scikit-learn &amp; Streamlit · For educational and portfolio purposes only
</div>
""", unsafe_allow_html=True)