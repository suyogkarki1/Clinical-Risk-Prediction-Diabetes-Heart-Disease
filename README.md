# Clinical Risk Prediction — Diabetes & Heart Disease

Two machine-learning models that estimate the likelihood of **diabetes** and **heart disease** from routine health measurements, each wrapped in an interactive Streamlit web app.

> ⚠️ **Educational / portfolio project only.** These models are trained on small public datasets and are **not** medical diagnostic tools. They cannot replace a blood test, an ECG, or a clinician.

---

## Overview

This project applies a consistent, end-to-end machine-learning workflow to two separate clinical classification tasks:

| Task | Dataset | Records | Features | Target |
|------|---------|---------|----------|--------|
| **Diabetes** | Pima Indians Diabetes | 768 | 8 | Diabetes vs No Diabetes |
| **Heart Disease** | UCI Cleveland | 302 | 13 | Disease vs No Disease |

Both follow the same pipeline: exploratory data analysis → preprocessing → model comparison → hyperparameter tuning → evaluation → deployment as a web app.

---

## Results

Metrics reported on the held-out test set.

**Diabetes — Logistic Regression (L1, C=0.1)**

| Metric | Score |
|--------|-------|
| Accuracy | 0.73 |
| Recall | 0.72 |
| F1 Score | 0.65 |
| ROC-AUC | 0.81 |
| PR-AUC | 0.81 |

**Heart Disease — Hard Voting Ensemble** (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVC, Decision Tree)

| Metric | Score |
|--------|-------|
| Accuracy | 0.79 |
| Precision | 0.83 |
| Recall | 0.75 |
| F1 Score | 0.79 |

\*ROC-AUC measured on the soft-voting variant, which exposes probabilities.

Because both are medical screening problems, **recall and PR-AUC** are emphasized over plain accuracy — catching true cases matters more than overall correctness.

---

## Approach

**EDA & Preprocessing**
- Distribution plots, box-plots against the target, and a correlation heatmap.
- Statistical testing (t-tests for numeric features, chi-square for categorical) run on the **training set only** to avoid leakage.
- A `ColumnTransformer` pipeline: median-impute + scale numeric features, one-hot encode categorical, most-frequent impute binary.
- Class imbalance handled with `class_weight='balanced'` across models.

**Modeling & Evaluation**
- Six models compared via 5-fold stratified cross-validation.
- Hyperparameters tuned with `GridSearchCV`, scored on PR-AUC.
- Final model selected per task; evaluated with accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix.

---

## Project Structure

```
Clinical-Risk-Prediction/
├── app.py                          # Streamlit entry point
├── pages/                          # additional app pages
├── dataset/                        # source CSVs
├── diabetes/
│   ├── diabetes_pipeline.joblib    # saved model + preprocessing
│   └── diabetes_column_config.json # feature names + order
├── heart/
│   ├── heart_disease_model.joblib
│   └── column_config.json
├── requirements.txt
└── README.md
```

The full preprocessing + model pipeline is saved together, so the apps only need to **load and predict** — they never re-implement scaling or encoding. A JSON config locks feature names and order so inputs are fed exactly as they were during training.

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/suyogkarki1/Clinical-Risk-Prediction-Diabetes-Heart-Disease.git
cd Clinical-Risk-Prediction-Diabetes-Heart-Disease

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Tech Stack

- **Python** · pandas, NumPy
- **scikit-learn** — pipelines, models, evaluation
- **Streamlit** — interactive web apps
- **matplotlib**, **seaborn** — visualization
- **joblib** — model persistence

---

## Datasets

- **Pima Indians Diabetes** — female patients of Pima Indian heritage, aged 21+. Features: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age.
- **UCI Cleveland Heart Disease** — cardiac patient records. Features: age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, ST slope, major vessels, thalassemia.

---

## Limitations & Further Work

**Limitations**
- Small datasets (302 and 768 rows) limit how well results generalize.
- No patient identifiers, so subject-level validation isn't possible.
- Modest test scores (0.73–0.79 accuracy) — realistic, not clinical-grade.
- Trained on specific populations; may not transfer to others.

**Next steps**
- Validate on larger, more diverse clinical datasets.
- Add probability-threshold tuning to reduce false negatives.
- Deploy both apps publicly on Streamlit Community Cloud.
- Add model explainability (SHAP) for per-prediction reasoning.

---

## License

Released for educational purposes. Datasets belong to their respective sources (UCI Machine Learning Repository).
