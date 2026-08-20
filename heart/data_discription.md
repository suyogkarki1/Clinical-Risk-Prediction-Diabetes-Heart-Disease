# Heart Disease Dataset — Data Description

**Source:** [johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (Kaggle)
**Origin:** Combined UCI Heart Disease data (Cleveland, Hungary, Switzerland, Long Beach VA)
**Rows:** ~1,025
**Features:** 13 predictors + 1 target

## Feature Dictionary

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Patient's age in years |
| `sex` | Categorical (binary) | 1 = male, 0 = female |
| `cp` | Categorical | Chest pain type: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) on hospital admission |
| `chol` | Numeric | Serum cholesterol (mg/dl) |
| `fbs` | Categorical (binary) | Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false |
| `restecg` | Categorical | Resting ECG results: 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved during exercise stress test |
| `exang` | Categorical (binary) | Exercise-induced angina: 1 = yes, 0 = no |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment: 0 = upsloping, 1 = flat, 2 = downsloping |
| `ca` | Numeric (0–3) | Number of major vessels colored by fluoroscopy |
| `thal` | Categorical | Thalassemia test result: 1 = normal, 2 = fixed defect, 3 = reversible defect |
| `target` | Categorical (binary) | 1 = heart disease present, 0 = absent *(verify direction — some versions are flipped)* |

## Notes

- **Categorical features stored as integers:** `cp`, `restecg`, `slope`, `thal` should be treated as categorical (one-hot/ordinal encoding), not continuous.
- **Strongest predictors (typically):** `cp`, `thalach`, `exang`, `oldpeak`, `ca`, `thal` — these reflect direct signs of impaired blood flow during stress, and usually correlate most strongly with `target`.
- **Data quality quirks:** `ca` and `thal` sometimes contain undocumented values (e.g., `ca = 4`, `thal = 0`) due to encoding issues in the original UCI collection — check `value_counts()` before modeling.
- **Expected distribution shape:** unlike synthetic datasets, `chol` and `oldpeak` should show a real right skew, and `age`/`trestbps` should cluster around typical clinical ranges rather than being uniform.