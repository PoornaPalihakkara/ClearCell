# ClearCell - Breast Cancer Classification

**Module:** IT2011 - Artificial Intelligence and Machine Learning  
**Year 2, Semester 1 (2026)** | Faculty of Computing, SLIIT

AI-assisted diagnostic support project that classifies breast tumour samples as **benign** or **malignant** using cytological features from the Breast Cancer Wisconsin (Original) dataset.

---

## Group Members

| Name                 | Preprocessing Role              | Model                    |
|----------------------|---------------------------------|--------------------------|
| Poorna Palihakkara (Me)|Dimensionality reduction (PCA) | Random Forest            |
| Thanthrige P.D.I.    | Missing data handling           | MLP (Keras)              |
| Rukmalgoda R.D.D.    | Categorical / target encoding   | Decision Tree            |
| Thashmina P.G.D.     | Outlier detection               | SVM                      |
| Dinali L.D.G.N.      | Feature selection               | KNN                      |
| Malka H.A.G.         | Normalization / scaling         | Logistic Regression      |

**Recommended model (group):** Random Forest

---

## Dataset

| Property | Detail |
|----------|--------|
| **Name** | Breast Cancer Wisconsin (Original) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original) |
| **Contributor** | Dr. William H. Wolberg, University of Wisconsin Hospitals |
| **Instances** | 699 |
| **Features** | 9 cytological measurements (scored 1-10) + sample ID + class |
| **Target** | `Class` - Benign (0) / Malignant (1) |
| **Class balance** | 458 benign (65.5%), 241 malignant (34.5%) |

**Predictive features:** Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Mitoses.

`Sample_code_number` is kept in the processed file for traceability and **dropped before model training**.

---

## Repository Structure

```text
ClearCell/
├── README.md
├── .gitignore
├── data/
│   ├── raw/                          # Assigned UCI dataset
├── notebooks/
│   ├── preprocessing/                # One notebook per member + integrated group preprocessing pipeline
│   └── models/                       # One model per member + comparison
├── results/
│   ├── eda_visualizations/           # EDA plots (PNG)
│   ├── logs/                         # Pipeline execution log
│   └── outputs/                      # processed_breast_cancer.csv
└── app/                              # Local Streamlit demo
    ├── app.py
    ├── requirements.txt
    └── processed_breast_cancer.csv
```

---

## Preprocessing & EDA

Each member implemented one technique; all were integrated in `group_pipeline.ipynb`.

| Technique | Decision |
|-----------|----------|
| Missing data | 16 missing values in `Bare_Nuclei` → median imputation |
| Encoding | Class labels `{2, 4}` → `{0, 1}` |
| Outliers | IQR flagged rows; **none removed** (mostly malignant clinical signal) |
| Scaling | `StandardScaler` on all 9 features (original columns retained) |
| Feature selection | All 9 features retained (positive correlation with target) |
| PCA | Exploratory only; PC1+PC2 ≈ 74.1% variance |

**Output:** `results/outputs/processed_breast_cancer.csv` (699 × 20)

---

## Models

Each member trained **at least four varieties** of their model (baseline, manual tuning, GridSearchCV / architecture variants, scaled vs original features) with stratified train/test split and appropriate metrics.

| Model | Member | Typical strengths |
|-------|--------|-------------------|
| Logistic Regression | Malka | Interpretable linear baseline |
| SVM | Thashmina | Strong margin-based accuracy |
| Decision Tree | Rukmalgoda | Fully interpretable paths |
| KNN | Dinali | Simple non-parametric baseline |
| **Random Forest** | **Palihakkara** | **Best overall balance (group choice)** |
| MLP (Keras) | Thanthrige | Non-linear deep-learning demo |

**Common evaluation metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
**Primary tuning metric:** F1-Score (mild class imbalance; missing malignant cases is costly)  
**Validation:** Stratified 80/20 split + Stratified 5-Fold CV inside GridSearchCV

---

## How to Run

### 1. Preprocessing

```bash
# Open and run any individual notebook under notebooks/preprocessing/
# Or run the integrated pipeline:
# group_pipeline.ipynb
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### 2. Models

```bash
# Open the relevant notebook under notebooks/models/
# Update the Google Drive path (Colab) or local path to:
# results/outputs/processed_breast_cancer.csv
# Run all cells top to bottom
```

### 3. Streamlit demo

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Opens a local UI to enter the 9 feature scores (1-10) and get a Benign / Malignant prediction using Random Forest.

---

## Key Results (summary)

- All six models achieve strong performance (typically **>94%** accuracy on the held-out test set).
- **Random Forest** was selected as the group’s recommended model for robustness, high F1 / Recall, feature importance, and no dependence on feature scaling.
- SVM and Logistic Regression remain strong alternatives (accuracy / interpretability).

Exact numbers are in the individual model notebooks and the group comparison notebook.

---

## Tech Stack

- Python 3  
- pandas, NumPy, Matplotlib, Seaborn  
- scikit-learn  
- TensorFlow / Keras (MLP only)  
- Jupyter / Google Colab  
- Streamlit (optional demo app)

---

## Disclaimer

This project is for **educational / decision-support demonstration** only.  
It is **not** a medical device and must not be used for clinical diagnosis.

---

IT2011 AIML | SLIIT Faculty of Computing | 2026
