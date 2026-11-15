"""
ClearCell – Local Breast Tumour Classification Interface
Group_26-MTR-IT2011-04 | IT2011 AI & ML

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClearCell – Breast Tumour Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "Clump_Thickness",
    "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape",
    "Marginal_Adhesion",
    "Single_Epithelial_Cell_Size",
    "Bare_Nuclei",
    "Bland_Chromatin",
    "Normal_Nucleoli",
    "Mitoses",
]

FEATURE_HELP = {
    "Clump_Thickness": "Thickness of cell clumps (1 = thin … 10 = thick)",
    "Uniformity_of_Cell_Size": "How uniform the cell sizes are (1 = uniform … 10 = highly variable)",
    "Uniformity_of_Cell_Shape": "How uniform the cell shapes are (1 = uniform … 10 = highly irregular)",
    "Marginal_Adhesion": "How well cells stick together at the edges (1 = high adhesion … 10 = low)",
    "Single_Epithelial_Cell_Size": "Size of individual epithelial cells (1 = small … 10 = large)",
    "Bare_Nuclei": "Nuclei without surrounding cytoplasm (1 = none … 10 = many)",
    "Bland_Chromatin": "Texture of the chromatin (1 = fine … 10 = coarse)",
    "Normal_Nucleoli": "Visibility / size of nucleoli (1 = small/invisible … 10 = prominent)",
    "Mitoses": "Rate of cell division (1 = rare … 10 = frequent)",
}

DATA_PATH = Path(__file__).parent / "processed_breast_cancer.csv"
MODEL_PATH = Path(__file__).parent / "random_forest_model.joblib"

# ---------------------------------------------------------------------------
# Train / load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load saved model or train Random Forest on the processed dataset."""
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["metrics"]

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }

    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return model, metrics


model, metrics = get_model()

# ---------------------------------------------------------------------------
# Sidebar – about & model info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 ClearCell")
    st.markdown("**Group_26-MTR-IT2011-04**")
    st.markdown("IT2011 – Artificial Intelligence & Machine Learning")
    st.divider()

    st.subheader("Model")
    st.markdown("**Random Forest** (recommended)")
    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | Accuracy | {metrics['Accuracy']} |
        | F1-Score | {metrics['F1-Score']} |
        | ROC-AUC  | {metrics['ROC-AUC']} |
        """
    )
    st.caption("Trained on Breast Cancer Wisconsin (Original) – UCI")
    st.divider()

    st.subheader("How to use")
    st.markdown(
        """
        1. Set each feature value (1–10) using the sliders  
        2. Click **Classify**  
        3. View the prediction and confidence  
        """
    )
    st.divider()
    st.caption(
        "⚠️ Decision-support only. Not a substitute for clinical diagnosis."
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Breast Tumour Classification")
st.markdown(
    "Enter the nine cytological feature scores from a fine-needle aspiration sample. "
    "Each feature is scored on a **1–10** scale."
)

# ---- Initialise session state BEFORE any widgets ----
BENIGN_PRESET = {
    "Clump_Thickness": 3, "Uniformity_of_Cell_Size": 1,
    "Uniformity_of_Cell_Shape": 1, "Marginal_Adhesion": 1,
    "Single_Epithelial_Cell_Size": 2, "Bare_Nuclei": 1,
    "Bland_Chromatin": 3, "Normal_Nucleoli": 1, "Mitoses": 1,
}
MALIGNANT_PRESET = {
    "Clump_Thickness": 8, "Uniformity_of_Cell_Size": 7,
    "Uniformity_of_Cell_Shape": 8, "Marginal_Adhesion": 6,
    "Single_Epithelial_Cell_Size": 6, "Bare_Nuclei": 9,
    "Bland_Chromatin": 7, "Normal_Nucleoli": 8, "Mitoses": 3,
}

for name in FEATURE_NAMES:
    if name not in st.session_state:
        st.session_state[name] = 5


def _set_benign():
    for k, v in BENIGN_PRESET.items():
        st.session_state[k] = v


def _set_malignant():
    for k, v in MALIGNANT_PRESET.items():
        st.session_state[k] = v


def _set_reset():
    for k in FEATURE_NAMES:
        st.session_state[k] = 5


# ---- Input form ----
st.subheader("Feature Inputs")

cols = st.columns(3)
values = {}

for i, name in enumerate(FEATURE_NAMES):
    with cols[i % 3]:
        values[name] = st.slider(
            label=name.replace("_", " "),
            min_value=1,
            max_value=10,
            step=1,
            help=FEATURE_HELP[name],
            key=name,  # value comes from session_state[name]
        )

# Quick-fill presets (on_click runs BEFORE widgets are re-created)
st.markdown("**Quick presets**")
pcol1, pcol2, pcol3, _ = st.columns([1, 1, 1, 2])
with pcol1:
    st.button("Typical Benign", use_container_width=True, on_click=_set_benign)
with pcol2:
    st.button("Typical Malignant", use_container_width=True, on_click=_set_malignant)
with pcol3:
    st.button("Reset to 5", use_container_width=True, on_click=_set_reset)

st.divider()

# ---- Classify ----
if st.button("🔍  Classify Sample", type="primary", use_container_width=True):
    x = np.array([[values[f] for f in FEATURE_NAMES]])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]  # [P(benign), P(malignant)]

    label = "Malignant" if pred == 1 else "Benign"
    confidence = proba[pred] * 100

    # Result card
    if pred == 1:
        st.error(f"### Prediction: **{label}**")
    else:
        st.success(f"### Prediction: **{label}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Class", label)
    m2.metric("Confidence", f"{confidence:.1f}%")
    m3.metric("Malignant probability", f"{proba[1]*100:.1f}%")

    # Probability bar
    st.markdown("**Class probabilities**")
    prob_df = pd.DataFrame(
        {"Class": ["Benign (0)", "Malignant (1)"], "Probability": [proba[0], proba[1]]}
    )
    st.bar_chart(prob_df.set_index("Class"), height=200)

    # Feature importance for this model (global)
    with st.expander("Feature importance (model-level)"):
        imp = pd.DataFrame(
            {"Feature": FEATURE_NAMES, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        st.bar_chart(imp.set_index("Feature"), horizontal=True, height=300)

    st.info(
        "This is a **decision-support** prediction based on the Random Forest model "
        "trained on the UCI Breast Cancer Wisconsin dataset. Always consult a qualified "
        "medical professional for diagnosis."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "ClearCell  |  Group_26-MTR-IT2011-04  |  IT2011 AI & ML  |  "
    "Faculty of Computing, SLIIT  |  Year 2 Semester 1 (2026)"
)
