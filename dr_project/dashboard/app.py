"""
dashboard/app.py
----------------
A Streamlit dashboard that demonstrates the trained diabetic retinopathy
classifier. Three pages, selectable from the sidebar:

  1. Classify an Image  -- upload a fundus photo (or pick a sample test
     image) and see the model's live prediction with confidence per class.
  2. Model Performance   -- the real evaluation results from training
     (curves, confusion matrix, classification report).
  3. Dataset Overview    -- class distribution and preprocessing/
     augmentation examples, for context.

"""

import os
import sys
import glob

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make src/ importable regardless of where streamlit is launched from
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(THIS_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import config
from data_preprocessing import preprocess_image, crop_to_circle_bounds, apply_clahe

st.set_page_config(
    page_title="Diabetic Retinopathy Stage Detection",
    page_icon="\U0001F441",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached loaders -- these run once per session, not on every interaction
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """
    Load the trained Keras model from MODEL_DIR. Cached with
    st.cache_resource so the (potentially large) model is loaded from disk
    only once per session, not on every button click / page interaction.
    Returns None if no trained model is found yet, so the app can show a
    helpful message instead of crashing.
    """
    import tensorflow as tf

    candidates = sorted(glob.glob(os.path.join(config.MODEL_DIR, "*.keras")))
    # Prefer the final model if it exists, otherwise fall back to whatever
    # checkpoint is available (useful mid-experimentation).
    preferred = os.path.join(config.MODEL_DIR, "dr_final_model.keras")
    path = preferred if os.path.exists(preferred) else (candidates[-1] if candidates else None)
    if path is None:
        return None, None
    return tf.keras.models.load_model(path), path


@st.cache_data(show_spinner=False)
def load_test_dataframe():
    """
    Load the test-split CSV that train.py saves (logs/test_split.csv), so
    the "pick a sample test image" dropdown can offer real, labelled test
    images without needing to re-run the train/val/test split here.
    """
    import pandas as pd

    path = os.path.join(config.LOG_DIR, "test_split.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def predict(model, filepath):
    """Run the full preprocessing pipeline + model prediction on one image
    file path, returning (preprocessed_image, probability_array)."""
    img = preprocess_image(filepath)
    probs = model.predict(img[None, ...], verbose=0)[0]
    return img, probs


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Diabetic Retinopathy Prototype")
page = st.sidebar.radio(
    "Go to",
    ["Classify an Image", "Model Performance", "Dataset Overview"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Computer Vision coursework prototype. Not a validated medical device -- "
    "for academic demonstration only."
)


# ---------------------------------------------------------------------------
# Page 1: Classify an Image
# ---------------------------------------------------------------------------
if page == "Classify an Image":
    st.title("Classify a Retinal Image")
    st.write(
        "Upload a fundus photograph, or pick a real test-set image below, "
        "to see the trained model's predicted diabetic retinopathy stage."
    )

    model, model_path = load_model()
    if model is None:
        st.error(
            "No trained model found in `saved_models/`. Run training "
            "(Sections 4-5 of the notebook, or `train.py`) first, then "
            "restart this dashboard."
        )
        st.stop()
    st.caption(f"Model loaded from: `{model_path}`")

    test_df = load_test_dataframe()

    col_left, col_right = st.columns([1, 1])
    filepath = None

    with col_left:
        st.subheader("Option A: upload your own image")
        uploaded = st.file_uploader("Fundus image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            tmp_path = os.path.join(config.OUTPUT_DIR, "_uploaded_preview.png")
            Image.open(uploaded).convert("RGB").save(tmp_path)
            filepath = tmp_path

    with col_right:
        st.subheader("Option B: pick a real test image")
        if test_df is None:
            st.info(
                "No `logs/test_split.csv` found yet -- this file is saved "
                "automatically by `train.py` / the notebook's training "
                "cells. Run training once to enable this option."
            )
        else:
            chosen_label = st.selectbox("Class", config.CLASS_NAMES)
            subset = test_df[test_df["label"] == chosen_label].reset_index(drop=True)
            idx = st.slider("Sample index", 0, max(len(subset) - 1, 0), 0)
            if len(subset) > 0:
                filepath = subset.loc[idx, "filepath"]
                st.caption(f"True label: **{chosen_label}**")

    st.markdown("---")

    if filepath is not None and os.path.exists(filepath):
        img, probs = predict(model, filepath)
        pred_idx = int(np.argmax(probs))
        pred_label = config.CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.image(img, caption="Preprocessed input (what the model sees)", use_container_width=True)
        with c2:
            st.metric("Predicted stage", pred_label, f"{confidence:.1%} confidence")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            colors = ["#4C72B0"] * len(config.CLASS_NAMES)
            colors[pred_idx] = "#DD8452"
            ax.barh(config.CLASS_NAMES, probs, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted probability")
            ax.invert_yaxis()
            for i, v in enumerate(probs):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)
            st.pyplot(fig)

        if confidence < 0.5:
            st.warning(
                "The model's top prediction has fairly low confidence -- "
                "treat this result with extra caution."
            )
    else:
        st.info("Upload an image or select a test sample above to see a prediction.")


# ---------------------------------------------------------------------------
# Page 2: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write(
        "Evaluation results from the reported training run, on the "
        "held-out test set."
    )

    def show_if_exists(filename, caption):
        path = os.path.join(config.OUTPUT_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"`{filename}` not found in outputs/ yet -- run evaluate.py / Section 6 of the notebook first.")

    tab1, tab2, tab3 = st.tabs(["Training Curves", "Confusion Matrix", "Precision / Recall / F1"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            show_if_exists("curves_phase1_head.png", "Phase 1 (frozen backbone)")
        with c2:
            show_if_exists("curves_phase2_finetune.png", "Phase 2 (fine-tuning)")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            show_if_exists("confusion_matrix_counts.png", "Raw counts")
        with c2:
            show_if_exists("confusion_matrix_normalised.png", "Row-normalised")

    with tab3:
        show_if_exists("per_class_metrics.png", "Per-class precision, recall, F1")
        report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
        if os.path.exists(report_path):
            with open(report_path) as f:
                st.code(f.read(), language=None)
        else:
            st.info("`classification_report.txt` not found in outputs/ yet.")


# ---------------------------------------------------------------------------
# Page 3: Dataset Overview
# ---------------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(
        "The dataset used to train this model: 2,750 fundus images across "
        "5 diabetic retinopathy stages, sourced from Kaggle."
    )

    def show_if_exists(filename, caption):
        path = os.path.join(config.OUTPUT_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"`{filename}` not found in outputs/ yet.")

    show_if_exists("class_distribution.png", "Class distribution (imbalanced: 1000 Healthy vs 190 Severe DR)")
    st.markdown("---")
    show_if_exists("preprocessing_demo.png", "Preprocessing pipeline: crop \u2192 CLAHE \u2192 edge enhancement \u2192 resize")
    st.markdown("---")
    show_if_exists("augmentation_demo.png", "Augmentation examples (rotation, flip, zoom, brightness/contrast jitter)")
