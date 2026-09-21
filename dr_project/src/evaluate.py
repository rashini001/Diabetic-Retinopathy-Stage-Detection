"""
evaluate.py
-----------

  - accuracy & loss curves for both training phases
  - confusion matrix (raw counts and row-normalised)
  - a full classification report (precision / recall / F1 per class,
    plus macro and weighted averages -- macro-average matters most here
    since the classes are imbalanced)
  - a small error-analysis helper that lists the most confidently wrong
    predictions.

All plotting functions save PNGs to OUTPUT_DIR and also return the
matplotlib Figure, so they can be used both from train.py/evaluate.py
scripts and interactively inside the notebook.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import CLASS_NAMES, OUTPUT_DIR, LOG_DIR


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history_dict: dict, phase_name: str, save: bool = True):
    """
    history_dict: a Keras History.history-style dict with keys
    'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_dict["accuracy"], label="train")
    axes[0].plot(history_dict["val_accuracy"], label="validation")
    axes[0].set_title(f"Accuracy — {phase_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_dict["loss"], label="train")
    axes[1].plot(history_dict["val_loss"], label="validation")
    axes[1].set_title(f"Loss — {phase_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, f"curves_{phase_name}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def plot_combined_history(combined_history_path: str = None):
    """Convenience wrapper that loads combined_history.json (from train.py)
    and plots both phases."""
    path = combined_history_path or os.path.join(LOG_DIR, "combined_history.json")
    with open(path) as f:
        combined = json.load(f)
    plot_training_curves(combined["phase1"], "phase1_head")
    plot_training_curves(combined["phase2"], "phase2_finetune")


# ---------------------------------------------------------------------------
# Predictions -> metrics
# ---------------------------------------------------------------------------
def get_predictions(model, dataset):
    """
    Run `model` over a tf.data `dataset` of (images, one_hot_labels)
    batches and return (y_true, y_pred, y_pred_probs) as numpy arrays of
    integer class indices (y_true/y_pred) and softmax probabilities
    (y_pred_probs).
    """
    y_true, y_pred_probs = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true).argmax(axis=1)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = y_pred_probs.argmax(axis=1)
    return y_true, y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred, normalize: bool = False, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d",
              xticks_rotation=45)
    title = "Confusion Matrix" + (" (row-normalised)" if normalize else "")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        suffix = "normalised" if normalize else "counts"
        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{suffix}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def get_classification_report(y_true, y_pred, save: bool = True) -> str:
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3
    )
    print(report)
    if save:
        path = os.path.join(OUTPUT_DIR, "classification_report.txt")
        with open(path, "w") as f:
            f.write(report)
        print(f"Saved {path}")
    return report


def get_classification_report_dict(y_true, y_pred) -> dict:
    return classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=3, output_dict=True
    )


def plot_per_class_metrics(report_dict: dict, save: bool = True):
    """Grouped bar chart of precision/recall/F1 per class -- a compact,
    reviewer-friendly way to show where the model struggles (usually the
    adjacent-stage confusions, e.g. Mild vs Moderate)."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        values = [report_dict[c][m] for c in CLASS_NAMES]
        ax.bar(x + (i - 1) * width, values, width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def most_confident_errors(y_true, y_pred, y_pred_probs, test_df, top_n: int = 10):
    wrong_mask = y_true != y_pred
    wrong_idx = np.where(wrong_mask)[0]
    confidences = y_pred_probs[wrong_idx, y_pred[wrong_idx]]
    order = np.argsort(-confidences)[:top_n]
    top_idx = wrong_idx[order]

    rows = []
    for i in top_idx:
        rows.append(
            {
                "filepath": test_df.iloc[i]["filepath"],
                "true_label": CLASS_NAMES[y_true[i]],
                "predicted_label": CLASS_NAMES[y_pred[i]],
                "confidence": float(y_pred_probs[i, y_pred[i]]),
            }
        )
    return rows


if __name__ == "__main__":
    # Example end-to-end usage once a model has been trained (train.py):
    #
    #   import tensorflow as tf
    #   from config import MODEL_DIR
    #   from data_preprocessing import build_dataframe
    #   from data_augmentation import build_tf_dataset
    #   import pandas as pd, os
    #
    #   model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "dr_final_model.keras"))
    #   test_df = pd.read_csv(os.path.join(LOG_DIR, "test_split.csv"))
    #   test_ds = build_tf_dataset(test_df, training=False)
    #
    #   y_true, y_pred, y_pred_probs = get_predictions(model, test_ds)
    #   plot_combined_history()
    #   plot_confusion_matrix(y_true, y_pred)
    #   plot_confusion_matrix(y_true, y_pred, normalize=True)
    #   report = get_classification_report(y_true, y_pred)
    #   plot_per_class_metrics(get_classification_report_dict(y_true, y_pred))
    #   errors = most_confident_errors(y_true, y_pred, y_pred_probs, test_df)
    print(__doc__)
