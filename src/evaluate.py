"""
evaluate.py
Loads the saved best model and generates:
  - Confusion matrix plot
  - ROC curve plot
  - SHAP feature importance summary

Run: python src/evaluate.py
Outputs saved to models/plots/
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_prep import prepare_pipeline

PLOTS_DIR = "models/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model():
    model = joblib.load("models/best_model.pkl")
    with open("models/meta.json") as f:
        meta = json.load(f)
    print(f"Loaded: {meta['best_model_name']}")
    return model, meta["best_model_name"]


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#185FA5", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_shap_summary(model, X_test, model_name):
    """
    SHAP summary plot — shows which features push predictions most.
    Separates the Pipeline steps to avoid SHAP compatibility bugs.
    """
    try:
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)

        # 1. Unpack the Pipeline safely
        if hasattr(model, "named_steps"):
            scaler = model.named_steps["scaler"]
            clf = model.named_steps["clf"]
            # Manually scale the sample data
            X_transformed = pd.DataFrame(
                scaler.transform(X_sample), 
                columns=X_sample.columns, 
                index=X_sample.index
            )
        else:
            X_transformed = X_sample
            clf = model

        # 2. Use the correct Explainer
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values_raw = explainer.shap_values(X_transformed)
        else:
            # Use LinearExplainer for Logistic Regression (faster and avoids the bug!)
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_values_raw = explainer.shap_values(X_transformed)

        # 3. Format SHAP values for plotting
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif len(np.shape(shap_values_raw)) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw

        # 4. Generate Plot (pass X_sample so we see the original unscaled values on the graph)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    except Exception as e:
        print(f"SHAP plot skipped: {e}")


def main():
    model, model_name = load_model()
    _, X_test, _, y_test = prepare_pipeline()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_prob, model_name)
    plot_shap_summary(model, X_test, model_name)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()