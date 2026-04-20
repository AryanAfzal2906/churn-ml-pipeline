"""
train.py
Trains multiple models, logs everything to MLflow, and saves the best model.

Run: python src/train.py
Then: mlflow ui   →  open http://localhost:5000 to see experiments
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_prep import prepare_pipeline

EXPERIMENT_NAME = "customer-churn-prediction"
MODEL_SAVE_DIR = "models"


def get_models():
    """Return dict of model name → model object."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
    }


def compute_metrics(y_true, y_pred, y_prob):
    """Return a dict of all evaluation metrics."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }


def train_and_log(model_name, model, X_train, X_test, y_train, y_test):
    """Train one model and log all params, metrics, and artifacts to MLflow."""
    with mlflow.start_run(run_name=model_name):

        # ── Train ──────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        # ── Metrics ────────────────────────────────────────────
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # ── Params ─────────────────────────────────────────────
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("features", X_train.shape[1])

        # ── Log model ──────────────────────────────────────────
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # ── Tags ───────────────────────────────────────────────
        mlflow.set_tag("dataset", "telco_churn")
        mlflow.set_tag("author", "aryan-afzal")

        # ── Classification report as artifact ──────────────────
        report = classification_report(y_test, y_pred, output_dict=False)
        report_path = f"models/{model_name}_report.txt"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n\n{report}")
        mlflow.log_artifact(report_path)

        run_id = mlflow.active_run().info.run_id
        print(f"  [{model_name}] ROC-AUC={metrics['roc_auc']} | F1={metrics['f1']} | run_id={run_id}")
        return metrics, run_id


def save_best_model(best_name, best_model):
    """Persist the best model to disk for API serving."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    path = os.path.join(MODEL_SAVE_DIR, "best_model.pkl")
    joblib.dump(best_model, path)
    meta = {"best_model_name": best_name, "model_path": path}
    with open(os.path.join(MODEL_SAVE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nBest model saved → {path}")


def main():
    # ── Data ───────────────────────────────────────────────────
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_pipeline()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    # ── MLflow setup ───────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment: '{EXPERIMENT_NAME}'")
    print("Run 'mlflow ui' to view results at http://localhost:5000\n")

    # ── Train all models ───────────────────────────────────────
    results = {}
    trained_models = {}
    for name, model in get_models().items():
        metrics, run_id = train_and_log(
            name, model, X_train, X_test, y_train, y_test
        )
        results[name] = (metrics["roc_auc"], model)

    # ── Pick best by ROC-AUC ───────────────────────────────────
    best_name = max(results, key=lambda k: results[k][0])
    best_model = results[best_name][1]
    print(f"\nBest model: {best_name} (ROC-AUC={results[best_name][0]})")

    save_best_model(best_name, best_model)

    # ── Save feature names for API ─────────────────────────────
    feature_path = os.path.join(MODEL_SAVE_DIR, "feature_names.json")
    with open(feature_path, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)


if __name__ == "__main__":
    main()
