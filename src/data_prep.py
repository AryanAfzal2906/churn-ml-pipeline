"""
data_prep.py
Loads the Telco Customer Churn dataset, cleans it, and engineers features.
Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
(or auto-downloaded via the function below)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_data(path: str = None) -> pd.DataFrame:
    """Load dataset from local path or download from URL."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print("Downloading Telco Churn dataset...")
        df = pd.read_csv(DATA_URL)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/raw_churn.csv", index=False)
        print("Saved to data/raw_churn.csv")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types and handle missing values."""
    df = df.copy()

    # TotalCharges has spaces instead of NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customer ID - not a feature
    df.drop(columns=["customerID"], inplace=True)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features to improve model signal."""
    df = df.copy()

    # Tenure buckets
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
    )

    # Charge per month ratio
    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    return df


def encode_features(df: pd.DataFrame):
    """Label-encode all categorical columns. Returns df + encoder dict."""
    df = df.copy()
    encoders = {}

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def get_splits(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    """Return train/test splits."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def prepare_pipeline(path: str = None):
    """Full prep pipeline — returns X_train, X_test, y_train, y_test."""
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_features(df)
    return get_splits(df)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_pipeline()
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.2%}")
