"""
test_api.py
Quick script to verify the FastAPI endpoint works.
Run AFTER: uvicorn api.main:app --reload

Usage: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# ── Health check ──────────────────────────────────────────────────────────────
resp = requests.get(f"{BASE_URL}/")
print("Health check:", resp.json())

# ── Predict: high churn risk customer ─────────────────────────────────────────
high_risk = {
    "gender": 0, "SeniorCitizen": 1, "Partner": 0,
    "Dependents": 0, "tenure": 2, "PhoneService": 1,
    "MultipleLines": 0, "InternetService": 1,
    "OnlineSecurity": 0, "OnlineBackup": 0,
    "DeviceProtection": 0, "TechSupport": 0,
    "StreamingTV": 0, "StreamingMovies": 0,
    "Contract": 0, "PaperlessBilling": 1,
    "PaymentMethod": 2, "MonthlyCharges": 75.50,
    "TotalCharges": 151.0, "tenure_group": 0,
    "charge_ratio": 75.5, "num_services": 2
}

resp = requests.post(f"{BASE_URL}/predict", json=high_risk)
print("\nHigh-risk customer prediction:")
print(json.dumps(resp.json(), indent=2))

# ── Predict: low churn risk customer ──────────────────────────────────────────
low_risk = {
    "gender": 1, "SeniorCitizen": 0, "Partner": 1,
    "Dependents": 1, "tenure": 60, "PhoneService": 1,
    "MultipleLines": 1, "InternetService": 1,
    "OnlineSecurity": 1, "OnlineBackup": 1,
    "DeviceProtection": 1, "TechSupport": 1,
    "StreamingTV": 1, "StreamingMovies": 1,
    "Contract": 2, "PaperlessBilling": 0,
    "PaymentMethod": 0, "MonthlyCharges": 99.90,
    "TotalCharges": 5994.0, "tenure_group": 4,
    "charge_ratio": 99.9, "num_services": 9
}

resp = requests.post(f"{BASE_URL}/predict", json=low_risk)
print("\nLow-risk customer prediction:")
print(json.dumps(resp.json(), indent=2))

# ── Model info ────────────────────────────────────────────────────────────────
resp = requests.get(f"{BASE_URL}/model-info")
print("\nModel info:", json.dumps(resp.json(), indent=2))
