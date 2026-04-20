"""
api/main.py
FastAPI app that serves predictions from the trained churn model.

Run:
    uvicorn api.main:app --reload
    
Then open:
    http://localhost:8000/docs   ← auto-generated Swagger UI
    http://localhost:8000/       ← health check
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML-powered API to predict customer churn probability.",
    version="1.0.0",
)

# ── Load model on startup ──────────────────────────────────────────────────────
MODEL = None
FEATURE_NAMES = None

@app.on_event("startup")
def load_model():
    global MODEL, FEATURE_NAMES
    model_path = "models/best_model.pkl"
    features_path = "models/feature_names.json"

    if not os.path.exists(model_path):
        raise RuntimeError(
            "Model not found. Run 'python src/train.py' first."
        )

    MODEL = joblib.load(model_path)
    with open(features_path) as f:
        FEATURE_NAMES = json.load(f)
    print(f"Model loaded from {model_path}")


# ── Input schema ──────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """All fields match the engineered feature set from data_prep.py"""
    gender: int = Field(..., description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=2)
    InternetService: int = Field(..., ge=0, le=2)
    OnlineSecurity: int = Field(..., ge=0, le=2)
    OnlineBackup: int = Field(..., ge=0, le=2)
    DeviceProtection: int = Field(..., ge=0, le=2)
    TechSupport: int = Field(..., ge=0, le=2)
    StreamingTV: int = Field(..., ge=0, le=2)
    StreamingMovies: int = Field(..., ge=0, le=2)
    Contract: int = Field(..., ge=0, le=2, description="0=Month-to-month, 1=One year, 2=Two year")
    PaperlessBilling: int = Field(..., ge=0, le=1)
    PaymentMethod: int = Field(..., ge=0, le=3)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    tenure_group: int = Field(..., ge=0, le=4)
    charge_ratio: float = Field(..., ge=0)
    num_services: int = Field(..., ge=0, le=9)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 1, "SeniorCitizen": 0, "Partner": 1,
                "Dependents": 0, "tenure": 12, "PhoneService": 1,
                "MultipleLines": 0, "InternetService": 1,
                "OnlineSecurity": 0, "OnlineBackup": 0,
                "DeviceProtection": 0, "TechSupport": 0,
                "StreamingTV": 1, "StreamingMovies": 1,
                "Contract": 0, "PaperlessBilling": 1,
                "PaymentMethod": 2, "MonthlyCharges": 70.35,
                "TotalCharges": 844.2, "tenure_group": 1,
                "charge_ratio": 65.7, "num_services": 4
            }
        }


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.
    Returns prediction, probability, and a risk label.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build input DataFrame in the same column order as training
    data = pd.DataFrame([customer.dict()])[FEATURE_NAMES]

    pred = int(MODEL.predict(data)[0])
    prob = round(float(MODEL.predict_proba(data)[0][1]), 4)

    if prob >= 0.7:
        risk = "HIGH"
        msg = "Customer is very likely to churn. Immediate retention action recommended."
    elif prob >= 0.4:
        risk = "MEDIUM"
        msg = "Customer shows moderate churn risk. Consider a proactive offer."
    else:
        risk = "LOW"
        msg = "Customer is likely to stay. Continue normal engagement."

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=prob,
        risk_level=risk,
        message=msg,
    )


@app.get("/model-info", tags=["Info"])
def model_info():
    """Returns metadata about the loaded model."""
    with open("models/meta.json") as f:
        meta = json.load(f)
    return {
        "model_name": meta.get("best_model_name"),
        "features_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
    }
