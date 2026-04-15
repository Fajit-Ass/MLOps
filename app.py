"""
API FastAPI — Sert le modèle Iris entraîné.
Endpoints :
  GET  /health   → vérification de santé
  POST /predict  → prédiction à partir de 4 features
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Iris Prediction API")

# Charger le modèle au démarrage
MODEL_PATH = "model/model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictResponse(BaseModel):
    prediction: str
    prediction_index: int
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model not loaded",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]])

    pred_index = int(model.predict(features)[0])
    probas = model.predict_proba(features)[0]

    return PredictResponse(
        prediction=TARGET_NAMES[pred_index],
        prediction_index=pred_index,
        probabilities={
            name: round(float(p), 4) for name, p in zip(TARGET_NAMES, probas)
        },
    )
