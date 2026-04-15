"""
API FastAPI — Sert le modèle Wine entraîné.
Endpoints :
  GET  /health   → vérification de santé
  POST /predict  → prédiction à partir de 13 features
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Wine Prediction API")

# Charger le modèle au démarrage
MODEL_PATH = "model/model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

TARGET_NAMES = ["class_0", "class_1", "class_2"]


class PredictRequest(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


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
        request.alcohol,
        request.malic_acid,
        request.ash,
        request.alcalinity_of_ash,
        request.magnesium,
        request.total_phenols,
        request.flavanoids,
        request.nonflavanoid_phenols,
        request.proanthocyanins,
        request.color_intensity,
        request.hue,
        request.od280_od315_of_diluted_wines,
        request.proline,
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
