git push origin develop# MLOps Mini-Projet — Wine Prediction Service

Projet MLOps complet : entraînement d'un modèle de classification (Wine), exposition via API FastAPI, conteneurisation Docker et pipeline CI GitHub Actions.

## Dataset

**Wine** (scikit-learn built-in) — 178 échantillons, 13 features, 3 classes (*class_0*, *class_1*, *class_2*).

## Structure du projet

```
├── train.py                  # Script d'entraînement
├── app.py                    # API FastAPI
├── requirements.txt          # Dépendances Python
├── Dockerfile                # Image Docker
├── .github/workflows/ci.yml  # Pipeline CI
└── model/                    # Artefacts (généré après entraînement)
    ├── model.pkl
    └── metrics.json
```

## Utilisation locale

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Entraîner le modèle
```bash
python train.py
```

### 3. Lancer l'API
```bash
uvicorn app:app --reload
```

### 4. Tester
```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 13.0, "malic_acid": 1.5, "ash": 2.3, "alcalinity_of_ash": 15.0, "magnesium": 120.0, "total_phenols": 2.8, "flavanoids": 3.0, "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.7, "color_intensity": 5.0, "hue": 1.0, "od280_od315_of_diluted_wines": 3.0, "proline": 1000.0}'
```

## Docker

```bash
docker build -t wine-api .
docker run -p 8000:8000 wine-api
```

## Pipeline CI (GitHub Actions)

| Branche | Étapes |
|---------|--------|
| `feature/*` | Install → Train |
| `develop` | Install → Train → Build Docker → Push to GHCR |
