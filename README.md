# MLOps Mini-Projet — Iris Prediction Service

Projet MLOps complet : entraînement d'un modèle de classification (Iris), exposition via API FastAPI, conteneurisation Docker et pipeline CI GitHub Actions.

## Dataset

**Iris** (scikit-learn built-in) — 150 échantillons, 4 features, 3 classes (*setosa*, *versicolor*, *virginica*).

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
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## Pipeline CI (GitHub Actions)

| Branche | Étapes |
|---------|--------|
| `feature/*` | Install → Train |
| `develop` | Install → Train → Build Docker → Push to GHCR |
