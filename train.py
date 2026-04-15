"""
Script d'entraînement — Dataset Wine (classification multi-classe)
Produit : model/model.pkl, model/metrics.json
"""

import os
import json
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline


def train():
    # 1. Charger le dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # 2. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Créer un pipeline (prétraitement + modèle)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42
        )),
    ])

    # 4. Entraîner
    pipeline.fit(X_train, y_train)

    # 5. Évaluer
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("=== Résultats d'évaluation ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score (weighted) : {f1:.4f}")
    print()
    print(classification_report(
        y_test, y_pred, target_names=wine.target_names
    ))

    # 6. Sauvegarder le modèle
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    # 7. Sauvegarder les métriques
    metrics = {"accuracy": round(accuracy, 4), "f1_weighted": round(f1, 4)}
    metrics_path = os.path.join("model", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées dans {metrics_path}")


if __name__ == "__main__":
    train()
