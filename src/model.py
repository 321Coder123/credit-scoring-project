import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

from src.preprocessing import load_data, get_train_test_data, create_preprocessor


def build_model_pipeline(preprocessor):
    """
    Crée le pipeline complet : Preprocessing -> Modèle
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])

    return model_pipeline


def train_and_evaluate(parquet_path, model_output_path):
    # 1. Chargement des données
    print("Chargement des données...")
    df = load_data(parquet_path)

    X_train, X_test, y_train, y_test = get_train_test_data(df)          # Separation en données d'entrainement et de test

    # 2. Création du préprocesseur (basé sur le train set uniquement !)
    print("Configuration du préprocesseur...")
    preprocessor = create_preprocessor(X_train)

    # 3. Construction du pipeline complet
    pipeline = build_model_pipeline(preprocessor)

    # 4. Entraînement
    print("Entraînement du modèle (ça peut prendre du temps)...")
    pipeline.fit(X_train, y_train)

    # 5. Évaluation
    print("Évaluation...")
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    print(f"\n--- RÉSULTATS ---")
    print(f"ROC AUC Score : {roc_auc:.4f}")

    # 6. Sauvegarde du modèle complet
    print(f"Sauvegarde du modèle dans {model_output_path}...")
    joblib.dump(pipeline, model_output_path)
    print("Terminé !")


if __name__ == "__main__":
    INPUT_FILE = "../data/processed/application_train.parquet"
    MODEL_FILE = "../models/credit_scoring_model.joblib"

    train_and_evaluate(INPUT_FILE, MODEL_FILE)