import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import ScoringRequest, ScoringResponse
from src.feature_engineering import feature_engineering

# Variable globale pour stocker le modèle
model = None

# Fonction qui s'exécute au démarrage et à l'arrêt de l'API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charge le modèle au démarrage
    global model
    try:
        model = joblib.load("models/credit_scoring_model.joblib")
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
    yield
    # Code à exécuter à l'arrêt (si besoin de nettoyer la mémoire)
    print("🛑 Arrêt de l'API")


# Initialisation de l'application
app = FastAPI(title="Credit Scoring API", version="1.0", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "API de Credit Scoring en ligne. Utilisez /docs pour tester."}


@app.post("/predict", response_model=ScoringResponse)
def predict(request: ScoringRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    try:
        # 1. Convertir les données JSON en DataFrame
        input_data = pd.DataFrame([request.features])

        # 2. Appliquer le features_engineering
        input_data_enriched = feature_engineering(input_data)

        # 3. Faire la prédiction (proba d'être en défaut de paiement)
        probability = model.predict_proba(input_data_enriched)[0][1]

        # 4. Règle métier (Threshold)
        threshold = 0.5
        decision = "Refusé" if probability > threshold else "Accordé"

        return {
            "probability": round(probability, 4),
            "decision": decision,
            "threshold_used": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")