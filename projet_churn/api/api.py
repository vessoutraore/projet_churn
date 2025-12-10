# api.py
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------
# Config & chargement modèle
# -------------------------

BASE_DIR = Path(__file__).resolve().parent           # dossier api/
MODEL_PATH = BASE_DIR.parent / "model" / "model.pkl" # dossier model/model.pkl

print("==> Chargement du modèle :", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="API Prédiction Churn Télécom – Guinée",
    description="API pour prédire la résiliation d’un client (churn = 1).",
    version="1.0.0",
)

# -------------------------
# Schéma d'entrée
# -------------------------

class ClientFeatures(BaseModel):
    region: Literal["Conakry", "Kankan", "Labé", "N’Zérékoré", "Boké", "Faranah"]
    sexe: Literal["M", "F"]
    age: int
    revenu_estime_gnf: float
    anciennete_mois: int
    type_abonnement: Literal["Prépayé", "Postpayé"]
    forfait_international: Literal["Oui", "Non"]
    messagerie_vocale: Literal["Oui", "Non"]
    recharge_mensuelle_moy_gnf: float
    minutes_jour: float
    minutes_nuit: float
    donnees_mo: float
    nombre_sms: int
    appels_service_client: int
    pannes_signalees_30j: int
    retard_paiement_jours: int
    minutes_internationales: float

# -------------------------
# Transformation en vecteur
# -------------------------

def make_feature_vector(data: ClientFeatures) -> np.ndarray:
    # À adapter si tu as aussi encodé tes variables catégorielles.
    vec = np.array(
        [
            data.age,
            data.revenu_estime_gnf,
            data.anciennete_mois,
            data.recharge_mensuelle_moy_gnf,
            data.minutes_jour,
            data.minutes_nuit,
            data.donnees_mo,
            data.nombre_sms,
            data.appels_service_client,
            data.pannes_signalees_30j,
            data.retard_paiement_jours,
            data.minutes_internationales,
        ],
        dtype=float,
    )
    return vec.reshape(1, -1)

# -------------------------
# Endpoint de prédiction
# -------------------------

@app.post("/predict")
def predict_churn(features: ClientFeatures):
    x = make_feature_vector(features)
    proba_churn = float(model.predict_proba(x)[0][1])
    pred = int(proba_churn >= 0.5)

    return {
        "prediction": pred,
        "proba_churn": proba_churn,
        "message": "1 = Client à risque, 0 = Client fidèle."
    }