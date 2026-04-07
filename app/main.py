"""
main.py — API FastAPI de prévision de température.

Endpoints :
    GET  /health          Statut de l'API et du modèle chargé
    POST /v1/predict      Prédiction de la température du lendemain

Le modèle est chargé depuis le MLflow Model Registry via l'alias @production.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime

import mlflow.sklearn
import pandas as pd
from datetime import timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "Weather_RF_Model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
API_VERSION = "1.0.0"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NOAA Weather Expert API",
    description=(
        "API de prévision de température quotidienne basée sur un modèle "
        "Random Forest entraîné sur des données NOAA-compatible. "
        "Le modèle est chargé depuis le MLflow Model Registry."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_production_model():
    """Charge le modèle @production depuis MLflow Model Registry.

    Stratégie en deux temps :
    1. Via le serveur HTTP MLflow (tracking URI configuré)
    2. Fallback direct sur /mlruns monté en volume Docker
       (nécessaire quand le modèle a été enregistré depuis l'hôte WSL
        avec un chemin file:// absolu non résolvable dans le container)
    """
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info("Chargement du modèle : %s", model_uri)

    # --- Tentative 1 : serveur HTTP MLflow ---
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle chargé avec succès depuis %s", model_uri)
        return model
    except Exception as exc:
        logger.warning("Chargement via serveur HTTP échoué : %s", exc)

    # --- Tentative 2 : chargement pickle direct depuis /mlruns (volume Docker) ---
    # Utilisé quand le modèle a été enregistré depuis l'hôte WSL avec un chemin
    # file:// absolu non résolvable dans le container Docker.
    try:
        import glob as _glob
        import pickle as _pickle
        # Cherche model.pkl dans toute l'arborescence /mlruns
        pkl_matches = _glob.glob("/mlruns/**/model.pkl", recursive=True)
        if pkl_matches:
            pkl_path = pkl_matches[0]
            with open(pkl_path, "rb") as f:
                model = _pickle.load(f)
            logger.info("Modèle chargé via pickle direct : %s", pkl_path)
            return model
        else:
            logger.error("Aucun model.pkl trouvé dans /mlruns")
    except Exception as exc2:
        logger.error("Échec du fallback pickle : %s", exc2)

    return None


model = load_production_model()

# ---------------------------------------------------------------------------
# Schémas Pydantic — validation stricte des entrées/sorties
# ---------------------------------------------------------------------------

class WeatherInput(BaseModel):
    """Données d'entrée pour la prédiction de température."""

    temp_today: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Température minimale du jour courant en °C (entre -50 et 60).",
        examples=[18.5],
    )
    temp_lag_1: float | None = Field(
        default=None,
        ge=-50.0,
        le=60.0,
        description="Température d'hier en °C (optionnel — amélioré la précision).",
        examples=[17.2],
    )

    model_config = {"json_schema_extra": {"example": {"temp_today": 18.5, "temp_lag_1": 17.2}}}


class WeatherPrediction(BaseModel):
    """Réponse structurée de la prédiction."""

    request_id: str = Field(description="Identifiant unique de la requête.")
    input_temp: float = Field(description="Température fournie en entrée (°C).")
    prediction_tomorrow: float = Field(description="Température prévue demain (°C).")
    model_alias: str = Field(description="Alias du modèle utilisé.")
    model_name: str = Field(description="Nom du modèle dans le registry.")
    timestamp: str = Field(description="Horodatage UTC de la prédiction.")


class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""

    status: str
    model_ready: bool
    model_name: str
    model_alias: str
    mlflow_uri: str
    api_version: str


# ---------------------------------------------------------------------------
# Middleware — logging des requêtes
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] %s %s", request_id, request.method, request.url.path)
    response = await call_next(request)
    logger.info("[%s] Statut : %d", request_id, response.status_code)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    tags=["Infrastructure"],
)
def health() -> HealthResponse:
    """Vérifie l'état de l'API et la disponibilité du modèle de production."""
    return HealthResponse(
        status="online",
        model_ready=model is not None,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        mlflow_uri=MLFLOW_TRACKING_URI,
        api_version=API_VERSION,
    )


@app.post(
    "/v1/predict",
    response_model=WeatherPrediction,
    summary="Prédire la température de demain",
    tags=["Prédiction"],
    responses={
        503: {"description": "Modèle non disponible — vérifier MLflow"},
        422: {"description": "Données d'entrée invalides (hors bornes)"},
    },
)
def predict(payload: WeatherInput) -> WeatherPrediction:
    """
    Prédit la température minimale du lendemain.

    Le modèle utilise la température du jour comme feature principale.
    Fournir `temp_lag_1` (température d'hier) améliore la précision.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modèle '{MODEL_NAME}@{MODEL_ALIAS}' non disponible. "
                "Vérifiez que MLflow est accessible et que le modèle est enregistré."
            ),
        )

    # Construction du vecteur de features
    # Les features non fournies sont imputées avec la valeur de temp_today
    temp_lag_1 = payload.temp_lag_1 if payload.temp_lag_1 is not None else payload.temp_today

    input_data = pd.DataFrame([{
        "temp": payload.temp_today,
        "temp_lag_1": temp_lag_1,
        "temp_lag_2": temp_lag_1,         # Approximation si non fourni
        "temp_lag_7": payload.temp_today,  # Approximation si non fourni
        "temp_rolling_mean_7": payload.temp_today,
        "temp_rolling_std_7": 1.0,
        "temp_rolling_mean_14": payload.temp_today,
        "day_of_year": datetime.now(timezone.utc).timetuple().tm_yday,
        "month": datetime.now(timezone.utc).month,
        "week_of_year": datetime.now(timezone.utc).isocalendar()[1],
    }])

    prediction = model.predict(input_data)
    pred_value = round(float(prediction[0]), 2)

    logger.info(
        "Prédiction : temp_today=%.1f°C → demain=%.2f°C",
        payload.temp_today,
        pred_value,
    )

    return WeatherPrediction(
        request_id=str(uuid.uuid4())[:8],
        input_temp=payload.temp_today,
        prediction_tomorrow=pred_value,
        model_alias=MODEL_ALIAS,
        model_name=MODEL_NAME,
        timestamp=datetime.now(timezone.utc).isoformat() + "Z",
    )
