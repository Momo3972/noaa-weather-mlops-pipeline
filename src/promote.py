"""
promote.py — Promotion du meilleur modèle vers l'alias @production.

Utilise l'API MLflow 2.x moderne (aliases), compatible avec app/main.py
qui charge le modèle via "models:/Weather_RF_Model@production".

⚠️  Les méthodes transition_model_version_stage() et get_latest_versions()
    sont dépréciées depuis MLflow 2.x. Ce fichier utilise exclusivement
    l'API par alias (set_registered_model_alias / get_model_version_by_alias).
"""

from __future__ import annotations

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "Weather_RF_Model")
PRODUCTION_ALIAS = "production"

# Seuil de qualité minimum pour autoriser la promotion
# Le modèle ne sera promu que si RMSE <= ce seuil (en °C)
RMSE_THRESHOLD = float(os.getenv("PROMOTION_RMSE_THRESHOLD", "3.0"))


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def promote_best_model(model_name: str = MODEL_NAME) -> None:
    """
    Identifie la dernière version du registre MLflow et la promeut
    vers l'alias @production après vérification du seuil de qualité.

    Stratégie de sélection :
        - Récupère toutes les versions disponibles du modèle
        - Prend la version avec le numéro le plus élevé (dernière enregistrée)
        - Vérifie que son RMSE est <= RMSE_THRESHOLD avant de promouvoir

    Args:
        model_name: Nom du modèle dans le MLflow Model Registry.

    Raises:
        ValueError: Si aucune version n'est disponible ou si le seuil RMSE
                    n'est pas atteint (la promotion est refusée).
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    logger.info("Recherche des versions disponibles pour '%s'...", model_name)

    # --- Récupération de toutes les versions (API moderne MLflow 2.x) ---
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:
        logger.error("Impossible de lister les versions du modèle : %s", exc)
        raise

    if not all_versions:
        raise ValueError(
            f"Aucune version trouvée pour le modèle '{model_name}'. "
            "Lancez d'abord src/train.py."
        )

    # Trier par numéro de version (descending) → prendre la plus récente
    latest_version = max(all_versions, key=lambda v: int(v.version))
    version_number = latest_version.version
    run_id = latest_version.run_id

    logger.info("Dernière version : v%s (run_id=%s)", version_number, run_id)

    # --- Vérification du seuil de qualité (Quality Gate) ---
    run_data = client.get_run(run_id).data
    rmse = run_data.metrics.get("rmse")

    if rmse is None:
        logger.warning(
            "Métrique 'rmse' absente du run %s. "
            "Promotion accordée sans vérification de seuil.",
            run_id,
        )
    elif rmse > RMSE_THRESHOLD:
        raise ValueError(
            f"Quality Gate ÉCHOUÉ — RMSE={rmse:.4f}°C > seuil={RMSE_THRESHOLD}°C. "
            f"La version v{version_number} ne sera PAS promue en production."
        )
    else:
        logger.info(
            "Quality Gate OK — RMSE=%.4f°C <= seuil=%.1f°C",
            rmse, RMSE_THRESHOLD,
        )

    # --- Promotion via alias (API MLflow 2.x moderne) ---
    client.set_registered_model_alias(
        name=model_name,
        alias=PRODUCTION_ALIAS,
        version=version_number,
    )

    logger.info(
        "✅ Modèle '%s' v%s promu avec alias '@%s'. RMSE=%.4f°C",
        model_name,
        version_number,
        PRODUCTION_ALIAS,
        rmse if rmse is not None else float("nan"),
    )

    # --- Vérification post-promotion ---
    promoted = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    logger.info(
        "Vérification : '@%s' pointe vers la version %s ✓",
        PRODUCTION_ALIAS,
        promoted.version,
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    promote_best_model()
