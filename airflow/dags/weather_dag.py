"""
weather_dag.py — DAG Airflow MLOps : pipeline complet en 5 étapes.

Pipeline :
    ingest_data → validate_data → train_model → promote_model → run_monitoring

Ce DAG orchestre l'intégralité du cycle de vie du modèle :
    1. Ingestion  : téléchargement et validation des données NOAA
    2. Validation : vérification de la qualité du dataset
    3. Entraînement : Random Forest avec feature engineering + tracking MLflow
    4. Promotion  : mise en production via alias @production (Quality Gate)
    5. Monitoring : rapport de dérive EvidentlyAI

Schedule : @daily (réentraînement automatique chaque nuit à minuit UTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arguments par défaut du DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "mlops",
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,   # Mettre True + configurer SMTP en production
    "email_on_retry": False,
    "depends_on_past": False,
}


# ---------------------------------------------------------------------------
# Fonctions Python appelées par chaque tâche
# ---------------------------------------------------------------------------

def task_ingest(**context) -> None:
    """Tâche 1 — Ingestion et téléchargement des données."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from ingestion import ingest_weather_data
    from pathlib import Path

    df = ingest_weather_data(output_path=Path("/opt/airflow/data/raw_weather.csv"))
    logger.info("Ingestion réussie : %d lignes.", len(df))

    # Passer le nombre de lignes à la tâche suivante via XCom
    context["ti"].xcom_push(key="n_rows", value=len(df))


def task_validate(**context) -> None:
    """Tâche 2 — Validation de la qualité des données."""
    import pandas as pd
    from pathlib import Path

    data_path = Path("/opt/airflow/data/raw_weather.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier manquant : {data_path}")

    df = pd.read_csv(data_path)

    # Validations
    assert "temp" in df.columns, "Colonne 'temp' manquante"
    assert len(df) > 100, f"Dataset trop petit : {len(df)} lignes"
    assert df["temp"].isnull().sum() == 0, "Valeurs nulles détectées dans 'temp'"

    null_rate = df.isnull().mean().max()
    assert null_rate < 0.05, f"Taux de nulls trop élevé : {null_rate:.1%}"

    logger.info(
        "Validation OK — %d lignes, taux de nulls=%.2f%%",
        len(df), null_rate * 100,
    )


def task_train(**context) -> None:
    """Tâche 3 — Entraînement du modèle Random Forest."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from train import train

    train()
    logger.info("Entraînement terminé.")


def task_promote(**context) -> None:
    """Tâche 4 — Promotion du meilleur modèle en production."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from promote import promote_best_model

    promote_best_model()
    logger.info("Promotion vers @production terminée.")


def task_monitor(**context) -> None:
    """Tâche 5 — Rapport de dérive des données EvidentlyAI."""
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from monitoring import run_monitoring

    result = run_monitoring()
    logger.info(
        "Monitoring terminé. Dérive détectée : %s (share=%.1f%%)",
        result["drift_detected"],
        result.get("drift_share", 0) * 100,
    )

    if result["drift_detected"]:
        logger.warning(
            "⚠️ Dérive significative détectée. "
            "Le prochain cycle de réentraînement corrigera le modèle."
        )


# ---------------------------------------------------------------------------
# Définition du DAG
# ---------------------------------------------------------------------------
with DAG(
    dag_id="NOAA_Weather_MLOps_Pipeline",
    description="Pipeline MLOps complet : ingestion → validation → entraînement → promotion → monitoring",
    default_args=default_args,
    schedule="@daily",          # Airflow 2.4+ — remplace schedule_interval
    catchup=False,
    max_active_runs=1,          # Évite les exécutions parallèles
    tags=["mlops", "weather", "random-forest", "production"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest,
        doc_md="Télécharge et valide les données météo depuis la source NOAA-compatible.",
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate,
        doc_md="Vérifie la qualité structurelle du dataset (colonnes, nulls, volume).",
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train,
        doc_md="Entraîne le Random Forest avec feature engineering et log les métriques dans MLflow.",
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote,
        doc_md="Promeut la dernière version en @production si le Quality Gate (RMSE) est passé.",
    )

    monitor = PythonOperator(
        task_id="run_monitoring",
        python_callable=task_monitor,
        doc_md="Génère le rapport de dérive EvidentlyAI entre données de référence et données récentes.",
    )

    # Pipeline séquentiel : chaque étape dépend de la précédente
    ingest >> validate >> train >> promote >> monitor
