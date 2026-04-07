"""
ingestion.py — Collecte et validation des données météorologiques.

Source : Dataset NOAA-compatible (températures quotidiennes, Melbourne 1981-1990).
Note   : Pour utiliser l'API NOAA CDO en production, remplacer l'URL par
         https://www.ncdc.noaa.gov/cdo-web/api/v2/data et fournir un token NOAA_API_TOKEN.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration du logger — production-ready (pas de print)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "daily-min-temperatures.csv"
)
EXPECTED_COLUMNS = {"date", "temp"}
MIN_ROWS = 100


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------

def ingest_weather_data(
    output_path: Path = Path("data/raw_weather.csv"),
    url: str = DATA_URL,
) -> pd.DataFrame:
    """
    Télécharge, valide et sauvegarde les données météo.

    Args:
        output_path: Chemin de sortie du CSV brut.
        url: URL source des données.

    Returns:
        DataFrame validé avec colonnes ['date', 'temp'].

    Raises:
        ValueError: Si les données téléchargées sont invalides ou incomplètes.
        requests.RequestException: Si le téléchargement échoue.
    """
    logger.info("Démarrage de l'ingestion depuis : %s", url)

    # --- Téléchargement ---
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        logger.error("Échec du téléchargement : %s", exc)
        raise

    # --- Normalisation des colonnes ---
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "temp"})

    # --- Validation structurelle ---
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le CSV source : {missing_cols}")

    # --- Validation des types ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

    # --- Validation de la qualité ---
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Valeurs nulles détectées :\n%s", null_counts[null_counts > 0])
        df = df.dropna()
        logger.info("Lignes nulles supprimées. Lignes restantes : %d", len(df))

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Données insuffisantes : {len(df)} lignes (minimum requis : {MIN_ROWS})"
        )

    # --- Validation des bornes de température ---
    temp_min, temp_max = df["temp"].min(), df["temp"].max()
    if not (-60.0 <= temp_min and temp_max <= 60.0):
        logger.warning(
            "Températures hors bornes plausibles : min=%.1f, max=%.1f", temp_min, temp_max
        )

    # --- Sauvegarde ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Ingestion terminée. %d lignes sauvegardées dans '%s'. "
        "Période : %s → %s | Temp. moy. : %.2f°C",
        len(df),
        output_path,
        df["date"].min().date(),
        df["date"].max().date(),
        df["temp"].mean(),
    )

    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ingest_weather_data()
