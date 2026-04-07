"""
monitoring.py — Surveillance de la dérive des données avec EvidentlyAI.

Stratégie de comparaison :
    - Référence  : première moitié du dataset (données historiques stables)
    - Courant    : dernière moitié du dataset (données récentes)
    - La colonne 'date' est EXCLUE de l'analyse (identifiant temporel unique,
      toujours en dérive par construction — non informatif)
    - Seule la colonne 'temp' est analysée pour la dérive de distribution

Un seuil de dérive est défini : si drift_score > DRIFT_THRESHOLD, une alerte
est loguée avec le niveau WARNING pour déclencher un réentraînement.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "raw_weather.csv")))
OUTPUT_DIR = Path(os.getenv("MONITORING_OUTPUT_DIR", str(BASE_DIR / "monitoring")))

# Colonnes à exclure de l'analyse de dérive
COLUMNS_TO_EXCLUDE = ["date"]

# Seuil d'alerte : proportion de colonnes en dérive au-dessus de laquelle
# on considère que les données ont changé significativement
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))

# Taille de la fenêtre de comparaison (nombre de jours dans chaque période)
WINDOW_SIZE = int(os.getenv("MONITORING_WINDOW_SIZE", "365"))


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def run_monitoring(
    data_path: Path = DATA_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Génère un rapport de dérive des données entre deux fenêtres temporelles.

    Fenêtres :
        - Référence : données de la première période (passé stable)
        - Courant   : données de la dernière période (période récente)

    La comparaison de mêmes périodes saisonnières évite les faux positifs
    liés à la saisonnalité naturelle des températures.

    Args:
        data_path: Chemin vers le CSV brut.
        output_dir: Dossier de sortie pour le rapport HTML et le JSON.

    Returns:
        Dictionnaire avec les résultats de dérive (drift_detected, drift_share).

    Raises:
        FileNotFoundError: Si le fichier de données est introuvable.
        ValueError: Si les données sont insuffisantes pour la comparaison.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Données introuvables : {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Chargement ---
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < WINDOW_SIZE * 2:
        raise ValueError(
            f"Données insuffisantes ({len(df)} lignes) pour comparer "
            f"deux fenêtres de {WINDOW_SIZE} jours."
        )

    # --- Sélection des fenêtres ---
    # Référence : première fenêtre dans le temps
    # Courant   : dernière fenêtre dans le temps
    reference = df.iloc[:WINDOW_SIZE].copy()
    current = df.iloc[-WINDOW_SIZE:].copy()

    logger.info(
        "Fenêtre de référence : %s → %s (%d lignes)",
        reference["date"].min().date(),
        reference["date"].max().date(),
        len(reference),
    )
    logger.info(
        "Fenêtre courante     : %s → %s (%d lignes)",
        current["date"].min().date(),
        current["date"].max().date(),
        len(current),
    )

    # --- Suppression des colonnes non analysables ---
    # La colonne 'date' est un identifiant temporel unique : toujours en dérive.
    # On analyse uniquement les features numériques pertinentes.
    cols_to_drop = [c for c in COLUMNS_TO_EXCLUDE if c in df.columns]
    reference_clean = reference.drop(columns=cols_to_drop)
    current_clean = current.drop(columns=cols_to_drop)

    logger.info("Colonnes analysées pour la dérive : %s", list(reference_clean.columns))

    # --- Rapport EvidentlyAI ---
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_clean, current_data=current_clean)

    # --- Sauvegarde HTML ---
    html_path = output_dir / "drift_report.html"
    report.save_html(str(html_path))
    logger.info("Rapport HTML sauvegardé : %s", html_path)

    # --- Export JSON pour automatisation ---
    metrics_dict = report.as_dict()
    json_path = output_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    logger.info("Métriques JSON sauvegardées : %s", json_path)

    # --- Interprétation et alerte ---
    drift_result = _extract_drift_summary(metrics_dict)

    if drift_result["drift_detected"]:
        logger.warning(
            "⚠️  DÉRIVE DÉTECTÉE — %.0f%% des features ont dérivé "
            "(seuil=%.0f%%). Un réentraînement est recommandé.",
            drift_result["drift_share"] * 100,
            DRIFT_THRESHOLD * 100,
        )
    else:
        logger.info(
            "✅ Pas de dérive significative — %.0f%% des features affectées "
            "(seuil=%.0f%%).",
            drift_result["drift_share"] * 100,
            DRIFT_THRESHOLD * 100,
        )

    return drift_result


def _extract_drift_summary(metrics_dict: dict) -> dict:
    """Extrait le résumé de dérive depuis le dictionnaire Evidently."""
    try:
        dataset_drift = metrics_dict["metrics"][0]["result"]
        return {
            "drift_detected": dataset_drift.get("dataset_drift", False),
            "drift_share": dataset_drift.get("share_of_drifted_columns", 0.0),
            "drifted_columns": dataset_drift.get("number_of_drifted_columns", 0),
            "total_columns": dataset_drift.get("number_of_columns", 0),
        }
    except (KeyError, IndexError) as exc:
        logger.error("Impossible d'extraire le résumé de dérive : %s", exc)
        return {"drift_detected": False, "drift_share": 0.0}


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_monitoring()
