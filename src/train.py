"""
train.py — Entraînement du modèle de prévision de température.

Pipeline :
    1. Chargement des données brutes
    2. Feature engineering (lags, rolling stats, saisonnalité)
    3. Split temporel strict (pas de data leakage)
    4. Entraînement Random Forest
    5. Logging MLflow (métriques, hyperparamètres, artefacts, feature importance)
    6. Enregistrement dans le Model Registry
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Workaround WSL2/DrvFs : shutil.copy2 échoue sur utime (timestamps nanoseconde
# non supportés sur filesystem Windows monté via DrvFs). On ignore l'erreur.
# ---------------------------------------------------------------------------
_orig_copystat = shutil.copystat

def _safe_copystat(src, dst, *, follow_symlinks=True):
    try:
        _orig_copystat(src, dst, follow_symlinks=follow_symlinks)
    except (PermissionError, OSError):
        pass  # utime non supporté sur DrvFs — le fichier est bien copié

shutil.copystat = _safe_copystat
# ---------------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_weather.csv"
MODEL_NAME = os.getenv("MODEL_NAME", "Weather_RF_Model")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://mlflow:5000")

# Hyperparamètres — modifiables via variables d'environnement
N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "200"))
MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "10"))
MIN_SAMPLES_LEAF = int(os.getenv("RF_MIN_SAMPLES_LEAF", "5"))
TEST_SIZE = float(os.getenv("TRAIN_TEST_SPLIT", "0.2"))
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Feature Engineering — cœur du pipeline Data Science
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features temporelles à partir de la colonne 'temp'.

    Features créées :
        - Lags          : temp_lag_1, temp_lag_2, temp_lag_7
        - Rolling stats : temp_rolling_mean_7, temp_rolling_std_7,
                          temp_rolling_mean_14
        - Saisonnalité  : day_of_year, month, week_of_year

    Args:
        df: DataFrame avec colonnes ['date', 'temp'], trié par date.

    Returns:
        DataFrame enrichi avec la target (temp du lendemain).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Target : température du lendemain ---
    df["target"] = df["temp"].shift(-1)

    # --- Lag features ---
    df["temp_lag_1"] = df["temp"].shift(1)
    df["temp_lag_2"] = df["temp"].shift(2)
    df["temp_lag_7"] = df["temp"].shift(7)

    # --- Rolling statistics ---
    df["temp_rolling_mean_7"] = df["temp"].shift(1).rolling(7).mean()
    df["temp_rolling_std_7"] = df["temp"].shift(1).rolling(7).std()
    df["temp_rolling_mean_14"] = df["temp"].shift(1).rolling(14).mean()

    # --- Saisonnalité ---
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Supprimer les lignes avec NaN (dues aux lags et rolling)
    df = df.dropna()

    return df


FEATURE_COLUMNS = [
    "temp",
    "temp_lag_1", "temp_lag_2", "temp_lag_7",
    "temp_rolling_mean_7", "temp_rolling_std_7", "temp_rolling_mean_14",
    "day_of_year", "month", "week_of_year",
]


# ---------------------------------------------------------------------------
# Split temporel — OBLIGATOIRE pour les séries temporelles
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split chronologique strict — aucune fuite de données futures.

    IMPORTANT : train_test_split(shuffle=True) est interdit pour les séries
    temporelles car il entraîne le modèle sur des données du futur.

    Args:
        df: DataFrame avec features et colonne 'target'.
        test_size: Proportion du jeu de test (derniers N% dans le temps).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    logger.info(
        "Split temporel : train=%d lignes, test=%d lignes (%.0f%% / %.0f%%)",
        len(train), len(test),
        (1 - test_size) * 100, test_size * 100,
    )

    return (
        train[FEATURE_COLUMNS],
        test[FEATURE_COLUMNS],
        train["target"],
        test["target"],
    )


# ---------------------------------------------------------------------------
# Entraînement principal
# ---------------------------------------------------------------------------

def train() -> None:
    """Lance l'entraînement complet avec tracking MLflow."""

    mlflow.set_tracking_uri(MLFLOW_URI)
    logger.info("MLflow tracking URI : %s", MLFLOW_URI)

    # --- Chargement des données ---
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {DATA_PATH}. "
            "Lancez d'abord src/ingestion.py."
        )

    df_raw = pd.read_csv(DATA_PATH)
    logger.info("Données chargées : %d lignes brutes.", len(df_raw))

    # --- Feature engineering ---
    df = build_features(df_raw)
    logger.info("Features construites : %d lignes exploitables.", len(df))

    # --- Split temporel ---
    X_train, X_test, y_train, y_test = temporal_train_test_split(df, TEST_SIZE)

    # --- Entraînement MLflow ---
    mlflow.set_experiment("NOAA_Weather_Pipeline")

    run_name = (
        f"rf_n{N_ESTIMATORS}_d{MAX_DEPTH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name):

        # Logging des hyperparamètres
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_features": len(FEATURE_COLUMNS),
            "features": ",".join(FEATURE_COLUMNS),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

        # Entraînement
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # --- Métriques complètes ---
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metrics({
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
        })

        logger.info(
            "Métriques | MSE=%.4f | RMSE=%.4f°C | MAE=%.4f°C | R²=%.4f",
            mse, rmse, mae, r2,
        )

        # --- Feature importance ---
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        importance_path = BASE_DIR / "monitoring" / "feature_importance.csv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        logger.info("Top 3 features :\n%s", importance_df.head(3).to_string(index=False))

        # --- Enregistrement du modèle ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="weather_model",
            registered_model_name=MODEL_NAME,
        )

        logger.info("Modèle enregistré sous '%s'. Run : %s", MODEL_NAME, run_name)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
