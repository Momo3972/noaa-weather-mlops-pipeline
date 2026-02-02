import mlflow
import mlflow.sklearn
import pandas as pd
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # UTILISATION DU SYSTÈME DE FICHIERS DIRECT (FILE://)
    # Cela évite les erreurs de permission /opt/mlflow
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_dir = os.path.join(base_dir, "mlruns")
    
    # On pointe directement vers le dossier local
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    logger.info(f"📁 Tracking local direct (SÉCURISÉ) : {mlruns_dir}")

    try:
        mlflow.set_experiment("NOAA_Weather_Pipeline")
        
        with mlflow.start_run(run_name="SUCCESSFUL_FINAL_RUN"):
            df = pd.read_csv(os.path.join(base_dir, "data", "raw_weather.csv"))
            df['target'] = df['temp'].shift(-1)
            df = df.dropna()
            
            X, y = df[['temp']], df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            mse = mean_squared_error(y_test, model.predict(X_test))
            mlflow.log_metric("mse", mse)
            
            # ENREGISTREMENT DU MODÈLE
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="weather_model",
                registered_model_name="Weather_RF_Model"
            )
            
            logger.info(f"🚀 RÉUSSITE TOTALE ! MSE: {mse:.4f}")

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")
        raise e

if __name__ == "__main__":
    train()