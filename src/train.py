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
    # Définition du chemin local pour éviter les erreurs de permission '/'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_weather.csv")
    mlruns_dir = os.path.join(base_dir, "mlruns")
    
    # Configuration du tracking URI vers le dossier local du projet
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    try:
        mlflow.set_experiment("NOAA_Weather_Project")
        
        with mlflow.start_run(run_name="Registry_Initialization"):
            df = pd.read_csv(data_path)
            df['target'] = df['temp'].shift(-1)
            df = df.dropna()
            
            X = df[['temp']]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            mse = mean_squared_error(y_test, model.predict(X_test))
            mlflow.log_metric("mse", mse)
            
            # Enregistrement CRUCIAL pour éviter l'erreur "Model not found"
            mlflow.sklearn.log_model(
                sk_model=model, 
                artifact_path="weather_model",
                registered_model_name="Weather_RF_Model"
            )
            
            logger.info(f"🚀 Modèle enregistré avec succès dans le registre local ! MSE: {mse:.4f}")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement : {e}")

if __name__ == "__main__":
    train()