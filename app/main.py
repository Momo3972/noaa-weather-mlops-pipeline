from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="NOAA Weather Expert API")

# --- CONFIGURATION MLFLOW DYNAMIQUE ---
# On utilise le nom du service 'mlflow' défini dans docker-compose au lieu d'une IP
MLFLOW_SERVER = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_SERVER)

def load_production_model():
    """Charge la version certifiée 'production' du registre MLflow"""
    try:
        # On utilise l'alias @production que vous avez activé dans l'interface
        model_uri = "models:/Weather_RF_Model@production"
        print(f"📡 Tentative de chargement du modèle : {model_uri}")
        
        # Chargement du modèle Random Forest
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"⚠️ Erreur critique de chargement : {e}")
        return None

# Chargement du modèle au démarrage de l'application
model = load_production_model()

@app.get("/Health")
def health():
    """Vérifie l'état de l'API et la disponibilité du modèle"""
    return {
        "status": "online", 
        "production_model_ready": model is not None,
        "mlflow_server": MLFLOW_SERVER,
        "model_path": "models:/Weather_RF_Model@production"
    }

@app.post("/predict")
def predict(temp_today: float):
    """Effectue une prédiction basée sur la température du jour"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Le modèle n'est pas encore chargé. Vérifiez la connexion avec MLflow."
        )
    
    try:
        # Préparation des données pour le Random Forest
        input_df = pd.DataFrame([[temp_today]], columns=['temp'])
        prediction = model.predict(input_df)
        
        return {
            "input_temp": temp_today,
            "prediction_tomorrow": round(float(prediction[0]), 2),
            "source": "MLflow Model Registry (Stage: production)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")