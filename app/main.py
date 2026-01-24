from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import os

app = FastAPI(title="NOAA Weather Expert API")

# Configuration dynamique du registre local
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlruns_dir = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_dir}")

def load_production_model():
    """Charge la version certifiée 'Production' du registre"""
    try:
        model_uri = "models:/Weather_RF_Model/Production"
        print(f"📡 Chargement du modèle depuis le registre : {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"⚠️ Erreur chargement Production : {e}")
        return None

# Chargement au démarrage
model = load_production_model()

@app.get("/")
def health():
    return {
        "status": "online", 
        "production_model_ready": model is not None,
        "registry_path": mlruns_dir
    }

@app.post("/predict")
def predict(temp_today: float):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle Production non chargé")
    
    input_df = pd.DataFrame([[temp_today]], columns=['temp'])
    prediction = model.predict(input_df)
    
    return {
        "input_temp": temp_today,
        "prediction_tomorrow": round(float(prediction[0]), 2),
        "source": "Model Registry (Production Stage)"
    }