import mlflow
from mlflow.tracking import MlflowClient
import os

def promote_latest_to_production(model_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_dir = os.path.join(base_dir, "mlruns")
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    client = MlflowClient()
    
    # Récupérer la dernière version du registre local
    try:
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            print(f"⚠️ Aucune version 'None' trouvée pour {model_name}.")
            return

        latest_v = versions[0].version
        
        # Transition vers Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✅ Modèle {model_name} v{latest_v} est maintenant en PRODUCTION !")
    except Exception as e:
        print(f"❌ Erreur de promotion : {e}")

if __name__ == "__main__":
    promote_latest_to_production("Weather_RF_Model")