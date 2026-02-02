import requests
import sys

def test_prediction():
    url = "http://localhost:8001/predict"
    params = {"temp_today": 25.0}
    
    try:
        response = requests.post(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "prediction_tomorrow" in data and isinstance(data["prediction_tomorrow"], (int, float)):
            print(f"✅ Test Réussi ! Prédiction reçue : {data['prediction_tomorrow']}")
            sys.exit(0)
        else:
            print("❌ Erreur : Format de réponse invalide.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Échec du test : Impossible de contacter l'API. Erreur : {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_prediction()
