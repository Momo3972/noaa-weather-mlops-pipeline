from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/Health") # Vérifie le endpoint de santé
    assert response.status_code == 200

def test_prediction():
    # Test avec le paramètre attendu par votre API : temp_today
    response = client.post("/predict?temp_today=22.5")
    assert response.status_code == 200
    assert "prediction" in response.json()