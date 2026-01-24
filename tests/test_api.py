from fastapi.testclient import TestClient
import sys
import os

# Ajout du chemin pour l'import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"