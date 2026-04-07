"""
test_api.py — Tests unitaires de l'API FastAPI.

Ces tests vérifient le comportement de l'API de bout en bout
sans dépendance à MLflow (modèle mocké via conftest.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Tests du endpoint /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests du endpoint GET /health."""

    def test_health_returns_200(self, api_client: TestClient) -> None:
        """L'endpoint /health doit toujours retourner HTTP 200."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, api_client: TestClient) -> None:
        """La réponse /health doit contenir tous les champs attendus."""
        response = api_client.get("/health")
        data = response.json()

        required_fields = {"status", "model_ready", "model_name", "model_alias", "api_version"}
        assert required_fields.issubset(data.keys()), (
            f"Champs manquants : {required_fields - data.keys()}"
        )

    def test_health_status_is_online(self, api_client: TestClient) -> None:
        """Le statut de l'API doit être 'online'."""
        response = api_client.get("/health")
        assert response.json()["status"] == "online"

    def test_health_model_ready(self, api_client: TestClient) -> None:
        """Le modèle doit être marqué comme prêt (mocké en session)."""
        response = api_client.get("/health")
        assert response.json()["model_ready"] is True


# ---------------------------------------------------------------------------
# Tests du endpoint /v1/predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    """Tests du endpoint POST /v1/predict."""

    def test_predict_returns_200_with_valid_input(self, api_client: TestClient) -> None:
        """Une requête valide doit retourner HTTP 200."""
        response = api_client.post("/v1/predict", json={"temp_today": 18.5})
        assert response.status_code == 200

    def test_predict_response_schema(self, api_client: TestClient) -> None:
        """La réponse doit contenir tous les champs du schéma WeatherPrediction."""
        response = api_client.post("/v1/predict", json={"temp_today": 18.5})
        data = response.json()

        required_fields = {
            "request_id", "input_temp", "prediction_tomorrow",
            "model_alias", "model_name", "timestamp",
        }
        assert required_fields.issubset(data.keys()), (
            f"Champs manquants : {required_fields - data.keys()}"
        )

    def test_predict_returns_float(self, api_client: TestClient) -> None:
        """La prédiction doit être un nombre flottant."""
        response = api_client.post("/v1/predict", json={"temp_today": 22.0})
        prediction = response.json()["prediction_tomorrow"]
        assert isinstance(prediction, (int, float))

    def test_predict_input_reflected(self, api_client: TestClient) -> None:
        """La température d'entrée doit être reflétée dans la réponse."""
        input_temp = 15.3
        response = api_client.post("/v1/predict", json={"temp_today": input_temp})
        assert response.json()["input_temp"] == input_temp

    def test_predict_with_lag_feature(self, api_client: TestClient) -> None:
        """La requête avec temp_lag_1 optionnel doit fonctionner."""
        response = api_client.post(
            "/v1/predict",
            json={"temp_today": 20.0, "temp_lag_1": 19.5},
        )
        assert response.status_code == 200

    # --- Tests de validation des entrées ---

    def test_predict_rejects_temp_too_high(self, api_client: TestClient) -> None:
        """Une température > 60°C doit retourner HTTP 422 (validation Pydantic)."""
        response = api_client.post("/v1/predict", json={"temp_today": 99.0})
        assert response.status_code == 422

    def test_predict_rejects_temp_too_low(self, api_client: TestClient) -> None:
        """Une température < -50°C doit retourner HTTP 422 (validation Pydantic)."""
        response = api_client.post("/v1/predict", json={"temp_today": -100.0})
        assert response.status_code == 422

    def test_predict_rejects_missing_body(self, api_client: TestClient) -> None:
        """Une requête sans corps doit retourner HTTP 422."""
        response = api_client.post("/v1/predict", json={})
        assert response.status_code == 422

    def test_predict_rejects_string_input(self, api_client: TestClient) -> None:
        """Une entrée non numérique doit retourner HTTP 422."""
        response = api_client.post("/v1/predict", json={"temp_today": "abc"})
        assert response.status_code == 422
