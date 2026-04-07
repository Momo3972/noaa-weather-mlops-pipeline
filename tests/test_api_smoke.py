"""
test_api_smoke.py — Smoke test d'intégration (API démarrée en local).

CE FICHIER N'EST PAS EXÉCUTÉ DANS LE CI (nécessite l'API locale sur :8001).
Usage : python tests/test_api_smoke.py  (quand docker-compose up est actif)

Différence avec test_api.py :
    - test_api.py   : tests unitaires avec modèle mocké — pour le CI
    - test_api_smoke.py : test d'intégration contre l'API réelle — en local
"""

from __future__ import annotations

import sys
import pytest
import requests


API_URL = "http://localhost:8001"


def is_api_running() -> bool:
    """Vérifie si l'API est accessible avant d'exécuter les smoke tests."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.mark.skipif(
    not is_api_running(),
    reason="API non disponible sur localhost:8001 — lancer docker-compose up d'abord",
)
class TestSmokeAPI:
    """Smoke tests contre l'API réelle (nécessite docker-compose up)."""

    def test_health_endpoint(self) -> None:
        response = requests.get(f"{API_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "online"

    def test_prediction_smoke(self) -> None:
        response = requests.post(
            f"{API_URL}/v1/predict",
            json={"temp_today": 25.0},
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction_tomorrow" in data
        assert isinstance(data["prediction_tomorrow"], (int, float))
        assert -50 <= data["prediction_tomorrow"] <= 60, (
            f"Prédiction hors bornes plausibles : {data['prediction_tomorrow']}°C"
        )


# ---------------------------------------------------------------------------
# Point d'entrée — utilisation directe (non pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not is_api_running():
        print("❌ API non disponible. Lancez 'make up' d'abord.")
        sys.exit(1)

    print(f"🔥 Smoke test contre {API_URL}...")
    r = requests.post(f"{API_URL}/v1/predict", json={"temp_today": 25.0}, timeout=5)
    r.raise_for_status()
    print(f"✅ Prédiction reçue : {r.json()['prediction_tomorrow']}°C")
