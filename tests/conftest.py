"""
conftest.py — Fixtures pytest partagées entre tous les tests.

Ce fichier est automatiquement chargé par pytest avant l'exécution des tests.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ajoute la racine du projet au path Python pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session", autouse=True)
def mock_mlflow_model():
    """
    Fixture de session : remplace le modèle MLflow par un mock.
    Permet de tester l'API sans avoir MLflow démarré ni de modèle enregistré.
    Appliquée automatiquement à tous les tests (autouse=True).
    """
    mock_model = MagicMock()
    # Le mock retourne une prédiction réaliste (température en °C)
    mock_model.predict.return_value = [18.75]

    with patch("app.main.load_production_model", return_value=mock_model):
        with patch("app.main.model", mock_model):
            yield mock_model


@pytest.fixture(scope="session")
def api_client(mock_mlflow_model):
    """Fournit un client TestClient FastAPI avec le modèle mocké."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client
