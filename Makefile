# ============================================================
# Makefile — NOAA Weather MLOps Pipeline
# Commandes rapides pour développement et opérations
# Usage : make <commande>
# ============================================================

.PHONY: help setup up down restart logs test train ingest promote monitor clean

# Couleurs terminal
BOLD  := \033[1m
GREEN := \033[32m
RESET := \033[0m

## ---------------------------------------------------------------
## Aide
## ---------------------------------------------------------------
help: ## Affiche cette aide
	@echo ""
	@echo "$(BOLD)NOAA Weather MLOps Pipeline$(RESET)"
	@echo "$(GREEN)Usage : make <commande>$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

## ---------------------------------------------------------------
## Environnement
## ---------------------------------------------------------------
setup: ## Crée l'environnement virtuel et installe les dépendances
	@echo "$(BOLD)Création de l'environnement Python 3.12...$(RESET)"
	python3.11 -m venv venv
	./venv/bin/pip install --upgrade pip
	@echo "Installation des packages ML & Data..."
	./venv/bin/pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.4.2 requests==2.32.3 python-dotenv==1.0.1
	@echo "Installation de MLflow..."
	./venv/bin/pip install mlflow==2.16.2
	@echo "Installation de FastAPI & tests..."
	./venv/bin/pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0" python-multipart==0.0.9 pytest==8.2.1 httpx==0.27.0
	@echo "Installation d'Evidently..."
	./venv/bin/pip install evidently==0.4.33
	@echo "$(GREEN)✅ Environnement prêt. Activez avec : source venv/bin/activate$(RESET)"

env: ## Copie .env.example vers .env (si .env n'existe pas)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✅ .env créé depuis .env.example. Renseignez vos valeurs.$(RESET)"; \
	else \
		echo ".env existe déjà."; \
	fi

## ---------------------------------------------------------------
## Docker
## ---------------------------------------------------------------
up: ## Démarre la stack complète (build + run en arrière-plan)
	@echo "$(BOLD)Démarrage de la stack MLOps...$(RESET)"
	docker-compose up -d --build
	@echo "$(GREEN)✅ Stack démarrée. Services disponibles :$(RESET)"
	@echo "   API FastAPI  : http://localhost:8001/docs"
	@echo "   MLflow UI    : http://localhost:5000"
	@echo "   Airflow UI   : http://localhost:8080 (admin/admin)"

down: ## Arrête la stack
	docker-compose down

restart: ## Redémarre la stack
	docker-compose down && docker-compose up -d --build

logs: ## Affiche les logs en temps réel
	docker-compose logs -f

logs-api: ## Logs de l'API uniquement
	docker-compose logs -f weather-api

logs-mlflow: ## Logs de MLflow uniquement
	docker-compose logs -f mlflow

## ---------------------------------------------------------------
## Pipeline ML (URI file:// pour exécution hors Docker)
## ---------------------------------------------------------------
# En local (hôte WSL), on écrit directement dans ./mlruns via file://
# Le serveur Docker MLflow lit le même dossier via son volume mount → UI synchronisée
# En production (dans Docker), les scripts utilisent http://mlflow:5000
LOCAL_MLFLOW_URI := file://$(CURDIR)/mlruns

ingest: ## Lance l'ingestion des données
	@echo "$(BOLD)Ingestion des données...$(RESET)"
	python src/ingestion.py

train: ## Lance l'entraînement du modèle
	@echo "$(BOLD)Entraînement du modèle Random Forest...$(RESET)"
	MLFLOW_TRACKING_URI=$(LOCAL_MLFLOW_URI) python src/train.py

promote: ## Promeut le meilleur modèle en production
	@echo "$(BOLD)Promotion du modèle...$(RESET)"
	MLFLOW_TRACKING_URI=$(LOCAL_MLFLOW_URI) python src/promote.py

monitor: ## Génère le rapport de dérive des données
	@echo "$(BOLD)Analyse de dérive (EvidentlyAI)...$(RESET)"
	MLFLOW_TRACKING_URI=$(LOCAL_MLFLOW_URI) python src/monitoring.py

pipeline: ingest train promote monitor ## Exécute le pipeline complet en local
	@echo "$(GREEN)✅ Pipeline complet terminé.$(RESET)"

## ---------------------------------------------------------------
## Tests
## ---------------------------------------------------------------
test: ## Lance les tests unitaires (pytest)
	@echo "$(BOLD)Exécution des tests unitaires...$(RESET)"
	pytest tests/test_api.py -v --tb=short

test-smoke: ## Lance les smoke tests (nécessite make up)
	@echo "$(BOLD)Smoke tests contre l'API locale...$(RESET)"
	python tests/test_api_smoke.py

test-all: test test-smoke ## Lance tous les tests

## ---------------------------------------------------------------
## Nettoyage
## ---------------------------------------------------------------
clean: ## Supprime les caches Python et fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé.$(RESET)"

clean-mlruns: ## Supprime les données MLflow locales (ATTENTION : irréversible)
	@echo "$(BOLD)⚠️  Suppression de mlruns/...$(RESET)"
	rm -rf mlruns/
	mkdir -p mlruns
	chmod 777 mlruns/
	@echo "$(GREEN)✅ mlruns/ réinitialisé.$(RESET)"
