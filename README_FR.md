# NOAA Weather — Pipeline MLOps Industriel End-to-End

> Cette documentation est également disponible en anglais : [README.md](README.md)

## Présentation générale

Ce projet met en œuvre un pipeline MLOps complet, industrialisé et orienté production pour la prévision de températures à partir des données météorologiques NOAA.  
Il couvre l’intégralité du cycle de vie d’un modèle de machine learning, depuis l’ingestion automatisée des données jusqu’au déploiement via API, en passant par le tracking des expériences, la gouvernance des modèles, l’orchestration et la surveillance de la dérive des données.

L’ensemble de l’infrastructure est conteneurisé, orchestré et automatisé, conformément aux standards MLOps en environnement professionnel.

---

## Objectifs & Valeur Métier

Ce projet vise à démontrer la conception d’un **système ML robuste, traçable et observable** :

- Automatisation complète - De l’ingestion NOAA à la mise en production du modèle
- Gouvernance des modèles - Suivi des expériences et gestion du cycle de vie via MLflow Model Registry
- Observabilité des données - Détection continue de la dérive des données avec EvidentlyAI
- Qualité production - CI/CD, tests, Dockerisation et reproductibilité
- Orchestration opérationnelle - Réentraînement planifié et workflows gérés par Apache Airflow

---

## Organisation du dépôt

```text
noaa-weather-mlops-pipeline/
├── .github/workflows/   # Pipelines CI/CD (tests, linting, push Docker)
├── airflow/             # Configuration Airflow et DAGs
├── app/                 # Service FastAPI et Dockerfile applicatif
├── data/                # Données NOAA (raw_weather.csv)
├── docs/
│   └── assets/          # Captures d’écran et preuves d’exécution
├── mlruns/              # Backend MLflow local (expériences & artefacts)
├── monitoring/          # Rapports de dérive EvidentlyAI
├── src/                 # Scripts du pipeline (ingestion, entraînement, promotion, monitoring)
├── tests/               # Tests unitaires et smoke tests de l’API
├── .gitignore           # Règles d’exclusion Git (env, artefacts, fichiers locaux)
├── docker-compose.yml   # Orchestration de la stack multi-conteneurs
├── requirements.txt     # Dépendances Python
├── README.md            # Documentation principale (anglais)
└── README_FR.md         # Documentation française
```

---

## Composants techniques

### 1. Ingestion & Préparation des données (`src/ingestion.py`)

- Récupération automatisée des données NOAA
- Nettoyage des valeurs manquantes
- Feature engineering et validation des jeux de données

### 2. Entraînement et Tracking des expériences (`src/train.py`)

- Modèle de régression Random Forest
- Journalisation des hyperparamètres, métriques (MSE) et artefacts dans MLflow

### 3. Promotion automatique du modèle (`src/promote.py`)

- Utilisation de `MlflowClient` pour identifier la dernière version validée
- Attribution automatique de l’alias `@production` dans le Model Registry MLflow

### 4. API de prédiction (`app/main.py`)

- Service FastAPI chargeant dynamiquement le modèle en production
- Endpoint `/predict` pour l’inférence
- Endpoint `/health` pour la supervision du service

### 5. Surveillance de la dérive des données (`src/monitoring.py`)

- Comparaison des données d’inférence avec les données de référence
- Génération automatique de rapports de dérive avec **EvidentlyAI**

---

## Figures et preuves d’exécution

Les captures ci-dessous sont disponibles dans `docs/assets/` et s’affichent directement sur GitHub

### Registre MLflow — Modèle certifié en production
![MLflow Registry](docs/assets/mlflow-registry.png)

### CI/CD — Pipeline GitHub Actions validé
![GitHub Actions](docs/assets/github-actions-success.png)

### Airflow — DAG de réentraînement planifié
![Airflow DAG](docs/assets/interface-apache-airflow.png)

### FastAPI — Documentation Swagger interactive
![Swagger UI](docs/assets/swagger-noaa.png)

### MLflow — Historique des runs et métriques
![MLflow Runs](docs/assets/mlflow-runs.png)

### Docker — Stack multi-conteneurs opérationnelle
![Docker Containers](docs/assets/docker-containers.png)

---

## Démarrage rapide

### 1. Initialisation des permissions (WSL / Linux)

```bash
docker-compose down
sudo rm -rf mlruns/ && mkdir -p mlruns/artifacts
chmod -R 777 mlruns/
```

### 2. Lancement de la stack

```bash
docker-compose up -d --build
```

### 3. Accès aux services

- **API FastAPI (Swagger)** : http://localhost:8001/docs
- **Interface MLflow** : http://localhost:5000
- **Interface Airflow** : http://localhost:8080  
  - Identifiants : `admin / admin`

---

## Tests & CI/CD

- Tests unitaires et tests API exécutés à chaque push
- Linting Dockerfile (Hadolint)
- Build et publication automatique des images Docker
- Pipeline CI/CD orchestré via GitHub Actions

---

## Concepts MLOps couverts

- Expériences reproductibles
- Versionnement et promotion des modèles
- Orchestration de pipelines ML
- Déploiement et serving de modèles
- Monitoring et détection de dérive
- Infrastructure as Code

---

## Auteur

**Mohamed Lamine OULD BOUYA**  
Data Engineering · MLOps · Machine Learning
