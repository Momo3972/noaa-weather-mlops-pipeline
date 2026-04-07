# ============================================================
# Dockerfile — MLflow Tracking Server
# Python 3.11-slim (Python 3.9 est en fin de vie depuis oct. 2025)
# ============================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Mohamed Lamine OULD BOUYA"
LABEL description="MLflow Tracking Server — NOAA Weather MLOps Pipeline"

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mlflow==2.16.2

# Patch DrvFs/WSL2 : ignore les erreurs utime/chmod sur filesystem Windows
COPY sitecustomize.py /usr/local/lib/python3.11/sitecustomize.py

# Création d'un utilisateur non-root (bonne pratique de sécurité)
RUN groupadd --system mlflow && \
    useradd --system --gid mlflow --no-create-home mlflow

# Répertoire MLflow avec les bonnes permissions
RUN mkdir -p /mlruns && chown -R mlflow:mlflow /mlruns

USER mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "/mlruns", \
     "--default-artifact-root", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
