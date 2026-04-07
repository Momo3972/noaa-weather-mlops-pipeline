# ============================================================
# app/Dockerfile — API FastAPI de prédiction
# Python 3.11-slim (Python 3.9 est en fin de vie depuis oct. 2025)
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Mohamed Lamine OULD BOUYA"
LABEL description="FastAPI Prediction Service — NOAA Weather MLOps Pipeline"

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif uniquement (pas tout le repo)
COPY app/ .

# Création d'un utilisateur non-root
RUN groupadd --system appuser && \
    useradd --system --gid appuser --no-create-home appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
