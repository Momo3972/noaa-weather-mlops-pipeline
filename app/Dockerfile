# Utilisation d'une image Python légère
FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances minimales
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application et des artefacts nécessaires
COPY app/ /app/app/
COPY mlruns/ /app/mlruns/

# Exposition du port de l'API
EXPOSE 8000

# Commande de lancement de production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]