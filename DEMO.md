# Live Demo Guide - NOAA Weather MLOps Pipeline

This guide walks you through running the complete MLOps pipeline end-to-end, from data ingestion to live model serving.
The entire stack runs locally via Docker - no cloud account or external service required.

---

## Prerequisites

| Tool             | Version   | Check command           |
|------------------|-----------|-------------------------|
| Docker Desktop   | ≥ 24.0    | `docker --version`      |
| Docker Compose   | ≥ 2.20    | `docker compose version`|
| Git              | any       | `git --version`         |

> **Windows users:** Docker Desktop with WSL2 backend is required
> **macOS / Linux:** Standard Docker Desktop or Docker Engine works as-is

---

## Step 1 - Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/noaa-weather-mlops-pipeline.git
cd noaa-weather-mlops-pipeline
```

---

## Step 2 - Configure the Environment

```bash
cp .env.example .env
```

No modifications needed. The default values are fully operational for a local demo.

---

## Step 3 - (WSL2 / Windows only) Initialize Volume Permissions

> Skip this step on macOS or native Linux

```bash
mkdir -p mlruns/artifacts
chmod -R 777 mlruns/
```

---

## Step 4 - Start the Full Stack

```bash
docker-compose up -d --build
```

This command builds and starts 5 services:

| Service                  | Role                                | URL                          |
|--------------------------|-------------------------------------|------------------------------|
| `mlflow_server`          | Experiment tracking & model registry| http://localhost:5000        |
| `weather_prediction_api` | FastAPI inference service           | http://localhost:8001/docs   |
| `airflow_webserver`      | Pipeline orchestration UI           | http://localhost:8080        |
| `airflow_scheduler`      | DAG scheduler (triggers at midnight)| —                            |
| `airflow_db` (postgres)  | Airflow metadata backend            | —                            |

**Wait ~3 minutes** for all health checks to pass, then verify:

```bash
docker ps
```

All containers should show `healthy` status

---

## Step 5 - Open the Interfaces

Open the following URLs in your browser:

- **Airflow UI** → http://localhost:8080 — Login: `admin` / `admin`
- **MLflow UI** → http://localhost:5000
- **FastAPI Swagger** → http://localhost:8001/docs

---

## Step 6 - Trigger the MLOps Pipeline

The DAG is scheduled to run automatically every day at midnight UTC.
To run it immediately for the demo:

1. Go to **http://localhost:8080**
2. Log in with `admin` / `admin`
3. Find the DAG named **`NOAA_Weather_MLOps_Pipeline`**
4. Click the **▶ Trigger DAG** button (top right)
5. Confirm the trigger

The pipeline executes 5 sequential tasks:

```
ingest_data -> validate_data -> train_model -> promote_model -> run_monitoring
```

| Task             | What it does                                                        | Duration  |
|------------------|---------------------------------------------------------------------|-----------|
| `ingest_data`    | Downloads NOAA-compatible temperature data, saves `raw_weather.csv` | ~5s       |
| `validate_data`  | Checks column presence, row count, null rate                        | ~2s       |
| `train_model`    | Feature engineering + Random Forest training + MLflow logging       | ~40s      |
| `promote_model`  | Quality Gate check (RMSE ≤ 3.0°C) → promotes model to `@production`| ~5s       |
| `run_monitoring` | EvidentlyAI drift report comparing reference vs current data        | ~10s      |

All 5 tasks should turn **green** within ~2 minutes

---

## Step 7 - Verify the Model Registry in MLflow

1. Go to **http://localhost:5000**
2. Click **Experiments** -> `NOAA_Weather_Pipeline`
3. The new run appears with source `airflow` - click it to inspect:
   - `rmse` (target: ≤ 3.0°C to pass the Quality Gate)
   - `mae`, `r2`, `mse`
   - Feature importance artifact
4. Click **Models** -> `Weather_RF_Model`:
   - `@production` alias points to the latest promoted version
   - This alias is what the API uses to load the model - no code change needed on new versions

---

## Step 8 - Test the Prediction API

### Option A - Swagger UI (no terminal needed)

Go to **http://localhost:8001/docs**

1. Click **GET /health** -> **Try it out** → **Execute**
   Expected response:
   ```json
   {
     "status": "online",
     "model_ready": true,
     "model_name": "Weather_RF_Model",
     "model_alias": "production"
   }
   ```

2. Click **POST /v1/predict** -> **Try it out** -> fill in the body:
   ```json
   {
     "temp_today": 18.5,
     "temp_lag_1": 17.2
   }
   ```
   Expected response:
   ```json
   {
     "request_id": "a1b2c3d4",
     "input_temp": 18.5,
     "prediction_tomorrow": 18.73,
     "model_alias": "production",
     "model_name": "Weather_RF_Model",
     "timestamp": "2026-04-06T14:00:00.000000Z"
   }
   ```

### Option B - curl

```bash
# Health check
curl http://localhost:8001/health

# Temperature prediction
curl -X POST http://localhost:8001/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"temp_today": 18.5, "temp_lag_1": 17.2}'
```

---

## Step 9 - Inspect the Drift Monitoring Report

The `run_monitoring` task generates an EvidentlyAI HTML report comparing the distribution
of the first 365 days (reference) vs the last 365 days (current) of the dataset

```bash
# Open the drift report in your browser
open monitoring/drift_report.html        # macOS
xdg-open monitoring/drift_report.html   # Linux / WSL2
start monitoring/drift_report.html      # Windows
```

The report shows whether the temperature distribution has significantly shifted,
which would trigger a retraining recommendation on the next pipeline run

---

## Step 10 - Run the Test Suite

Unit tests run without Docker (mock model - no MLflow dependency):

```bash
pip install -r requirements.txt
pytest tests/test_api.py -v
```

Integration smoke tests run against the live stack:

```bash
python tests/test_api_smoke.py
```

The CI/CD pipeline (GitHub Actions) runs `test_api.py` automatically on every push to `main`.

---

## Stop the Stack

```bash
docker-compose down
```

To also remove persistent volumes (full reset):

```bash
docker-compose down -v
```

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         Apache Airflow (port 8080)       │
                    │   DAG: NOAA_Weather_MLOps_Pipeline       │
                    │   Schedule: @daily (midnight UTC)        │
                    └────────────┬────────────────────────────┘
                                 │ orchestrates
              ┌──────────────────▼──────────────────────────┐
              │                                             │
   ┌──────────▼──────────┐              ┌──────────────────▼──────────┐
   │   src/ingestion.py  │              │      src/train.py           │
   │   Downloads NOAA    │              │  Feature engineering        │
   │   data → CSV        │              │  Random Forest (80/20)      │
   └──────────┬──────────┘              │  MLflow tracking            │
              │                        └──────────────────┬──────────┘
              │                                           │
   ┌──────────▼──────────┐              ┌──────────────────▼──────────┐
   │  src/validate.py    │              │     src/promote.py          │
   │  Quality checks     │              │  Quality Gate RMSE ≤ 3.0°C  │
   │  on the dataset     │              │  → alias @production        │
   └─────────────────────┘              └──────────────────┬──────────┘
                                                           │
              ┌────────────────────────────────────────────▼──────────┐
              │                   MLflow (port 5000)                  │
              │         Experiment tracking + Model Registry          │
              │         @production alias → loaded by API             │
              └────────────────────────────────────────────┬──────────┘
                                                           │
              ┌────────────────────────────────────────────▼──────────┐
              │              FastAPI (port 8001)                      │
              │   GET  /health     → service status                   │
              │   POST /v1/predict → temperature forecast             │
              └───────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────────────────┐
              │         src/monitoring.py (EvidentlyAI)               │
              │  Drift detection: reference vs current distribution   │
              │  Output: monitoring/drift_report.html                 │
              └───────────────────────────────────────────────────────┘
```

---

## Troubleshooting

**MLflow UI shows no experiments after starting**
→ The `mlruns/` directory is gitignored. It is generated at runtime when the pipeline runs.
Trigger the DAG once (Step 6) to populate MLflow.

**API returns `model_ready: false`**
→ No model is registered yet. Trigger the DAG to run the full pipeline including `promote_model`.

**Airflow DAG not visible**
→ Wait 30–60 seconds after `docker-compose up` for the scheduler to scan the DAGs folder.

**Permission errors on WSL2**
→ Run the Step 3 commands to reset volume permissions.

**Port already in use**
→ Check for conflicting services: `lsof -i :5000` / `lsof -i :8080` / `lsof -i :8001`

---

## Author

**Mohamed Lamine OULD BOUYA**
Data Engineering · MLOps · Machine Learning
