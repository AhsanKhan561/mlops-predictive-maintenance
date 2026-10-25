# Task 10 — End-to-End MLOps Pipeline

FastAPI service for the AI4I 2020 Industrial Predictive Maintenance model from Task 9.

## API inputs

`POST /predict` accepts JSON with these exact fields:

```json
{
  "Type": "M",
  "Air_temperature": 298.1,
  "Process_temperature": 308.6,
  "Rotational_speed": 1551,
  "Torque": 42.8,
  "Tool_wear": 0
}
```

`Type` must be `L`, `M`, or `H`. Sensor fields must be numeric. Missing fields, wrong data types, invalid type values, and unknown extra fields return HTTP 400.

## Run locally

Use Python 3.11 and install the pinned requirements (especially scikit-learn 1.6.1, which matches the notebook's serialized model version):

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for Swagger UI.

## Run tests

```bash
pytest -q
```

## Run with Docker

```bash
docker build -t predictive-maintenance-api .
docker run --rm -p 8000:8000 predictive-maintenance-api
```

Then open http://localhost:8000/docs.

## Notes

- The model bundle is included in `models/app_bundle.joblib`.
- The API preserves the main and failure-type thresholds saved in the bundle.
- Pickle/joblib files should only be loaded from trusted sources.
