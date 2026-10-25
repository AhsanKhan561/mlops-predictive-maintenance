import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH", "models/app_bundle.joblib")

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description="FastAPI serving the Task 9 AI4I predictive-maintenance model.",
    version="1.0.0",
)

# The assignment requires invalid payloads to return HTTP 400 (FastAPI defaults to 422).
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Type: Literal["L", "M", "H"]
    Air_temperature: StrictInt | StrictFloat = Field(..., description="Air temperature in kelvin")
    Process_temperature: StrictInt | StrictFloat = Field(..., description="Process temperature in kelvin")
    Rotational_speed: StrictInt | StrictFloat = Field(..., description="Rotational speed in rpm")
    Torque: StrictInt | StrictFloat = Field(..., description="Torque in Nm")
    Tool_wear: StrictInt | StrictFloat = Field(..., description="Tool wear in minutes")


_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(BUNDLE_PATH):
            raise FileNotFoundError(
                f"Model bundle not found at {BUNDLE_PATH}. Place app_bundle.joblib in the models folder."
            )
        _bundle = joblib.load(BUNDLE_PATH)
    return _bundle


@app.get("/")
def root():
    return {"message": "Industrial Predictive Maintenance API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    try:
        bundle = get_bundle()
        return {"status": "ok", "model_loaded": bundle.get("model") is not None}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.post("/predict")
def predict(payload: PredictionInput):
    bundle = get_bundle()
    row = payload.model_dump()
    features = pd.DataFrame([row], columns=bundle["feature_cols"])

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    probability = float(model.predict_proba(features)[0, 1])
    failed = probability >= threshold

    mode_predictions = {}
    for mode, mode_model in bundle.get("mode_models", {}).items():
        mode_threshold = float(bundle.get("mode_thresholds", {}).get(mode, 0.5))
        mode_probability = float(mode_model.predict_proba(features)[0, 1])
        mode_predictions[mode] = {
            "failure_type": {
                "TWF": "Tool Wear Failure",
                "HDF": "Heat Dissipation Failure",
                "PWF": "Power Failure",
                "OSF": "Overstrain Failure",
                "RNF": "Random Failure",
            }.get(mode, mode),
            "probability": round(mode_probability, 4),
            "threshold": round(mode_threshold, 4),
            "predicted": bool(mode_probability >= mode_threshold),
        }

    return {
        "machine_failure": bool(failed),
        "failure_probability": round(probability, 4),
        "decision_threshold": round(threshold, 4),
        "status": "Failure risk detected" if failed else "No failure detected at configured threshold",
        "failure_types": mode_predictions,
    }
