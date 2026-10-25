import os
from fastapi.testclient import TestClient

# Tests expect the model bundle to be present at models/app_bundle.joblib.
from app import app

client = TestClient(app)

SAMPLE = {
    "Type": "M",
    "Air_temperature": 298.1,
    "Process_temperature": 308.6,
    "Rotational_speed": 1551,
    "Torque": 42.8,
    "Tool_wear": 0,
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["machine_failure"], bool)
    assert 0 <= body["failure_probability"] <= 1
    assert "failure_types" in body


def test_text_rejected_for_numeric_sensor_with_400():
    invalid = {**SAMPLE, "Torque": "heavy"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_invalid_type_rejected_with_400():
    invalid = {**SAMPLE, "Type": "X"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_missing_field_rejected_with_400():
    invalid = {k: v for k, v in SAMPLE.items() if k != "Tool_wear"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400


def test_extra_field_rejected_with_400():
    invalid = {**SAMPLE, "unknown_sensor": 123}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 400
