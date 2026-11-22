"""End-to-end tests of the REST API with FastAPI's TestClient and a temporary MLflow store."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from bike_demand.api import EXAMPLE_RECORD, app, create_app
from bike_demand.config import Settings
from bike_demand.request_log import RequestLog

SMALL_GRID = ({"n_estimators": 10, "max_depth": 5},)


@pytest.fixture
def client(settings: Settings, raw_df: pd.DataFrame) -> TestClient:
    raw_df.to_csv(settings.data_path, index=False)
    return TestClient(create_app(settings, param_grid=SMALL_GRID))


def test_module_level_app_exists() -> None:
    assert app.title == "Seoul Bike Demand API"


def test_health_before_training(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False, "model_version": None}


def test_predict_without_model_returns_503(client: TestClient) -> None:
    response = client.post("/predict", json={"records": [EXAMPLE_RECORD]})
    assert response.status_code == 503


def test_predict_rejects_invalid_input(client: TestClient) -> None:
    invalid = {**EXAMPLE_RECORD, "hour": 25, "seasons": "Monsoon"}
    response = client.post("/predict", json={"records": [invalid]})
    assert response.status_code == 422


def test_predict_rejects_empty_request(client: TestClient) -> None:
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_train_then_predict(client: TestClient, settings: Settings) -> None:
    train_response = client.post("/train")
    assert train_response.status_code == 200
    trained = train_response.json()
    assert trained["promoted"] is True
    assert set(trained["metrics"]) == {"rmse", "mae", "r2"}

    predict_response = client.post("/predict", json={"records": [EXAMPLE_RECORD, EXAMPLE_RECORD]})
    assert predict_response.status_code == 200
    body = predict_response.json()
    assert body["model_version"] == trained["model_version"]
    assert len(body["predictions"]) == 2

    # the request is logged both to CSV (for Evidently) and to MLflow
    logged = RequestLog(settings.log_dir).read_predictions()
    assert len(logged) == 2
    assert "prediction" in logged.columns
    assert MlflowClient().get_run(body["mlflow_run_id"]).data.params["model_version"] == trained["model_version"]
    assert client.get("/health").json()["model_loaded"] is True


def test_second_training_is_logged(client: TestClient, settings: Settings) -> None:
    client.post("/train")
    client.post("/train")
    training_log = pd.read_csv(RequestLog(settings.log_dir).training_path)
    assert len(training_log) == 2
