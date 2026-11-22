"""REST API (FastAPI) with ``/train``, ``/predict`` and ``/health`` endpoints.

Run locally:  uvicorn bike_demand.api:app --reload   ->  Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import datetime
from dataclasses import asdict
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bike_demand import __version__
from bike_demand.config import Settings
from bike_demand.predict import ModelNotAvailableError, ModelService, log_predictions_to_mlflow
from bike_demand.request_log import RequestLog
from bike_demand.train import ParamGrid, run_training

EXAMPLE_RECORD: dict[str, Any] = {
    "date": "2018-06-15",
    "hour": 18,
    "temperature": 24.5,
    "humidity": 55,
    "wind_speed": 1.8,
    "visibility": 1800,
    "dew_point_temperature": 14.6,
    "solar_radiation": 0.9,
    "rainfall": 0.0,
    "snowfall": 0.0,
    "seasons": "Summer",
    "holiday": "No Holiday",
    "functioning_day": "Yes",
}


class BikeRecord(BaseModel):
    """One hour of weather and calendar data (input of the model)."""

    date: datetime.date
    hour: int = Field(ge=0, le=23)
    temperature: float = Field(description="°C")
    humidity: float = Field(ge=0, le=100, description="%")
    wind_speed: float = Field(ge=0, description="m/s")
    visibility: float = Field(ge=0, description="10 m units")
    dew_point_temperature: float = Field(description="°C")
    solar_radiation: float = Field(ge=0, description="MJ/m2")
    rainfall: float = Field(ge=0, description="mm")
    snowfall: float = Field(ge=0, description="cm")
    seasons: Literal["Winter", "Spring", "Summer", "Autumn"]
    holiday: Literal["Holiday", "No Holiday"]
    functioning_day: Literal["Yes", "No"]


class PredictRequest(BaseModel):
    records: list[BikeRecord] = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(json_schema_extra={"examples": [{"records": [EXAMPLE_RECORD]}]})


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow field names starting with "model_"

    model_version: str | None
    predictions: list[float]
    mlflow_run_id: str


class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow field names starting with "model_"

    run_id: str
    model_version: str
    params: dict[str, Any]
    metrics: dict[str, float]
    promoted: bool
    previous_champion_rmse: float | None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow field names starting with "model_"

    status: str
    model_loaded: bool
    model_version: str | None


def create_app(settings: Settings | None = None, param_grid: ParamGrid | None = None) -> FastAPI:
    """Application factory (the settings and the parameter grid can be overridden in tests)."""
    app_settings = settings if settings is not None else Settings.from_env()
    service = ModelService(app_settings)
    request_log = RequestLog(app_settings.log_dir)

    app = FastAPI(
        title="Seoul Bike Demand API",
        version=__version__,
        description="Hourly bike rental demand prediction. Models are tracked and served from MLflow.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=service.is_loaded, model_version=service.model_version)

    @app.post("/train", response_model=TrainResponse)
    def train() -> TrainResponse:
        """Retrain on the dataset; the new model is only served if it is not worse than the champion."""
        result = run_training(app_settings, param_grid)
        request_log.log_training(result)
        if result.promoted:
            service.load()
        return TrainResponse(**asdict(result))

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Predict the number of rented bikes for each record with the champion model."""
        features = pd.DataFrame([record.model_dump(mode="json") for record in request.records])
        try:
            predictions = service.predict(features)
        except ModelNotAvailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        request_log.log_prediction(features, predictions, service.model_version)
        run_id = log_predictions_to_mlflow(app_settings, features, predictions, service.model_version)
        return PredictResponse(model_version=service.model_version, predictions=predictions, mlflow_run_id=run_id)

    return app


app = create_app()
