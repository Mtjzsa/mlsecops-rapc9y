"""Serving the champion model from the MLflow model registry and logging predictions to MLflow."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from bike_demand.config import Settings
from bike_demand.train import MODEL_ALIAS


class ModelNotAvailableError(RuntimeError):
    """Raised when there is no champion model in the registry yet."""


def predictions_experiment_name(settings: Settings) -> str:
    """MLflow experiment that stores the prediction requests (separate from training runs)."""
    return f"{settings.experiment_name}-predictions"


class ModelService:
    """Loads the ``champion`` model version lazily and serves predictions with it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_version: str | None = None
        self._model: Pipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """(Re)load the current champion. Returns ``False`` if there is no champion yet."""
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        try:
            version = MlflowClient().get_model_version_by_alias(self.settings.model_name, MODEL_ALIAS)
        except MlflowException:
            return False
        self._model = mlflow.sklearn.load_model(f"models:/{self.settings.model_name}@{MODEL_ALIAS}")
        self.model_version = str(version.version)
        return True

    def predict(self, features: pd.DataFrame) -> list[float]:
        """Predict hourly rentals (non-negative, rounded to 1 decimal)."""
        if self._model is None:
            self.load()
        if self._model is None:
            raise ModelNotAvailableError("No champion model in the registry - call /train first.")
        predictions = self._model.predict(features)
        return [round(max(float(value), 0.0), 1) for value in predictions]


def log_predictions_to_mlflow(
    settings: Settings, features: pd.DataFrame, predictions: Sequence[float], model_version: str | None
) -> str:
    """Store one prediction request (inputs + outputs) as an MLflow run artifact. Returns the run id."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(predictions_experiment_name(settings))
    with mlflow.start_run(run_name="predict") as run:
        mlflow.log_param("model_version", model_version)
        mlflow.log_metrics(
            {"n_records": float(len(predictions)), "mean_prediction": float(sum(predictions) / len(predictions))}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "predictions.csv"
            features.assign(prediction=list(predictions)).to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="predictions")
    return run.info.run_id
