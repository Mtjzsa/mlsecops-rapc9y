"""Tests for serving the champion model and logging predictions to MLflow."""

from __future__ import annotations

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from bike_demand.config import Settings
from bike_demand.data import split_features_target
from bike_demand.predict import (
    ModelNotAvailableError,
    ModelService,
    log_predictions_to_mlflow,
    predictions_experiment_name,
)
from bike_demand.train import to_api_format, train_and_log

SMALL_GRID = ({"n_estimators": 10, "max_depth": 5},)


@pytest.fixture
def api_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    features, _ = split_features_target(clean_df)
    return to_api_format(features.head(5))


def test_predictions_experiment_name(settings: Settings) -> None:
    assert predictions_experiment_name(settings) == "test-experiment-predictions"


def test_service_without_champion(settings: Settings, api_features: pd.DataFrame) -> None:
    service = ModelService(settings)
    assert not service.load()
    assert not service.is_loaded
    with pytest.raises(ModelNotAvailableError):
        service.predict(api_features)


def test_service_loads_champion_lazily(
    settings: Settings, clean_df: pd.DataFrame, api_features: pd.DataFrame
) -> None:
    result = train_and_log(clean_df, settings, SMALL_GRID)
    service = ModelService(settings)

    predictions = service.predict(api_features)  # loads the champion on the first call

    assert service.is_loaded
    assert service.model_version == result.model_version
    assert len(predictions) == len(api_features)
    assert all(value >= 0 for value in predictions)


def test_log_predictions_to_mlflow(settings: Settings, api_features: pd.DataFrame) -> None:
    predictions = [100.0, 200.0, 300.0, 400.0, 500.0]

    run_id = log_predictions_to_mlflow(settings, api_features, predictions, "3")

    run = MlflowClient().get_run(run_id)
    assert run.data.params["model_version"] == "3"
    assert run.data.metrics["mean_prediction"] == pytest.approx(300.0)
    assert [a.path for a in MlflowClient().list_artifacts(run_id, "predictions")] == ["predictions/predictions.csv"]
