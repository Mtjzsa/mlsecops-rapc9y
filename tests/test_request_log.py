"""Tests for the CSV request logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bike_demand.request_log import RequestLog
from bike_demand.train import TrainingResult


def test_read_predictions_without_log_is_empty(tmp_path: Path) -> None:
    assert RequestLog(tmp_path).read_predictions().empty


def test_log_prediction_appends_rows_with_single_header(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "logs")
    features = pd.DataFrame({"hour": [8, 9], "temperature": [10.5, 11.0]})

    log.log_prediction(features, [120.0, 150.5], "1")
    log.log_prediction(features.head(1), [99.0], "2")

    logged = log.read_predictions()
    assert list(logged.columns) == ["timestamp", "model_version", "hour", "temperature", "prediction"]
    assert logged["prediction"].tolist() == [120.0, 150.5, 99.0]
    assert logged["model_version"].tolist() == [1, 1, 2]


def test_log_training(tmp_path: Path) -> None:
    log = RequestLog(tmp_path)
    result = TrainingResult(
        run_id="abc",
        model_version="4",
        params={"n_estimators": 10},
        metrics={"rmse": 180.0, "mae": 110.0, "r2": 0.9},
        promoted=False,
        previous_champion_rmse=175.0,
    )

    log.log_training(result)

    logged = pd.read_csv(log.training_path)
    assert logged.loc[0, "run_id"] == "abc"
    assert not logged.loc[0, "promoted"]
    assert logged.loc[0, "rmse"] == 180.0
