"""Tests for training, MLflow logging and the guarded champion promotion.

MLflow runs against a throw-away SQLite tracking store in a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from bike_demand.config import DATE_COLUMN, Settings
from bike_demand.data import split_features_target
from bike_demand.train import (
    MODEL_ALIAS,
    REFERENCE_ARTIFACT_DIR,
    build_model,
    evaluate,
    get_champion_rmse,
    main,
    run_training,
    to_api_format,
    train_and_log,
)

GOOD_GRID = ({"n_estimators": 20, "max_depth": 8}, {"n_estimators": 10, "max_depth": 3})
BAD_GRID = ({"n_estimators": 5, "max_depth": 1},)


def test_evaluate_returns_regression_metrics(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    model = build_model({"n_estimators": 10}).fit(features, target)
    metrics = evaluate(model, features, target)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1


def test_to_api_format_converts_date_to_iso_string(clean_df: pd.DataFrame) -> None:
    result = to_api_format(clean_df)
    assert result[DATE_COLUMN].iloc[0] == "2018-01-01"
    assert clean_df[DATE_COLUMN].dtype != result[DATE_COLUMN].dtype


def test_first_model_becomes_champion(clean_df: pd.DataFrame, settings: Settings) -> None:
    result = train_and_log(clean_df, settings, GOOD_GRID)

    assert result.promoted
    assert result.previous_champion_rmse is None
    assert result.params in GOOD_GRID
    client = MlflowClient()
    champion = client.get_model_version_by_alias(settings.model_name, MODEL_ALIAS)
    assert str(champion.version) == result.model_version
    assert get_champion_rmse(client, settings.model_name) == pytest.approx(result.metrics["rmse"])


def test_worse_model_is_not_promoted(clean_df: pd.DataFrame, settings: Settings) -> None:
    good = train_and_log(clean_df, settings, GOOD_GRID)
    bad = train_and_log(clean_df, settings, BAD_GRID)

    assert not bad.promoted
    assert bad.previous_champion_rmse == pytest.approx(good.metrics["rmse"])
    client = MlflowClient()
    champion = client.get_model_version_by_alias(settings.model_name, MODEL_ALIAS)
    assert str(champion.version) == good.model_version
    assert client.get_model_version(settings.model_name, bad.model_version).tags["promoted"] == "false"


def test_reference_dataset_is_logged(clean_df: pd.DataFrame, settings: Settings) -> None:
    result = train_and_log(clean_df, settings, GOOD_GRID)
    artifacts = MlflowClient().list_artifacts(result.run_id, REFERENCE_ARTIFACT_DIR)
    assert [Path(a.path).name for a in artifacts] == ["reference.csv"]


def test_champion_rmse_is_none_without_registered_model(settings: Settings) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    assert get_champion_rmse(MlflowClient(), "does-not-exist") is None


def test_run_training_loads_csv(raw_df: pd.DataFrame, settings: Settings) -> None:
    raw_df.to_csv(settings.data_path, index=False)
    result = run_training(settings, GOOD_GRID)
    assert result.promoted


def test_main_cli(
    raw_df: pd.DataFrame, settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_df.to_csv(settings.data_path, index=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", settings.experiment_name)
    monkeypatch.setattr("bike_demand.train.DEFAULT_PARAM_GRID", GOOD_GRID)

    assert main(["--data", str(settings.data_path)]) == 0
    assert "promoted to champion" in capsys.readouterr().out
