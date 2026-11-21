"""Tests for the configuration module."""

from pathlib import Path

import pytest

from bike_demand import __version__
from bike_demand.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    NUMERIC_FEATURES,
    RAW_TO_CLEAN_COLUMNS,
    TARGET,
    Settings,
)


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_feature_lists_are_disjoint() -> None:
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_every_clean_column_is_target_date_or_feature() -> None:
    expected = {TARGET, DATE_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES}
    assert set(RAW_TO_CLEAN_COLUMNS.values()) == expected


def test_settings_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.data_path == Path("data/seoul_bike_5000.csv")
    assert settings.log_dir == Path("logs")
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.experiment_name == "seoul-bike-demand"
    assert settings.model_name == "seoul-bike-demand-model"


def test_settings_read_from_environment() -> None:
    env = {
        "DATA_PATH": "/data/other.csv",
        "LOG_DIR": "/logs",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_EXPERIMENT_NAME": "exp",
        "MLFLOW_MODEL_NAME": "model",
    }
    settings = Settings.from_env(env)
    assert settings.data_path == Path("/data/other.csv")
    assert settings.log_dir == Path("/logs")
    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.experiment_name == "exp"
    assert settings.model_name == "model"


def test_settings_fall_back_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "from-os-env")
    assert Settings.from_env().model_name == "from-os-env"
