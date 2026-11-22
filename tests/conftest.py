"""Shared test fixtures: a small synthetic dataset in the original UCI format."""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from bike_demand.config import Settings
from bike_demand.data import clean


def make_raw_df(n_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the original UCI column names and formats."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_rows, freq="h")
    temperature = rng.normal(12, 10, n_rows).round(1)
    hour = dates.hour.to_numpy()
    count = np.clip(300 + 25 * temperature + 40 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 30, n_rows), 0, None)
    return pd.DataFrame(
        {
            "Date": dates.strftime("%d/%m/%Y"),
            "Rented Bike Count": count.astype(int),
            "Hour": hour,
            "Temperature(°C)": temperature,
            "Humidity(%)": rng.integers(10, 99, n_rows),
            "Wind speed (m/s)": rng.uniform(0, 5, n_rows).round(1),
            "Visibility (10m)": rng.integers(100, 2000, n_rows),
            "Dew point temperature(°C)": (temperature - rng.uniform(0, 10, n_rows)).round(1),
            "Solar Radiation (MJ/m2)": rng.uniform(0, 3, n_rows).round(2),
            "Rainfall(mm)": rng.choice([0.0, 0.0, 0.0, 0.5, 2.0], n_rows),
            "Snowfall (cm)": rng.choice([0.0, 0.0, 0.0, 0.3], n_rows),
            "Seasons": rng.choice(["Winter", "Spring", "Summer", "Autumn"], n_rows),
            "Holiday": rng.choice(["No Holiday", "Holiday"], n_rows, p=[0.9, 0.1]),
            "Functioning Day": rng.choice(["Yes", "No"], n_rows, p=[0.95, 0.05]),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing to a throw-away SQLite MLflow store in a temporary directory."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    # Artifacts go to the temporary directory instead of ./mlruns
    for experiment in ("test-experiment", "test-experiment-predictions"):
        mlflow.create_experiment(experiment, artifact_location=(tmp_path / "artifacts" / experiment).as_uri())
    return Settings(
        data_path=tmp_path / "data.csv",
        log_dir=tmp_path / "logs",
        mlflow_tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        model_name="test-model",
    )
