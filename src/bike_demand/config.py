"""Project-wide configuration: column names, feature lists and runtime settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Original UCI column names -> clean snake_case names used everywhere in the code
RAW_TO_CLEAN_COLUMNS: dict[str, str] = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(°C)": "dew_point_temperature",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "seasons",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}

TARGET = "rented_bike_count"
DATE_COLUMN = "date"
DATE_FORMAT = "%d/%m/%Y"

NUMERIC_FEATURES: tuple[str, ...] = (
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point_temperature",
    "solar_radiation",
    "rainfall",
    "snowfall",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("seasons", "holiday", "functioning_day")

RANDOM_STATE = 42


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from environment variables (with local defaults)."""

    data_path: Path
    log_dir: Path
    mlflow_tracking_uri: str
    experiment_name: str
    model_name: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to ``os.environ``)."""
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            data_path=Path(source.get("DATA_PATH", "data/seoul_bike_5000.csv")),
            log_dir=Path(source.get("LOG_DIR", "logs")),
            mlflow_tracking_uri=source.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            experiment_name=source.get("MLFLOW_EXPERIMENT_NAME", "seoul-bike-demand"),
            model_name=source.get("MLFLOW_MODEL_NAME", "seoul-bike-demand-model"),
        )
