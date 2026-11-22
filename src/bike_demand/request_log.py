"""Append-only CSV logs of the API requests - the input of the Evidently drift report."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bike_demand.train import TrainingResult

PREDICTION_LOG_FILE = "predictions.csv"
TRAINING_LOG_FILE = "training.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


class RequestLog:
    """Logs every ``/predict`` and ``/train`` request into CSV files in ``log_dir``."""

    def __init__(self, log_dir: Path) -> None:
        self.predictions_path = log_dir / PREDICTION_LOG_FILE
        self.training_path = log_dir / TRAINING_LOG_FILE

    def log_prediction(self, features: pd.DataFrame, predictions: Sequence[float], model_version: str | None) -> None:
        rows = features.copy()
        rows.insert(0, "timestamp", _now())
        rows.insert(1, "model_version", model_version)
        rows["prediction"] = list(predictions)
        _append(self.predictions_path, rows)

    def log_training(self, result: TrainingResult) -> None:
        row = {
            "timestamp": _now(),
            "run_id": result.run_id,
            "model_version": result.model_version,
            "promoted": result.promoted,
            "previous_champion_rmse": result.previous_champion_rmse,
            **result.metrics,
        }
        _append(self.training_path, pd.DataFrame([row]))

    def read_predictions(self) -> pd.DataFrame:
        """All logged prediction requests (empty dataframe if there are none yet)."""
        if not self.predictions_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.predictions_path)
