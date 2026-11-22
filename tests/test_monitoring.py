"""Tests for the Evidently drift report and the request simulator."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from bike_demand.config import DATE_COLUMN, TARGET, Settings
from bike_demand.monitoring import (
    MONITORED_FEATURES,
    NotEnoughDataError,
    build_drift_report,
    generate_drift_report,
    load_reference_data,
    monitoring_experiment_name,
    send_prediction_requests,
    simulate_records,
    summarize_drift,
)
from bike_demand.predict import ModelNotAvailableError
from bike_demand.request_log import RequestLog
from bike_demand.train import to_api_format, train_and_log

SMALL_GRID = ({"n_estimators": 10, "max_depth": 5},)


def test_simulate_records_without_shift(clean_df: pd.DataFrame) -> None:
    records = simulate_records(clean_df, 30, seed=1)
    assert len(records) == 30
    assert TARGET not in records.columns
    assert isinstance(records.loc[0, DATE_COLUMN], str)


def test_simulate_records_shifts_temperature(clean_df: pd.DataFrame) -> None:
    normal = simulate_records(clean_df, 30, seed=1)
    shifted = simulate_records(clean_df, 30, temperature_shift=10.0, seed=1)
    assert (shifted["temperature"] - normal["temperature"]).round(6).eq(10.0).all()
    assert (shifted["dew_point_temperature"] - normal["dew_point_temperature"]).round(6).eq(10.0).all()


def test_simulate_more_records_than_rows(clean_df: pd.DataFrame) -> None:
    assert len(simulate_records(clean_df, len(clean_df) + 50, seed=1)) == len(clean_df) + 50


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


def test_send_prediction_requests_in_batches(clean_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> FakeResponse:
        calls.append({"url": url, "n": len(json["records"])})
        return FakeResponse()

    monkeypatch.setattr("bike_demand.monitoring.requests.post", fake_post)

    sent = send_prediction_requests("http://api:8000", simulate_records(clean_df, 120, seed=1), batch_size=50)

    assert sent == 120
    assert [call["n"] for call in calls] == [50, 50, 20]
    assert calls[0]["url"] == "http://api:8000/predict"


def test_monitoring_experiment_name(settings: Settings) -> None:
    assert monitoring_experiment_name(settings) == "test-experiment-monitoring"


def test_report_needs_logged_requests(settings: Settings) -> None:
    with pytest.raises(NotEnoughDataError):
        generate_drift_report(settings)


def test_reference_data_needs_champion(settings: Settings) -> None:
    with pytest.raises(ModelNotAvailableError):
        load_reference_data(settings)


def test_drift_report_detects_temperature_shift(clean_df: pd.DataFrame) -> None:
    reference = to_api_format(clean_df)
    current = simulate_records(clean_df, 100, temperature_shift=25.0, seed=2)

    summary = summarize_drift(build_drift_report(reference, current))

    assert summary.n_columns == len(MONITORED_FEATURES)
    assert "temperature" in summary.drifted_columns
    assert summary.n_drifted_columns == len(summary.drifted_columns)


def test_generate_drift_report_end_to_end(settings: Settings, clean_df: pd.DataFrame) -> None:
    train_and_log(clean_df, settings, SMALL_GRID)
    records = simulate_records(clean_df, 40, temperature_shift=25.0, seed=3)
    RequestLog(settings.log_dir).log_prediction(records, [100.0] * len(records), "1")

    summary, html = generate_drift_report(settings)

    assert "temperature" in summary.drifted_columns
    assert "<html" in html.lower()
    experiment = MlflowClient().get_experiment_by_name("test-experiment-monitoring")
    assert experiment is not None
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.metrics["n_current_rows"] == 40
