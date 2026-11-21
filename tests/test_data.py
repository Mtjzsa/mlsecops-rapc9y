"""Tests for loading, validating and cleaning the dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bike_demand.config import DATE_COLUMN, TARGET
from bike_demand.data import (
    REQUIRED_COLUMNS,
    DataValidationError,
    clean,
    load_csv,
    main,
    preprocess_file,
    rename_columns,
    split_features_target,
    train_test_split_df,
    validate_columns,
)


def test_load_csv_reads_cp1252_file(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    path = tmp_path / "raw.csv"
    raw_df.to_csv(path, index=False, encoding="cp1252")
    loaded = load_csv(path, encoding="cp1252")
    assert "Temperature(°C)" in loaded.columns
    assert len(loaded) == len(raw_df)


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv")


def test_rename_columns_to_snake_case(raw_df: pd.DataFrame) -> None:
    renamed = rename_columns(raw_df)
    assert set(REQUIRED_COLUMNS) == set(renamed.columns)


def test_validate_columns_reports_missing(clean_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="humidity"):
        validate_columns(clean_df.drop(columns=["humidity"]))


def test_clean_output_schema(clean_df: pd.DataFrame) -> None:
    assert list(clean_df.columns) == list(REQUIRED_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(clean_df[DATE_COLUMN])
    assert pd.api.types.is_integer_dtype(clean_df[TARGET])
    assert clean_df.isna().sum().sum() == 0


def test_clean_is_idempotent_on_clean_names(clean_df: pd.DataFrame) -> None:
    as_text = clean_df.assign(date=clean_df[DATE_COLUMN].dt.strftime("%d/%m/%Y"))
    assert len(clean(as_text)) == len(clean_df)


def test_clean_drops_bad_rows_and_duplicates(raw_df: pd.DataFrame) -> None:
    dirty = raw_df.copy()
    dirty.loc[0, "Date"] = "not a date"
    dirty.loc[1, "Rented Bike Count"] = -5
    dirty.loc[2, "Rented Bike Count"] = None
    dirty.loc[3, "Seasons"] = "  Winter "
    dirty = pd.concat([dirty, dirty.iloc[[10]]], ignore_index=True)

    cleaned = clean(dirty)

    assert len(cleaned) == len(raw_df) - 3
    assert set(cleaned["seasons"]) <= {"Winter", "Spring", "Summer", "Autumn"}


def test_split_features_target(clean_df: pd.DataFrame) -> None:
    features, target = split_features_target(clean_df)
    assert TARGET not in features.columns
    assert target.name == TARGET
    assert len(features) == len(target)


def test_train_test_split_df(clean_df: pd.DataFrame) -> None:
    train, test = train_test_split_df(clean_df, test_size=0.25)
    assert len(train) + len(test) == len(clean_df)
    assert len(test) == pytest.approx(len(clean_df) * 0.25, abs=1)


def test_preprocess_file_writes_utf8_csv(tmp_path: Path, raw_df: pd.DataFrame) -> None:
    src = tmp_path / "raw.csv"
    dst = tmp_path / "out" / "processed.csv"
    raw_df.to_csv(src, index=False, encoding="cp1252")

    result = preprocess_file(src, dst, encoding="cp1252")

    reloaded = pd.read_csv(dst, encoding="utf-8")
    assert len(reloaded) == len(result)
    assert reloaded[DATE_COLUMN].iloc[0] == "2018-01-01"


def test_main_cli(tmp_path: Path, raw_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "raw.csv"
    dst = tmp_path / "processed.csv"
    raw_df.to_csv(src, index=False, encoding="cp1252")

    exit_code = main([str(src), str(dst), "--encoding", "cp1252"])

    assert exit_code == 0
    assert dst.exists()
    assert "Saved" in capsys.readouterr().out
