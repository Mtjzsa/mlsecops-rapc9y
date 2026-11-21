"""Loading, validating and cleaning the Seoul bike sharing dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from bike_demand.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    DATE_FORMAT,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_TO_CLEAN_COLUMNS,
    TARGET,
)

REQUIRED_COLUMNS: tuple[str, ...] = (DATE_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET)


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected schema."""


def load_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a CSV file. The original UCI file needs ``encoding="cp1252"``."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, encoding=encoding)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the original UCI column names to snake_case (already clean names are kept)."""
    return df.rename(columns=RAW_TO_CLEAN_COLUMNS)


def validate_columns(df: pd.DataFrame, required: Sequence[str] = REQUIRED_COLUMNS) -> None:
    """Raise :class:`DataValidationError` if any required column is missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset.

    Steps: rename columns, validate the schema, parse the date, strip text values,
    coerce numeric columns, drop rows with an invalid date / target and duplicated rows.
    """
    result = rename_columns(df.copy())
    validate_columns(result)

    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], format=DATE_FORMAT, errors="coerce")
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str).str.strip()
    for column in (*NUMERIC_FEATURES, TARGET):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=[DATE_COLUMN, TARGET])
    result = result[result[TARGET] >= 0]
    result = result.drop_duplicates().reset_index(drop=True)
    result[TARGET] = result[TARGET].astype(int)
    return cast(pd.DataFrame, result[list(REQUIRED_COLUMNS)])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features ``X`` and target ``y``."""
    validate_columns(df)
    return df.drop(columns=[TARGET]), cast(pd.Series, df[TARGET])


def train_test_split_df(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split a cleaned dataframe into train and test parts."""
    test = df.sample(frac=test_size, random_state=random_state)
    train = cast(pd.DataFrame, df[~df.index.isin(test.index)])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def preprocess_file(src: Path, dst: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load ``src``, clean it and save the result as a UTF-8 CSV to ``dst``."""
    cleaned = clean(load_csv(src, encoding=encoding))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(dst, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    return cleaned


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: ``python -m bike_demand.data <src> <dst> [--encoding ENC]``."""
    parser = argparse.ArgumentParser(description="Clean the Seoul bike sharing dataset.")
    parser.add_argument("src", type=Path, help="input CSV (original or sampled dataset)")
    parser.add_argument("dst", type=Path, help="output CSV for the cleaned dataset")
    parser.add_argument("--encoding", default="utf-8", help="encoding of the input file")
    args = parser.parse_args(argv)

    cleaned = preprocess_file(args.src, args.dst, encoding=args.encoding)
    print(f"Saved {len(cleaned)} rows x {cleaned.shape[1]} columns to {args.dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin entry point, main() itself is tested
    raise SystemExit(main())
