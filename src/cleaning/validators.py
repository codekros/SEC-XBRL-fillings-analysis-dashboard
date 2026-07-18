from __future__ import annotations

from typing import Any

import pandas as pd

from .config import TABLE_CONFIG


def validate_schema(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    config = TABLE_CONFIG[table_name]

    expected = set(
        config["business_keys"]
        + config["date_columns"]
        + config["numeric_columns"]
        + config["categorical_columns"]
        + config["nullable_columns"]
    )

    actual = set(df.columns)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    return {
        "passed": len(missing) == 0,
        "missing_columns": missing,
        "unexpected_columns": unexpected,
    }


def check_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:

    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    return missing.sort_values(
        "missing_count",
        ascending=False,
    )


def check_duplicate_rows(
    df: pd.DataFrame,
) -> dict[str, Any]:

    count = int(df.duplicated().sum())

    return {
        "passed": count == 0,
        "duplicate_rows": count,
    }


def check_duplicate_business_keys(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    keys = TABLE_CONFIG[table_name]["business_keys"]

    missing_keys = [key for key in keys if key not in df.columns]

    if missing_keys:
        return {
            "passed": False,
            "business_keys": keys,
            "duplicate_keys": None,
            "error": f"Missing business key columns: {missing_keys}",
        }

    duplicates = int(
        df.duplicated(
            subset=keys,
            keep=False,
        ).sum()
    )

    return {
        "passed": duplicates == 0,
        "business_keys": keys,
        "duplicate_keys": duplicates,
    }


def validate_date_columns(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    results = {}

    for column in TABLE_CONFIG[table_name]["date_columns"]:

        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            format="%Y%m%d",
            errors="coerce",
        )

        invalid = int(parsed.isna().sum())

        results[column] = {
            "invalid_dates": invalid,
            "passed": invalid == 0,
        }

    return results


def validate_numeric_columns(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    results = {}

    for column in TABLE_CONFIG[table_name]["numeric_columns"]:

        if column not in df.columns:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid = int(converted.isna().sum())

        results[column] = {
            "invalid_numeric": invalid,
            "passed": invalid == 0,
        }

    return results


def validate_categorical_columns(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    results = {}

    for column in TABLE_CONFIG[table_name]["categorical_columns"]:

        if column not in df.columns:
            continue

        results[column] = {
            "unique_values": int(
                df[column].nunique(dropna=True)
            ),
            "missing_values": int(
                df[column].isna().sum()
            ),
        }

    return results


def run_validations(
    df: pd.DataFrame,
    table_name: str,
) -> dict[str, Any]:

    return {
        "schema": validate_schema(
            df,
            table_name,
        ),
        "missing": check_missing_values(
            df,
        ),
        "duplicate_rows": check_duplicate_rows(
            df,
        ),
        "duplicate_keys": check_duplicate_business_keys(
            df,
            table_name,
        ),
        "dates": validate_date_columns(
            df,
            table_name,
        ),
        "numeric": validate_numeric_columns(
            df,
            table_name,
        ),
        "categorical": validate_categorical_columns(
            df,
            table_name,
        ),
    }