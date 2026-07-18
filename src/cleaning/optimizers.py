from __future__ import annotations

import pandas as pd

from .config import TABLE_CONFIG


def strip_whitespace(
    df: pd.DataFrame,
) -> pd.DataFrame:

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in string_columns:
        df[column] = df[column].str.strip()

    return df


def convert_datetime_columns(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:

    config = TABLE_CONFIG[table_name]

    for column in config["date_columns"]:

        if column not in df.columns:
            continue

        df[column] = pd.to_datetime(
            df[column],
            format="%Y%m%d",
            errors="coerce",
        )

    return df


def convert_categorical_columns(
    df: pd.DataFrame,
    table_name: str,
    threshold: float = 0.50,
) -> pd.DataFrame:

    config = TABLE_CONFIG[table_name]

    for column in config["categorical_columns"]:

        if column not in df.columns:
            continue

        if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
            continue

        unique_ratio = df[column].nunique(dropna=False) / len(df)

        if unique_ratio <= threshold:
            df[column] = df[column].astype("category")

    return df


def downcast_integer_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    integer_columns = df.select_dtypes(
        include=["int", "int64", "Int64"]
    ).columns

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            downcast="integer",
        )

    return df


def downcast_float_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    float_columns = df.select_dtypes(
        include=["float", "float64"]
    ).columns

    for column in float_columns:

        df[column] = pd.to_numeric(
            df[column],
            downcast="float",
        )

    return df


def optimize_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:

    df = strip_whitespace(df)

    df = convert_datetime_columns(
        df,
        table_name,
    )

    df = convert_categorical_columns(
        df,
        table_name,
    )

    df = downcast_integer_columns(df)

    df = downcast_float_columns(df)

    return df