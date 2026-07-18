from __future__ import annotations

import pandas as pd

from .config import TABLE_CONFIG


def rename_columns(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:

    rename_mapping = TABLE_CONFIG[table_name]["rename_columns"]

    return df.rename(columns=rename_mapping)


def standardize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )

    return df


def standardize_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:

    df = standardize_column_names(df)

    df = rename_columns(
        df,
        table_name,
    )

    return df