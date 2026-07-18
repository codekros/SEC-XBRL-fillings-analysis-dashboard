from __future__ import annotations

from typing import Any

import pandas as pd

from .standardizer import standardize_dataframe
from .optimizers import optimize_dataframe
from .utils import (
    dataset_overview,
    get_memory_usage,
    print_quality_summary,
)
from .validators import run_validations


def clean_dataframe(
    df: pd.DataFrame,
    table_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    overview = dataset_overview(df)

    df = standardize_dataframe(
        df=df,
        table_name=table_name,
    )

    validation_before = run_validations(
        df=df,
        table_name=table_name,
    )

    if not validation_before["schema"]["passed"]:
        raise ValueError(
            f"Schema validation failed for '{table_name}'.\n"
            f"Missing Columns: {validation_before['schema']['missing_columns']}"
        )

    memory_before = get_memory_usage(df)

    cleaned_df = optimize_dataframe(
        df=df.copy(),
        table_name=table_name,
    )

    validation_after = run_validations(
        df=cleaned_df,
        table_name=table_name,
    )

    memory_after = get_memory_usage(cleaned_df)

    duplicate_rows = validation_after["duplicate_rows"]["duplicate_rows"]

    duplicate_keys = validation_after["duplicate_keys"]["duplicate_keys"]

    null_count = int(
        cleaned_df.isna().sum().sum()
    )

    print_quality_summary(
        table_name=table_name,
        rows=overview["rows"],
        columns=overview["columns"],
        memory_before=memory_before,
        memory_after=memory_after,
        duplicate_rows=duplicate_rows,
        duplicate_keys=duplicate_keys,
        null_count=null_count,
    )

    report = {
        "overview": overview,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
    }

    return cleaned_df, report