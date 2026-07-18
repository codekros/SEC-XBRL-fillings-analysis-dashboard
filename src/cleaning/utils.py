from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logger(name: str = "xbrl_cleaning") -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logger()


def read_txt_file(
    file_path: str | Path,
    separator: str = "\t",
) -> pd.DataFrame:

    file_path = Path(file_path)

    logger.info("Reading %s", file_path.name)

    return pd.read_csv(
        file_path,
        sep=separator,
        low_memory=False,
    )


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    logger.info("Saved parquet -> %s", output_path)


def get_memory_usage(df: pd.DataFrame) -> float:

    return round(
        df.memory_usage(deep=True).sum() / (1024**2),
        2,
    )


def dataset_overview(
    df: pd.DataFrame,
) -> dict[str, Any]:

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": get_memory_usage(df),
    }


def print_quality_summary(
    *,
    table_name: str,
    rows: int,
    columns: int,
    memory_before: float,
    memory_after: float,
    duplicate_rows: int,
    duplicate_keys: int,
    null_count: int,
) -> None:

    reduction = memory_before - memory_after

    logger.info("")
    logger.info("=" * 70)
    logger.info("TABLE               : %s", table_name.upper())
    logger.info("=" * 70)

    logger.info("Rows                : %s", format(rows, ","))
    logger.info("Columns             : %d", columns)
    logger.info("Null Values         : %s", format(null_count, ","))
    logger.info("Duplicate Rows      : %s", format(duplicate_rows, ","))
    logger.info("Duplicate Keys      : %s", format(duplicate_keys, ","))

    logger.info("Memory Before (MB)  : %.2f", memory_before)
    logger.info("Memory After  (MB)  : %.2f", memory_after)
    logger.info("Memory Saved  (MB)  : %.2f", reduction)

    logger.info("=" * 70)