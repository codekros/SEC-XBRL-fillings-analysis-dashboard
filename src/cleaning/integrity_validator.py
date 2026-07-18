from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_master_table(
    sub_df: pd.DataFrame,
) -> dict[str, Any]:

    duplicate_ids = int(
        sub_df["filing_id"].duplicated().sum()
    )

    return {
        "rows": len(sub_df),
        "duplicate_filing_ids": duplicate_ids,
        "passed": duplicate_ids == 0,
    }


def validate_child_table(
    child_df: pd.DataFrame,
    valid_filing_ids: set[str],
    table_name: str,
) -> dict[str, Any]:

    orphan_mask = ~child_df["filing_id"].isin(valid_filing_ids)

    orphan_count = int(orphan_mask.sum())

    return {
        "table": table_name,
        "rows_checked": len(child_df),
        "orphan_filing_ids": orphan_count,
        "passed": orphan_count == 0,
    }


def validate_quarter_integrity(
    processed_quarter_dir: Path,
) -> dict[str, Any]:

    required_files = {
        "sub": processed_quarter_dir / "sub.parquet",
        "num": processed_quarter_dir / "num.parquet",
        "pre": processed_quarter_dir / "pre.parquet",
    }

    missing_files = [
        file.name
        for file in required_files.values()
        if not file.exists()
    ]

    if missing_files:
        return {
            "quarter": processed_quarter_dir.name,
            "status": "SKIPPED",
            "passed": False,
            "reason": "Required parquet files are missing.",
            "missing_files": missing_files,
        }

    sub_df = load_parquet(required_files["sub"])
    num_df = load_parquet(required_files["num"])
    pre_df = load_parquet(required_files["pre"])

    valid_filing_ids = set(sub_df["filing_id"])

    sub_report = validate_master_table(sub_df)

    num_report = validate_child_table(
        num_df,
        valid_filing_ids,
        "num",
    )

    pre_report = validate_child_table(
        pre_df,
        valid_filing_ids,
        "pre",
    )

    passed = all(
        (
            sub_report["passed"],
            num_report["passed"],
            pre_report["passed"],
        )
    )

    return {
        "quarter": processed_quarter_dir.name,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "sub": sub_report,
        "num": num_report,
        "pre": pre_report,
    }