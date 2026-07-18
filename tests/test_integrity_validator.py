import pandas as pd
import pytest
from cleaning.standardizer import standardize_dataframe
from cleaning.integrity_validator import (
    validate_master_table,
    validate_child_table,
    validate_quarter_integrity,
)

def test_validate_master_table(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    report = validate_master_table(df)
    
    assert report["rows"] == 2
    assert report["duplicate_filing_ids"] == 0
    assert report["passed"] is True
    
    # Introduce duplicate filing_id
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report_fail = validate_master_table(df_dup)
    assert report_fail["passed"] is False
    assert report_fail["duplicate_filing_ids"] == 1

def test_validate_child_table(raw_sub_df, raw_num_df):
    sub_df = standardize_dataframe(raw_sub_df, "sub")
    num_df = standardize_dataframe(raw_num_df, "num")
    
    valid_ids = set(sub_df["filing_id"])
    report = validate_child_table(num_df, valid_ids, "num")
    
    # All filing_ids in raw_num_df ("0000000000-20-000001", "0000000000-20-000002") 
    # exist in raw_sub_df
    assert report["rows_checked"] == 3
    assert report["orphan_filing_ids"] == 0
    assert report["passed"] is True
    
    # Add an orphan
    num_df_orphan = num_df.copy()
    num_df_orphan.loc[0, "filing_id"] = "orphan-id"
    report_orphan = validate_child_table(num_df_orphan, valid_ids, "num")
    assert report_orphan["passed"] is False
    assert report_orphan["orphan_filing_ids"] == 1

def test_validate_quarter_integrity_missing_files(tmp_path):
    # Empty directory, missing parquet files
    report = validate_quarter_integrity(tmp_path)
    assert report["status"] == "SKIPPED"
    assert report["passed"] is False
    assert "sub.parquet" in report["missing_files"]

def test_validate_quarter_integrity_success(tmp_path, raw_sub_df, raw_num_df, raw_pre_df):
    sub_df = standardize_dataframe(raw_sub_df, "sub")
    num_df = standardize_dataframe(raw_num_df, "num")
    pre_df = standardize_dataframe(raw_pre_df, "pre")
    
    # Save as parquet
    sub_df.to_parquet(tmp_path / "sub.parquet")
    num_df.to_parquet(tmp_path / "num.parquet")
    pre_df.to_parquet(tmp_path / "pre.parquet")
    
    report = validate_quarter_integrity(tmp_path)
    
    assert report["status"] == "PASS"
    assert report["passed"] is True
    assert report["sub"]["passed"] is True
    assert report["num"]["passed"] is True
    assert report["pre"]["passed"] is True

def test_validate_quarter_integrity_failure(tmp_path, raw_sub_df, raw_num_df, raw_pre_df):
    sub_df = standardize_dataframe(raw_sub_df, "sub")
    num_df = standardize_dataframe(raw_num_df, "num")
    pre_df = standardize_dataframe(raw_pre_df, "pre")
    
    # Make num have an orphan
    num_df.loc[0, "filing_id"] = "orphan-id"
    
    sub_df.to_parquet(tmp_path / "sub.parquet")
    num_df.to_parquet(tmp_path / "num.parquet")
    pre_df.to_parquet(tmp_path / "pre.parquet")
    
    report = validate_quarter_integrity(tmp_path)
    
    assert report["status"] == "FAIL"
    assert report["passed"] is False
    assert report["num"]["passed"] is False
    assert report["num"]["orphan_filing_ids"] == 1
