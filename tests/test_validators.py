import pandas as pd
import pytest
from cleaning.standardizer import standardize_dataframe
from cleaning.validators import (
    validate_schema,
    check_missing_values,
    check_duplicate_rows,
    check_duplicate_business_keys,
    validate_date_columns,
    validate_numeric_columns,
    validate_categorical_columns,
    run_validations,
)

def test_validate_schema(raw_sub_df):
    # Standarized sub dataframe has correct columns
    df = standardize_dataframe(raw_sub_df, "sub")
    schema_report = validate_schema(df, "sub")
    
    assert schema_report["passed"] is True
    assert len(schema_report["missing_columns"]) == 0
    
    # Drop a column to make it fail
    df_missing = df.drop(columns=["filing_id"])
    schema_report_fail = validate_schema(df_missing, "sub")
    assert schema_report_fail["passed"] is False
    assert "filing_id" in schema_report_fail["missing_columns"]

def test_check_missing_values():
    df = pd.DataFrame({
        "col_a": [1, 2, None],
        "col_b": [None, None, None]
    })
    
    missing_report = check_missing_values(df)
    assert missing_report.loc["col_b", "missing_count"] == 3
    assert missing_report.loc["col_b", "missing_percent"] == 100.0
    assert missing_report.loc["col_a", "missing_count"] == 1
    assert missing_report.loc["col_a", "missing_percent"] == 33.33

def test_check_duplicate_rows():
    df = pd.DataFrame({
        "col_a": [1, 2, 2],
        "col_b": ["a", "b", "b"]
    })
    
    dup_report = check_duplicate_rows(df)
    assert dup_report["passed"] is False
    assert dup_report["duplicate_rows"] == 1

def test_check_duplicate_business_keys(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    
    # Unique keys -> pass
    dup_keys_report = check_duplicate_business_keys(df, "sub")
    assert dup_keys_report["passed"] is True
    assert dup_keys_report["duplicate_keys"] == 0
    
    # Non-unique keys -> fail
    df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dup_keys_report_fail = check_duplicate_business_keys(df_dup, "sub")
    assert dup_keys_report_fail["passed"] is False
    assert dup_keys_report_fail["duplicate_keys"] == 2  # both duplicate rows returned with keep=False

def test_validate_date_columns(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    
    # All date columns are valid YYYYMMDD in fixture
    report = validate_date_columns(df, "sub")
    assert report["filing_date"]["passed"] is True
    assert report["filing_date"]["invalid_dates"] == 0
    
    # Introduce invalid date string
    df_invalid = df.copy()
    df_invalid.loc[0, "filing_date"] = "not-a-date"
    report_fail = validate_date_columns(df_invalid, "sub")
    assert report_fail["filing_date"]["passed"] is False
    assert report_fail["filing_date"]["invalid_dates"] == 1

def test_validate_numeric_columns(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    
    report = validate_numeric_columns(df, "sub")
    assert report["fiscal_year"]["passed"] is True
    assert report["fiscal_year"]["invalid_numeric"] == 0
    
    # Introduce invalid numeric value
    df_invalid = df.copy()
    df_invalid["fiscal_year"] = df_invalid["fiscal_year"].astype(object)
    df_invalid.loc[0, "fiscal_year"] = "not-a-number"
    report_fail = validate_numeric_columns(df_invalid, "sub")
    assert report_fail["fiscal_year"]["passed"] is False
    assert report_fail["fiscal_year"]["invalid_numeric"] == 1

def test_validate_categorical_columns(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    report = validate_categorical_columns(df, "sub")
    
    # country has values ["US", "CA"]
    assert report["country"]["unique_values"] == 2
    assert report["country"]["missing_values"] == 0

def test_run_validations(raw_sub_df):
    df = standardize_dataframe(raw_sub_df, "sub")
    full_report = run_validations(df, "sub")
    
    assert "schema" in full_report
    assert "missing" in full_report
    assert "duplicate_rows" in full_report
    assert "duplicate_keys" in full_report
    assert "dates" in full_report
    assert "numeric" in full_report
    assert "categorical" in full_report
    assert full_report["schema"]["passed"] is True
