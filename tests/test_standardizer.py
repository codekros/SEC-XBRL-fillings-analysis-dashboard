import pandas as pd
import pytest
from cleaning.standardizer import (
    standardize_column_names,
    rename_columns,
    standardize_dataframe,
)

def test_standardize_column_names():
    # DataFrame with non-standard column names
    df = pd.DataFrame(
        columns=["  Column A  ", "Column B", "COLUMN C", "Column  D"]
    )
    standardized = standardize_column_names(df)
    
    # Expected columns: ['column_a', 'column_b', 'column_c', 'column__d']
    # Note: str.replace(" ", "_") on "Column  D" after strip gives "column__d" due to two spaces
    assert list(standardized.columns) == ["column_a", "column_b", "column_c", "column__d"]

def test_rename_columns_sub(raw_sub_df):
    # standardize columns first since rename expects standardized keys
    df = standardize_column_names(raw_sub_df)
    renamed = rename_columns(df, "sub")
    
    assert "filing_id" in renamed.columns  # adsh -> filing_id
    assert "company_id" in renamed.columns  # cik -> company_id
    assert "company_name" in renamed.columns  # name -> company_name
    assert "fiscal_year" in renamed.columns  # fy -> fiscal_year

def test_rename_columns_num(raw_num_df):
    df = standardize_column_names(raw_num_df)
    renamed = rename_columns(df, "num")
    
    assert "filing_id" in renamed.columns  # adsh -> filing_id
    assert "financial_tag" in renamed.columns  # tag -> financial_tag
    assert "reported_value" in renamed.columns  # value -> reported_value

def test_standardize_dataframe(raw_sub_df):
    standardized = standardize_dataframe(raw_sub_df, "sub")
    
    # Verify standardizations & renaming occurred together
    assert "filing_id" in standardized.columns
    assert "company_name" in standardized.columns
    # Verify name value is not stripped by standardizer (only columns)
    assert standardized["company_name"].iloc[0] == " COMPANY A "
