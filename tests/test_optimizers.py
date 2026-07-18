import pandas as pd
import pytest
from cleaning.standardizer import standardize_dataframe
from cleaning.optimizers import (
    strip_whitespace,
    convert_datetime_columns,
    convert_categorical_columns,
    downcast_integer_columns,
    downcast_float_columns,
    optimize_dataframe,
)

def test_strip_whitespace():
    df = pd.DataFrame({
        "col_str": ["  hello ", "world  ", "  both  ", None],
        "col_int": [1, 2, 3, 4]
    })
    
    cleaned = strip_whitespace(df)
    
    assert cleaned["col_str"].iloc[0] == "hello"
    assert cleaned["col_str"].iloc[1] == "world"
    assert cleaned["col_str"].iloc[2] == "both"
    assert pd.isna(cleaned["col_str"].iloc[3])
    assert cleaned["col_int"].iloc[0] == 1

def test_convert_datetime_columns():
    df = pd.DataFrame({
        "filing_date": ["20260415", "20260416", "invalid_date", None]
    })
    
    # We need a table config context, using "sub" which has "filing_date" as date column
    cleaned = convert_datetime_columns(df, "sub")
    
    assert pd.api.types.is_datetime64_any_dtype(cleaned["filing_date"])
    assert cleaned["filing_date"].iloc[0] == pd.Timestamp("2026-04-15")
    assert cleaned["filing_date"].iloc[1] == pd.Timestamp("2026-04-16")
    assert pd.isna(cleaned["filing_date"].iloc[2])
    assert pd.isna(cleaned["filing_date"].iloc[3])

def test_convert_categorical_columns():
    # Low cardinality column
    df = pd.DataFrame({
        "form_type": ["10-Q", "10-Q", "10-Q", "10-K"]  # 2 unique out of 4 (ratio = 0.5)
    })
    
    # Threshold default is 0.50. So 0.50 unique ratio <= 0.50 should trigger conversion.
    cleaned = convert_categorical_columns(df, "sub", threshold=0.50)
    assert isinstance(cleaned["form_type"].dtype, pd.CategoricalDtype)
    
    # High cardinality column
    df_high = pd.DataFrame({
        "form_type": ["10-Q", "10-K", "8-K", "S-1"]  # 4 unique out of 4 (ratio = 1.0)
    })
    cleaned_high = convert_categorical_columns(df_high, "sub", threshold=0.50)
    assert not isinstance(cleaned_high["form_type"].dtype, pd.CategoricalDtype)

def test_downcast_integer_columns():
    df = pd.DataFrame({
        "int_col": [1, 2, 100, 5]
    }, dtype="int64")
    
    cleaned = downcast_integer_columns(df)
    
    # Large default integers should be downcast to int8/int16/etc.
    assert cleaned["int_col"].dtype == "int8"

def test_downcast_float_columns():
    df = pd.DataFrame({
        "float_col": [1.1, 2.2, 3.3]
    }, dtype="float64")
    
    cleaned = downcast_float_columns(df)
    
    assert cleaned["float_col"].dtype == "float32"

def test_optimize_dataframe(raw_sub_df):
    standardized = standardize_dataframe(raw_sub_df, "sub")
    optimized = optimize_dataframe(standardized, "sub")
    
    # Verify strings stripped
    assert optimized["company_name"].iloc[0] == "COMPANY A"
    assert optimized["company_name"].iloc[1] == "COMPANY B"
    
    # Verify datetimes parsed
    assert pd.api.types.is_datetime64_any_dtype(optimized["filing_date"])
    assert pd.api.types.is_datetime64_any_dtype(optimized["report_period"])
    
    # Verify category conversion (threshold=0.50)
    # country has values ["US", "CA"] (2 unique out of 2, ratio = 1.0) -> not converted
    # form_type has values ["10-Q", "10-Q"] (1 unique out of 2, ratio = 0.5) -> converted
    assert not isinstance(optimized["country"].dtype, pd.CategoricalDtype)
    assert isinstance(optimized["form_type"].dtype, pd.CategoricalDtype)
    
    # Verify downcasting
    assert optimized["fiscal_year"].dtype == "int16"  # 2026 fits in int16
