import pytest
import pandas as pd
from cleaning.cleaner import clean_dataframe

def test_clean_dataframe_success(raw_sub_df):
    cleaned_df, report = clean_dataframe(raw_sub_df, "sub")
    
    # Assertions on cleaned DataFrame
    assert "filing_id" in cleaned_df.columns
    assert "company_name" in cleaned_df.columns
    # Check that strings were stripped by optimizer
    assert cleaned_df["company_name"].iloc[0] == "COMPANY A"
    assert cleaned_df["company_name"].iloc[1] == "COMPANY B"
    
    # Assertions on report structure
    assert "overview" in report
    assert "validation_before" in report
    assert "validation_after" in report
    assert "memory_before_mb" in report
    assert "memory_after_mb" in report
    
    # Validation checks
    assert report["validation_after"]["schema"]["passed"] is True

def test_clean_dataframe_schema_failure():
    # Pass a dataframe missing required columns but with string columns
    invalid_df = pd.DataFrame(columns=["some_random_column"])
    
    with pytest.raises(ValueError) as excinfo:
        clean_dataframe(invalid_df, "sub")
        
    assert "Schema validation failed" in str(excinfo.value)
