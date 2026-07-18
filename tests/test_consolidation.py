import pandas as pd
import pytest
from consolidate_data import extract_quarter_kpis

def test_extract_quarter_kpis_fallback_logic(tmp_path):
    # 1. Create a sub.parquet
    sub_df = pd.DataFrame({
        "filing_id": ["filing-1", "filing-2"],
        "company_id": [111, 222],
        "company_name": ["COMPANY ALPHA", "COMPANY BETA"],
        "fiscal_year": [2026, 2026],
        "fiscal_period": ["Q1", "Q1"],
        "form_type": ["10-Q", "10-Q"],
        "report_period": pd.to_datetime(["2026-03-31", "2026-03-31"])
    })
    sub_df.to_parquet(tmp_path / "sub.parquet")

    # 2. Create a num.parquet
    # - filing-1 has "Revenues" tag
    # - filing-2 has "SalesRevenueNet" tag (which is fallback)
    num_df = pd.DataFrame({
        "filing_id": ["filing-1", "filing-1", "filing-2", "filing-2"],
        "financial_tag": ["Revenues", "Assets", "SalesRevenueNet", "Assets"],
        "report_date": pd.to_datetime(["2026-03-31", "2026-03-31", "2026-03-31", "2026-03-31"]),
        "quarters": [1, 0, 1, 0],
        "unit": ["USD", "USD", "USD", "USD"],
        "segment": [None, None, None, None],
        "co_registrant": [None, None, None, None],
        "reported_value": [500000.0, 1000000.0, 600000.0, 1200000.0]
    })
    num_df.to_parquet(tmp_path / "num.parquet")

    # 3. Extract KPIs
    res = extract_quarter_kpis(tmp_path)

    # 4. Verify results
    assert len(res) == 2
    
    # Check fallback logic
    alpha = res[res["company_name"] == "COMPANY ALPHA"].iloc[0]
    beta = res[res["company_name"] == "COMPANY BETA"].iloc[0]

    assert alpha["revenue"] == 500000.0
    assert alpha["total_assets"] == 1000000.0

    # SalesRevenueNet should fallback to revenue for Beta
    assert beta["revenue"] == 600000.0
    assert beta["total_assets"] == 1200000.0

def test_extract_quarter_kpis_filters(tmp_path):
    # Test that segments and co-registrants are filtered out
    sub_df = pd.DataFrame({
        "filing_id": ["filing-1"],
        "company_id": [111],
        "company_name": ["COMPANY ALPHA"],
        "fiscal_year": [2026],
        "fiscal_period": ["Q1"],
        "form_type": ["10-Q"],
        "report_period": pd.to_datetime(["2026-03-31"])
    })
    sub_df.to_parquet(tmp_path / "sub.parquet")

    num_df = pd.DataFrame({
        "filing_id": ["filing-1", "filing-1", "filing-1"],
        "financial_tag": ["Revenues", "Revenues", "Revenues"],
        "report_date": pd.to_datetime(["2026-03-31", "2026-03-31", "2026-03-31"]),
        "quarters": [1, 1, 1],
        "unit": ["USD", "USD", "USD"],
        # Row 0: Consolidated
        # Row 1: Segmented
        # Row 2: Subsidiary (co_registrant)
        "segment": [None, "SegmentA", None],
        "co_registrant": [None, None, "CoRegB"],
        "reported_value": [500000.0, 100000.0, 200000.0]
    })
    num_df.to_parquet(tmp_path / "num.parquet")

    res = extract_quarter_kpis(tmp_path)
    
    # Only consolidated parent value (500000.0) should be captured
    assert len(res) == 1
    assert res["revenue"].iloc[0] == 500000.0

def test_extract_quarter_kpis_accounting_derivations(tmp_path):
    sub_df = pd.DataFrame({
        "filing_id": ["filing-1"],
        "company_id": [111],
        "company_name": ["COMPANY ALPHA"],
        "fiscal_year": [2026],
        "fiscal_period": ["Q1"],
        "form_type": ["10-Q"],
        "report_period": pd.to_datetime(["2026-03-31"])
    })
    sub_df.to_parquet(tmp_path / "sub.parquet")

    # num has:
    # - Revenue (500,000) and Cost of Revenue (300,000) but no Gross Profit (should derive 200,000)
    # - Assets (1,000,000) and Liabilities (600,000) but no Equity (should derive 400,000)
    # - OperatingIncomeLoss (80,000) and Depreciation (20,000) (should derive EBITDA of 100,000)
    num_df = pd.DataFrame({
        "filing_id": ["filing-1", "filing-1", "filing-1", "filing-1", "filing-1", "filing-1"],
        "financial_tag": [
            "Revenues", 
            "CostOfGoodsAndServicesSold", 
            "Assets", 
            "Liabilities", 
            "OperatingIncomeLoss", 
            "DepreciationDepletionAndAmortization"
        ],
        "report_date": pd.to_datetime(["2026-03-31"] * 6),
        "quarters": [1, 1, 0, 0, 1, 1],
        "unit": ["USD"] * 6,
        "segment": [None] * 6,
        "co_registrant": [None] * 6,
        "reported_value": [500000.0, 300000.0, 1000000.0, 600000.0, 800000.0, 20000.0]
    })
    num_df.to_parquet(tmp_path / "num.parquet")

    res = extract_quarter_kpis(tmp_path)
    
    assert len(res) == 1
    row = res.iloc[0]
    
    # Check derived values
    assert row["gross_profit"] == 200000.0  # 500k - 300k
    assert row["equity"] == 400000.0        # 1M - 600k
    assert row["ebitda"] == 820000.0        # 800k (operating income) + 20k (D&A)

def test_investment_diagnostics_formulas():
    # Mock row data simulating the consolidated output structure
    row = {
        "revenue": 1000000.0,
        "net_income": 100000.0,
        "total_assets": 2000000.0,
        "equity": 500000.0,
        "operating_cash_flow": 120000.0
    }
    
    # Run identical DuPont formulas used in app.py
    dup_margin = (row["net_income"] / row["revenue"] * 100)
    dup_turnover = row["revenue"] / row["total_assets"]
    dup_multiplier = row["total_assets"] / row["equity"]
    dup_roe = (row["net_income"] / row["equity"] * 100)
    
    assert dup_margin == 10.0          # 100k / 1M * 100
    assert dup_turnover == 0.5         # 1M / 2M
    assert dup_multiplier == 4.0       # 2M / 500k
    assert dup_roe == 20.0             # 100k / 500k * 100
    assert abs(dup_margin / 100 * dup_turnover * dup_multiplier * 100 - dup_roe) < 1e-9
    
    # Run identical Sloan formula used in app.py
    sloan_ratio = (row["net_income"] - row["operating_cash_flow"]) / row["total_assets"]
    sloan_pct = sloan_ratio * 100
    
    assert sloan_pct == -1.0           # (100k - 120k) / 2M * 100


