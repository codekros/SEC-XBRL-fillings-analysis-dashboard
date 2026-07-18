import sys
from pathlib import Path
import pytest
import pandas as pd

# Add src to the system path to allow importing cleaning package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

@pytest.fixture
def raw_sub_df():
    """
    Returns a sample raw 'sub' table DataFrame.
    """
    return pd.DataFrame({
        "adsh": ["0000000000-20-000001", "0000000000-20-000002"],
        "cik": [1000, 2000],
        "name": [" COMPANY A ", "COMPANY B  "],
        "sic": [1234, 5678],
        "countryba": ["US", "CA"],
        "stprba": ["NY", "ON"],
        "cityba": ["New York", "Toronto"],
        "zipba": ["10001", "M5V 2T6"],
        "bas1": ["123 Broadway", "456 King St"],
        "bas2": [None, "Suite 100"],
        "period": ["20260331", "20260331"],
        "fy": [2026, 2026],
        "fp": ["Q1", "Q1"],
        "filed": ["20260415", "20260416"],
        "form": ["10-Q", "10-Q"],
        "accepted": ["2026-04-15 17:30:00", "2026-04-16 16:00:00"],
        "countryinc": ["US", "CA"],
        "stprinc": ["DE", "ON"]
    })

@pytest.fixture
def raw_num_df():
    """
    Returns a sample raw 'num' table DataFrame.
    """
    return pd.DataFrame({
        "adsh": ["0000000000-20-000001", "0000000000-20-000001", "0000000000-20-000002"],
        "tag": ["Assets", "Liabilities", "Assets"],
        "version": ["us-gaap/2025", "us-gaap/2025", "us-gaap/2025"],
        "ddate": ["20260331", "20260331", "20260331"],
        "qtrs": [0, 0, 0],
        "uom": ["USD", "USD", "USD"],
        "coreg": [None, None, "Sub-1"],
        "value": [1000000.0, 500000.0, 2000000.0],
        "footnote": [None, "Footnote text", None]
    })

@pytest.fixture
def raw_tag_df():
    """
    Returns a sample raw 'tag' table DataFrame.
    """
    return pd.DataFrame({
        "tag": ["Assets", "Liabilities"],
        "version": ["us-gaap/2025", "us-gaap/2025"],
        "custom": [0, 0],
        "abstract": [0, 0],
        "datatype": ["monetary", "monetary"],
        "iord": ["I", "I"],
        "crdr": ["D", "C"],
        "tlabel": ["Assets Label", "Liabilities Label"],
        "doc": ["Assets doc", "Liabilities doc"]
    })

@pytest.fixture
def raw_pre_df():
    """
    Returns a sample raw 'pre' table DataFrame.
    """
    return pd.DataFrame({
        "adsh": ["0000000000-20-000001", "0000000000-20-000001"],
        "report": [1, 1],
        "line": [10, 20],
        "stmt": ["BS", "BS"],
        "inpth": [0, 0],
        "rfile": [1, 1],
        "tag": ["Assets", "Liabilities"],
        "version": ["us-gaap/2025", "us-gaap/2025"],
        "plabel": ["Assets Presentation Label", "Liabilities Presentation Label"]
    })
