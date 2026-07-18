from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_TABLES = (
    "sub",
    "num",
    "tag",
    "pre",
)

TABLE_CONFIG = {

    "sub": {

        "rename_columns": {
            "adsh": "filing_id",
            "cik": "company_id",
            "name": "company_name",
            "sic": "sic_code",
            "countryba": "country",
            "stprba": "state",
            "cityba": "city",
            "zipba": "zip_code",
            "bas1": "address_line1",
            "bas2": "address_line2",
            "period": "report_period",
            "fy": "fiscal_year",
            "fp": "fiscal_period",
            "filed": "filing_date",
            "form": "form_type",
        },

        "business_keys": [
            "filing_id",
        ],

        "date_columns": [
            "accepted",
            "filing_date",
            "report_period",
        ],

        "categorical_columns": [
            "form_type",
            "country",
            "countryinc",
            "state",
            "stprinc",
        ],

        "numeric_columns": [
            "fiscal_year",
        ],

        "nullable_columns": [],
    },

    "num": {

        "rename_columns": {
            "adsh": "filing_id",
            "tag": "financial_tag",
            "version": "taxonomy_version",
            "ddate": "report_date",
            "qtrs": "quarters",
            "uom": "unit",
            "segments": "segment",
            "coreg": "co_registrant",
            "value": "reported_value",
            "footnote": "footnote",
        },

        "business_keys": [
            "filing_id",
            "financial_tag",
            "taxonomy_version",
            "report_date",
            "quarters",
            "unit",
        ],

        "date_columns": [
            "report_date",
        ],

        "categorical_columns": [
            "financial_tag",
            "taxonomy_version",
            "unit",
            "co_registrant",
        ],

        "numeric_columns": [
            "reported_value",
            "quarters",
        ],

        "nullable_columns": [
            "co_registrant",
            "footnote",
        ],
    },

    "tag": {

        "rename_columns": {
            "tag": "financial_tag",
            "version": "taxonomy_version",
            "custom": "is_custom",
            "abstract": "is_abstract",
            "datatype": "data_type",
            "iord": "balance_type",
            "crdr": "normal_balance",
            "tlabel": "label",
            "doc": "description",
        },

        "business_keys": [
            "financial_tag",
            "taxonomy_version",
        ],

        "date_columns": [],

        "categorical_columns": [
            "data_type",
            "normal_balance",
        ],

        "numeric_columns": [],

        "nullable_columns": [
            "is_custom",
        ],
    },

    "pre": {

        "rename_columns": {
            "adsh": "filing_id",
            "report": "report_number",
            "line": "line_number",
            "stmt": "statement_type",
            "inpth": "is_parenthetical",
            "rfile": "report_file",
            "tag": "financial_tag",
            "version": "taxonomy_version",
            "plabel": "presentation_label",
        },

        "business_keys": [
            "filing_id",
            "report_number",
            "line_number",
            "financial_tag",
        ],

        "date_columns": [],

        "categorical_columns": [
            "statement_type",
            "financial_tag",
            "taxonomy_version",
        ],

        "numeric_columns": [
            "line_number",
            "report_number",
            "is_parenthetical",
            "report_file",
        ],

        "nullable_columns": [],
    },
}