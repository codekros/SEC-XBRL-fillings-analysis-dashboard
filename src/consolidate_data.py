from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("consolidate_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "consolidated_kpis.parquet"

KPI_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "OperatingRevenues",
        "Revenue",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractsWithCustomers",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfSales",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
    ],
    "income_before_tax": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeignAndDomestic",
        "IncomeLossBeforeIncomeTaxesMinorityInterestAndExtraordinaryItems",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "AccountsAndNotesReceivableNetCurrent",
        "AccountsReceivableNet",
    ],
    "payables": [
        "AccountsPayableCurrent",
        "AccountsPayable",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryRawMaterials",
        "Inventories",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "investing_cash_flow": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],
    "financing_cash_flow": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
}

BALANCE_SHEET_KPIS = {"total_assets", "total_liabilities", "equity", "receivables", "payables", "inventory"}

def extract_quarter_kpis(quarter_dir: Path) -> pd.DataFrame:
    sub_path = quarter_dir / "sub.parquet"
    num_path = quarter_dir / "num.parquet"

    if not sub_path.exists() or not num_path.exists():
        logger.warning("Missing parquet files in %s. Skipping...", quarter_dir.name)
        return pd.DataFrame()

    logger.info("Consolidating quarter: %s", quarter_dir.name)

    sub_df = pd.read_parquet(
        sub_path,
        columns=[
            "filing_id",
            "company_id",
            "company_name",
            "fiscal_year",
            "fiscal_period",
            "form_type",
            "report_period",
        ],
    )

    num_df = pd.read_parquet(
        num_path,
        columns=[
            "filing_id",
            "financial_tag",
            "report_date",
            "quarters",
            "unit",
            "segment",
            "co_registrant",
            "reported_value",
        ],
    )

    num_df = num_df[
        (num_df["segment"].isna() | (num_df["segment"] == "")) &
        (num_df["co_registrant"].isna() | (num_df["co_registrant"] == ""))
    ]

    merged = pd.merge(num_df, sub_df, on="filing_id")

    merged = merged[merged["report_date"] == merged["report_period"]]

    if merged.empty:
        return pd.DataFrame()

    all_tags = []
    tag_to_kpi = {}
    tag_priority = {}
    
    for kpi, tags in KPI_TAGS.items():
        all_tags.extend(tags)
        for idx, tag in enumerate(tags):
            tag_to_kpi[tag] = kpi
            tag_priority[tag] = idx

    merged = merged[merged["financial_tag"].isin(all_tags)].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["kpi"] = merged["financial_tag"].map(tag_to_kpi)
    merged["tag_priority"] = merged["financial_tag"].map(tag_priority)

    is_bs = merged["kpi"].isin(BALANCE_SHEET_KPIS)
    
    bs_df = merged[is_bs & (merged["quarters"] == 0)].copy()
    bs_df["quarter_score"] = 0
    
    flow_df = merged[~is_bs & (merged["quarters"] > 0)].copy()
    
    flow_df["quarter_score"] = 1
    is_10k = flow_df["form_type"] == "10-K"
    flow_df.loc[is_10k & (flow_df["quarters"] == 4), "quarter_score"] = 0
    flow_df.loc[(~is_10k) & (flow_df["quarters"] == 1), "quarter_score"] = 0

    processed = pd.concat([bs_df, flow_df], ignore_index=True)
    if processed.empty:
        return pd.DataFrame()

    processed = processed.sort_values(
        by=["filing_id", "kpi", "quarter_score", "tag_priority"]
    )

    best_matches = processed.drop_duplicates(subset=["filing_id", "kpi"], keep="first")

    kpi_pivot = best_matches.pivot(
        index="filing_id",
        columns="kpi",
        values="reported_value"
    )

    metadata = sub_df.set_index("filing_id")
    final_df = metadata.join(kpi_pivot, how="inner").reset_index()

    for kpi in KPI_TAGS.keys():
        if kpi not in final_df.columns:
            final_df[kpi] = None

    if "ebitda" not in final_df.columns:
        final_df["ebitda"] = None

    final_df["company_name"] = final_df["company_name"].str.strip()

    cost_of_rev_fill = final_df["cost_of_revenue"].fillna(0.0)
    final_df["gross_profit"] = final_df["gross_profit"].fillna(
        final_df["revenue"] - cost_of_rev_fill
    )

    final_df["equity"] = final_df["equity"].fillna(
        final_df["total_assets"] - final_df["total_liabilities"]
    )
    final_df["total_liabilities"] = final_df["total_liabilities"].fillna(
        final_df["total_assets"] - final_df["equity"]
    )
    final_df["total_assets"] = final_df["total_assets"].fillna(
        final_df["total_liabilities"] + final_df["equity"]
    )

    da_fill = final_df["depreciation_amortization"].fillna(0.0)
    derived_ebitda = final_df["operating_income"] + da_fill
    derived_ebitda = derived_ebitda.fillna(final_df["net_income"])
    final_df["ebitda"] = final_df["ebitda"].fillna(derived_ebitda)

    derived_ebt = final_df["operating_income"].fillna(final_df["net_income"])
    final_df["income_before_tax"] = final_df["income_before_tax"].fillna(derived_ebt)

    return final_df

def main() -> None:
    logger.info("Starting SEC XBRL Data Consolidation Pipeline")

    if not PROCESSED_DIR.exists():
        logger.error("Processed directory not found: %s", PROCESSED_DIR)
        return

    quarter_dirs = sorted(
        [d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.lower() != "consolidated"]
    )

    all_quarter_dfs = []
    for q_dir in quarter_dirs:
        name = q_dir.name.lower()
        if len(name) == 6 and name[:4].isdigit() and name[4] == "q" and name[5].isdigit():
            q_df = extract_quarter_kpis(q_dir)
            if not q_df.empty:
                all_quarter_dfs.append(q_df)

    if not all_quarter_dfs:
        logger.error("No processed quarter data found to consolidate.")
        return

    consolidated_df = pd.concat(all_quarter_dfs, ignore_index=True)

    consolidated_df = consolidated_df.sort_values(
        by=["company_name", "report_period"], ascending=[True, True]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    consolidated_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Consolidated KPIs successfully saved -> %s", OUTPUT_FILE)
    logger.info("Total rows: %d, Companies: %d", len(consolidated_df), consolidated_df["company_name"].nunique())

if __name__ == "__main__":
    main()
