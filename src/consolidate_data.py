from pathlib import Path
import pandas as pd
import logging
import re

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
        "SalesRevenueServicesNet",
        "OperatingRevenues",
        "Revenue",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RevenueFromContractsWithCustomers",
        "InterestAndDividendIncomeOperating",
        "InterestIncomeOperating",
        "RevenuesTotal",
        "RealEstateRevenue",
        "OilAndGasRevenue",
        "FinancialServicesRevenue",
        "ElectricUtilityRevenue",
        "ServiceRevenue",
        "TotalRevenues",
        "RevenuesNetOfInterestExpense",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfSales",
        "CostOfServices",
        "CostOfServicesSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
        "CostOfSalesTotal",
        "CostOfDirectMaterialsAndLabor",
        "CostOfGoodsAndServicesSoldSubjectToOperatingLeases",
    ],
    "gross_profit": [
        "GrossProfit",
        "GrossMargin",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "ProfitLossFromOperatingActivities",
        "OperatingProfitLoss",
        "OperatingIncome",
        "OperatingLoss",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "AmortizationOfIntangibleAssets",
        "DepreciationAmortizationAndOther",
        "DepreciationAmortizationAndImpairment",
        "DepreciationExpense",
        "AmortizationExpense",
        "DepreciationAndAmortizationExpense",
        "AmortisationExpense",
        "DepreciationAmortisationExpense",
        "AmortisationOfIntangibleAssets",
        "DepreciationDepletionAndAmortizationExpense",
        "DepreciationNonproduction",
        "DepreciationProduction",
    ],
    "income_before_tax": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeignAndDomestic",
        "IncomeLossBeforeIncomeTaxesMinorityInterestAndExtraordinaryItems",
        "ProfitLossBeforeTax",
        "ProfitLossBeforeIncomeTax",
        "IncomeLossBeforeIncomeTaxes",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitLossAttributableToOwners",
        "ComprehensiveIncomeNetOfTax",
    ],
    "total_assets": [
        "Assets",
        "AssetsTotal",
    ],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesTotal",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity",
        "EquityAttributableToOwnersOfParent",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "AccountsAndNotesReceivableNetCurrent",
        "AccountsReceivableNet",
        "TradeAndOtherCurrentReceivables",
        "TradeReceivables",
        "CurrentTradeReceivables",
    ],
    "payables": [
        "AccountsPayableCurrent",
        "AccountsPayable",
        "TradeAndOtherCurrentPayables",
        "TradePayables",
        "CurrentTradePayables",
    ],
    "inventory": [
        "InventoryNet",
        "InventoryRawMaterials",
        "Inventories",
        "InventoriesTotal",
        "InventoryFinishedGoods",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivities",
        "CashFlowFromUsedInOperatingActivities",
        "NetCashFlowsFromUsedInOperatingActivities",
    ],
    "investing_cash_flow": [
        "NetCashProvidedByUsedInInvestingActivities",
        "CashFlowsFromUsedInInvestingActivities",
        "NetCashFlowsFromUsedInInvestingActivities",
    ],
    "financing_cash_flow": [
        "NetCashProvidedByUsedInFinancingActivities",
        "CashFlowsFromUsedInFinancingActivities",
        "NetCashFlowsFromUsedInFinancingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
        "PurchaseOfPropertyPlantAndEquipment",
        "PurchaseOfIntangibleAssets",
        "PaymentsForPurchaseOfPropertyPlantAndEquipment",
    ],
}

BALANCE_SHEET_KPIS = {"total_assets", "total_liabilities", "equity", "receivables", "payables", "inventory"}

def extract_quarter_kpis(quarter_dir: Path) -> pd.DataFrame:
    sub_path = quarter_dir / "sub.parquet"
    num_path = quarter_dir / "num.parquet"
    pre_path = quarter_dir / "pre.parquet"
    tag_path = quarter_dir / "tag.parquet"

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

    # --- TIER 2: PRESENTATION & TAXONOMY LABEL MATCHING ---
    # Load presentation (pre.parquet) and taxonomy (tag.parquet) if available to catch custom tags
    if pre_path.exists():
        try:
            pre_df = pd.read_parquet(pre_path, columns=["filing_id", "statement_type", "financial_tag", "presentation_label"])
            pre_df = pre_df[pre_df["presentation_label"].notna()]
            
            # Key label mappings
            label_rules = {
                "depreciation_amortization": (pre_df["statement_type"].isin(["IS", "CF"])) & (pre_df["presentation_label"].str.contains(r"depreciat|amortiz", case=False, na=False)),
                "capex": (pre_df["statement_type"] == "CF") & (pre_df["presentation_label"].str.contains(r"property.*plant.*equipment|capital expenditure", case=False, na=False)),
                "cost_of_revenue": (pre_df["statement_type"] == "IS") & (pre_df["presentation_label"].str.contains(r"cost of sales|cost of revenue|cost of goods", case=False, na=False)),
                "gross_profit": (pre_df["statement_type"] == "IS") & (pre_df["presentation_label"].str.contains(r"gross profit|gross margin", case=False, na=False)),
                "operating_income": (pre_df["statement_type"] == "IS") & (pre_df["presentation_label"].str.contains(r"operating income|operating loss|operating profit", case=False, na=False)),
            }
            
            for kpi, mask in label_rules.items():
                matched_tags = pre_df[mask]["financial_tag"].unique()
                for tag in matched_tags:
                    if tag not in tag_to_kpi:
                        tag_to_kpi[tag] = kpi
                        tag_priority[tag] = 90  # Higher priority than raw regex fallback
                        all_tags.append(tag)
        except Exception as e:
            logger.debug("Could not read pre.parquet in %s: %s", quarter_dir.name, e)

    if tag_path.exists():
        try:
            tag_df = pd.read_parquet(tag_path, columns=["financial_tag", "label"])
            tag_df = tag_df[tag_df["label"].notna()]
            
            tag_label_rules = {
                "depreciation_amortization": tag_df["label"].str.contains(r"depreciation.*amortization|depreciation expense|amortization expense", case=False, na=False),
                "revenue": tag_df["label"].str.contains(r"revenue from contract with customer|sales revenue net|total revenues", case=False, na=False),
            }
            for kpi, mask in tag_label_rules.items():
                matched_tags = tag_df[mask]["financial_tag"].unique()
                for tag in matched_tags:
                    if tag not in tag_to_kpi:
                        tag_to_kpi[tag] = kpi
                        tag_priority[tag] = 95
                        all_tags.append(tag)
        except Exception as e:
            logger.debug("Could not read tag.parquet in %s: %s", quarter_dir.name, e)

    # --- DYNAMIC REGEX FALLBACK TAG MATCHING ---
    unique_merged_tags = merged["financial_tag"].dropna().unique()
    unmapped_tags = [t for t in unique_merged_tags if t not in tag_to_kpi]
    
    forbidden = r"^(?!.*(?:IncreaseDecrease|Proceeds|Payments|Receipts|CashProvided|Adjustments|Reconciliation|EffectOf|CashEquivalents|Inflow|Outflow|OperatingActivities|InvestingActivities|FinancingActivities|Amortization|Depreciation))"
    
    fallback_rules = {
        "revenue": re.compile(forbidden + r".*(?:Revenue|Sales|Turnover).*$", re.IGNORECASE),
        "cost_of_revenue": re.compile(forbidden + r".*(?:CostOfGoods|CostOfSales|CostOfRevenue|CostOfServices).*$", re.IGNORECASE),
        "gross_profit": re.compile(forbidden + r"^(?:GrossProfit|GrossMargin)$", re.IGNORECASE),
        "operating_income": re.compile(forbidden + r"^(?:OperatingIncomeLoss|OperatingIncome|OperatingProfitLoss|ProfitLossFromOperatingActivities)$", re.IGNORECASE),
        "depreciation_amortization": re.compile(r".*(?:DepreciationAndAmortization|DepreciationExpense|AmortizationExpense|AmortisationExpense).*$", re.IGNORECASE),
        "income_before_tax": re.compile(forbidden + r".*(?:BeforeTax|BeforeIncomeTax|BeforeIncomeTaxes).*$", re.IGNORECASE),
        "net_income": re.compile(forbidden + r"^(?:NetIncomeLoss|ProfitLoss|NetIncome|ComprehensiveIncome)$", re.IGNORECASE),
        "total_assets": re.compile(forbidden + r"^(?:Assets|AssetsTotal)$", re.IGNORECASE),
        "total_liabilities": re.compile(forbidden + r"^(?:Liabilities|LiabilitiesTotal)$", re.IGNORECASE),
        "equity": re.compile(forbidden + r"^(?:StockholdersEquity|Equity|EquityAttributableToOwners)$", re.IGNORECASE),
        "receivables": re.compile(forbidden + r".*(?:AccountsReceivable|TradeReceivables|CurrentReceivables).*$", re.IGNORECASE),
        "payables": re.compile(forbidden + r".*(?:AccountsPayable|TradePayables|CurrentPayables).*$", re.IGNORECASE),
        "inventory": re.compile(forbidden + r".*(?:InventoryNet|Inventories|InventoryRawMaterials).*$", re.IGNORECASE),
        "operating_cash_flow": re.compile(r".*(?:CashProvidedByUsedInOperatingActivities|CashFlowsFromUsedInOperatingActivities|CashFlowFromUsedInOperatingActivities).*$", re.IGNORECASE),
        "investing_cash_flow": re.compile(r".*(?:CashProvidedByUsedInInvestingActivities|CashFlowsFromUsedInInvestingActivities).*$", re.IGNORECASE),
        "financing_cash_flow": re.compile(r".*(?:CashProvidedByUsedInFinancingActivities|CashFlowsFromUsedInFinancingActivities).*$", re.IGNORECASE),
        "capex": re.compile(r".*(?:PaymentsToAcquirePropertyPlantAndEquipment|PurchaseOfPropertyPlantAndEquipment|PaymentsForPurchaseOfPropertyPlantAndEquipment).*$", re.IGNORECASE),
    }
    
    for tag in unmapped_tags:
        for kpi, regex in fallback_rules.items():
            if regex.match(tag):
                tag_to_kpi[tag] = kpi
                tag_priority[tag] = 99  # Lowest priority so exact standard match is always preferred
                all_tags.append(tag)
                break

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

    # --- BI-DIRECTIONAL ACCOUNTING DERIVATIONS ---
    # Income Statement Identities
    cost_of_rev_fill = final_df["cost_of_revenue"].fillna(0.0)
    final_df["gross_profit"] = final_df["gross_profit"].fillna(
        final_df["revenue"] - cost_of_rev_fill
    )
    final_df["cost_of_revenue"] = final_df["cost_of_revenue"].fillna(
        final_df["revenue"] - final_df["gross_profit"]
    )
    final_df["revenue"] = final_df["revenue"].fillna(
        final_df["gross_profit"] + final_df["cost_of_revenue"]
    )

    # Balance Sheet Identities
    final_df["equity"] = final_df["equity"].fillna(
        final_df["total_assets"] - final_df["total_liabilities"]
    )
    final_df["total_liabilities"] = final_df["total_liabilities"].fillna(
        final_df["total_assets"] - final_df["equity"]
    )
    final_df["total_assets"] = final_df["total_assets"].fillna(
        final_df["total_liabilities"] + final_df["equity"]
    )

    # EBITDA & EBT Derivations
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

    # --- SYSTEMATIC DEDUPLICATION ---
    # We group by (company_name, report_period) and keep the single most complete filing.
    # We prioritize primary filing forms (10-K, 10-Q, 20-F) over secondary disclosures/amendments.
    
    # 1. Count non-null KPIs for each row
    kpi_cols = [
        "revenue", "cost_of_revenue", "gross_profit", "operating_income", 
        "depreciation_amortization", "income_before_tax", "net_income", 
        "total_assets", "total_liabilities", "equity", "receivables", 
        "payables", "inventory", "operating_cash_flow", "investing_cash_flow", 
        "financing_cash_flow", "capex", "ebitda"
    ]
    kpis_present = [col for col in kpi_cols if col in consolidated_df.columns]
    consolidated_df["_non_null_kpi_count"] = consolidated_df[kpis_present].notna().sum(axis=1)

    # 2. Assign form priority (lower priority value means higher preference)
    def get_form_priority(form):
        form = str(form).upper()
        if "10-K" in form or "20-F" in form:
            return 0
        elif "10-Q" in form:
            return 1
        elif "6-K" in form:
            return 2
        elif "8-K" in form:
            return 4
        else:
            return 3  # F-1, S-1, etc.

    consolidated_df["_form_priority"] = consolidated_df["form_type"].apply(get_form_priority)

    # 3. Sort:
    #   - company_name (ascending)
    #   - report_period (ascending)
    #   - _non_null_kpi_count (descending, so most complete first)
    #   - _form_priority (ascending, so primary forms first)
    #   - filing_id (descending, so latest filing id if equal completeness)
    consolidated_df = consolidated_df.sort_values(
        by=["company_name", "report_period", "_non_null_kpi_count", "_form_priority", "filing_id"],
        ascending=[True, True, False, True, False]
    )

    # 4. Drop duplicates, keeping the first (most complete primary form)
    initial_rows = len(consolidated_df)
    consolidated_df = consolidated_df.drop_duplicates(subset=["company_name", "report_period"], keep="first")
    removed_rows = initial_rows - len(consolidated_df)
    logger.info("Deduplication: Removed %d duplicate period rows.", removed_rows)

    # 5. Clean up temporary sorting columns
    consolidated_df = consolidated_df.drop(columns=["_non_null_kpi_count", "_form_priority"])
    consolidated_df = consolidated_df.reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    consolidated_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Consolidated KPIs successfully saved -> %s", OUTPUT_FILE)
    logger.info("Total rows: %d, Companies: %d", len(consolidated_df), consolidated_df["company_name"].nunique())

if __name__ == "__main__":
    main()
