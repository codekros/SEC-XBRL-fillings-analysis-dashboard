#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "data" / "processed" / "company_missing_values.csv"

def print_help():
    print("Usage:")
    print("  python check_missing.py <company_name>       Search for a specific company's missing KPIs.")
    print("  python check_missing.py --empty              List all companies missing ALL 18 KPIs.")
    print("  python check_missing.py --high               List companies with 15+ missing KPIs.")
    print("  python check_missing.py --complete           List a sample of companies with 100% complete KPIs.")
    print("\nExamples:")
    print("  python check_missing.py \"MICROSOFT\"")
    print("  python check_missing.py --empty")

def main():
    if not CSV_PATH.exists():
        print(f"Error: Stats file not found at {CSV_PATH}.")
        print("Please run the consolidation and analysis pipeline first.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    arg = sys.argv[1].strip().lower()
    df = pd.read_csv(CSV_PATH)

    # Command: --empty (List companies with 18 fully missing KPIs)
    if arg == "--empty":
        empty_cos = df[df["fully_missing_kpis_count"] == 18]
        print("=" * 70)
        print(f"COMPANIES MISSING ALL 18 KPIs ({len(empty_cos)}):")
        print("=" * 70)
        for idx, row in empty_cos.iterrows():
            print(f" - {row['company_name']} ({row['total_filings']} filings)")
        print("=" * 70)
        sys.exit(0)

    # Command: --high (List companies with 15+ fully missing KPIs)
    elif arg == "--high":
        high_cos = df[df["fully_missing_kpis_count"] >= 15]
        print("=" * 70)
        print(f"COMPANIES WITH 15+ MISSING KPIs ({len(high_cos)}):")
        print("=" * 70)
        # Group by count to print neatly
        for count in sorted(high_cos["fully_missing_kpis_count"].unique(), reverse=True):
            sub = high_cos[high_cos["fully_missing_kpis_count"] == count]
            print(f"\nMissing {count} KPIs ({len(sub)} companies):")
            for idx, row in sub.head(10).iterrows():
                print(f"  - {row['company_name']} ({row['total_filings']} filings)")
            if len(sub) > 10:
                print(f"    ... and {len(sub) - 10} more (view full list in CSV)")
        print("=" * 70)
        sys.exit(0)

    # Command: --complete (List sample of companies with 0 fully missing KPIs)
    elif arg == "--complete":
        complete_cos = df[df["fully_missing_kpis_count"] == 0]
        print("=" * 70)
        print(f"COMPANIES WITH 100% COMPLETE KPIs ({len(complete_cos)}):")
        print("=" * 70)
        print("Sample of complete companies:")
        for idx, row in complete_cos.head(30).iterrows():
            print(f"  - {row['company_name']} ({row['total_filings']} filings)")
        print(f"\n... and {len(complete_cos) - 30} more. Open company_missing_values.csv to see all.")
        print("=" * 70)
        sys.exit(0)

    # Check for help flags
    elif arg in ["-h", "--help"]:
        print_help()
        sys.exit(0)

    # Default: Search for a specific company
    search_query = " ".join(sys.argv[1:]).strip().upper()
    matches = df[df["company_name"].str.contains(search_query, case=False, na=False)]

    if matches.empty:
        print(f"No companies found matching '{search_query}'.")
        sys.exit(0)

    if len(matches) > 1:
        print(f"Found {len(matches)} matches. Please be more specific:")
        for idx, row in matches.head(15).iterrows():
            print(f" - {row['company_name']} (Filings: {row['total_filings']})")
        if len(matches) > 15:
            print(f" ... and {len(matches) - 15} more.")
        sys.exit(0)

    # Single match found
    row = matches.iloc[0]
    company_name = row["company_name"]
    total_filings = int(row["total_filings"])

    print("=" * 70)
    print(f"MISSINGNESS REPORT: {company_name}")
    print(f"Total Filings: {total_filings}")
    print("=" * 70)

    # Always missing
    always_miss = str(row["always_missing_kpis"]).split(", ") if pd.notna(row["always_missing_kpis"]) else []
    print(f"\n❌ ALWAYS MISSING KPIs ({len(always_miss)}):")
    if always_miss:
        for kpi in always_miss:
            print(f"  - {kpi}")
    else:
        print("  None! (All KPIs are populated in at least one filing)")

    # Partially missing
    part_miss = str(row["partially_missing_kpis"]).split(", ") if pd.notna(row["partially_missing_kpis"]) else []
    print(f"\n⚠️ PARTIALLY MISSING KPIs ({len(part_miss)}):")
    if part_miss:
        for kpi in part_miss:
            missing_count = int(row[f"{kpi}_missing_count"])
            missing_pct = float(row[f"{kpi}_missing_pct"])
            print(f"  - {kpi}: missing in {missing_count}/{total_filings} filings ({missing_pct}%)")
    else:
        print("  None!")

    # Fully complete KPIs
    kpi_cols = [
        "revenue", "cost_of_revenue", "gross_profit", "operating_income", 
        "depreciation_amortization", "income_before_tax", "net_income", 
        "total_assets", "total_liabilities", "equity", "receivables", 
        "payables", "inventory", "operating_cash_flow", "investing_cash_flow", 
        "financing_cash_flow", "capex", "ebitda"
    ]
    kpi_cols = [c for c in kpi_cols if f"{c}_missing_count" in row.index]
    
    complete_kpis = [k for k in kpi_cols if int(row[f"{k}_missing_count"]) == 0]
    print(f"\n✅ 100% COMPLETE KPIs ({len(complete_kpis)}):")
    if complete_kpis:
        for i in range(0, len(complete_kpis), 3):
            print("  - " + ", ".join(complete_kpis[i:i+3]))
    else:
        print("  None!")
    print("=" * 70)

if __name__ == "__main__":
    main()
