import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import joblib



st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    h1, h2, h3, h4 {
        color: #0F172A !important;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 15px 20px;
        flex: 1;
        min-width: 170px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    
    .kpi-card:hover {
        border: 1px solid #CBD5E1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .delta-positive {
        color: #059669;
    }
    
    .delta-negative {
        color: #DC2626;
    }
    
    .delta-neutral {
        color: #64748B;
    }
    
    .dashboard-section {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .tag-highlight {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #475569;
    }
    </style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "consolidated_kpis.parquet"

@st.cache_data
def load_financial_data(file_path: Path, mtime: float):
    if not file_path.exists():
        st.error(f"Consolidated KPIs data file not found. Please run the consolidation script first.")
        st.stop()
    df = pd.read_parquet(file_path)
    
    # Filter out companies missing >= 15 KPIs (SPACs, ETFs, IFRS, and inactive shell companies)
    kpis = [
        "revenue", "cost_of_revenue", "gross_profit", "operating_income", 
        "depreciation_amortization", "income_before_tax", "net_income", 
        "total_assets", "total_liabilities", "equity", "receivables", 
        "payables", "inventory", "operating_cash_flow", "investing_cash_flow", 
        "financing_cash_flow", "capex", "ebitda"
    ]
    kpis_present = [col for col in kpis if col in df.columns]
    company_na_counts = df.groupby("company_name")[kpis_present].apply(lambda x: x.isna().all().sum())
    valid_companies = company_na_counts[company_na_counts < 15].index
    df = df[df["company_name"].isin(valid_companies)].copy()

    df["report_period"] = pd.to_datetime(df["report_period"])
    
    df["receivables"] = df["receivables"].fillna(0.0)
    df["inventory"] = df["inventory"].fillna(0.0)
    df["payables"] = df["payables"].fillna(0.0)
    df["capex"] = df["capex"].fillna(0.0)

    assets_safe = df["total_assets"].replace(0, np.nan)
    liabilities_safe = df["total_liabilities"].replace(0, np.nan)
    equity_safe = df["equity"].replace(0, np.nan)
    revenue_safe = df["revenue"].replace(0, np.nan)

    df["debt_to_assets"] = (df["total_liabilities"] / assets_safe).fillna(0.0)
    df["operating_margin"] = (df["operating_income"] / revenue_safe).fillna(0.0)
    df["roa"] = (df["net_income"] / assets_safe).fillna(0.0)
    df["asset_turnover"] = (df["revenue"] / assets_safe).fillna(0.0)
    df["equity_multiplier"] = (df["total_assets"] / equity_safe).fillna(1.0)
    df["sloan_ratio"] = ((df["net_income"] - df["operating_cash_flow"]) / assets_safe).fillna(0.0)
    df["receivables_to_payables"] = (df["receivables"] / df["payables"].replace(0, np.nan)).fillna(0.0)
    df["cash_flow_to_debt"] = (df["operating_cash_flow"] / liabilities_safe).fillna(0.0)

    FEATURE_COLUMNS_PRE = [
        "debt_to_assets",
        "operating_margin",
        "roa",
        "asset_turnover",
        "equity_multiplier",
        "sloan_ratio",
        "receivables_to_payables",
        "cash_flow_to_debt"
    ]
    for col in FEATURE_COLUMNS_PRE:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        
    return df

df = load_financial_data(DATA_PATH, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)
all_companies = sorted(df["company_name"].unique())

FEATURE_COLUMNS_PRE = [
    "debt_to_assets",
    "operating_margin",
    "roa",
    "asset_turnover",
    "equity_multiplier",
    "sloan_ratio",
    "receivables_to_payables",
    "cash_flow_to_debt"
]
feature_medians = df[FEATURE_COLUMNS_PRE].median().to_dict()
feature_stds = df[FEATURE_COLUMNS_PRE].std().replace(0.0, 1.0).to_dict()

feature_directions = {
    "debt_to_assets": 1,
    "equity_multiplier": 1,
    "sloan_ratio": 1,
    "roa": -1,
    "operating_margin": -1,
    "asset_turnover": -1,
    "cash_flow_to_debt": -1,
    "receivables_to_payables": -1
}


@st.cache_resource
def load_ml_model():
    model_path = PROJECT_ROOT / "src" / "models" / "distress_classifier.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

ml_model_payload = load_ml_model()

def format_currency(val):
    if pd.isna(val) or val is None:
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f} B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f} M"
    else:
        return f"{sign}${abs_val:,.0f}"

def format_percent_val(val):
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.1f}%"

def format_de_val(val):
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.2f}x"

st.sidebar.markdown("<h3 style='text-align: center; color:#0F172A;'>Financial Intelligence</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("Select Focus Company")
selected_company = st.sidebar.selectbox(
    "Security Lookup",
    all_companies,
    index=all_companies.index("MICROSOFT CORP") if "MICROSOFT CORP" in all_companies else 0
)

company_df = df[df["company_name"] == selected_company].sort_values("report_period").reset_index(drop=True)
latest_idx = len(company_df) - 1
latest_row = company_df.iloc[latest_idx]

st.sidebar.markdown(f"""
<div style='background-color:#F8FAFC; padding:15px; border-radius:6px; border:1px solid #E2E8F0; margin-top:10px; color:#0F172A;'>
    <p style='margin:0; font-size:0.85rem; color:#64748B;'><b>Filer Identifiers:</b></p>
    <p style='margin:4px 0; font-size:0.9rem;'><b>CIK:</b> {latest_row['company_id']}</p>
    <p style='margin:4px 0; font-size:0.9rem;'><b>Filing Form:</b> {latest_row['form_type']}</p>
    <p style='margin:4px 0; font-size:0.9rem;'><b>Filing Period:</b> {latest_row['report_period'].strftime('%Y-%m-%d')}</p>
    <p style='margin:4px 0; font-size:0.9rem;'><b>Fiscal Period:</b> FY{latest_row['fiscal_year']} {latest_row['fiscal_period']}</p>
</div>
""", unsafe_allow_html=True)

# --- DATA INTEGRITY PROFILE ---
comp_name_upper = selected_company.upper()
form_upper = str(latest_row['form_type']).upper()

labels = []
explanations = []

is_etf = any(kw in comp_name_upper for kw in ["ETF", "TRUST", "FUND", "POOL", "GRANTOR"])
is_ifrs = "20-F" in form_upper
is_spac = any(kw in comp_name_upper for kw in ["ACQUISITION", "SPAC", "SHELL", "BLANK CHECK"])
is_financial = any(kw in comp_name_upper for kw in ["BANK", "INSURANCE", "FINANCIAL", "INVESTMENT", "MUTUAL", "CAPITAL", "LENDING", "CREDIT", "SAVINGS"]) and not is_etf

nan_cols = [col for col in ["revenue", "cost_of_revenue", "receivables", "payables", "inventory", "capex"] if col in latest_row.index and pd.isna(latest_row[col])]

if is_etf:
    labels.append("ETF / Investment Trust")
    explanations.append("💼 **ETF/Trust:** Holds commodities (e.g. gold, crypto) or stock baskets, and does not have standard commercial operational revenues, inventory, cost of revenue, or capex.")
elif is_ifrs:
    labels.append("Foreign Private Issuer (IFRS)")
    explanations.append("🌍 **IFRS Taxonomy:** Files using non-US GAAP standards. US GAAP metrics are mostly blank except shared tags like Assets.")
elif is_spac:
    labels.append("SPAC / Shell Entity")
    explanations.append("🐚 **Shell / SPAC:** Pre-merger acquisition vehicle or inactive company with no standard commercial operations.")
elif is_financial:
    labels.append("Financial Institution")
    explanations.append("🏦 **Financial Operations:** Banks and investment firms report interest/fee income rather than standard revenue/cost of sales, and do not carry physical inventory.")
elif "inventory" in nan_cols:
    labels.append("Service / Technology Firm")
    explanations.append("💻 **Services/Software:** Naturally holds no physical inventory, which explains why the inventory metric is N/A.")

if not labels:
    labels.append("Standard Operating Corporation")

if nan_cols:
    expl = f"⚠️ **N/A Metrics:** {', '.join(nan_cols)} are N/A. "
    if is_etf:
        expl += "Expected due to trust structure."
    elif is_ifrs:
        expl += "Expected due to IFRS reporting."
    elif is_spac:
        expl += "Expected due to shell/SPAC status."
    elif is_financial:
        expl += "Expected due to banking/financial model."
    elif "inventory" in nan_cols and len(nan_cols) == 1:
        expl += "Expected due to service-oriented business."
    else:
        expl += "Due to non-standard or missing disclosures in raw SEC tables."
    explanations.append(expl)
else:
    explanations.append("✅ **100% Data Integrity:** All core financial metrics are successfully populated.")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Integrity Profile")
for label in labels:
    st.sidebar.markdown(f"<span class='tag-highlight'>{label}</span>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
for exp in explanations:
    st.sidebar.info(exp)


peer_companies = [c for c in all_companies if c != selected_company]

def get_derived_ratios(row):
    rev = row["revenue"]
    has_rev = not pd.isna(rev) and rev != 0.0
    
    ebitda = row["ebitda"] if not pd.isna(row["ebitda"]) else 0.0
    ocf = row["operating_cash_flow"] if not pd.isna(row["operating_cash_flow"]) else None
    capex = row["capex"] if not pd.isna(row["capex"]) else None
    fcf = (ocf - capex) if ocf is not None and capex is not None else 0.0
    
    g_margin = (row["gross_profit"] / rev * 100) if has_rev and not pd.isna(row["gross_profit"]) else None
    ebitda_margin = (ebitda / rev * 100) if has_rev and not pd.isna(row["ebitda"]) else None
    n_margin = (row["net_income"] / rev * 100) if has_rev and not pd.isna(row["net_income"]) else None
    
    equity = row["equity"]
    de = (row["total_liabilities"] / equity) if not pd.isna(equity) and equity != 0.0 and not pd.isna(row["total_liabilities"]) else None
    
    fcf_conv = (fcf / ebitda * 100) if ebitda != 0 else None
    
    return g_margin, ebitda_margin, n_margin, de, fcf_conv, fcf


@st.cache_data
def build_history_df(company_df: pd.DataFrame, mtime: float) -> pd.DataFrame:
    rows = []
    for _, row in company_df.iterrows():
        gm, em, nm, de, fcfc, fcf = get_derived_ratios(row)
        rows.append({
            "Period": row["report_period"].strftime("%Y-%m-%d"),
            "Quarter": f"FY{row['fiscal_year']} {row['fiscal_period']}",
            "Revenue": row["revenue"],
            "Gross Margin (%)": gm,
            "EBITDA Margin (%)": em,
            "Net Profit Margin (%)": nm,
            "Total Assets": row["total_assets"],
            "Total Liabilities": row["total_liabilities"],
            "Equity": row["equity"],
            "Debt to Equity": de,
            "Operating Cash Flow": row["operating_cash_flow"],
            "Free Cash Flow": fcf,
            "FCF Conversion (%)": fcfc
        })
    return pd.DataFrame(rows)

latest_gm, latest_em, latest_nm, latest_de, latest_fcfc, latest_fcf = get_derived_ratios(latest_row)

prev_row = company_df.iloc[latest_idx - 1] if latest_idx > 0 else None
if prev_row is not None:
    prev_gm, prev_em, prev_nm, prev_de, prev_fcfc, prev_fcf = get_derived_ratios(prev_row)
else:
    prev_gm, prev_em, prev_nm, prev_de, prev_fcfc, prev_fcf = None, None, None, None, None, None

lat_assets = float(latest_row["total_assets"]) if not pd.isna(latest_row["total_assets"]) and latest_row["total_assets"] != 0 else 1.0
lat_equity = float(latest_row["equity"]) if not pd.isna(latest_row["equity"]) and latest_row["equity"] != 0 else 1.0
lat_rev = float(latest_row["revenue"]) if not pd.isna(latest_row["revenue"]) and latest_row["revenue"] != 0 else 1.0
lat_liab = float(latest_row["total_liabilities"]) if not pd.isna(latest_row["total_liabilities"]) and latest_row["total_liabilities"] != 0 else 1.0

header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown(f"<h1 style='margin-bottom:0;'>{selected_company.title()}</h1>", unsafe_allow_html=True)
    
    net_inc = latest_row["net_income"]
    net_inc_str = format_currency(net_inc)
    
    if prev_row is not None and prev_row["net_income"] and net_inc is not None:
        pct_change = ((net_inc - prev_row["net_income"]) / abs(prev_row["net_income"])) * 100
        delta_class = "delta-positive" if pct_change >= 0 else "delta-negative"
        delta_icon = "▲" if pct_change >= 0 else "▼"
        delta_html = f"<span class='{delta_class}' style='font-size: 1.1rem; font-weight: 600; margin-left: 10px;'>{delta_icon} {pct_change:+.1f}%</span>"
    else:
        delta_html = "<span class='delta-neutral' style='font-size: 1.1rem; margin-left: 10px;'>• No prior period</span>"
        
    st.markdown(f"<div style='font-size: 1.8rem; font-weight: 700; margin-top: 5px; color:#0F172A;'>{net_inc_str} {delta_html}</div>", unsafe_allow_html=True)
    
    cleaned_name = selected_company.split()[0].lower()
    website_link = f"https://www.{cleaned_name}.com"
    sec_link = f"https://www.sec.gov/edgar/searchedgar/companysearch?CIK={latest_row['company_id']}"
    st.markdown(f"""
    <div style='margin-top: 10px; font-size: 0.9rem; color:#64748B; margin-bottom: 20px;'>
        <a href='{website_link}' target='_blank' style='margin-right: 15px; color:#2563EB; text-decoration:none;'>🌐 {cleaned_name}.com</a>
        <a href='{sec_link}' target='_blank' style='margin-right: 15px; color:#2563EB; text-decoration:none;'>🏛️ SEC CIK: {latest_row['company_id']}</a>
        <span style='margin-right: 15px;'>📄 Form: {latest_row['form_type']}</span>
        <span>📅 Period: {latest_row['report_period'].strftime('%Y-%m-%d')}</span>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    hist_df = build_history_df(company_df, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)
    csv_content = hist_df.to_csv(index=False).encode('utf-8')
    
    st.markdown("<div style='text-align: right; margin-top: 20px;'>", unsafe_allow_html=True)
    st.download_button(
        label="Export Report Data (CSV)",
        data=csv_content,
        file_name=f"{selected_company.lower().replace(' ', '_')}_financial_report.csv",
        mime="text/csv",
        key="header_export_btn"
    )
    st.markdown("</div>", unsafe_allow_html=True)

tab_overview, tab_peer, tab_ml = st.tabs([
    "Overview",
    "Peers",
    "Machine Learning"
])

def get_growth_html(curr_val, prev_val, higher_is_better=True, formatter=format_currency):
    if prev_val is None or curr_val is None or pd.isna(curr_val) or pd.isna(prev_val) or prev_val == 0:
        return '<span class="kpi-delta delta-neutral">• No prior period</span>'
    
    pct = ((curr_val - prev_val) / abs(prev_val)) * 100
    is_positive = pct >= 0
    
    if (is_positive and higher_is_better) or (not is_positive and not higher_is_better):
        delta_class = "delta-positive"
        icon = "▲"
    else:
        delta_class = "delta-negative"
        icon = "▼"
        
    return f'<span class="kpi-delta {delta_class}">{icon} {pct:+.1f}%</span>'

with tab_overview:
    dup_margin = (latest_row["net_income"] / lat_rev * 100) if not pd.isna(latest_row["net_income"]) and lat_rev != 0 else 0.0
    dup_turnover = lat_rev / lat_assets
    dup_multiplier = lat_assets / lat_equity
    dup_roe = (latest_row["net_income"] / lat_equity * 100) if not pd.isna(latest_row["net_income"]) and lat_equity != 0 else 0.0
    
    if latest_row["net_income"] is not None and latest_row["operating_cash_flow"] is not None and latest_row["total_assets"] and latest_row["total_assets"] != 0:
        sloan_ratio = (latest_row["net_income"] - latest_row["operating_cash_flow"]) / latest_row["total_assets"]
        sloan_pct = sloan_ratio * 100
        if -10.0 <= sloan_pct <= 10.0:
            sloan_status = "SAFE"
            sloan_color = "#059669"
            sloan_desc = "High Earnings Quality: Net income matches cash collection. Low accruals risk."
        elif sloan_pct > 10.0:
            sloan_status = "ACCRUAL WARNING"
            sloan_color = "#DC2626"
            sloan_desc = "Low Earnings Quality: Profits driven by non-cash accruals. High cash conversion risk."
        else:
            sloan_status = "CASH ACCRETIVE"
            sloan_color = "#D97706"
            sloan_desc = "Cash Accretive: Cash generation significantly outstrips reported net income."
    else:
        sloan_pct = None
        sloan_status = "INSUFFICIENT DATA"
        sloan_color = "#64748B"
        sloan_desc = "Accruals check requires Net Income, Operating Cash Flow, and Total Assets."

    opex_alerts = []
    opex_curr = latest_row["revenue"] - latest_row["operating_income"] if latest_row["revenue"] is not None and latest_row["operating_income"] is not None else None
    opex_prev = prev_row["revenue"] - prev_row["operating_income"] if prev_row is not None and prev_row["revenue"] is not None and prev_row["operating_income"] is not None else None
    rev_growth = ((latest_row["revenue"] - prev_row["revenue"]) / abs(prev_row["revenue"]) * 100) if prev_row is not None and prev_row["revenue"] and latest_row["revenue"] is not None else None
    
    if opex_curr is not None and opex_prev is not None and opex_prev != 0:
        opex_growth = ((opex_curr - opex_prev) / abs(opex_prev) * 100)
    else:
        opex_growth = None
        
    if latest_gm is not None and prev_gm is not None and latest_gm < prev_gm:
        opex_alerts.append(("Gross Margin Compression", f"Gross margin declined by {prev_gm - latest_gm:.1f}% period-over-period."))
    
    if opex_growth is not None and rev_growth is not None and opex_growth > rev_growth:
        opex_alerts.append(("Negative Operating Leverage", f"OpEx grew at {opex_growth:.1f}% outstripping revenue growth of {rev_growth:.1f}%."))
        
    if rev_growth is not None and rev_growth < 0:
        opex_alerts.append(("Sales Contraction", f"Revenues contracted by {abs(rev_growth):.1f}% compared to prior period."))
        
    pros = []
    cons = []
    
    if sloan_status == "SAFE":
        pros.append("High earnings quality: net income balances well with operating cash flow (low accrual ratio).")
    elif sloan_status == "ACCRUAL WARNING":
        cons.append("Earnings quality warning: net income is driven by non-cash accruals rather than cash collections.")
        
    if dup_roe > 15.0:
        pros.append(f"Strong return on equity (ROE) of {dup_roe:.1f}% representing efficient capital returns.")
    elif dup_roe < 5.0:
        cons.append(f"Suboptimal Return on Equity: ROE is low at {dup_roe:.1f}%.")
        
    if latest_de is not None:
        if latest_de < 0.5:
            pros.append(f"Conservative capital structure: leverage is low with a Debt-to-Equity of {latest_de:.2f}x.")
        elif latest_de > 1.5:
            cons.append(f"Elevated debt profile: Debt-to-Equity is high ({latest_de:.2f}x). Watch solvency cushions.")
            
    if latest_em is not None:
        if latest_em > 20.0:
            pros.append(f"Healthy operating profitability: EBITDA margin is strong at {latest_em:.1f}%.")
        elif latest_em < 5.0:
            cons.append(f"Compressed operating profitability: EBITDA margin is low at {latest_em:.1f}%.")
            
    for title, desc in opex_alerts:
        cons.append(f"Concern: {title} – {desc}")
        
    if not pros:
        pros.append("Baseline financial structure is stable. Solvency and operations parameters are standard.")
    if not cons:
        cons.append("No major operating alerts or margin compressions detected for the current period.")

    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    grid_col, about_col = st.columns([2, 1])
    
    sloan_val_str = f"{sloan_pct:+.2f}%" if sloan_pct is not None else "N/A"
    roe_val_str = f"{dup_roe:.2f}%" if dup_roe != 0 else "N/A"
    
    with grid_col:
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background-color: #E2E8F0; border: 1px solid #E2E8F0; border-radius: 6px; overflow: hidden;">
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Total Assets</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_currency(latest_row['total_assets'])}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Revenue</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_currency(latest_row['revenue'])}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Gross Margin</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_percent_val(latest_gm)}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">EBITDA Margin</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_percent_val(latest_em)}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Net Margin</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_percent_val(latest_nm)}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Debt to Equity</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_de_val(latest_de)}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">FCF Conversion</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{format_percent_val(latest_fcfc)}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">Sloan Accruals</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{sloan_val_str}</span>
            </div>
            <div style="background-color: #FFFFFF; padding: 15px 20px;">
                <span style="font-size: 0.75rem; color: #64748B; display: block; margin-bottom: 5px; font-weight:600;">ROE</span>
                <span style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{roe_val_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with about_col:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; height: 100%; color:#0F172A;">
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">About</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin:0;">
                The company is a registered filer with the SEC. Active registration exhibits filing period ending <b>{latest_row['report_period'].strftime('%Y-%m-%d')}</b> under form type <b>{latest_row['form_type']}</b>.
            </p>
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-top: 12px; margin-bottom: 4px;">Key Insights</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin:0;">
                • Accruals Quality is deemed <b><span style='color:{sloan_color};'>{sloan_status}</span></b>.<br/>
                • Operating efficiency shows <b>{len(opex_alerts)} alert(s)</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    st.markdown("### Strengths & Concerns")
    
    pro_col, con_col = st.columns(2)
    with pro_col:
        pros_html = "<div style='background-color:#E6FFFA; border:1px solid #319795; padding:20px; border-radius:6px; min-height:180px; color:#234E52;'>"
        pros_html += "<h4 style='color:#234E52; margin-top:0; font-weight:600;'>PROS</h4>"
        pros_html += "<ul style='margin-bottom:0; font-size:0.85rem; line-height:1.5; padding-left:20px;'>"
        for p in pros:
            pros_html += f"<li style='margin-bottom:6px;'>{p}</li>"
        pros_html += "</ul></div>"
        st.markdown(pros_html, unsafe_allow_html=True)
        
    with con_col:
        cons_html = "<div style='background-color:#FFF5F5; border:1px solid #E53E3E; padding:20px; border-radius:6px; min-height:180px; color:#742A2A;'>"
        cons_html += "<h4 style='color:#742A2A; margin-top:0; font-weight:600;'>CONS</h4>"
        cons_html += "<ul style='margin-bottom:0; font-size:0.85rem; line-height:1.5; padding-left:20px;'>"
        for c in cons:
            cons_html += f"<li style='margin-bottom:6px;'>{c}</li>"
        cons_html += "</ul></div>"
        st.markdown(cons_html, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    st.markdown("### Income Statement Trajectory")
    
    fig_is = go.Figure()
    fig_is.add_trace(go.Scatter(
        x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
        y=company_df["revenue"],
        name="Revenue",
        mode="lines+markers",
        line=dict(color="#2563EB", width=2),
        marker=dict(size=5)
    ))
    fig_is.add_trace(go.Scatter(
        x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
        y=company_df["gross_profit"],
        name="Gross Profit",
        mode="lines+markers",
        line=dict(color="#10B981", width=2),
        marker=dict(size=5)
    ))
    fig_is.add_trace(go.Scatter(
        x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
        y=company_df["net_income"],
        name="Net Income",
        mode="lines+markers",
        line=dict(color="#475569", width=2),
        marker=dict(size=5)
    ))
    
    fig_is.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", family="Inter"),
        xaxis=dict(gridcolor="#E2E8F0", tickangle=-30),
        yaxis=dict(gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_is, width="stretch", theme=None)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    sec2_col1, sec2_col2 = st.columns([1, 1])
    
    with sec2_col1:
        st.markdown("### Cash Generation Dynamics")
        fcf_values = company_df["operating_cash_flow"] - company_df["capex"]
        capex_display = -company_df["capex"]
    
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Scatter(
            x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
            y=company_df["operating_cash_flow"],
            name="Operating Cash Flow",
            mode="lines+markers",
            line=dict(color="#059669", width=2),
            marker=dict(size=5)
        ))
        fig_cf.add_trace(go.Scatter(
            x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
            y=capex_display,
            name="CapEx (outflow)",
            mode="lines+markers",
            line=dict(color="#E11D48", width=2, dash="dash"),
            marker=dict(size=5)
        ))
        fig_cf.add_trace(go.Scatter(
            x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
            y=fcf_values,
            name="Free Cash Flow",
            mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=5)
        ))
        
        fig_cf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Inter"),
            xaxis=dict(gridcolor="#E2E8F0", tickangle=-30),
            yaxis=dict(gridcolor="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_cf, width="stretch", theme=None)
    
    with sec2_col2:
        st.markdown("### Capital Structure Composition")
        
        fig_bs = go.Figure()
        fig_bs.add_trace(go.Bar(
            x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
            y=company_df["total_liabilities"],
            name="Total Liabilities",
            marker_color="#94A3B8"
        ))
        fig_bs.add_trace(go.Bar(
            x=company_df["report_period"].dt.strftime('%Y-%m-%d'),
            y=company_df["equity"],
            name="Total Equity",
            marker_color="#059669"
        ))
        
        fig_bs.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Inter"),
            xaxis=dict(gridcolor="#E2E8F0", tickangle=-30),
            yaxis=dict(gridcolor="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_bs, width="stretch", theme=None)
    
    st.markdown("</div>", unsafe_allow_html=True)


with tab_peer:
    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    st.markdown("### Peer Valuation Benchmark")
    st.write("Compare the focus company's performance against industry competitors in real-time.")
    
    default_peer_idx = 0
    preferred_defaults = ["APPLE INC", "MICROSOFT CORP", "ALPHABET INC", "AMAZON.COM, INC.", "NVIDIA CORP"]
    for preferred in preferred_defaults:
        if preferred in peer_companies:
            default_peer_idx = peer_companies.index(preferred)
            break
            
    compare_peer = st.selectbox(
        "Select Comparison Peer",
        peer_companies,
        index=default_peer_idx
    )
    st.write("")
    
    peer_df = df[df["company_name"] == compare_peer].sort_values("report_period").reset_index(drop=True)
    if not peer_df.empty:
        peer_latest = peer_df.iloc[-1]
        peer_gm, peer_em, peer_nm, peer_de, peer_fcfc, peer_fcf = get_derived_ratios(peer_latest)
        
        comp_metrics = {
            "Metric": [
                "Report Period",
                "Fiscal Year/Period",
                "Revenue",
                "Gross Margin (%)",
                "EBITDA Margin (%)",
                "Net Profit Margin (%)",
                "Total Assets",
                "Total Liabilities",
                "Total Equity",
                "Debt to Equity Ratio (x)",
                "FCF Conversion (%)"
            ],
            selected_company: [
                latest_row["report_period"].strftime('%Y-%m-%d'),
                f"FY{latest_row['fiscal_year']} {latest_row['fiscal_period']}",
                format_currency(latest_row["revenue"]),
                format_percent_val(latest_gm),
                format_percent_val(latest_em),
                format_percent_val(latest_nm),
                format_currency(latest_row["total_assets"]),
                format_currency(latest_row["total_liabilities"]),
                format_currency(latest_row["equity"]),
                format_de_val(latest_de),
                format_percent_val(latest_fcfc)
            ],
            compare_peer: [
                peer_latest["report_period"].strftime('%Y-%m-%d'),
                f"FY{peer_latest['fiscal_year']} {peer_latest['fiscal_period']}",
                format_currency(peer_latest["revenue"]),
                format_percent_val(peer_gm),
                format_percent_val(peer_em),
                format_percent_val(peer_nm),
                format_currency(peer_latest["total_assets"]),
                format_currency(peer_latest["total_liabilities"]),
                format_currency(peer_latest["equity"]),
                format_de_val(peer_de),
                format_percent_val(peer_fcfc)
            ]
        }
        comp_df = pd.DataFrame(comp_metrics)
        st.dataframe(comp_df, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab_ml:
    if ml_model_payload is not None:
        model = ml_model_payload["model"]
        feature_names = ml_model_payload["feature_names"]
        
        lat_net   = float(latest_row["net_income"])         if not pd.isna(latest_row["net_income"])         else 0.0
        lat_cf    = float(latest_row["operating_cash_flow"]) if not pd.isna(latest_row["operating_cash_flow"]) else 0.0
        lat_rec   = float(latest_row["receivables"])         if not pd.isna(latest_row["receivables"])         else 0.0
        lat_pay   = float(latest_row["payables"])            if not pd.isna(latest_row["payables"])            else 0.0
        lat_opinc = float(latest_row["operating_income"])   if not pd.isna(latest_row["operating_income"])   else 0.0
    
        debt_to_assets = lat_liab / lat_assets
        operating_margin = lat_opinc / lat_rev if lat_rev != 0 else 0.0
        roa = lat_net / lat_assets
        asset_turnover = lat_rev / lat_assets
        equity_multiplier = lat_assets / lat_equity if lat_equity != 0 else 1.0
        sloan_ratio = (lat_net - lat_cf) / lat_assets
        receivables_to_payables = lat_rec / lat_pay if lat_pay != 0 else 0.0
        cash_flow_to_debt = lat_cf / lat_liab if lat_liab != 0 else 0.0
    
        feature_dict = {
            "debt_to_assets": debt_to_assets,
            "operating_margin": operating_margin,
            "roa": roa,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
            "sloan_ratio": sloan_ratio,
            "receivables_to_payables": receivables_to_payables,
            "cash_flow_to_debt": cash_flow_to_debt
        }
    
        X_inf = pd.DataFrame([feature_dict])[feature_names]
        X_inf = X_inf.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
        probs = model.predict_proba(X_inf)[0]
        risk_idx = model.predict(X_inf)[0]
    
        risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
        risk_colors = ["#10B981", "#F59E0B", "#EF4444"]
        risk_descs = [
            "The company shows excellent solvency, healthy margins, and positive cash flow ratios. Financial distress risk is minimal.",
            "The company is in a transitionary phase. Some leverage or operational margins show mild caution; monitor credit terms.",
            "The company exhibits severe distress indicators, negative equity buffers, or persistent cash outflow. Action recommended."
        ]
    
        predicted_risk = risk_labels[risk_idx]
        predicted_color = risk_colors[risk_idx]
        predicted_desc = risk_descs[risk_idx]
    
        st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
        st.markdown("### Machine Learning Distress Predictor")
        
        st.markdown("#### AI Risk Assessment")
        st.markdown(f"""
        <div style='background-color:#F8FAFC; border:1px solid #E2E8F0; padding:20px; border-radius:6px; margin-bottom:20px;'>
            <div style='font-size:0.75rem; color:#64748B; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;'>AI Prediction Label</div>
            <div style='font-size:2rem; font-weight:700; color:{predicted_color}; margin:10px 0;'>{predicted_risk}</div>
            <div style='font-size:0.85rem; color:#475569; line-height:1.5;'>{predicted_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Prediction Confidence Levels")
        fig_prob = go.Figure(go.Bar(
            x=[probs[0]*100, probs[1]*100, probs[2]*100],
            y=["Low Risk", "Medium Risk", "High Risk"],
            orientation="h",
            marker=dict(color=["#10B981", "#F59E0B", "#EF4444"])
        ))
        fig_prob.update_layout(
            xaxis=dict(title="Confidence (%)", range=[0, 100], gridcolor="#E2E8F0"),
            yaxis=dict(gridcolor="#E2E8F0"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Inter"),
            height=220,
            margin=dict(l=120, r=20, t=10, b=50)
        )
        st.plotly_chart(fig_prob, width="stretch", theme=None)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
        st.markdown("### Feature Risk Contribution (Explainable AI)")
        st.markdown(
            "<p style='font-size:0.85rem; color:#64748B; margin-top:-10px; margin-bottom:15px;'>"
            "Shows how much each financial ratio drives the prediction toward distress (red/right) or safety (green/left) relative to the market baseline."
            "</p>",
            unsafe_allow_html=True
        )
        
        importances = model.feature_importances_
        
        local_contributions = []
        for feat in feature_names:
            val = feature_dict[feat]
            median_val = feature_medians[feat]
            std_val = feature_stds[feat]
            importance = importances[feature_names.index(feat)]
            
            deviation = (val - median_val) / std_val
            contribution = deviation * feature_directions[feat] * importance
            local_contributions.append(contribution)
            
        local_contributions = np.array(local_contributions)
        
        abs_sorted_idx = np.argsort(np.abs(local_contributions))
        
        sorted_features = [feature_names[i] for i in abs_sorted_idx]
        sorted_contribs = local_contributions[abs_sorted_idx]
        
        bar_colors = ["#EF4444" if c >= 0 else "#10B981" for c in sorted_contribs]
        
        fig_imp = go.Figure(go.Bar(
            x=sorted_contribs,
            y=[f.replace("_", " ").title() for f in sorted_features],
            orientation="h",
            marker_color=bar_colors
        ))
        fig_imp.update_layout(
            xaxis=dict(title="Risk Contribution Score", gridcolor="#E2E8F0"),
            yaxis=dict(gridcolor="#E2E8F0"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Inter"),
            height=320,
            margin=dict(l=180, r=20, t=10, b=50)
        )
        st.plotly_chart(fig_imp, width="stretch", theme=None)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='dashboard-section'>", unsafe_allow_html=True)
    st.markdown("### Historical Financial Statements Table")
    
    hist_df_table = build_history_df(company_df, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)
    
    formatted_df = hist_df_table.copy()
    curr_cols = ["Revenue", "Total Assets", "Total Liabilities", "Equity", "Operating Cash Flow", "Free Cash Flow"]
    for col in curr_cols:
        formatted_df[col] = formatted_df[col].apply(format_currency)
        
    pct_cols = ["Gross Margin (%)", "EBITDA Margin (%)", "Net Profit Margin (%)", "FCF Conversion (%)"]
    for col in pct_cols:
        formatted_df[col] = formatted_df[col].apply(format_percent_val)
        
    formatted_df["Debt to Equity"] = formatted_df["Debt to Equity"].apply(format_de_val)
    
    st.dataframe(formatted_df, width="stretch", hide_index=True)
    
    csv_content_table = hist_df_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Report Data (CSV)",
        data=csv_content_table,
        file_name=f"{selected_company.lower().replace(' ', '_')}_financial_report.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
