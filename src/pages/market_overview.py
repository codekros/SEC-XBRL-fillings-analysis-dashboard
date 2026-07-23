import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path



st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

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
        font-weight: 700;
    }

    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    .section-badge-rising {
        display: inline-block;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 14px;
    }

    .section-badge-consistent {
        display: inline-block;
        background: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 14px;
    }

    .matrix-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.78rem;
        margin-top: 10px;
    }
    .matrix-table th {
        background: #F1F5F9;
        color: #475569;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        padding: 9px 12px;
        border-bottom: 2px solid #E2E8F0;
        white-space: nowrap;
        position: sticky;
        top: 0;
        z-index: 2;
        min-width: 100px;
    }
    .matrix-table th.metric-col {
        background: #0F172A;
        color: #F8FAFC;
        text-align: left;
        min-width: 170px;
        position: sticky;
        left: 0;
        z-index: 3;
    }
    .matrix-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #F1F5F9;
        text-align: right;
        color: #334155;
        font-weight: 500;
        white-space: nowrap;
        min-width: 100px;
    }
    .matrix-table td.metric-label {
        text-align: left;
        color: #475569;
        font-weight: 600;
        background: #FAFAFA;
        white-space: nowrap;
        position: sticky;
        left: 0;
        z-index: 1;
        border-right: 2px solid #E2E8F0;
    }
    .matrix-table tr:hover td {
        background: #F0F9FF;
    }
    .matrix-table tr:hover td.metric-label {
        background: #E0F2FE;
    }
    .val-pos { color: #059669; font-weight: 700; }
    .val-neg { color: #DC2626; font-weight: 700; }
    .val-neutral { color: #94A3B8; }
    .rank-badge {
        display: inline-block;
        background: #E2E8F0;
        border-radius: 4px;
        padding: 1px 5px;
        font-size: 0.65rem;
        font-weight: 700;
        color: #475569;
        margin-right: 5px;
        margin-bottom: 2px;
    }
    .company-header {
        display: block;
        max-width: 110px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.72rem;
    }
    </style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "consolidated_kpis.parquet"


@st.cache_data
def load_data(file_path: Path, mtime: float):
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

    rev_safe   = df["revenue"].replace(0, np.nan)
    assets_safe = df["total_assets"].replace(0, np.nan)
    equity_safe = df["equity"].replace(0, np.nan)
    liab_safe   = df["total_liabilities"].replace(0, np.nan)

    df["gross_margin_pct"]   = (df["gross_profit"]   / rev_safe * 100).replace([np.inf, -np.inf], np.nan)
    df["ebitda_margin_pct"]  = (df["ebitda"]          / rev_safe * 100).replace([np.inf, -np.inf], np.nan)
    df["net_margin_pct"]     = (df["net_income"]      / rev_safe * 100).replace([np.inf, -np.inf], np.nan)
    df["roa_pct"]            = (df["net_income"]      / assets_safe * 100).replace([np.inf, -np.inf], np.nan)
    df["debt_to_equity"]     = (df["total_liabilities"] / equity_safe).replace([np.inf, -np.inf], np.nan)
    df["fcf"]                = df["operating_cash_flow"] - df["capex"]
    df["ocf"]                = df["operating_cash_flow"]

    df["quarter_label"] = df["report_period"].dt.to_period("Q").astype(str)
    return df


df = load_data(DATA_PATH, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)

all_quarters = sorted(df["quarter_label"].unique(), reverse=True)

def fmt_currency(val):
    if pd.isna(val) or val is None:
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.0f}K"
    return f"{sign}${abs_val:.0f}"

def fmt_pct(val):
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.1f}%"

def fmt_x(val):
    if pd.isna(val) or val is None:
        return "N/A"
    return f"{val:.2f}x"

def color_class(val, higher_is_better=True):
    if pd.isna(val):
        return "val-neutral"
    if (val > 0 and higher_is_better) or (val < 0 and not higher_is_better):
        return "val-pos"
    if (val < 0 and higher_is_better) or (val > 0 and not higher_is_better):
        return "val-neg"
    return "val-neutral"

ALL_METRICS = {
    "Revenue ($)":           ("revenue",          fmt_currency, True),
    "Gross Profit ($)":      ("gross_profit",      fmt_currency, True),
    "EBITDA ($)":            ("ebitda",            fmt_currency, True),
    "Net Income ($)":        ("net_income",        fmt_currency, True),
    "Total Assets ($)":      ("total_assets",      fmt_currency, True),
    "Total Liabilities ($)": ("total_liabilities", fmt_currency, False),
    "Equity ($)":            ("equity",            fmt_currency, True),
    "Op. Cash Flow ($)":     ("ocf",               fmt_currency, True),
    "Free Cash Flow ($)":    ("fcf",               fmt_currency, True),
    "Gross Margin (%)":      ("gross_margin_pct",  fmt_pct,      True),
    "EBITDA Margin (%)":     ("ebitda_margin_pct", fmt_pct,      True),
    "Net Margin (%)":        ("net_margin_pct",    fmt_pct,      True),
    "ROA (%)":               ("roa_pct",           fmt_pct,      True),
    "Debt / Equity (x)":     ("debt_to_equity",    fmt_x,        False),
}

RANK_METRICS = {
    "Revenue Growth (%)":    "revenue",
    "Gross Margin (%)":      "gross_margin_pct",
    "Net Profit Margin (%)" : "net_margin_pct",
    "EBITDA Margin (%)":     "ebitda_margin_pct",
    "ROA (%)":               "roa_pct",
    "Operating Cash Flow":   "ocf",
    "Free Cash Flow":        "fcf",
    "Debt to Equity":        "debt_to_equity",
}

RANK_HIGHER_BETTER = {
    "Revenue Growth (%)":    True,
    "Gross Margin (%)":      True,
    "Net Profit Margin (%)": True,
    "EBITDA Margin (%)":     True,
    "ROA (%)":               True,
    "Operating Cash Flow":   True,
    "Free Cash Flow":        True,
    "Debt to Equity":        False,
}

@st.cache_data
def compute_deltas(rank_col: str, current_q: str, prev_q: str, mtime: float):
    curr_df = df[df["quarter_label"] == current_q][["company_name", rank_col]].copy()
    prev_df = df[df["quarter_label"] == prev_q][["company_name", rank_col]].copy()

    curr_df = curr_df.groupby("company_name")[rank_col].mean().reset_index()
    prev_df = prev_df.groupby("company_name")[rank_col].mean().reset_index()

    merged = curr_df.merge(prev_df, on="company_name", suffixes=("_curr", "_prev"))
    merged = merged.dropna(subset=[f"{rank_col}_curr", f"{rank_col}_prev"])

    if rank_col == "revenue":
        merged = merged[merged[f"{rank_col}_prev"].abs() > 1e3]
        merged["delta"] = (
            (merged[f"{rank_col}_curr"] - merged[f"{rank_col}_prev"])
            / merged[f"{rank_col}_prev"].abs() * 100
        )
        merged["delta"] = merged["delta"].clip(-999, 999)
    else:
        merged["delta"] = merged[f"{rank_col}_curr"] - merged[f"{rank_col}_prev"]

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta"])
    return merged[["company_name", "delta"]].sort_values("delta", ascending=False)


@st.cache_data
def compute_consistency(rank_col: str, last_4_quarters: tuple, mtime: float):
    subset = df[df["quarter_label"].isin(last_4_quarters)][["company_name", "quarter_label", rank_col]]
    pivot = subset.groupby(["company_name", "quarter_label"])[rank_col].mean().unstack()
    pivot = pivot.dropna(thresh=4)
    row_means = pivot.abs().mean(axis=1)
    if rank_col == "revenue":
        pivot = pivot[row_means > 1e5]
    else:
        pivot = pivot[row_means > 0.01]
    std_ser = pivot.std(axis=1).dropna()
    std_df = std_ser.reset_index()
    std_df.columns = ["company_name", "std_dev"]
    return std_df.sort_values("std_dev")


def get_company_snapshot(companies: list, quarter: str) -> pd.DataFrame:
    cols = ["company_name"] + [col for col, _, _ in ALL_METRICS.values()]
    q_df = df[df["quarter_label"] == quarter][cols].copy()
    q_df = q_df.groupby("company_name").mean(numeric_only=True).reset_index()
    return q_df[q_df["company_name"].isin(companies)]


def render_bar_chart(companies, deltas, colors, title_color):
    fig = go.Figure(go.Bar(
        x=deltas,
        y=companies,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f"{d:+.2f}" if not pd.isna(d) else "N/A" for d in deltas],
        textposition="outside",
        textfont=dict(size=10, family="Inter")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", family="Inter", size=11),
        height=360,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(gridcolor="#E2E8F0", showgrid=True, zeroline=True,
                   zerolinecolor="#94A3B8", zerolinewidth=1.5),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
    )
    return fig


def render_consistency_chart(companies, std_devs):
    colors = ["#3B82F6" if i < 3 else "#93C5FD" for i in range(len(companies))]
    fig = go.Figure(go.Bar(
        x=std_devs,
        y=companies,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in std_devs],
        textposition="outside",
        textfont=dict(size=10, family="Inter")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A", family="Inter", size=11),
        height=360,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(title="Std Dev (lower = more stable)", gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
    )
    return fig


def build_matrix_html(snapshot_df: pd.DataFrame, companies: list,
                       delta_map: dict = None, badge_color: str = "#059669",
                       higher_is_better: bool = True) -> str:
    def abbrev(name: str) -> str:
        stop = {"INC", "INC.", "CORP", "CORP.", "LLC", "LTD", "LTD.",
                "CO.", "CO", "PLC", "GROUP", "HOLDINGS", "HOLDING"}
        words = [w for w in name.split() if w.upper() not in stop]
        if not words:
            words = name.split()
        return " ".join(words[:3]).title()

    html = '<div style="overflow-x:auto; border-radius:8px; border:1px solid #E2E8F0;">'
    html += '<table class="matrix-table">'

    html += '<thead><tr>'
    html += '<th class="metric-col">METRIC</th>'
    for i, co in enumerate(companies):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
        html += (
            f'<th title="{co}">'
            f'<span class="rank-badge">{medal}</span>'
            f'<span class="company-header">{abbrev(co)}</span>'
            f'</th>'
        )
    html += '</tr></thead>'

    html += '<tbody>'

    if delta_map is not None:
        html += '<tr style="background:#F8FAFC;">'
        html += ('<td class="metric-label" '
                 'style="font-weight:700; color:#0F172A; border-top:2px solid #CBD5E1;">'
                 'QoQ Δ (Rank Metric)</td>')
        for co in companies:
            d = delta_map.get(co, np.nan)
            if pd.isna(d):
                html += '<td class="val-neutral" style="border-top:2px solid #CBD5E1;">—</td>'
            else:
                cls = color_class(d, higher_is_better)
                sign = "+" if d > 0 else ""
                unit = "%" if higher_is_better else "x"
                html += (f'<td class="{cls}" style="border-top:2px solid #CBD5E1;">'
                         f'{sign}{d:.2f}{unit}</td>')
        html += '</tr>'

    for label, (col, formatter, hib) in ALL_METRICS.items():
        html += '<tr>'
        html += f'<td class="metric-label">{label}</td>'
        for co in companies:
            row = snapshot_df[snapshot_df["company_name"] == co]
            if row.empty:
                html += '<td class="val-neutral">—</td>'
                continue
            val = row.iloc[0].get(col, np.nan)
            if pd.isna(val):
                html += '<td class="val-neutral">N/A</td>'
            else:
                formatted = formatter(val)
                if col in ("net_income", "gross_profit", "ebitda", "ocf", "fcf",
                           "gross_margin_pct", "ebitda_margin_pct", "net_margin_pct", "roa_pct"):
                    cls = "val-pos" if val > 0 else ("val-neg" if val < 0 else "")
                elif col == "debt_to_equity":
                    cls = "val-neg" if val > 2 else ("val-pos" if val < 0.5 else "")
                else:
                    cls = ""
                td_class = f' class="{cls}"' if cls else ""
                html += f'<td{td_class}>{formatted}</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    return html


st.sidebar.markdown("<h3 style='color:#0F172A;'>🌐 Market Overview</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

selected_metric_label = st.sidebar.selectbox(
    "📊 Rank Companies By",
    list(RANK_METRICS.keys()),
    index=0
)

if len(all_quarters) >= 2:
    current_q_default = all_quarters[0]
    prev_q_default    = all_quarters[1]
else:
    current_q_default = all_quarters[0]
    prev_q_default    = all_quarters[0]

current_q = st.sidebar.selectbox(
    "📅 Current Quarter",
    all_quarters,
    index=0
)

available_prev = [q for q in all_quarters if q < current_q]
if not available_prev:
    available_prev = all_quarters[1:] if len(all_quarters) > 1 else all_quarters

prev_q = st.sidebar.selectbox(
    "📅 vs. Prior Quarter",
    available_prev,
    index=0
)

last_4 = tuple(sorted([q for q in all_quarters if q <= current_q], reverse=True)[:4])

st.sidebar.markdown("---")
st.sidebar.info(f"**Current:** {current_q}\n\n**Prior:** {prev_q}\n\n**Trailing:** {', '.join(last_4)}")

st.markdown("""
<div style='margin-bottom:24px;'>
    <h1 style='margin-bottom:4px;'>📈 Market Overview</h1>
    <p style='color:#64748B; font-size:0.95rem; margin:0;'>
        Cross-market performance intelligence — Top 10 movers and stable performers vs. the last quarter.
    </p>
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:14px 18px;'>
        <div style='font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;'>Current Quarter</div>
        <div style='font-size:1.3rem; font-weight:700; color:#0F172A;'>{current_q}</div>
    </div>""", unsafe_allow_html=True)
with col_info2:
    st.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:14px 18px;'>
        <div style='font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;'>Prior Quarter</div>
        <div style='font-size:1.3rem; font-weight:700; color:#0F172A;'>{prev_q}</div>
    </div>""", unsafe_allow_html=True)
with col_info3:
    st.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:14px 18px;'>
        <div style='font-size:0.7rem; color:#64748B; font-weight:700; text-transform:uppercase;'>Ranking Metric</div>
        <div style='font-size:1.3rem; font-weight:700; color:#0F172A;'>{selected_metric_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

rank_col    = RANK_METRICS[selected_metric_label]
hib         = RANK_HIGHER_BETTER[selected_metric_label]

delta_df    = compute_deltas(rank_col, current_q, prev_q, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)
consist_df  = compute_consistency(rank_col, last_4, DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)

rising_df   = delta_df.head(10)

rising_companies   = rising_df["company_name"].tolist()
consistent_companies = consist_df.head(10)["company_name"].tolist()

rising_snap   = get_company_snapshot(rising_companies,    current_q)
consist_snap  = get_company_snapshot(consistent_companies, current_q)

rising_delta_map  = dict(zip(rising_df["company_name"],  rising_df["delta"]))

st.markdown('<span class="section-badge-rising">🚀 TOP 10 RISING</span>', unsafe_allow_html=True)
st.markdown(f"*Companies with the largest positive QoQ improvement in **{selected_metric_label}** from {prev_q} → {current_q}.*")

with st.container():
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    chart_col, _ = st.columns([3, 1])
    with chart_col:
        colors_r = ["#059669" if i < 3 else "#34D399" if i < 6 else "#A7F3D0"
                    for i in range(len(rising_companies))]
        fig_r = render_bar_chart(
            rising_companies,
            rising_df["delta"].tolist(),
            colors_r,
            "#059669"
        )
        st.plotly_chart(fig_r, width="stretch", theme=None)

    st.markdown("#### 📋 Full Metric Matrix — Rising Companies")
    matrix_html = build_matrix_html(
        rising_snap, rising_companies,
        delta_map=rising_delta_map,
        badge_color="#059669",
        higher_is_better=hib
    )
    st.markdown(matrix_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)



st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<span class="section-badge-consistent">📊 TOP 10 CONSISTENT</span>', unsafe_allow_html=True)
st.markdown(f"*Companies with the lowest standard deviation in **{selected_metric_label}** across the trailing 4 quarters ({', '.join(last_4)}).*")

with st.container():
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    chart_col3, _ = st.columns([3, 1])
    with chart_col3:
        std_vals = consist_df[consist_df["company_name"].isin(consistent_companies)].head(10)
        std_vals_ordered = [
            consist_df[consist_df["company_name"] == c]["std_dev"].values[0]
            if c in consist_df["company_name"].values else np.nan
            for c in consistent_companies
        ]
        fig_c = render_consistency_chart(consistent_companies, std_vals_ordered)
        st.plotly_chart(fig_c, width="stretch", theme=None)

    st.markdown("#### 📋 Full Metric Matrix — Consistent Companies")
    matrix_html_c = build_matrix_html(
        consist_snap, consistent_companies,
        delta_map=None,
        badge_color="#2563EB",
        higher_is_better=hib
    )
    st.markdown(matrix_html_c, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.78rem; padding:20px 0; border-top:1px solid #E2E8F0; margin-top:10px;'>
    Financial Intelligence Terminal · Market Overview · Data sourced from SEC XBRL filings
</div>
""", unsafe_allow_html=True)
