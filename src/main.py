import streamlit as st

st.set_page_config(
    page_title="Financial Intelligence Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    {
        "Analytics": [
            st.Page("app.py",            title="Company Analysis",  icon="🏢", default=True),
            st.Page("pages/market_overview.py", title="Market Overview",   icon="📈"),
        ],
    }
)

pages.run()
