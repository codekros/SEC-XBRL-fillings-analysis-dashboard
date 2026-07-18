import streamlit as st
import sys
from pathlib import Path

# Add src/ to the Python path to make imports cleaner
src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

st.set_page_config(
    page_title="Financial Intelligence Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation(
    {
        "Analytics": [
            st.Page("src/app.py", title="Company Analysis", icon="🏢", default=True),
            st.Page("src/pages/market_overview.py", title="Market Overview", icon="📈"),
        ],
    }
)

pages.run()
