"""Entry point: run with `streamlit run app.py`."""

import streamlit as st
from dotenv import load_dotenv

from travel_assistant.auth import require_login
from travel_assistant.pages import PAGES
from travel_assistant.notices import SIDEBAR_DISCLAIMER

load_dotenv()
st.set_page_config(page_title="Travel Policy Assistant", page_icon="✈️", layout="wide")

if require_login():
    with st.sidebar:
        st.title("✈️ Travel Policy")
        st.caption("AI Bootcamp Capstone · Proof of Concept")
        selection = st.radio("Navigation", list(PAGES), key="navigation")
        st.divider()
        st.caption(SIDEBAR_DISCLAIMER)
        st.caption("Use synthetic policy documents only.")
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    PAGES[selection]()
