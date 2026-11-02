"""Entry point: run with `streamlit run app.py`."""

import streamlit as st
from dotenv import load_dotenv

from travel_assistant.auth import require_login
from travel_assistant.pages import PAGES

load_dotenv()
st.set_page_config(page_title="Travel Policy Assistant", page_icon="✈️", layout="wide")

if require_login():
    with st.sidebar:
        st.title("✈️ Travel Policy")
        st.caption("AI Compliance Assistant · Phase 2")
        selection = st.radio("Navigation", list(PAGES))
        st.divider()
        st.caption("Use synthetic policy documents only.")
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    PAGES[selection]()
