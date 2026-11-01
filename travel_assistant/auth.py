"""Session-based demonstration login, not production authentication."""

import hmac
import os

import streamlit as st


def get_setting(name: str) -> str:
    """Prefer environment variables, then top-level Streamlit secrets."""
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""


def require_login() -> bool:
    """Render login until credentials match; keep access in session state."""
    if st.session_state.get("authenticated", False):
        return True

    st.title("✈️ AI Travel Policy Compliance Assistant")
    st.caption("Sign in to explore the Phase 1 demonstration.")
    expected_username = get_setting("DEMO_USERNAME")
    expected_password = get_setting("DEMO_PASSWORD")
    if not expected_username or not expected_password:
        st.error("Demo login is not configured. Set DEMO_USERNAME and DEMO_PASSWORD in .env or Streamlit secrets, then restart the app.")
        return False

    with st.form("login", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        username_matches = hmac.compare_digest(username.encode(), expected_username.encode())
        password_matches = hmac.compare_digest(password.encode(), expected_password.encode())
        if username_matches and password_matches:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.info("Demonstration access only. Use synthetic data; this is not a production security system.")
    return False
