"""Existing two-column travel form with validated advisory assessments."""

import streamlit as st

from travel_assistant.auth import get_setting
from travel_assistant.compliance import APPROVALS, CABINS, CURRENCIES, advisory_summary, assess_trip, validate_trip
from travel_assistant.documents import PolicyError
from travel_assistant.qa import DEFAULT_CHAT_MODEL
from travel_assistant.vector_store import get_active_policy


def _show_section(label: str, result: dict) -> None:
    st.subheader(label)
    st.text(result["status"])
    for finding in result.get("display_findings", result["findings"]):
        if finding.get("context_only"):
            st.caption("Policy applicability context — not a separate requirement.")
        else:
            if finding.get("requirement"):
                st.write(finding["requirement"])
            st.text(f"{finding['status']} — {finding['explanation']}")
        if finding.get("validation_error"):
            st.warning("Evidence validation failed for this finding. It requires review before the requirement can be assessed.")
        for part in finding.get("parts", []):
            role = part["role"]
            if part.get("coverage_missing"):
                role += " (coverage incomplete)"
            st.text(f"{role}: {part['status']} — {part['explanation']}")
        if finding["citations"]:
            with st.expander("Policy evidence"):
                for citation in finding["citations"]:
                    st.text(f"[{citation['number']}] {citation['source']} — page {citation['page']}")
                    st.text(citation["quote"])


def compliance():
    st.title("Travel Policy Compliance Checker")
    st.info("Advisory assessment using the active fictional policy. Formal approval remains with the approving authority.")
    st.caption("Use fictional travel details only. Approval statuses and additional context are self-reported. Unknown details can result in Insufficient information.")
    st.caption("Origin and destination are required to begin. Other details are assessed against applicable policy requirements.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Flight details")
        origin = st.text_input("Origin", max_chars=150)
        destination = st.text_input("Destination", max_chars=150)
        cabin = st.selectbox("Cabin class", [None, *CABINS], format_func=lambda value: value or "Unknown")
        amount_column, currency_column = st.columns(2)
        with amount_column:
            airfare = st.number_input("Estimated airfare", min_value=0.0, value=None, step=0.01)
        with currency_column:
            airfare_currency = st.selectbox("Airfare currency", [None, *CURRENCIES], format_func=lambda value: value or "Unknown")
        flight_approval = st.selectbox("Flight approval status (self-reported)", APPROVALS)
    with right, st.container(border=True):
        st.subheader("Accommodation details")
        hotel = st.text_input("Hotel / accommodation", max_chars=150)
        nights = st.number_input("Number of nights", min_value=1, value=1, step=1)
        amount_column, currency_column = st.columns(2)
        with amount_column:
            nightly_rate = st.number_input("Nightly rate", min_value=0.0, value=None, step=0.01)
        with currency_column:
            nightly_currency = st.selectbox("Nightly rate currency", [None, *CURRENCIES], format_func=lambda value: value or "Unknown")
        stars = st.selectbox("Hotel star rating", [None, 1, 2, 3, 4, 5], format_func=lambda value: str(value) if value else "Unknown")
        accommodation_approval = st.selectbox("Accommodation approval status (self-reported)", APPROVALS)
    context = st.text_area("Additional context", placeholder="Example: travel dates, claimed approval details, or business purpose", max_chars=1500)
    raw = {"origin": origin, "destination": destination, "cabin": cabin, "airfare": airfare,
           "airfare_currency": airfare_currency, "hotel": hotel, "nights": nights,
           "nightly_rate": nightly_rate, "nightly_rate_currency": nightly_currency, "stars": stars,
           "flight_approval": flight_approval, "accommodation_approval": accommodation_approval, "context": context}
    validation_error = None
    try:
        validate_trip(raw)
    except PolicyError as error:
        validation_error = str(error)
    active, storage_error = None, None
    try:
        active = get_active_policy()
    except PolicyError as error:
        storage_error = str(error)
        st.error(storage_error)
    if active is None and storage_error is None:
        st.info("Go to Policy Documents and process a synthetic policy PDF before checking compliance.")
    if active is not None:
        st.text(f"Active policy: {active.get('filename', 'Active synthetic policy')}")
    key = get_setting("OPENAI_API_KEY")
    if not key.strip():
        st.caption("Set OPENAI_API_KEY in .env or Streamlit secrets and restart the app.")
    if validation_error:
        st.caption(validation_error)
    submitted = st.button("Check compliance", type="primary", use_container_width=True,
                          disabled=validation_error is not None or active is None or not key.strip())
    st.subheader("Assessment")
    if submitted:
        try:
            with st.spinner("Retrieving flight and accommodation requirements and assessing evidence…"):
                result = assess_trip(raw, key, get_setting("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL,
                                     expected_collection=active["collection"])
            st.info(advisory_summary(result))
            st.caption(f"Self-reported flight approval — {flight_approval}. Self-reported accommodation approval — {accommodation_approval}.")
            first, second = st.columns(2)
            with first:
                _show_section("Flight assessment", result["flight"])
            with second:
                _show_section("Accommodation assessment", result["accommodation"])
        except PolicyError as error:
            st.error(str(error))
    else:
        with st.container(border=True):
            st.write("Your advisory assessment will appear here.")
            st.caption("Complete the required origin and destination and select Check compliance to assess the supplied details.")
