"""Streamlit ingestion interface, kept separate from local ingestion services."""

import streamlit as st

from travel_assistant.auth import get_setting
from travel_assistant.documents import PolicyError
from travel_assistant.vector_store import get_active_policy, process_policy, retrieve_policy


def documents():
    st.title("Document Upload")
    st.warning("Use synthetic travel policy documents only. Do not upload internal agency documents.")
    st.info("Processing sends extracted text to OpenAI for embeddings. Text, vectors, and source metadata are stored locally. One policy is active at a time.")
    uploaded = st.file_uploader("Synthetic policy PDF (maximum 20 MB / 300 pages)", type=["pdf"])
    confirmed = st.checkbox("I confirm this is a synthetic or fictional policy suitable for external processing.")
    if st.button("Process policy", type="primary", disabled=uploaded is None or not confirmed):
        try:
            with st.spinner("Extracting text, creating embeddings, and storing the policy…"):
                result = process_policy(uploaded.getvalue(), uploaded.name, get_setting("OPENAI_API_KEY"))
            st.success("Processing complete. The new policy is active.")
            if result.get("cleanup_warning"):
                st.warning(result["cleanup_warning"])
        except PolicyError as error:
            st.error(str(error))

    st.subheader("Active policy")
    try:
        active = get_active_policy()
    except PolicyError as error:
        st.error(str(error))
        return
    if active is None:
        st.caption("Status: No active policy. Select a synthetic PDF and process it.")
        return
    st.write(f"**Filename:** {active['filename']}")
    first, second, third = st.columns(3)
    first.metric("PDF pages", active["page_count"])
    second.metric("Chunks", active["chunk_count"])
    third.metric("Status", active["status"])
    st.caption(f"Processed at (UTC): {active['processed_at']}")
    if active["text_page_count"] < active["page_count"]:
        st.warning(f"Only {active['text_page_count']} of {active['page_count']} pages contain extractable text. Other pages were skipped; no OCR is performed.")

    with st.expander("Development only: test semantic retrieval"):
        st.caption("Embeds your query using OpenAI and displays up to three matching passages. No chatbot or compliance decision is generated.")
        with st.form("retrieval_test"):
            query = st.text_input("Search query", max_chars=1000)
            submitted = st.form_submit_button("Retrieve passages")
        if submitted:
            try:
                with st.spinner("Searching the active policy…"):
                    results = retrieve_policy(query, get_setting("OPENAI_API_KEY"))
                for result in results:
                    st.write(f"**{result['source']} — page {result['page']}**")
                    st.text(result["text"])
                    st.caption(f"Cosine distance: {result['distance']:.4f} (lower means closer; not a compliance score)")
            except PolicyError as error:
                st.error(str(error))
