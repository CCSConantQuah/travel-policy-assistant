"""Session-scoped chat UI for the active synthetic policy."""

import streamlit as st

from travel_assistant.auth import get_setting
from travel_assistant.documents import PolicyError
from travel_assistant.qa import DEFAULT_CHAT_MODEL, ask_policy, sync_conversation
from travel_assistant.vector_store import get_active_policy


def _show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        # Plain text prevents untrusted user/document/model content becoming links or HTML.
        st.text(message["content"])
        if message.get("citations"):
            with st.expander("Policy sources"):
                for citation in message["citations"]:
                    st.text(f"[{citation['number']}] {citation['source']} — page {citation['page']}")
                    st.text(citation["quote"])


def questions():
    st.title("Travel Policy Q&A")
    st.caption("Ask about the active synthetic policy. Verify answers against the cited passages. Do not enter internal agency information.")
    try:
        active = get_active_policy()
    except PolicyError as error:
        sync_conversation(st.session_state, None)
        st.error(str(error))
        return
    version = active["collection"] if active else None
    sync_conversation(st.session_state, version)
    if notice := st.session_state.pop("qa_notice", None):
        st.info(notice)
    if st.button("Clear conversation"):
        st.session_state.qa_messages = []
    if active is None:
        st.info("No active policy. Go to Document Upload in the sidebar and process a synthetic policy PDF first.")
        return
    st.text(f"Active policy: {active['filename']}")
    for message in st.session_state.qa_messages:
        _show_message(message)
    key = get_setting("OPENAI_API_KEY")
    if not key.strip():
        st.info("Set OPENAI_API_KEY in .env or Streamlit secrets and restart the app to ask questions.")
    question = st.chat_input("Ask a travel policy question", max_chars=1000, disabled=not key.strip())
    if question:
        try:
            with st.spinner("Finding policy passages and preparing an answer…"):
                answer = ask_policy(
                    question, key, get_setting("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL,
                    history=st.session_state.qa_messages, expected_collection=version,
                )
            st.session_state.qa_messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer["answer"], "citations": answer["citations"]},
            ])
            st.rerun()
        except PolicyError as error:
            # A replacement during generation clears the old conversation, too.
            try:
                current = get_active_policy()
                sync_conversation(st.session_state, current["collection"] if current else None)
            except PolicyError:
                sync_conversation(st.session_state, None)
            if st.session_state.qa_policy != version:
                st.session_state.qa_notice = "The active policy changed or became unavailable. Conversation cleared; please ask again."
                st.rerun()
            st.error(str(error))
