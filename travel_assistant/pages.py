"""Page views. Compliance checking remains a placeholder."""

import streamlit as st

from travel_assistant.upload_page import documents
from travel_assistant.chat_page import questions


def home():
    st.title("Official travel, clearer decisions")
    st.write("An AI Bootcamp capstone exploring how officers can understand travel policy and review proposed official travel arrangements.")
    left, right = st.columns(2)
    with left:
        st.subheader("💬 Travel Policy Q&A")
        st.write("Ask natural-language questions and receive policy-grounded answers with source references.")
    with right:
        st.subheader("✅ Compliance Checker")
        st.write("Planned: review flight and accommodation details, with explanations and policy citations.")
    st.info("Phase 3 adds policy Q&A with cited evidence. Synthetic PDF ingestion and local retrieval are available; compliance assessment remains a placeholder.")
    st.subheader("Get started")
    st.write("Explore the pages using the sidebar. Use Document Upload to process a synthetic travel policy.")


def compliance():
    st.title("Travel Policy Compliance Checker")
    st.info("Interface preview: compliance assessment is not implemented yet.")
    left, right = st.columns(2)
    with left:
        st.subheader("Flight details")
        st.text_input("Origin")
        st.text_input("Destination")
        st.selectbox("Cabin class", ["Select a cabin class", "Economy", "Premium economy", "Business", "First"])
        st.text_input("Estimated airfare and currency", placeholder="Example: SGD 800")
    with right:
        st.subheader("Accommodation details")
        st.text_input("Hotel / accommodation")
        st.number_input("Number of nights", min_value=1, value=1, step=1)
        st.text_input("Nightly rate and currency", placeholder="Example: SGD 180")
    st.text_area("Additional context", placeholder="Example: travel dates, approval details, or business purpose")
    st.button("Check compliance", disabled=True)
    st.subheader("Assessment")
    st.caption("Future results: compliant, non-compliant, or insufficient information, with explanations and policy citations.")


def about():
    st.title("About Us")
    st.write("This individual AI Bootcamp capstone is a proof of concept for helping officers plan official travel and understand policy requirements.")
    st.subheader("Purpose and scope")
    st.write("The planned assistant combines policy Q&A with an AI-assisted review of proposed flights and accommodation. It supports understanding and review; officers remain responsible for approvals and final decisions.")
    st.subheader("Features")
    st.markdown("- Policy Q&A with source references (available)\n- Compliance assessments with explanations and citations (planned)\n- Synthetic PDF ingestion and development retrieval (available)\n- Demo login and sidebar navigation (available)")
    st.subheader("Technology roadmap")
    st.write("Python and Streamlit power the interface. PyMuPDF extracts PDF text, LangChain splits it into chunks, OpenAI creates embeddings, and ChromaDB stores them locally. OpenAI generates Q&A answers from retrieved passages. Compliance logic remains planned.")


def methodology():
    st.title("Methodology")
    st.info("Phase 3 implements ingestion, retrieval, and grounded policy Q&A. Compliance checking remains a future use case.")
    st.subheader("1. Proposed architecture")
    st.code("Streamlit interface → Application services → Policy retrieval → AI response\nSynthetic policy PDF → PDF processing → Embeddings → Local vector store", language="text")
    st.caption("PDF processing, embeddings, local vector storage, and Q&A answer generation are implemented. Compliance assessment remains planned.")
    st.subheader("2. Use-case flowchart: Policy Q&A")
    st.code("Question + recent history → Embed query → Retrieve top 3 active-policy passages\n→ Generate structured claims or insufficient-evidence result\n→ Validate passage IDs and exact evidence quotes → Show answer and metadata citations", language="text")
    st.write("Questions, history, and document text are untrusted user data, separated from the system grounding rules. Citations come from retrieved metadata. Conversation history resets when the active collection changes; results are discarded if the policy changes while answering.")
    st.subheader("3. Use-case flowchart: Compliance checking")
    st.code("Travel details → Retrieve rules → Assess available evidence\n→ Compliant / Non-compliant / Insufficient information → Explain with citations", language="text")


PAGES = {
    "Home": home,
    "Travel Policy Q&A": questions,
    "Travel Policy Compliance Checker": compliance,
    "Document Upload": documents,
    "About Us": about,
    "Methodology": methodology,
}
