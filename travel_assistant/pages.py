"""Page views and the navigation map."""

import streamlit as st

from travel_assistant.upload_page import documents
from travel_assistant.chat_page import questions
from travel_assistant.compliance_page import compliance


def home():
    st.title("Official travel, clearer decisions")
    st.write("An AI Bootcamp capstone exploring how officers can understand travel policy and review proposed official travel arrangements.")
    left, right = st.columns(2)
    with left:
        st.subheader("💬 Travel Policy Q&A")
        st.write("Ask natural-language questions and receive policy-grounded answers with source references.")
    with right:
        st.subheader("✅ Compliance Checker")
        st.write("Review flight and accommodation details, with advisory findings and policy citations.")
    st.info("Assess proposed travel arrangements and ask policy questions using an uploaded fictional policy and cited evidence.")
    st.subheader("Get started")
    st.write("Explore the pages using the sidebar. Use Document Upload to process a synthetic travel policy.")


def about():
    st.title("About Us")
    st.write("This individual AI Bootcamp capstone is a proof of concept for helping officers plan official travel and understand policy requirements.")
    st.subheader("Purpose and scope")
    st.write("The assistant combines policy Q&A with an AI-assisted advisory review of proposed flights and accommodation. Officers remain responsible for approvals and final decisions.")
    st.subheader("Features")
    st.markdown("- Policy Q&A with source references\n- Advisory flight and accommodation assessments with cited evidence\n- Synthetic PDF ingestion and development retrieval\n- Demo login and sidebar navigation")
    st.subheader("Technology roadmap")
    st.write("Python and Streamlit power the interface. PyMuPDF extracts PDF text, LangChain splits it into chunks, OpenAI creates embeddings, and ChromaDB stores them locally. OpenAI generates grounded Q&A answers and structured compliance findings, which the application validates against retrieved evidence.")


def methodology():
    st.title("Methodology")
    st.info("Phase 4 adds advisory flight and accommodation assessments to document ingestion, retrieval, and grounded policy Q&A.")
    st.subheader("1. Proposed architecture")
    st.code("Streamlit interface → Application services → Policy retrieval → AI response\nSynthetic policy PDF → PDF processing → Embeddings → Local vector store", language="text")
    st.caption("Policy ingestion, local retrieval, Q&A, and advisory compliance assessment use the same active knowledge base.")
    st.subheader("2. Use-case flowchart: Policy Q&A")
    st.code("Question + recent history → Embed query → Retrieve top 3 active-policy passages\n→ Generate structured claims or insufficient-evidence result\n→ Validate passage IDs and exact evidence quotes → Show answer and metadata citations", language="text")
    st.write("Questions, history, and document text are untrusted user data, separated from the system grounding rules. Citations come from retrieved metadata. Conversation history resets when the active collection changes; results are discarded if the policy changes while answering.")
    st.subheader("3. Use-case flowchart: Compliance checking")
    st.code("Travel inputs → Validate amounts, currencies, and structured fields\n→ Retrieve top 3 flight passages and top 3 accommodation passages\n→ Generate structured findings → Validate exact quotes and source IDs\n→ Check numeric comparisons, approval dependencies, and missing details\n→ Separate flight/accommodation statuses → Advisory summary and evidence", language="text")
    st.write("Amounts are compared only in matching currencies. Approval claims remain self-reported. Unsupported evidence, unresolved exceptions, and missing information produce Insufficient information. A validated violation is retained when other requirements remain unresolved. Findings apply to the supplied inputs and retrieved evidence, with formal approval handled by the approving authority.")


PAGES = {
    "Home": home,
    "Travel Policy Q&A": questions,
    "Travel Policy Compliance Checker": compliance,
    "Document Upload": documents,
    "About Us": about,
    "Methodology": methodology,
}
