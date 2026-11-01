"""Phase 1 views. No AI calls, file processing, or policy decisions occur."""

import streamlit as st


def home():
    st.title("Official travel, clearer decisions")
    st.write("An AI Bootcamp capstone exploring how officers can understand travel policy and review proposed official travel arrangements.")
    left, right = st.columns(2)
    with left:
        st.subheader("💬 Travel Policy Q&A")
        st.write("Planned: ask natural-language questions and receive policy-grounded answers with source references.")
    with right:
        st.subheader("✅ Compliance Checker")
        st.write("Planned: review flight and accommodation details, with explanations and policy citations.")
    st.info("Phase 1 demonstrates login, navigation, and interface layouts. AI features are not connected.")
    st.subheader("Get started")
    st.write("Explore the pages using the sidebar. Future phases will use synthetic travel policies only.")


def questions():
    st.title("Travel Policy Q&A")
    st.write("Ask about official travel requirements, allowances, or approvals.")
    st.info("Interface preview: question answering and source references will be implemented in a later phase.")
    st.text_area("Your policy question", placeholder="Example: What approval is required before booking a flight?")
    st.button("Ask policy assistant", disabled=True)
    st.subheader("Answer and policy sources")
    st.caption("Future answers and document references will appear here.")


def compliance():
    st.title("Travel Policy Compliance Checker")
    st.info("Interface preview: no compliance assessment is performed in Phase 1.")
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


def documents():
    st.title("Document Upload")
    st.warning("Use synthetic travel policy documents only. Do not upload internal agency documents.")
    st.info("Upload is disabled in Phase 1. No files are processed, stored, or sent to external services.")
    st.file_uploader("Synthetic policy PDF", type=["pdf"], disabled=True)
    st.caption("Future phases will add PDF validation, document processing, and policy retrieval.")


def about():
    st.title("About Us")
    st.write("This individual AI Bootcamp capstone is a proof of concept for helping officers plan official travel and understand policy requirements.")
    st.subheader("Purpose and scope")
    st.write("The planned assistant combines policy Q&A with an AI-assisted review of proposed flights and accommodation. It supports understanding and review; officers remain responsible for approvals and final decisions.")
    st.subheader("Features")
    st.markdown("- Policy Q&A with source references (planned)\n- Compliance assessments with explanations and citations (planned)\n- Synthetic policy document upload (planned)\n- Demo login and sidebar navigation (available)")
    st.subheader("Technology roadmap")
    st.write("Python and Streamlit power this interface. Later phases may add OpenAI, LangChain, ChromaDB, and PyMuPDF. These integrations are not implemented in Phase 1.")


def methodology():
    st.title("Methodology")
    st.info("Design placeholders only. The architecture and use-case flowcharts will be expanded in later phases.")
    st.subheader("1. Proposed architecture")
    st.code("Streamlit interface → Application services → Policy retrieval → AI response\nSynthetic policy PDF → PDF processing → Embeddings → Local vector store", language="text")
    st.caption("Planned architecture; only the Streamlit interface exists today.")
    st.subheader("2. Use-case flowchart: Policy Q&A")
    st.code("Question → Retrieve policy passages → Generate grounded answer → Show citations", language="text")
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
