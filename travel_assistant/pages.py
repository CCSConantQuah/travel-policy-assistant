"""Page views and the navigation map."""

import streamlit as st

from travel_assistant.upload_page import documents
from travel_assistant.chat_page import questions
from travel_assistant.compliance_page import compliance
from travel_assistant.documents import PolicyError
from travel_assistant.vector_store import get_active_policy
from travel_assistant import methodology_diagrams as diagrams


def _navigate(page: str) -> None:
    """Home buttons select the same navigation widget as the sidebar."""
    st.session_state.navigation = page


def home():
    st.title("Plan official travel with policy in view")
    st.write("Explore your fictional travel policy and review proposed flights and accommodation with cited evidence.")
    with st.container(border=True):
        st.subheader("Your policy workspace")
        try:
            active = get_active_policy()
        except PolicyError as error:
            active = None
            st.error(str(error))
        if active:
            st.text(f"Active policy: {active['filename']}")
            st.success(f"Policy status: {active['status']} — available for questions and advisory assessment.")
        else:
            st.info("No active policy is available. Start by processing a fictional PDF in Policy Documents.")
        st.button("Open Policy Documents", on_click=_navigate, args=("Policy Documents",))
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("💬 Travel Policy Q&A")
        st.write("Ask natural-language questions and receive policy-grounded answers with source references.")
        st.button("Explore policy Q&A", type="primary", on_click=_navigate, args=("Travel Policy Q&A",), use_container_width=True)
    with right, st.container(border=True):
        st.subheader("✅ Compliance Checker")
        st.write("Review flight and accommodation details, with advisory findings and policy citations.")
        st.button("Review travel compliance", type="primary", on_click=_navigate, args=("Travel Policy Compliance Checker",), use_container_width=True)
    st.caption("Advisory guidance only. Review the evidence; formal approval remains with the approving authority.")
    st.subheader("Get started")
    upload, explore, review = st.columns(3)
    with upload:
        st.markdown("**1. Upload**")
        st.write("Process one fictional policy in Policy Documents.")
    with explore:
        st.markdown("**2. Explore**")
        st.write("Ask policy questions and read the cited passages.")
    with review:
        st.markdown("**3. Review**")
        st.write("Assess a fictional proposal and review findings and unresolved requirements.")


def about():
    st.title("About Us")
    st.subheader("AI Travel Policy Compliance Assistant")
    st.write("An individual AI Bootcamp capstone proof of concept designed to help officers understand travel policies and make informed official travel planning decisions.")

    with st.container(border=True):
        st.subheader("The problem")
        st.write("Officers planning official travel may need to manually review lengthy policy documents to identify applicable requirements, expenditure limits and approval conditions. This can be time-consuming and may lead to inconsistent interpretations, overlooked requirements or additional administrative effort.")
        st.subheader("Our approach")
        st.write("The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant policy passages and generate evidence-supported answers and advisory compliance assessments. Both use cases operate against the active uploaded fictional travel policy.")

    st.subheader("Core use cases")
    questions_card, compliance_card = st.columns(2)
    with questions_card, st.container(border=True):
        st.subheader("Travel Policy Q&A")
        st.markdown("- Ask natural-language policy questions.\n- Receive answers grounded in retrieved policy passages.\n- Check source references to verify the answer.")
    with compliance_card, st.container(border=True):
        st.subheader("Travel Policy Compliance Checker")
        st.markdown("- Review proposed flight and accommodation arrangements.\n- Assess applicable policy requirements.\n- Receive independent advisory findings for flights and accommodation, with supporting evidence.\n- Identify violations and unresolved requirements.")

    st.subheader("Expected impact")
    st.caption("Potential benefits of the proof of concept; these are not measured business outcomes.")
    st.markdown("- **Improved efficiency:** Reduce time spent manually searching policy documents.\n- **Improved compliance awareness:** Identify potential violations and unresolved requirements before booking.\n- **Greater transparency:** Support answers and findings with traceable policy evidence.\n- **Better decision support:** Help officers make more informed travel planning decisions.")

    st.subheader("Technology stack")
    application, intelligence = st.columns(2)
    with application, st.container(border=True):
        st.markdown("**Application and document processing**")
        st.markdown("- **Python** — Application logic\n- **Streamlit** — User interface\n- **PyMuPDF** — PDF text extraction")
    with intelligence, st.container(border=True):
        st.markdown("**AI and retrieval**")
        st.markdown("- **OpenAI** — Language model and embeddings\n- **ChromaDB** — Vector storage and semantic retrieval\n- **LangChain** — Text chunking")

    with st.container(border=True):
        st.subheader("Scope and limitations")
        st.markdown("- This is an educational proof of concept. Use only synthetic or fictional travel policy documents and travel details.\n- Assessments are advisory and do not constitute formal travel approval.\n- Missing information or insufficient evidence may require manual review.\n- Complex policy interpretation remains outside the intended POC scope.\n- The application does not provide live travel prices, booking services or independent approval verification.")


def methodology():
    st.title("Methodology")
    st.caption("How policy retrieval, AI generation and application-level checks work together.")
    st.write("Retrieval-Augmented Generation (RAG) supplies relevant document passages to a language model at question or assessment time. Both use cases retrieve evidence from the active uploaded fictional policy. OpenAI generates structured answers or findings, and Python validates evidence and applies supported compliance checks before displaying results.")
    st.write("Embeddings represent text as numerical vectors for semantic similarity search. Ingestion creates a searchable knowledge base; the application does not train or fine-tune the language model.")

    with st.container(border=True):
        st.subheader("1. System architecture")
        st.graphviz_chart(diagrams.ARCHITECTURE, width="stretch", alt="System architecture separating local Streamlit, Python services and ChromaDB from external OpenAI embedding and language model processing.")
        st.write("Streamlit presents the interface, while Python coordinates ingestion, retrieval and validation. ChromaDB stores policy text, embeddings and filename/page metadata locally. OpenAI processes document chunks and search queries for embeddings, and questions, recent conversation, travel inputs and retrieved passages for generation. API responses return to Python for validation and display.")
        st.caption("Solid arrows show processing and data exchange. Dashed arrows supply local evidence or coverage context. This diagram groups services; the flowcharts below show their sequence.")

    with st.container(border=True):
        st.subheader("2. Document ingestion")
        st.graphviz_chart(diagrams.INGESTION, width="stretch", alt="Validated fictional PDFs are extracted, chunked, embedded externally and stored locally before atomic activation; failures preserve the previous policy.")
        st.write("The backend checks PDF type, a 20 MB size limit, encryption and a 1–300 page count before extracting selectable text. Each page is split separately, preserving its filename and physical page number. Scanned/image-only documents require OCR, which this POC does not provide.")
        st.write("After embeddings are returned, Python builds a new ChromaDB collection and checks its stored chunk count. Only successful storage activates the new policy for both use cases. A failed upload preserves the previous active knowledge base; superseded collection cleanup occurs after activation.")

    with st.container(border=True):
        st.subheader("3. Travel Policy Q&A")
        st.graphviz_chart(diagrams.QA, width="stretch", alt="Question and follow-up context lead to semantic retrieval and structured generation, with a branch between validated cited answers and insufficient evidence.")
        st.write("For short follow-up questions, retrieval can include the previous user question. Generation receives up to eight recent messages as untrusted context, never as policy evidence. The model returns structured claims, passage IDs and verbatim quotations, or indicates insufficient evidence.")
        st.write("Python validates retrieved IDs and contiguous source quotations, allowing whitespace differences only. Citations use the validated passage's filename and physical page number rather than model-invented references. Missing or unusable evidence produces an insufficient-evidence response. History resets on policy replacement, and stale results are rejected.")

    with st.container(border=True):
        st.subheader("4. Travel Policy Compliance Checker")
        st.graphviz_chart(diagrams.COMPLIANCE, width="stretch", alt="Separate flight and accommodation retrieval feeds structured findings and deterministic checks; each component returns Compliant, Non-compliant or Insufficient information.")
        st.caption("Outcome colours: green — Compliant; red — Non-compliant; amber — Insufficient information. Status decisions apply separately to each component.")
        st.write("After input validation, separate queries retrieve up to three flight and three accommodation passages. Python also reads the complete active-policy text locally for requirement coverage; this is not an additional LLM call. Structured findings are checked against validated quotations, applicable requirements, cabin permissions, comparison operators, currencies and exception scope.")
        st.write("Supported numerical comparisons use Decimal values and policy-backed thresholds. A validated violation takes Non-compliant precedence within its component even when other requirements remain unresolved. Otherwise, incomplete coverage, missing evidence or unresolved conditions lead to Insufficient information. Compliant requires complete applicable coverage and resolved checks.")
        st.write("General travel authorisation remains separate from cabin exceptions. Approval selections and additional context are self-reported, not independent verification. Presentation consolidates related findings after status calculation while retaining the checked ledger and warnings. The summary is advisory and cannot formally approve travel.")

    st.subheader("5. AI and conventional programming")
    ai, programming = st.columns(2)
    with ai, st.container(border=True):
        st.subheader("AI responsibilities")
        st.markdown("- Generate text embeddings.\n- Interpret natural-language questions and retrieved policy context.\n- Generate policy-grounded structured answer claims.\n- Propose structured compliance findings for validation.")
    with programming, st.container(border=True):
        st.subheader("Conventional programming responsibilities")
        st.markdown("- Validate inputs and manage the active policy.\n- Query ChromaDB for semantically similar passages.\n- Validate evidence quotations and source metadata.\n- Check supported numerical comparisons and approval dependencies.\n- Check applicable requirement coverage.\n- Determine component statuses and render citations.")

    with st.container(border=True):
        st.subheader("6. Reliability, safeguards and POC limits")
        st.markdown("- **Version isolation:** Retrieval, coverage and final result checks use the same immutable active collection; replaced-policy results are rejected.\n- **Evidence and coverage:** Exact source spans and metadata are validated. Missing applicable requirements cannot be treated as satisfied.\n- **Conservative fallback:** Unsupported evidence or interpretations require manual review rather than a guessed conclusion.\n- **Untrusted inputs:** Policy text, questions, history and travel context are separated from system instructions. They cannot serve as independently verified approval.\n- **Human oversight:** Findings are advisory. Officers must review the evidence and obtain formal approval through the appropriate authority.\n- **Fictional data only:** OpenAI processing is external. Do not submit internal agency documents or real sensitive travel information.")
        st.write("These safeguards reduce unsupported outputs; they do not eliminate hallucinations or guarantee accuracy. Complex policy interpretation is outside the intended educational POC scope. Inclusive hotel star-rating ranges remain best effort: live responses can abstain despite retrieving the correct range clause. Incomplete coverage and manual-review warnings must be respected; Insufficient information is not permission to book.")


PAGES = {
    "Home": home,
    "Travel Policy Q&A": questions,
    "Travel Policy Compliance Checker": compliance,
    "Policy Documents": documents,
    "About Us": about,
    "Methodology": methodology,
}
