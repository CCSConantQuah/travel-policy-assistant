"""Static DOT diagrams for the Methodology page; no backend processing."""

# Streamlit accepts DOT strings directly and renders them in its native chart.
# Shared styling keeps diagrams legible without CSS or additional dependencies.
STYLE = '''
    graph [bgcolor="transparent", rankdir=TB, nodesep=0.3, ranksep=0.4, pad=0.2];
    node [shape=box, style="rounded,filled", fillcolor="#EFF6FF",
          color="#CBD5E1", fontcolor="#0F172A", fontname="Arial", fontsize=14,
          margin="0.16,0.12"];
    edge [color="#64748B", fontcolor="#475569", fontname="Arial", fontsize=12,
          arrowsize=0.7];
'''

ARCHITECTURE = 'digraph architecture {' + STYLE + r'''
    officer [label="Officer / user"];
    subgraph cluster_local {
        label="LOCAL APPLICATION AND KNOWLEDGE BASE";
        color="#CBD5E1"; fontname="Arial"; fontcolor="#475569";
        ui [label="Streamlit user interface"];
        services [label="Python application services"];
        ingestion [label="Document ingestion\nPyMuPDF extraction + LangChain chunking"];
        qa [label="Travel Policy Q&A"];
        compliance [label="Travel Policy Compliance Checker"];
        chroma [label="ChromaDB\nLocal text, vectors and source metadata\nSemantic retrieval", fillcolor="#FFFFFF"];
        validation [label="Application-level validation\nSources, evidence and compliance checks"];
        outputs [label="Cited answers / advisory findings\nOr insufficient evidence and manual review"];
        ui -> services;
        services -> ingestion;
        services -> qa;
        services -> compliance;
        ingestion -> chroma [label="store embedded chunks"];
        qa -> chroma [label="retrieve passages", dir=both];
        compliance -> chroma [label="retrieve + read coverage", dir=both];
        validation -> outputs;
    }
    subgraph cluster_external {
        label="EXTERNAL OPENAI API PROCESSING";
        color="#1D4ED8"; fontname="Arial"; fontcolor="#1D4ED8";
        embeddings [label="Embedding service\nDocument chunks and search queries"];
        llm [label="Language model\nStructured answers and findings"];
    }
    officer -> ui;
    services -> embeddings [label="text / returned vectors", dir=both];
    qa -> llm [label="question, recent history + passages"];
    compliance -> llm [label="travel inputs + passages"];
    llm -> validation [label="structured output"];
    chroma -> validation [label="source metadata / local coverage", style=dashed];
}'''

INGESTION = 'digraph ingestion {' + STYLE + r'''
    upload [label="Upload fictional policy PDF\nConfirm suitability for external processing"];
    validate [label="Validate file and document\nPDF, size, encryption and page count"];
    extract [label="PyMuPDF: extract selectable page text\nRetain filename + physical page number"];
    chunks [label="LangChain: split each page into chunks\n1,000 characters; up to 200 overlap"];
    embeddings [label="OpenAI API: generate text embeddings\nExternal processing", color="#1D4ED8"];
    store [label="Store in a new local ChromaDB collection\nVectors + text + source metadata"];
    complete [label="Storage complete?", shape=diamond];
    activate [label="Atomically activate the new policy\nAvailable to Q&A and compliance", fillcolor="#DCFCE7", color="#15803D"];
    preserve [label="Report processing failure\nKeep the previous active policy", fillcolor="#FEF3C7", color="#B45309"];
    upload -> validate -> extract -> chunks -> embeddings -> store -> complete;
    complete -> activate [label="Yes"];
    complete -> preserve [label="No"];
    validate -> preserve [label="Invalid", style=dashed];
    extract -> preserve [label="No usable text", style=dashed];
    embeddings -> preserve [label="API failure", style=dashed];
}'''

QA = 'digraph qa {' + STYLE + r'''
    question [label="Officer submits a natural-language question"];
    query [label="Check input + active policy version\nBuild retrieval query with previous user question\nwhen applicable"];
    embed [label="OpenAI API: embed retrieval query", color="#1D4ED8"];
    retrieve [label="ChromaDB: retrieve up to 3 passages\nFrom the same active policy"];
    llm [label="OpenAI API: structured answer generation\nQuestion + up to 8 recent messages + passages", color="#1D4ED8"];
    validate [label="Check evidence sufficiency, passage IDs\nand contiguous source quotations"];
    enough [label="Sufficient validated evidence?", shape=diamond];
    answer [label="Grounded answer + source citations", fillcolor="#DCFCE7", color="#15803D"];
    insufficient [label="Insufficient-evidence response\nDo not infer missing policy rules", fillcolor="#FEF3C7", color="#B45309"];
    version [label="Recheck active version before displaying\nDiscard stale work; clear old-policy history"];
    question -> query -> embed -> retrieve -> llm -> validate -> enough;
    retrieve -> insufficient [label="No passages", style=dashed];
    enough -> answer [label="Yes"];
    enough -> insufficient [label="No"];
    answer -> version;
    insufficient -> version;
}'''

COMPLIANCE = 'digraph compliance {' + STYLE + r'''
    inputs [label="Proposed flight and accommodation details\nApproval claims remain self-reported"];
    valid [label="Inputs valid and active policy available?", shape=diamond];
    correct [label="Explain missing setup / invalid inputs\nDo not request assessment", fillcolor="#FEF3C7", color="#B45309"];
    retrieval [label="Embed separate flight and accommodation queries\nRetrieve up to 3 passages for each component"];
    snapshot [label="Read complete active-policy text locally\nFor applicable requirement coverage"];
    llm [label="OpenAI API: structured flight and hotel findings\nUsing travel inputs + retrieved passages", color="#1D4ED8"];
    checks [label="Validate evidence, sources and requirement coverage\nCheck cabin permissions and exception scope\nDecimal comparisons: supported operators + currencies\nApply approval dependencies; do not verify claims"];
    violation [label="Validated violation\nin this component?", shape=diamond];
    complete [label="Applicable coverage complete\nand all checks resolved?", shape=diamond];
    noncompliant [label="Non-compliant", fillcolor="#FEE2E2", color="#B91C1C"];
    compliant [label="Compliant", fillcolor="#DCFCE7", color="#15803D"];
    insufficient [label="Insufficient information", fillcolor="#FEF3C7", color="#B45309"];
    output [label="Recheck policy version\nIndependent flight / accommodation statuses\nAdvisory summary + findings + validated evidence"];
    inputs -> valid;
    valid -> correct [label="No"];
    valid -> retrieval [label="Yes"];
    retrieval -> snapshot -> llm -> checks -> violation;
    snapshot -> checks [label="local coverage inventory", style=dashed];
    violation -> noncompliant [label="Yes"];
    violation -> complete [label="No"];
    complete -> compliant [label="Yes"];
    complete -> insufficient [label="No"];
    noncompliant -> output;
    compliant -> output;
    insufficient -> output;
}'''
