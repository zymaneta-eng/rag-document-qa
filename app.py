"""
Streamlit Web UI for RAG Document Q&A Prototype.
Provides an interactive, visual interface for uploading documents and asking grounded questions.
"""

import os
import time
from pathlib import Path
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Document Q&A Prototype",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    .citation-card {
        background-color: #1E293B;
        border-left: 4px solid #38BDF8;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine(
        in_memory=True,
        chunk_size=600,
        chunk_overlap=100
    )
    # Pre-index sample document if available
    sample_pdf = Path(__file__).parent / "sample_docs" / "enterprise_sla_policy.pdf"
    if sample_pdf.exists():
        st.session_state.engine.ingest_pdf(sample_pdf.read_bytes(), filename="enterprise_sla_policy.pdf")

engine: RAGEngine = st.session_state.engine

# Header
st.title("📚 RAG Document Q&A Prototype")
st.caption("Upload a PDF/text file, index into ChromaDB, and ask grounded questions with source page citations.")
st.warning(
    "🔒 **Privacy Notice:** Documents are indexed in an ephemeral in-memory vector store (not persisted to disk). "
    "Without an API key, the system returns exact extracted source passages. "
    "If an OpenAI API key is configured, relevant excerpts are sent to OpenAI (GPT-4o-mini) to generate conversational answers. "
    "Please test using public or sample documents. Max file size: 5MB."
)

# Sidebar: Document Management
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT Document (Max 5MB)", type=["pdf", "txt", "md"])

if uploaded_file is not None:
    if st.sidebar.button("⚡ Index Uploaded File", type="primary", use_container_width=True):
        content = uploaded_file.getvalue()
        if len(content) > 5 * 1024 * 1024:
            st.sidebar.error("File exceeds the 5MB size limit for this prototype demo.")
            st.stop()
        if uploaded_file.name.lower().endswith(".pdf"):
            res = engine.ingest_pdf(content, filename=uploaded_file.name)
        else:
            text = content.decode("utf-8", errors="replace")
            res = engine.ingest_text(text, filename=uploaded_file.name)
        st.session_state.active_doc_name = uploaded_file.name
        st.sidebar.success(f"Indexed: {res.total_chunks} chunks ({res.total_pages} pages)")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Indexed Chunks:** `{len(engine.chunks)}`")
st.sidebar.markdown(f"**Vector Store:** `ChromaDB (In-Memory HNSW)`")

# Quota Limiter & LLM Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Engine & Quota")

custom_openai_key = st.sidebar.text_input(
    "Custom OpenAI Key (optional)",
    type="password",
    help="Enter your personal OpenAI key for unlimited queries"
)

MAX_FREE_LLM_CALLS = 5
st.session_state.setdefault("llm_calls_made", 0)
remaining_calls = max(0, MAX_FREE_LLM_CALLS - st.session_state.llm_calls_made)

server_key = os.getenv("OPENAI_API_KEY")

if custom_openai_key:
    st.sidebar.success("🔑 Personal OpenAI Key Active (Unlimited)")
    effective_key = custom_openai_key
    allow_llm = True
elif server_key:
    if remaining_calls > 0:
        st.sidebar.info(f"⚡ Demo Quota: {remaining_calls}/{MAX_FREE_LLM_CALLS} free LLM queries remaining")
        effective_key = server_key
        allow_llm = True
    else:
        st.sidebar.warning(f"⚠️ Demo Quota Exhausted: 0/{MAX_FREE_LLM_CALLS} left (Falling back to Extractive Mode)")
        effective_key = None
        allow_llm = False
else:
    st.sidebar.caption("ℹ️ Extractive Mode Active (No server OpenAI key configured)")
    effective_key = None
    allow_llm = False

# Main Interface
col_main, col_info = st.columns([2, 1])

with col_main:
    # Active document badge
    active_doc_name = getattr(st.session_state, "active_doc_name", "enterprise_sla_policy.pdf (Preloaded Sample)")
    st.info(f"📄 **Active Document in ChromaDB:** `{active_doc_name}`")

    st.subheader("Ask a Question")

    # Sample Question buttons
    st.markdown("**Quick test queries (from sample SLA document):**")
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    preset_q = None
    with btn_c1:
        if st.button("⏱️ Severity 1 response time?"):
            preset_q = "What is the initial response time for Severity 1 outages?"
    with btn_c2:
        if st.button("🔒 Data encryption at rest?"):
            preset_q = "What encryption standard is used for data at rest?"
    with btn_c3:
        if st.button("🍪 Unanswerable question"):
            preset_q = "What is the recipe for chocolate chip cookies?"

    user_query = st.text_input(
        "Enter your question:",
        value=preset_q if preset_q else "",
        placeholder="e.g. What is the monthly uptime percentage guarantee?",
    )

    ask_btn = st.button("🔍 Submit Question", type="primary")

    if (ask_btn or preset_q) and user_query:
        # Rate limit cooldown check (at least 2 seconds between clicks)
        current_time = time.time()
        last_time = st.session_state.get("last_query_ts", 0)
        if current_time - last_time < 2.0:
            st.warning("Please wait a moment before sending another query.")
            st.stop()
        st.session_state.last_query_ts = current_time

        with st.spinner("Searching ChromaDB and processing response..."):
            response = engine.query(
                user_query,
                top_k=3,
                min_relevance=0.20,
                api_key=effective_key,
                allow_llm=allow_llm,
            )
            # Increment quota count if a free server LLM query was consumed
            if response.answer_mode == "llm_generative" and not custom_openai_key:
                st.session_state.llm_calls_made += 1

        st.markdown("---")
        if response.answered_from_document:
            if response.answer_mode == "llm_generative":
                st.info("🤖 **Mode:** Generative LLM Synthesis (OpenAI GPT-4o-mini)")
            else:
                st.info("🔍 **Mode:** Extractive Passage Retrieval (Verified Document Excerpt)")
            st.success(f"**Result (Processed in {response.latency_ms} ms):**")
            st.markdown(response.answer)

            st.markdown("#### 📖 Source Citations (Provenience)")
            for cit in response.citations:
                with st.expander(f"📄 Page {cit.page_number} (Relevance Score: {cit.relevance_score * 100:.1f}%)"):
                    st.write(cit.snippet)
        else:
            st.info("ℹ️ **Information Not Found in Document**")
            st.markdown(
                f"The requested answer could not be found in `{st.session_state.active_doc_name}`."
            )
            st.caption(
                "The vector relevance score fell below the retrieval threshold. "
                "Answers are only returned when directly supported by verified document passages."
            )

with col_info:
    st.subheader("ℹ️ How This Works")
    st.markdown("""
1. **Extraction:** `pypdf` extracts text while preserving page numbers.
2. **Chunking:** Hierarchical recursive text splitting (600 chars, 100 overlap).
3. **Vector Store:** **ChromaDB** stores dense vectors and metadata.
4. **Accuracy Guard:** Rejects queries with cosine distance > 0.80 when verified passages are absent.
5. **Citations:** Every answer cites the exact source page and excerpt.
    """)
