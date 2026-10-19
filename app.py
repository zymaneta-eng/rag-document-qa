"""
Streamlit Web UI for RAG Document Q&A Prototype.
Provides an interactive, visual interface for uploading documents and asking grounded questions.
"""

import streamlit as st
from pathlib import Path
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
    st.session_state.engine = RAGEngine(persist_dir="./chroma_db", chunk_size=600, chunk_overlap=100)
    # Pre-index sample document if available
    sample_pdf = Path(__file__).parent / "sample_docs" / "enterprise_sla_policy.pdf"
    if sample_pdf.exists():
        st.session_state.engine.ingest_pdf(sample_pdf.read_bytes(), filename="enterprise_sla_policy.pdf")

engine: RAGEngine = st.session_state.engine

# Header
st.title("📚 RAG Document Q&A Prototype")
st.caption("Upload a PDF/text file, index into ChromaDB, and ask grounded questions with source page citations.")

# Sidebar: Document Management
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt", "md"])

if uploaded_file is not None:
    if st.sidebar.button("⚡ Index Uploaded File", type="primary", use_container_width=True):
        content = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".pdf"):
            res = engine.ingest_pdf(content, filename=uploaded_file.name)
        else:
            text = content.decode("utf-8", errors="replace")
            res = engine.ingest_text(text, filename=uploaded_file.name)
        st.sidebar.success(f"Indexed: {res.total_chunks} chunks ({res.total_pages} pages)")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Indexed Chunks:** `{len(engine.chunks)}`")
st.sidebar.markdown(f"**Vector Store:** `ChromaDB (HNSW Cosine)`")

# Main Interface
col_main, col_info = st.columns([2, 1])

with col_main:
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
        with st.spinner("Searching ChromaDB and generating answer..."):
            response = engine.query(user_query, top_k=3, min_relevance=0.20)

        st.markdown("---")
        if response.answered_from_document:
            st.success(f"**Answer (Generated in {response.latency_ms} ms):**")
            st.markdown(response.answer)

            st.markdown("#### 📖 Source Citations (Provenience)")
            for cit in response.citations:
                with st.expander(f"📄 Page {cit.page_number} (Relevance Score: {cit.relevance_score * 100:.1f}%)"):
                    st.write(cit.snippet)
        else:
            st.warning("⚠️ **Out of Context / Refusal:**")
            st.markdown(response.answer)
            st.caption("The similarity threshold check prevented a hallucinated answer because the document does not contain this information.")

with col_info:
    st.subheader("ℹ️ How This Works")
    st.markdown("""
1. **Extraction:** `pypdf` extracts text while preserving page numbers.
2. **Chunking:** Hierarchical recursive text splitting (600 chars, 100 overlap).
3. **Vector Store:** **ChromaDB** stores dense vectors and metadata.
4. **Accuracy Guard:** Rejects questions with cosine distance > 0.80 to prevent hallucinations.
5. **Citations:** Every answer cites the exact source page and excerpt.
    """)
