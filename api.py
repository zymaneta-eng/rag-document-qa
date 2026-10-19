"""
FastAPI Microservice for Document Q&A RAG Prototype.
Provides API-first endpoints for document ingestion, semantic search, and grounded Q&A.
"""

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import QueryRequest, QueryResponse, IngestionResponse
from rag_engine import RAGEngine

app = FastAPI(
    title="Document Q&A RAG Prototype API",
    description="API-first Retrieval-Augmented Generation prototype with PDF chunking, ChromaDB vector indexing, and source citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine(persist_dir="./chroma_db", chunk_size=600, chunk_overlap=100)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "RAG Document Q&A",
        "version": "1.0.0",
        "chromadb_active": engine.collection is not None,
        "indexed_chunks": len(engine.chunks),
    }


@app.post("/api/v1/upload", response_model=IngestionResponse, tags=["Document Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document.
    Extracts text, segments into overlapping chunks with page metadata,
    and indexes into ChromaDB vector store.
    """
    content = await file.read()
    filename = file.filename or "uploaded_doc"

    if filename.lower().endswith(".pdf"):
        try:
            return engine.ingest_pdf(content, filename=filename)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to process PDF: {str(exc)}")
    elif filename.lower().endswith((".txt", ".md")):
        text = content.decode("utf-8", errors="replace")
        return engine.ingest_text(text, filename=filename)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Please upload a .pdf, .txt, or .md file."
        )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Question & Answering"])
def ask_question(request: QueryRequest):
    """
    Ask a natural language question about the uploaded document.
    Retrieves top-k relevant chunks from ChromaDB and generates a grounded response.
    Returns refusal message if context does not support the answer.
    """
    if len(engine.chunks) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Upload a document to /api/v1/upload first."
        )

    return engine.query(
        request_question=request.question,
        top_k=request.top_k,
        min_relevance=request.min_relevance,
    )


@app.get("/api/v1/chunks", tags=["Debugging & Inspection"])
def list_indexed_chunks(limit: int = 10):
    """Inspect stored chunks and their provenance page numbers."""
    all_chunks = list(engine.chunks.values())
    return [
        {
            "chunk_id": c.chunk_id,
            "page_number": c.page_number,
            "source_file": c.source_file,
            "preview": c.text[:120] + "...",
        }
        for c in all_chunks[:limit]
    ]
