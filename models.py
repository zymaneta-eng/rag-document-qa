"""
RAG Document Q&A Pydantic Schemas.
Defines structured data contracts for chunking, citations, query requests, and responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Chunk of extracted document text with provenance metadata."""
    chunk_id: str = Field(description="Unique identifier for chunk")
    text: str = Field(description="Text segment of the document")
    page_number: int = Field(default=1, description="1-indexed source document page")
    source_file: str = Field(description="Original filename")
    chunk_index: int = Field(default=0, description="Sequential position in document")


class SourceCitation(BaseModel):
    """Source reference returned with an answer for verification."""
    chunk_id: str
    page_number: int
    snippet: str
    relevance_score: float = Field(description="Cosine similarity relevance score (0.0 to 1.0)")


class QueryRequest(BaseModel):
    """Question submission request."""
    question: str = Field(description="Natural language question about the document", min_length=2)
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")
    min_relevance: float = Field(default=0.20, ge=0.0, le=1.0, description="Minimum similarity threshold")


class QueryResponse(BaseModel):
    """Grounded answer with citations and confidence metadata."""
    question: str
    answer: str
    answered_from_document: bool = Field(description="True if context was sufficient to answer")
    answer_mode: str = Field(default="extractive_passage", description="'llm_generative' (OpenAI) or 'extractive_passage' (snippet extraction)")
    citations: List[SourceCitation] = Field(default_factory=list)
    latency_ms: float = 0.0


class IngestionResponse(BaseModel):
    """Document upload & indexing summary."""
    filename: str
    total_pages: int
    total_chunks: int
    status: str
    message: str
