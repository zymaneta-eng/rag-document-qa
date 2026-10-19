"""
Automated Pytest Suite for RAG Document Q&A.
Verifies PDF parsing, page metadata tracking, vector retrieval, answer generation,
out-of-scope refusal logic, and FastAPI endpoints.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from rag_engine import RAGEngine
from api import app, engine


SAMPLE_PDF_PATH = Path(__file__).parent / "sample_docs" / "enterprise_sla_policy.pdf"


@pytest.fixture
def rag_instance(tmp_path):
    chroma_dir = str(tmp_path / "test_chroma")
    return RAGEngine(persist_dir=chroma_dir, chunk_size=500, chunk_overlap=80)


def test_pdf_ingestion_and_page_metadata(rag_instance):
    assert SAMPLE_PDF_PATH.exists()
    content = SAMPLE_PDF_PATH.read_bytes()

    res = rag_instance.ingest_pdf(content, filename="enterprise_sla_policy.pdf")
    assert res.total_pages == 3
    assert res.total_chunks >= 3

    # Check page numbers are properly assigned (1, 2, and 3)
    pages_recorded = {c.page_number for c in rag_instance.chunks.values()}
    assert 1 in pages_recorded
    assert 2 in pages_recorded
    assert 3 in pages_recorded


def test_answerable_query_page_1(rag_instance):
    content = SAMPLE_PDF_PATH.read_bytes()
    rag_instance.ingest_pdf(content, filename="enterprise_sla_policy.pdf")

    # Ask about Severity 1 response time (Page 1)
    response = rag_instance.query("What is the initial response time for Severity 1 outages?")
    assert response.answered_from_document is True
    assert len(response.citations) > 0
    assert any("15 minutes" in c.snippet for c in response.citations) or "15 minutes" in response.answer


def test_answerable_query_page_2(rag_instance):
    content = SAMPLE_PDF_PATH.read_bytes()
    rag_instance.ingest_pdf(content, filename="enterprise_sla_policy.pdf")

    # Ask about encryption and SOC 2 (Page 2)
    response = rag_instance.query("What encryption standard is used for data at rest?")
    assert response.answered_from_document is True
    assert len(response.citations) > 0
    assert response.citations[0].page_number == 2
    assert "AES-256" in response.answer or "AES-256" in response.citations[0].snippet


def test_unanswerable_query_triggers_refusal(rag_instance):
    content = SAMPLE_PDF_PATH.read_bytes()
    rag_instance.ingest_pdf(content, filename="enterprise_sla_policy.pdf")

    # Ask completely unrelated question not present in the document
    response = rag_instance.query("What is the recipe for chocolate chip cookies?", min_relevance=0.45)
    assert response.answered_from_document is False
    assert "could not be found" in response.answer.lower()
    assert len(response.citations) == 0


def test_fastapi_endpoints():
    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "RAG Document Q&A"

    # Upload PDF
    with open(SAMPLE_PDF_PATH, "rb") as f:
        r = client.post("/api/v1/upload", files={"file": ("enterprise_sla_policy.pdf", f, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["total_pages"] == 3

    # Query endpoint
    r = client.post("/api/v1/query", json={"question": "What is the monthly uptime percentage guarantee?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answered_from_document"] is True
    assert len(data["citations"]) > 0
    assert data["citations"][0]["page_number"] == 1
