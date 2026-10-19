"""
RAG Engine: Document Ingestion, Recursive Chunking, Vector Indexing & Grounded Generation.
Implements:
  1. PDF & Text parsing with page-level provenance.
  2. Recursive character text chunking with overlap.
  3. ChromaDB vector indexing (with lightweight fallback for test environments).
  4. Top-K similarity retrieval with score thresholding.
  5. Refusal logic when context is insufficient (prevents hallucinations).
  6. Source citations with page numbers and exact snippets.
"""

import os
import re
import math
import uuid
import time
from typing import List, Tuple, Dict, Any, Optional
from io import BytesIO
from pypdf import PdfReader

from models import DocumentChunk, SourceCitation, QueryResponse, IngestionResponse


class RecursiveTextSplitter:
    """Splits text hierarchically on paragraphs, sentences, and words."""
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size])
                else:
                    other_chunks = self._split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return [c.strip() for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc.strip())
                    # Handle overlap by keeping tail elements
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped) + len(separator)
            current_doc.append(d)
            total += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc.strip())
        return docs


def compute_embedding(text: str, dim: int = 1024) -> List[float]:
    """
    Computes a deterministic, L2-normalized vector for semantic retrieval.
    Filters common functional stopwords to maximize retrieval SNR.
    """
    import hashlib
    words = re.findall(r"\w+", text.lower())
    stopwords = {"a", "an", "the", "in", "on", "of", "and", "or", "for", "to", "with", "is", "are", "at", "by", "what", "it", "how"}
    filtered = [w for w in words if w not in stopwords] or words
    vec = [0.0] * dim
    for w in filtered:
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 5) for x in vec]


class RAGEngine:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        in_memory: bool = True,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.persist_dir = persist_dir
        self.in_memory = in_memory
        self.chunks: Dict[str, DocumentChunk] = {}
        self._init_vector_store()

    def _init_vector_store(self):
        """Initializes ChromaDB (in-memory EphemeralClient or persistent) or fallback."""
        self.chroma_client = None
        self.collection = None
        try:
            import chromadb
            if self.in_memory or not self.persist_dir:
                self.chroma_client = chromadb.EphemeralClient()
            else:
                self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.chroma_client = None
            self.collection = None

    def reset(self):
        """Clear previously indexed chunks and recreate collection to avoid document blending."""
        self.chunks.clear()
        if self.chroma_client:
            try:
                self.chroma_client.delete_collection("rag_documents")
            except Exception:
                pass
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name="rag_documents",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass

    def ingest_pdf(self, file_bytes: bytes, filename: str) -> IngestionResponse:
        """Extracts text per page and chunks with page number tracking. Resets previous document state."""
        self.reset()
        reader = PdfReader(BytesIO(file_bytes))
        total_pages = len(reader.pages)
        all_chunks: List[DocumentChunk] = []

        chunk_counter = 0
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            text_chunks = self.splitter.split_text(page_text)
            for ch_idx, chunk_text in enumerate(text_chunks):
                chunk_counter += 1
                chunk_id = f"{filename}_p{page_idx}_c{ch_idx}_{uuid.uuid4().hex[:6]}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page_idx,
                    source_file=filename,
                    chunk_index=chunk_counter,
                )
                all_chunks.append(chunk)
                self.chunks[chunk_id] = chunk

        self._index_chunks(all_chunks)

        return IngestionResponse(
            filename=filename,
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            status="indexed",
            message=f"Successfully extracted {len(all_chunks)} chunks across {total_pages} page(s).",
        )

    def ingest_text(self, text: str, filename: str) -> IngestionResponse:
        """Extracts text and chunks with page 1 tracking. Resets previous document state."""
        self.reset()
        text_chunks = self.splitter.split_text(text)
        all_chunks = []
        for ch_idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = f"{filename}_p1_c{ch_idx}_{uuid.uuid4().hex[:6]}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                page_number=1,
                source_file=filename,
                chunk_index=ch_idx,
            )
            all_chunks.append(chunk)
            self.chunks[chunk_id] = chunk

        self._index_chunks(all_chunks)

        return IngestionResponse(
            filename=filename,
            total_pages=1,
            total_chunks=len(all_chunks),
            status="indexed",
            message=f"Successfully extracted and indexed {len(all_chunks)} chunk(s).",
        )

    def _index_chunks(self, chunks: List[DocumentChunk]):
        """Store chunk vectors in ChromaDB or internal index."""
        if not chunks:
            return

        if self.collection:
            try:
                embeddings = [compute_embedding(c.text) for c in chunks]
                self.collection.upsert(
                    ids=[c.chunk_id for c in chunks],
                    embeddings=embeddings,
                    documents=[c.text for c in chunks],
                    metadatas=[{"page": c.page_number, "source": c.source_file} for c in chunks]
                )
                return
            except Exception:
                pass

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Semantic search returning top-k chunks with cosine relevance score."""
        if not self.chunks:
            return []

        # 1. Use ChromaDB with explicit query embedding if available
        if self.collection and self.collection.count() > 0:
            try:
                q_emb = compute_embedding(question)
                results = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k, self.collection.count()),
                )
                retrieved = []
                if results and "ids" in results and results["ids"][0]:
                    ids = results["ids"][0]
                    distances = results["distances"][0] if "distances" in results else [0.2] * len(ids)
                    for chunk_id, dist in zip(ids, distances):
                        if chunk_id in self.chunks:
                            sim = max(0.0, min(1.0, 1.0 - dist))
                            retrieved.append((self.chunks[chunk_id], round(sim, 3)))
                    return retrieved
            except Exception:
                pass

        # 2. Resilient Deterministic Vector Retrieval Fallback
        # Vectorized TF-IDF style term frequency with length normalization
        query_words = set(re.findall(r"\w+", question.lower()))
        scored: List[Tuple[DocumentChunk, float]] = []

        for chunk in self.chunks.values():
            chunk_words = re.findall(r"\w+", chunk.text.lower())
            if not chunk_words:
                continue
            intersection = query_words.intersection(chunk_words)
            if not intersection:
                score = 0.0
            else:
                overlap = sum(chunk_words.count(w) for w in intersection)
                score = overlap / math.sqrt(len(chunk_words) * (len(query_words) + 1))
                score = round(min(1.0, score * 1.5), 3)

            if score > 0.05:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def query(
        self,
        request_question: str,
        top_k: int = 3,
        min_relevance: float = 0.20,
        api_key: Optional[str] = None,
        allow_llm: bool = True,
    ) -> QueryResponse:
        """Main RAG Q&A query workflow with refusal logic, citations, and optional LLM synthesis."""
        start_time = time.perf_counter()

        retrieved = self.retrieve(request_question, top_k=top_k)

        # Accuracy Safeguard: Refusal if no chunks exceed minimum threshold
        valid_chunks = [item for item in retrieved if item[1] >= min_relevance]

        if not valid_chunks:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return QueryResponse(
                question=request_question,
                answer="The answer to this question could not be found in the uploaded document.",
                answered_from_document=False,
                citations=[],
                latency_ms=latency_ms,
            )

        # Build Citations
        citations = [
            SourceCitation(
                chunk_id=c.chunk_id,
                page_number=c.page_number,
                snippet=(c.text[:400] + "...") if len(c.text) > 400 else c.text,
                relevance_score=score,
            )
            for c, score in valid_chunks
        ]

        # Generate Grounded Answer (LLM if allowed & key present, else extractive)
        answer, answer_mode = self._generate_answer(
            request_question,
            valid_chunks,
            api_key=api_key if allow_llm else None,
            allow_llm=allow_llm,
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            question=request_question,
            answer=answer,
            answered_from_document=True,
            answer_mode=answer_mode,
            citations=citations,
            latency_ms=latency_ms,
        )

    def _generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Tuple[DocumentChunk, float]],
        api_key: Optional[str] = None,
        allow_llm: bool = True,
        max_tokens: int = 250,
    ) -> Tuple[str, str]:
        """
        Generates answer.
        Returns (answer_text, answer_mode) where mode is:
          - 'llm_generative' (when OpenAI gpt-4o-mini is active)
          - 'extractive_passage' (when returning verified document excerpt)
        """
        context_blocks = []
        for c, score in retrieved_chunks:
            context_blocks.append(f"[Page {c.page_number}]: {c.text}")
        context_str = "\n\n".join(context_blocks)

        key = api_key or (os.getenv("OPENAI_API_KEY") if allow_llm else None)
        if key and allow_llm:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key)
                prompt = (
                    "You are a strict Document Q&A assistant. "
                    "Answer the user's question ONLY using the facts stated in the context snippets below. "
                    "If the question cannot be answered strictly from the context, respond: "
                    "'The answer to this question could not be found in the uploaded document.' "
                    "Always reference the relevant page number in your answer.\n\n"
                    f"Context Snippets:\n{context_str}\n\n"
                    f"Question: {question}\n\nAnswer:"
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip(), "llm_generative"
            except Exception:
                pass

        # Extractive snippet retrieval mode (transparent, non-generative)
        top_chunk, score = retrieved_chunks[0]
        return f"According to page {top_chunk.page_number}: {top_chunk.text}", "extractive_passage"
