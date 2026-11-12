"""Qdrant access: plan settings and the policy documents uploaded by an administrator (the RAG store).

Nothing is seeded: policy chunks are inserted only when a policy document is uploaded.

Connection (first match wins): QDRANT_URL (+ QDRANT_API_KEY) for the `qdrant/qdrant` Docker server; or QDRANT_PATH for
Qdrant's embedded local mode (a directory, or ":memory:"), which needs no server but allows only one process at a time.
"""
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterable

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from mcp_server import embeddings

logger = logging.getLogger(__name__)

SETTINGS = "plan_settings"
DOCUMENTS = "uploaded_documents"
_VECTOR_COLLECTIONS = (DOCUMENTS,)
_PAYLOAD_ONLY = (SETTINGS,)  # Qdrant needs a vector per point; these carry a 1-number placeholder
_NAMESPACE = uuid.UUID("6f1d5c1e-52a0-4c58-8a43-6d1f0d6c7a11")
_BATCH = 128
POLICY_TOP_K = 5
_PAYLOAD_INDEXES = {DOCUMENTS: {"doc_id": "keyword"}}

_client: "QdrantClient | _LockedClient | None" = None
_client_lock = threading.Lock()
_hit_lock = threading.Lock()


class _LockedClient:
    """Embedded mode has no internal locking, so serialise every call."""

    def __init__(self, inner: QdrantClient):
        self._inner, self._lock = inner, threading.RLock()

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return call


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'", ""))).strip()


def enabled() -> bool:
    if os.getenv("QDRANT_ENABLED", "1") == "0":
        return False
    return bool(os.getenv("QDRANT_URL") or os.getenv("QDRANT_PATH"))


def _is_local() -> bool:
    return not os.getenv("QDRANT_URL")


def client():
    global _client
    with _client_lock:
        if _client is None:
            url = os.getenv("QDRANT_URL")
            if url:
                _client = QdrantClient(
                    url=url, api_key=os.getenv("QDRANT_API_KEY") or None,
                    timeout=int(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")),
                    prefer_grpc=os.getenv("QDRANT_PREFER_GRPC") == "1",
                )
            else:
                local = os.environ["QDRANT_PATH"]
                _client = _LockedClient(QdrantClient(location=local) if local == ":memory:" else QdrantClient(path=local))
        return _client


def reset_connection_cache() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None


def _id(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, "\x1f".join(parts)))


def _eq(key: str, value) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _scroll(collection: str, flt: models.Filter | None = None):
    offset = None
    while True:
        points, offset = client().scroll(collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
        yield from (p.payload for p in points)
        if offset is None:
            return


def _upsert(collection: str, points: list[models.PointStruct]) -> None:
    for i in range(0, len(points), _BATCH):
        client().upsert(collection, points=points[i:i + _BATCH], wait=True)


# --- schema -----------------------------------------------------------------------------------------------------

def create_schema() -> None:
    c = client()
    for name in _VECTOR_COLLECTIONS + _PAYLOAD_ONLY:
        if c.collection_exists(name):
            continue
        size, distance = (embeddings.DIMENSIONS, models.Distance.COSINE) if name in _VECTOR_COLLECTIONS else (1, models.Distance.DOT)
        try:
            c.create_collection(name, vectors_config=models.VectorParams(size=size, distance=distance))
        except Exception:
            if not c.collection_exists(name):  # another worker may have created it first
                raise
    if not _is_local():  # embedded mode has no payload indexes
        for name, fields in _PAYLOAD_INDEXES.items():
            for field, kind in fields.items():
                c.create_payload_index(name, field, models.PayloadSchemaType(kind))


def ensure_ready() -> None:
    create_schema()


# --- uploaded documents -----------------------------------------------------------------------------------------

def ingest_document(doc_id: str, filename: str, batches: Iterable[list[str]]) -> dict:
    """Embed and store a document's chunks batch by batch, as each batch arrives.

    The doc_id is a hash of the file, so a document that is already stored is left alone. If a later batch fails,
    the chunks stored so far are removed so a retry starts clean.
    """
    if not client().collection_exists(DOCUMENTS):
        create_schema()
    own = models.Filter(must=[_eq("doc_id", doc_id)])
    existing = client().count(DOCUMENTS, count_filter=own, exact=True).count
    if existing:
        return {"doc_id": doc_id, "chunks": existing, "status": "unchanged"}
    total, now = 0, time.time()
    try:
        for batch in batches:
            _upsert(DOCUMENTS, [
                models.PointStruct(
                    id=_id("document", doc_id, str(total + n)), vector=vec,
                    payload={"doc_id": doc_id, "filename": filename, "chunk_index": total + n, "text": text,
                             "uploaded_at": now, "metadata": {"filename": filename, "chunk_index": total + n}},
                )
                for n, (text, vec) in enumerate(zip(batch, embeddings.embed(batch)))
            ])
            total += len(batch)
    except Exception:
        client().delete(DOCUMENTS, points_selector=models.FilterSelector(filter=own))
        raise
    return {"doc_id": doc_id, "chunks": total, "status": "inserted"}

# --- settings ---------------------------------------------------------------------------------------------------

def get_settings() -> dict[str, str]:
    return {p["key"]: p["value"] for p in _scroll(SETTINGS)}


def set_setting(key: str, value: str) -> None:
    client().set_payload(SETTINGS, {"value": value}, points=[_id("setting", key)])



# --- policy search (RAG) ----------------------------------------------------------------------------------------

class _LangchainEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embeddings.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embeddings.embed([text])[0]


def search_policy(question: str, k: int = POLICY_TOP_K) -> list[str]:
    """Similarity search over the uploaded policy chunks (LangChain QdrantVectorStore); returns the top-k chunk texts."""
    question = question.strip()
    if not question or not client().collection_exists(DOCUMENTS):
        return []
    c = client()
    store = QdrantVectorStore(
        client=getattr(c, "_inner", c), collection_name=DOCUMENTS, embedding=_LangchainEmbeddings(),
        content_payload_key="text", metadata_payload_key="metadata", validate_collection_config=False,
    )
    hits = store.similarity_search_with_score(question, k=k)
    print(f"[RAG] question={question!r} -> top {len(hits)} chunks from Qdrant")
    for rank, (doc, score) in enumerate(hits, 1):
        print(f"[RAG] {rank}. score={score:.4f} file={doc.metadata.get('filename')} text={doc.page_content[:120]!r}")
    return [doc.page_content for doc, _ in hits]
