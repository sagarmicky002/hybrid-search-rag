"""Three retrieval methods over the same indexed chunks: keyword (BM25),
vector/semantic (embeddings + Chroma), and hybrid (both, combined via
Reciprocal Rank Fusion).
"""
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "codebase"
RRF_K = 60  # standard constant used in Reciprocal Rank Fusion

_model = None
_bm25 = None
_bm25_docs = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def index_chunks(chunks):
    model = get_model()
    collection = get_collection()

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    ids = [f"{c.file_path}:{c.start_line}-{c.end_line}::{i}" for i, c in enumerate(chunks)]
    metadatas = [
        {"file_path": c.file_path, "start_line": c.start_line, "end_line": c.end_line}
        for c in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    _load_bm25(force=True)
    return len(chunks)


def _load_bm25(force=False):
    global _bm25, _bm25_docs
    if _bm25 is not None and not force:
        return
    collection = get_collection()
    data = collection.get()
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    if not ids:
        _bm25, _bm25_docs = None, []
        return
    _bm25 = BM25Okapi([tokenize(d) for d in docs])
    _bm25_docs = [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]


def vector_search(query, top_k=5):
    model = get_model()
    collection = get_collection()
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    ids = results["ids"][0] if results["ids"] else []
    return [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]


def keyword_search(query, top_k=5):
    _load_bm25()
    if not _bm25_docs:
        return []
    scores = _bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [_bm25_docs[i] for i in ranked if scores[i] > 0]


def hybrid_search(query, top_k=5):
    """Reciprocal Rank Fusion: a doc's score is the sum of 1/(RRF_K + rank)
    across every ranked list it appears in. No score tuning needed —
    just combines rank positions from each method.
    """
    vec_hits = vector_search(query, top_k=top_k * 2)
    kw_hits = keyword_search(query, top_k=top_k * 2)

    fused_scores = {}
    doc_lookup = {}
    for ranked_list in (vec_hits, kw_hits):
        for rank, hit in enumerate(ranked_list):
            fused_scores[hit["id"]] = fused_scores.get(hit["id"], 0) + 1 / (RRF_K + rank + 1)
            doc_lookup[hit["id"]] = hit

    ranked_ids = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
    return [doc_lookup[doc_id] for doc_id, _ in ranked_ids]
