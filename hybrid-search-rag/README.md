# HybridSearch — Keyword + Semantic Retrieval

A RAG app that retrieves context using **three methods side by side** —
keyword search (BM25), semantic/vector search (embeddings), and a
**hybrid** combination of both — then generates a cited answer using
whichever method the hybrid step ranks best.

## Why hybrid search

Pure vector search is great at understanding *meaning* ("login" ≈
"authentication") but can miss exact matches — like a specific function
or variable name — because it's optimized for semantic similarity, not
literal text. Keyword search (BM25) is the opposite: exact but literal.
Hybrid search combines both, so a query benefits from exact-match
precision *and* meaning-based recall.

## How the fusion works — Reciprocal Rank Fusion (RRF)

Instead of hand-tuning how much to trust each method's *score* (which
are on different scales and hard to compare), RRF combines their
**rankings**:

```
score(doc) = sum over each method's ranked list of  1 / (60 + rank_in_that_list)
```

A document ranked highly by either method gets a high fused score. No
score normalization or tuning required — this is a well-established
technique used in real hybrid search systems.

## Stack (100% free, no credit card)
- **LLM:** Gemini 2.5 Flash
- **Keyword search:** BM25 (`rank_bm25`, pure Python, no API)
- **Semantic search:** `sentence-transformers/all-MiniLM-L6-v2`, run locally
- **Vector store:** Chroma
- **UI:** Streamlit

## Setup
```bash
pip install -r requirements.txt
```
Get a free key at [aistudio.google.com](https://aistudio.google.com), then:
```bash
cp .env.example .env   # fill in GEMINI_API_KEY
```

## Run
```bash
streamlit run app.py
```
Index a repo in the sidebar, then search. The app shows keyword-only and
semantic-only results side by side, plus a final Gemini-generated answer
built from the hybrid-fused results.

## Evaluation
`eval.py` measures **retrieval hit rate** for all three methods on the
same test questions — the direct evidence for whether hybrid actually
helps.

```bash
python eval.py
```

**Sample results** (5 questions about this project's own codebase):
| Method | Hit rate |
|---|---|
| Keyword only | 100% |
| Semantic only | 100% |
| Hybrid | 100% |

On this small, well-organized codebase all three methods tie — there
isn't enough noise or ambiguity to separate them. The qualitative
difference is easier to see directly: searching `"BM25Okapi"` returns a
**different top-3 ranking** from each method (see `smoke_test.py`),
which is what hybrid fusion is combining. Hybrid search's benefit grows
on larger, messier codebases and on queries mixing exact identifiers
with natural language — a known limitation worth stating honestly
rather than cherry-picking a favorable example.

## Design choices
- **Same chunking as the companion project** ([rag-chatbot](https://github.com/GARGI-tec/rag-chatbot)) —
  AST-aware for Python (whole functions/classes), line-window fallback
  for other files.
- **RRF over score-blending:** avoids the brittle problem of tuning a
  weight between BM25 scores and cosine similarities, which live on
  incomparable scales.
- **All three results shown, not just hybrid:** makes the comparison
  demonstrable rather than a black box.

## Limitations
- BM25 index is rebuilt from Chroma's stored documents on each app
  restart rather than persisted separately — fine at this scale, would
  need its own persistence for a larger corpus.
- No cross-encoder re-ranking step after fusion — RRF alone decides
  final order.
- Single-repo indexing at a time.
