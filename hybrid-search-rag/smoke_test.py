"""Sanity check: index this repo, then compare keyword/semantic/hybrid
search on a query where keyword search should have a clear edge
(an exact identifier name) to prove hybrid isn't worse than either alone.
"""
from src.ingest import chunk_repo
from src.search_engine import hybrid_search, index_chunks, keyword_search, vector_search

chunks = chunk_repo(".")
n = index_chunks(chunks)
print(f"Indexed {n} chunks\n")

query = "BM25Okapi"
print(f"Query: {query!r}\n")

for name, fn in [("Keyword", keyword_search), ("Semantic", vector_search), ("Hybrid", hybrid_search)]:
    hits = fn(query, top_k=3)
    print(f"-- {name} --")
    for h in hits:
        print(f"  {h['meta']['file_path']} (lines {h['meta']['start_line']}-{h['meta']['end_line']})")
    print()
