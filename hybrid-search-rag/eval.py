"""Compares retrieval hit-rate across keyword-only, semantic-only, and
hybrid search on the same test questions — the core proof that hybrid
search helps.

Fill in eval_questions.json with real questions about a repo you've
indexed, then run: python eval.py
"""
import json

from src.search_engine import hybrid_search, keyword_search, vector_search


def hit_rate(method_fn, cases):
    hits = 0
    for case in cases:
        results = method_fn(case["question"], top_k=5)
        retrieved_files = {r["meta"]["file_path"] for r in results}
        if case["expected_file"] in retrieved_files:
            hits += 1
    return hits / len(cases) if cases else 0


def run_eval(eval_file="eval_questions.json"):
    with open(eval_file) as f:
        cases = json.load(f)

    keyword_rate = hit_rate(keyword_search, cases)
    vector_rate = hit_rate(vector_search, cases)
    hybrid_rate = hit_rate(hybrid_search, cases)

    print(f"Keyword-only hit rate:  {keyword_rate:.0%}")
    print(f"Semantic-only hit rate: {vector_rate:.0%}")
    print(f"Hybrid hit rate:        {hybrid_rate:.0%}")

    return {"keyword": keyword_rate, "semantic": vector_rate, "hybrid": hybrid_rate}


if __name__ == "__main__":
    run_eval()
