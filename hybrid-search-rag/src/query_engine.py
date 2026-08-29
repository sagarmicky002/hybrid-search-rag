"""Generates a cited answer from Gemini using hybrid-retrieved chunks."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from src.search_engine import hybrid_search

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY as an environment variable (see .env.example)")
        _client = genai.Client(api_key=api_key)
    return _client


def build_prompt(question, hits):
    context_blocks = []
    for h in hits:
        loc = f"{h['meta']['file_path']} (lines {h['meta']['start_line']}-{h['meta']['end_line']})"
        context_blocks.append(f"[Source: {loc}]\n{h['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know.
Cite the source file and line numbers you used.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question, top_k=5):
    hits = hybrid_search(question, top_k=top_k)
    if not hits:
        return "No indexed content found. Index a repo first.", []
    prompt = build_prompt(question, hits)
    client = get_client()
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    except errors.ClientError as e:
        if e.code == 429:
            return "Gemini's free-tier request limit was hit for today. The search results above are still valid — try the answer again later.", hits
        raise
    return response.text, hits
