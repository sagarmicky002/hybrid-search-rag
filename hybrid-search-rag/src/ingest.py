"""Walks a local repo and splits it into retrievable chunks."""
import ast
import os
from dataclasses import dataclass
from pathlib import Path

CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".md", ".txt", ".json"}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", "chroma_db"}


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    text: str


def iter_repo_files(repo_path):
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if Path(f).suffix in CODE_EXTENSIONS:
                yield os.path.join(root, f)


def chunk_by_lines(path, text, window=40, overlap=5):
    lines = text.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + window, len(lines))
        snippet = "\n".join(lines[i:end])
        if snippet.strip():
            chunks.append(Chunk(path, i + 1, end, snippet))
        if end == len(lines):
            break
        i += window - overlap
    return chunks


def chunk_python_file(path, text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_by_lines(path, text)
    lines = text.splitlines()
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not nodes:
        return chunk_by_lines(path, text)
    chunks = []
    for node in nodes:
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        snippet = "\n".join(lines[start - 1:end])
        if snippet.strip():
            chunks.append(Chunk(path, start, end, snippet))
    return chunks


def chunk_file(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if Path(path).suffix == ".py":
        return chunk_python_file(path, text)
    return chunk_by_lines(path, text)


def chunk_repo(repo_path):
    all_chunks = []
    for file_path in iter_repo_files(repo_path):
        all_chunks.extend(chunk_file(file_path))
    return all_chunks
