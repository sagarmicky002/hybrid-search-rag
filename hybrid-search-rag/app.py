import streamlit as st

from src.ingest import chunk_repo
from src.query_engine import answer_question
from src.search_engine import index_chunks, keyword_search, vector_search

st.set_page_config(page_title="HybridSearch — Keyword + Semantic Retrieval", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background:
            linear-gradient(0deg, rgba(20,184,166,0.06) 1px, transparent 1px) 0 0 / 100% 42px,
            linear-gradient(135deg, #f4fbfa 0%, #eef6ff 45%, #fdf6ee 100%);
        background-attachment: fixed;
        color: #0f2027;
    }
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label { color: #0f2027; }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 2px solid #0f2027;
    }

    .hero-band {
        background: #0f2027;
        border-radius: 18px;
        padding: 1.4rem 1.7rem;
        margin-bottom: 1.4rem;
        border: 2px solid #0f2027;
    }
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .hero-title .accent { color: #2dd4bf; }
    .hero-subtitle { color: rgba(255,255,255,0.75); font-size: 0.98rem; }

    .section-badge {
        display: inline-block;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #0f2027;
        background: #ffe8b3;
        border: 2px solid #0f2027;
        border-radius: 6px;
        padding: 0.2rem 0.7rem;
        margin-bottom: 0.6rem;
        box-shadow: 3px 3px 0 #0f2027;
    }

    .stButton > button {
        background: #2dd4bf;
        color: #0f2027;
        border: 2px solid #0f2027;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
        box-shadow: 4px 4px 0 #0f2027;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0 #0f2027;
        color: #0f2027;
    }

    [data-testid="stTextInput"] input,
    input[type="text"] {
        background: #ffffff !important;
        border: 2px solid #0f2027 !important;
        border-radius: 10px !important;
        color: #0f2027 !important;
        caret-color: #0f2027 !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: rgba(15,32,39,0.4) !important; }

    .method-card {
        background: #ffffff;
        border: 2px solid #0f2027;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 5px 5px 0 rgba(15,32,39,0.15);
    }
    .method-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px dashed rgba(15,32,39,0.2);
    }
    .result-line { font-size: 0.85rem; color: #0f2027; margin-bottom: 0.4rem; }

    .answer-card {
        background: #fffaf0;
        border: 2px solid #0f2027;
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 6px 6px 0 rgba(15,32,39,0.12);
    }

    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 2px solid #0f2027 !important;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-band"><div class="hero-title">🔎 Hybrid<span class="accent">Search</span></div>'
    '<div class="hero-subtitle">Compare keyword search, semantic search, and hybrid retrieval side by side — then get a cited AI answer built from the best of both.</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<span class="section-badge">Step 1</span>', unsafe_allow_html=True)
    st.subheader("📂 Index a repo")
    repo_path = st.text_input("Local repo path", placeholder=r"D:\my-project")
    if st.button("⚡ Index Repo") and repo_path:
        with st.spinner("Chunking, embedding, and building keyword index..."):
            chunks = chunk_repo(repo_path)
            n = index_chunks(chunks)
        st.success(f"Indexed {n} chunks from `{repo_path}`")

    st.divider()
    st.caption("Keyword: BM25 · Semantic: local embeddings + Chroma · Fusion: Reciprocal Rank Fusion")

st.markdown('<span class="section-badge">Step 2</span>', unsafe_allow_html=True)
st.subheader("🔍 Search & compare")
query = st.text_input("Search query or question", placeholder="e.g. BM25Okapi OR how does hybrid ranking work")
go = st.button("Compare + Ask")

if go and query:
    with st.spinner("Running all three search methods..."):
        kw_hits = keyword_search(query, top_k=5)
        vec_hits = vector_search(query, top_k=5)

    st.markdown('<span class="section-badge">Retrieval comparison</span>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">🔤 Keyword search (BM25)</div>', unsafe_allow_html=True)
        if kw_hits:
            for h in kw_hits:
                st.markdown(
                    f'<div class="result-line">📄 {h["meta"]["file_path"]} (lines {h["meta"]["start_line"]}-{h["meta"]["end_line"]})</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="result-line">No keyword matches.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.markdown('<div class="method-title">🧠 Semantic search (embeddings)</div>', unsafe_allow_html=True)
        if vec_hits:
            for h in vec_hits:
                st.markdown(
                    f'<div class="result-line">📄 {h["meta"]["file_path"]} (lines {h["meta"]["start_line"]}-{h["meta"]["end_line"]})</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="result-line">No semantic matches.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Fusing both + generating answer..."):
        answer, hybrid_hits = answer_question(query, top_k=5)

    st.markdown('<span class="section-badge">Hybrid answer</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    if hybrid_hits:
        st.markdown('<span class="section-badge">Hybrid sources (rank-fused)</span>', unsafe_allow_html=True)
        for h in hybrid_hits:
            loc = f"📄 {h['meta']['file_path']}  (lines {h['meta']['start_line']}-{h['meta']['end_line']})"
            with st.expander(loc):
                st.code(h["text"])
else:
    st.markdown(
        '<div class="method-card">👋 Index a repo in the sidebar, then run a search to compare methods side by side.</div>',
        unsafe_allow_html=True,
    )
