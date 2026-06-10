"""
Quant RAG Web Server — persistent FastAPI backend

Loads the embedding model, ChromaDB collection, and BM25 index ONCE on
startup and keeps them in memory for the lifetime of the process.
Every query hits these warm in-memory objects, cutting cold-start latency
from ~15 s to ~2-3 s.

Usage (from the rag/ directory, inside quant_venv):
    pip install fastapi uvicorn
    uvicorn rag_server:app --host 0.0.0.0 --port 8000

Then open: http://localhost:8000

For auto-reload during development:
    uvicorn rag_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import uuid

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from rag_build import get_collection, get_embedder
from rag_generate import (
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    XAI_API_BASE,
    build_context_block,
)
from rag_query import (
    FETCH_MULTIPLIER,
    bm25_retrieve,
    load_bm25_index,
    order_for_attention,
    rrf_merge,
    select_diverse_top_k,
    trim_to_sentences,
)

# ── Env ───────────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "rag_db")
COLLECTION_NAME = "quant_rag"
BUFFER_TURNS = 4  # last N turns kept verbatim in session memory

# ── Global in-memory state (loaded once at startup) ───────────────────────────
_embedder = None
_collection = None
_bm25_data = None
_api_key: str = ""
_sessions: dict = {}  # session_id → {"turns": [...]}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quant RAG Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global _embedder, _collection, _bm25_data, _api_key
    print("=== Quant RAG Server starting ===", flush=True)
    print("Loading embedding model...", flush=True)
    _embedder = get_embedder()
    print("Loading ChromaDB collection...", flush=True)
    _collection = get_collection(DB_PATH, COLLECTION_NAME, embedder=_embedder)
    print(f"  {_collection.count()} chunks loaded.", flush=True)
    _bm25_data = load_bm25_index(DB_PATH)
    if _bm25_data:
        print(f"  BM25 index loaded ({len(_bm25_data['ids'])} entries) — hybrid search active.", flush=True)
    else:
        print("  BM25 index not found — dense-only mode. Run build_bm25_only.py to enable hybrid.", flush=True)
    _api_key = os.getenv("XAI_API_KEY", "")
    if not _api_key:
        print("  WARNING: XAI_API_KEY not set in .env — Grok calls will fail.", flush=True)
    print("=== Server ready at http://localhost:8000 ===", flush=True)


# ── Request model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    k: int = 8
    no_hybrid: bool = False
    model: str = DEFAULT_MODEL
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET


# ── Retrieval (uses global warm state) ───────────────────────────────────────
def _retrieve(query: str, k: int, no_hybrid: bool):
    fetch_n = max(k * FETCH_MULTIPLIER, k)
    results = _collection.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )
    dense_docs  = results.get("documents", [[]])[0]
    dense_metas = results.get("metadatas",  [[]])[0]
    dense_dists = results.get("distances",  [[]])[0]
    dense_ids   = results.get("ids",        [[]])[0]

    if not dense_docs:
        return []

    docs, metas, dists = dense_docs, dense_metas, dense_dists

    if not no_hybrid and _bm25_data is not None:
        dense_ranked  = list(zip(dense_ids, [1 - d for d in dense_dists]))
        sparse_ranked = bm25_retrieve(query, _bm25_data, fetch_n)
        rrf_scores    = rrf_merge(dense_ranked, sparse_ranked)

        id_to_chunk = {
            cid: (doc, meta, dist)
            for cid, doc, meta, dist in zip(dense_ids, dense_docs, dense_metas, dense_dists)
        }
        bm25_id_map = {cid: i for i, cid in enumerate(_bm25_data["ids"])}
        for cid, _ in sparse_ranked:
            if cid not in id_to_chunk:
                idx = bm25_id_map.get(cid)
                if idx is not None:
                    id_to_chunk[cid] = (
                        _bm25_data["docs"][idx],
                        _bm25_data["metas"][idx],
                        1.0,
                    )

        top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:fetch_n]
        docs  = [id_to_chunk[cid][0] for cid, _ in top_rrf if cid in id_to_chunk]
        metas = [id_to_chunk[cid][1] for cid, _ in top_rrf if cid in id_to_chunk]
        dists = [id_to_chunk[cid][2] for cid, _ in top_rrf if cid in id_to_chunk]

    ranked = select_diverse_top_k(docs, metas, dists, k)
    return order_for_attention(ranked)


# ── Sources payload for frontend ─────────────────────────────────────────────
def _build_sources(rows):
    out = []
    for score, sim, doc, meta, _dist in rows:
        out.append({
            "title":       meta.get("title", "untitled"),
            "source_type": meta.get("source_type", "unknown"),
            "trust":       meta.get("trust_tier", "?"),
            "sim":         round(float(sim), 3),
            "score":       round(float(score), 3),
            "arxiv_id":    meta.get("arxiv_id", ""),
            "published":   (meta.get("published", "") or "")[:10],
            "snippet":     trim_to_sentences(doc, max_chars=400),
        })
    return out


# ── Grok streaming (sync generator → SSE strings) ────────────────────────────
def _grok_stream(api_key: str, model: str, messages: list):
    """
    Sync generator.  Yields SSE-formatted strings for each token from Grok,
    then a final 'done' event.  FastAPI runs this in a thread pool automatically
    via StreamingResponse.
    """
    url = f"{XAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "stream": True, "messages": messages}

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Grok {resp.status_code}: {resp.text[:300]}'})}\n\n"
                return
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                payload_str = raw_line[len("data: "):]
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ── Session memory ────────────────────────────────────────────────────────────
def _memory_block(session: dict) -> str:
    turns = session.get("turns", [])[-BUFFER_TURNS:]
    if not turns:
        return ""
    lines = ["## Recent conversation"]
    for t in turns:
        lines.append(f"**Q:** {t['query']}")
        preview = t["answer"][:500] + " [...]" if len(t["answer"]) > 500 else t["answer"]
        lines.append(f"**A:** {preview}")
    return "\n".join(lines)


# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    if not _api_key:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not configured in .env")
    if _collection is None:
        raise HTTPException(status_code=503, detail="Server still loading — try again in a moment")

    # Session
    sid = req.session_id or str(uuid.uuid4())
    session = _sessions.setdefault(sid, {"turns": []})

    # Retrieve + assemble context
    rows = _retrieve(req.query, req.k, req.no_hybrid)
    sources = _build_sources(rows)
    context_block, included, dropped = build_context_block(rows, req.token_budget)

    # Memory
    mem = _memory_block(session)

    # Prompt
    parts = []
    if mem:
        parts.append(mem)
    parts.append(f"## Grounding context (retrieved from local Quant RAG index)\n\n{context_block}")
    parts.append(f"## Question\n{req.query}\n\nAnswer using the rules in your system prompt.")
    user_prompt = "\n\n".join(parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    answer_parts: list = []

    def generate():
        # 1. Send metadata (sources, session id) before first token
        yield f"data: {json.dumps({'type': 'meta', 'session_id': sid, 'sources': sources, 'included': included, 'dropped': dropped})}\n\n"

        # 2. Stream tokens from Grok, collect for session memory
        for sse_line in _grok_stream(_api_key, req.model, messages):
            if '"type": "token"' in sse_line:
                try:
                    answer_parts.append(json.loads(sse_line[6:]).get("content", ""))
                except Exception:
                    pass
            yield sse_line

        # 3. Save turn to session memory
        session["turns"].append({"query": req.query, "answer": "".join(answer_parts)})

        # 4. Done
        yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":   "ok",
        "chunks":   _collection.count() if _collection else 0,
        "hybrid":   _bm25_data is not None,
        "sessions": len(_sessions),
    }


# ── Clear session ─────────────────────────────────────────────────────────────
@app.post("/api/sessions/clear")
def clear_session(session_id: str = ""):
    if session_id and session_id in _sessions:
        _sessions.pop(session_id)
    return {"cleared": True}


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
