"""
RAG Chat — Conversational interface over the Quant RAG + Grok pipeline

Wraps rag_generate.py with multi-turn conversation memory so follow-up
questions ("what about combining that with funding rate?") build on prior
answers rather than starting cold each time.

Memory architecture (three layers):
  1. Buffer memory  — last BUFFER_TURNS full Q&A pairs injected verbatim
                      into every new prompt. Grok sees exactly what was
                      asked and answered recently.
  2. Summary memory — turns older than the buffer are compressed into a
                      rolling summary (via Grok itself) so the model keeps
                      long-range context without blowing the token budget.
  3. Entity memory  — key topics/signals mentioned across the session are
                      tracked and surfaced as a compact "session focus" note,
                      nudging retrieval toward consistent themes.

Session persistence: each session is saved as a JSON file in ./rag_sessions/
so you can resume a prior conversation with --session <id>.

Usage:
    # Start a new session (interactive REPL)
    python3 rag_chat.py

    # Resume a prior session
    python3 rag_chat.py --session 20240610_143022

    # Single-shot (non-interactive) with memory context
    python3 rag_chat.py --query "does SOPR confirm the outflow signal?"

    # Show full session history
    python3 rag_chat.py --session 20240610_143022 --history
"""

import argparse
import datetime
import json
import os
import re
import sys
import uuid

from dotenv import load_dotenv

from rag_build import get_collection, get_embedder
from rag_generate import (
    build_context_block,
    stream_grok,
    SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
)
from rag_query import (
    FETCH_MULTIPLIER,
    select_diverse_top_k,
    order_for_attention,
    load_bm25_index,
    bm25_retrieve,
    rrf_merge,
)

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

# --- Memory constants ---
BUFFER_TURNS = 4          # number of recent Q&A pairs injected verbatim
SUMMARY_MAX_TOKENS = 400  # rough token cap for the rolling summary block
SESSION_DIR = os.path.join(os.path.dirname(__file__), "rag_sessions")

# Signals/entities we track across turns to nudge retrieval focus.
# Extend this list as your strategy vocabulary grows.
ENTITY_PATTERNS = [
    r"\bexchange\s+(?:net)?flow\b",
    r"\bSOPR\b",
    r"\bMVRV\b",
    r"\bNUPL\b",
    r"\bfunding\s+rate\b",
    r"\bwhal[e]?\b",
    r"\bopen\s+interest\b",
    r"\bRSI\b",
    r"\bBollinger\b",
    r"\bMACD\b",
    r"\bminer\s+outflow\b",
    r"\brealized\s+price\b",
    r"\bexchange\s+reserve\b",
    r"\bliquidation\b",
    r"\bfear\s+and\s+greed\b",
    r"\bvolatility\b",
    r"\bmomentum\b",
    r"\bsupport\b",
    r"\bresistance\b",
]
_ENTITY_RE = re.compile("|".join(ENTITY_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def session_path(session_id: str) -> str:
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, f"{session_id}.json")


def load_session(session_id: str) -> dict:
    path = session_path(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "session_id": session_id,
        "created": datetime.datetime.now().isoformat(),
        "turns": [],
        "rolling_summary": "",
        "entities": [],
    }


def save_session(session: dict) -> None:
    with open(session_path(session["session_id"]), "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def new_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[str]:
    """Pull out known trading/on-chain signal mentions from text."""
    found = set()
    for m in _ENTITY_RE.finditer(text):
        found.add(m.group(0).strip().lower())
    return sorted(found)


def update_entities(session: dict, query: str, answer: str) -> None:
    new = extract_entities(query + " " + answer)
    existing = set(session.get("entities", []))
    existing.update(new)
    session["entities"] = sorted(existing)


# ---------------------------------------------------------------------------
# Memory prompt assembly
# ---------------------------------------------------------------------------

def build_memory_block(session: dict) -> str:
    """
    Assemble the memory context block injected before grounding context:
      - Rolling summary of turns older than the buffer
      - Verbatim buffer of the last BUFFER_TURNS Q&A pairs
      - Entity/focus note listing signals mentioned this session
    """
    parts = []

    if session.get("rolling_summary"):
        parts.append(
            "## Prior conversation (summary)\n"
            f"{session['rolling_summary']}"
        )

    turns = session.get("turns", [])
    buffer_turns = turns[-BUFFER_TURNS:] if len(turns) >= BUFFER_TURNS else turns
    if buffer_turns:
        lines = ["## Recent conversation (verbatim)"]
        for t in buffer_turns:
            lines.append(f"**Q:** {t['query']}")
            # Truncate long answers to keep the buffer compact.
            answer_preview = t["answer"]
            if len(answer_preview) > 600:
                answer_preview = answer_preview[:600].rstrip() + " […]"
            lines.append(f"**A:** {answer_preview}")
        parts.append("\n".join(lines))

    entities = session.get("entities", [])
    if entities:
        parts.append(
            "## Session focus (signals / topics mentioned so far)\n"
            + ", ".join(entities)
        )

    if not parts:
        return ""
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Rolling summary (compress old turns via Grok)
# ---------------------------------------------------------------------------

def maybe_summarize(session: dict, api_key: str, model: str) -> None:
    """
    When the number of turns exceeds the buffer, compress the oldest turns
    (those that will fall out of the verbatim buffer) into the rolling summary.
    Uses a tiny Grok call — non-streaming, just a short summary request.
    """
    turns = session.get("turns", [])
    overflow = len(turns) - BUFFER_TURNS
    if overflow <= 0:
        return  # nothing to summarize yet

    to_summarize = turns[:overflow]
    prior_summary = session.get("rolling_summary", "")

    history_text = "\n".join(
        f"Q: {t['query']}\nA: {t['answer'][:400]}" for t in to_summarize
    )

    summarize_prompt = (
        "Below is a partial conversation between a quant developer and a RAG assistant "
        "about Bitcoin trading signals and strategy.\n\n"
        f"{'Prior summary:\n' + prior_summary + chr(10) + chr(10) if prior_summary else ''}"
        f"New turns to incorporate:\n{history_text}\n\n"
        f"Write a compact summary (under {SUMMARY_MAX_TOKENS * 4} characters) that captures "
        "the key topics, signals discussed, conclusions reached, and any open questions. "
        "This summary will be injected into future prompts as memory context."
    )

    import requests
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": False,
        "max_tokens": SUMMARY_MAX_TOKENS,
        "messages": [
            {"role": "user", "content": summarize_prompt},
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            summary = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if summary:
                session["rolling_summary"] = summary
                # Drop the turns we just summarized — buffer turns stay.
                session["turns"] = turns[overflow:]
    except Exception as e:
        # Non-fatal — just skip summarization this turn.
        print(f"  [memory] Summary generation skipped: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core retrieval (reused from rag_generate, kept local for flexibility)
# ---------------------------------------------------------------------------

def retrieve(query, db_path, collection_name, embedder, k, min_trust, source_type,
             no_hybrid=False):
    collection = get_collection(db_path, collection_name, embedder=embedder)
    where = {}
    if min_trust:
        where["trust_tier"] = {"$gte": min_trust}
    if source_type:
        where["source_type"] = source_type

    fetch_n = max(k * FETCH_MULTIPLIER, k)

    # Dense retrieval
    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    dense_docs  = results.get("documents", [[]])[0]
    dense_metas = results.get("metadatas",  [[]])[0]
    dense_dists = results.get("distances",  [[]])[0]
    dense_ids   = results.get("ids",        [[]])[0]
    if not dense_docs:
        return []

    # Hybrid fusion (BM25 + RRF)
    docs, metas, dists = dense_docs, dense_metas, dense_dists
    if not no_hybrid:
        bm25_data = load_bm25_index(db_path)
        if bm25_data is not None:
            dense_ranked  = list(zip(dense_ids, [1 - d for d in dense_dists]))
            sparse_ranked = bm25_retrieve(query, bm25_data, fetch_n)
            rrf_scores    = rrf_merge(dense_ranked, sparse_ranked)

            id_to_chunk: dict = {}
            for cid, doc, meta, dist in zip(dense_ids, dense_docs, dense_metas, dense_dists):
                id_to_chunk[cid] = (doc, meta, dist)
            bm25_id_map = {cid: i for i, cid in enumerate(bm25_data["ids"])}
            for cid, _ in sparse_ranked:
                if cid not in id_to_chunk:
                    idx = bm25_id_map.get(cid)
                    if idx is not None:
                        id_to_chunk[cid] = (
                            bm25_data["docs"][idx],
                            bm25_data["metas"][idx],
                            1.0,
                        )

            top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:fetch_n]
            docs  = [id_to_chunk[cid][0] for cid, _ in top_rrf if cid in id_to_chunk]
            metas = [id_to_chunk[cid][1] for cid, _ in top_rrf if cid in id_to_chunk]
            dists = [id_to_chunk[cid][2] for cid, _ in top_rrf if cid in id_to_chunk]

    ranked = select_diverse_top_k(docs, metas, dists, k)
    return order_for_attention(ranked)


# ---------------------------------------------------------------------------
# Single turn
# ---------------------------------------------------------------------------

def run_turn(
    query: str,
    session: dict,
    api_key: str,
    db_path: str,
    collection_name: str,
    embedder,
    model: str,
    k: int,
    min_trust: int,
    source_type,
    token_budget: int,
    show_context: bool,
) -> str:
    """Run one Q&A turn, inject memory, stream the answer, return answer text."""

    # 1. Retrieve grounding context for this query.
    rows = retrieve(query, db_path, collection_name, embedder, k, min_trust, source_type)
    if not rows:
        msg = "No context retrieved — has the index been built? Run rag_build.py first."
        print(msg)
        return msg

    context_block, included, dropped = build_context_block(rows, token_budget)

    if show_context:
        print("--- Assembled context block ---")
        print(context_block)
        print(f"--- ({included} chunks included, {dropped} dropped) ---\n")

    # 2. Build memory block from session history.
    memory_block = build_memory_block(session)

    # 3. Assemble user prompt: memory → grounding → question.
    prompt_parts = []
    if memory_block:
        prompt_parts.append(memory_block)
    prompt_parts.append(
        f"## Grounding context (retrieved from local Quant RAG index)\n\n{context_block}"
    )
    prompt_parts.append(
        f"## Question\n{query}\n\n"
        "Answer using the rules in your system prompt, and take into account "
        "the conversation memory above if relevant."
    )
    user_prompt = "\n\n".join(prompt_parts)

    # 4. Stream answer.
    present_types = {row[3].get("source_type") for row in rows}
    section_count = sum(
        1 for s in ["arxiv_paper", "onchain_reference", "strategy_notes", "blog_or_news"]
        if s in present_types
    )
    memory_note = f" | memory: {len(session['turns'])} prior turns" if session["turns"] else ""
    print(f"\nQ: {query}")
    print(f"({included} chunks, {section_count} source types{memory_note} — streaming from {model}...)\n")

    # Capture the streamed answer text for session storage.
    answer_tokens = []
    _original_write = sys.stdout.write

    def capturing_write(s):
        answer_tokens.append(s)
        _original_write(s)

    sys.stdout.write = capturing_write
    try:
        stream_grok(api_key, model, SYSTEM_PROMPT, user_prompt)
    finally:
        sys.stdout.write = _original_write

    answer = "".join(answer_tokens).strip()
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn conversational RAG chat with memory"
    )
    parser.add_argument("--query", default=None,
                        help="Single query (non-interactive mode)")
    parser.add_argument("--session", default=None,
                        help="Session ID to resume (default: new session)")
    parser.add_argument("--history", action="store_true",
                        help="Print session history and exit")
    parser.add_argument("--db-path", default="./rag_db")
    parser.add_argument("--collection", default="quant_rag")
    parser.add_argument("--model-embed", default="default")
    parser.add_argument("--model-llm", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--min-trust", type=int, default=0)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("XAI_API_KEY not found. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    session_id = args.session or new_session_id()
    session = load_session(session_id)

    if args.history:
        print(f"Session: {session_id}  ({len(session['turns'])} turns)")
        if session.get("rolling_summary"):
            print(f"\nSummary:\n{session['rolling_summary']}\n")
        for i, t in enumerate(session["turns"], 1):
            print(f"[{i}] Q: {t['query']}")
            print(f"    A: {t['answer'][:300]}{'…' if len(t['answer']) > 300 else ''}\n")
        return

    embedder = get_embedder(args.model_embed)

    def do_turn(query: str):
        # Compress old turns into rolling summary before each turn.
        maybe_summarize(session, api_key, args.model_llm)

        answer = run_turn(
            query=query,
            session=session,
            api_key=api_key,
            db_path=args.db_path,
            collection_name=args.collection,
            embedder=embedder,
            model=args.model_llm,
            k=args.k,
            min_trust=args.min_trust,
            source_type=args.source_type,
            token_budget=args.token_budget,
            show_context=args.show_context,
        )

        # Store the turn and update entity memory.
        session["turns"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "answer": answer,
        })
        update_entities(session, query, answer)
        save_session(session)
        return answer

    if args.query:
        # Non-interactive single-shot with memory context.
        do_turn(args.query)
        print(f"\n[Session saved: {session_id}]")
        return

    # Interactive REPL.
    print(f"Quant RAG Chat  |  session: {session_id}")
    print(f"Model: {args.model_llm}  |  k={args.k}  |  budget={args.token_budget} tokens")
    print("Type your question, or 'exit' / Ctrl-C to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[Session saved: {session_id}]")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print(f"[Session saved: {session_id}]")
            break

        do_turn(query)
        print()


if __name__ == "__main__":
    main()
