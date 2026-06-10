"""
RAG Generation — Grok-powered answer generation over the local Quant RAG index

This is the "generation side" of context engineering: it takes the
re-ranked, deduplicated chunks produced by rag_query.py's retrieval
pipeline, assembles them into a structured, labeled prompt (grouped by
source type, ordered for LLM attention, budgeted by token count), sends
that to Grok (xAI), and streams the answer back token-by-token.

Usage:
    python3 rag_generate.py "why does bitcoin spike after large exchange outflows?"
    python3 rag_generate.py "funding rate squeeze patterns" --k 8 --min-trust 2

Setup:
    1. Add to .env (project root):  XAI_API_KEY=your_key_here
    2. pip install requests python-dotenv --break-system-packages
       (chromadb / sentence-transformers already required by rag_query.py)
"""

import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv

# Reuse the retrieval, re-ranking, dedup, and trimming logic verbatim —
# the generation layer should never re-implement what retrieval already
# solved well. This also guarantees both the CLI (`rag_query.py`) and the
# generation path see identical context for the same query.
from rag_build import get_collection, get_embedder
from rag_query import (
    FETCH_MULTIPLIER,
    select_diverse_top_k,
    order_for_attention,
    trim_to_sentences,
    load_bm25_index,
    bm25_retrieve,
    rrf_merge,
)

# Load XAI_API_KEY from the project-root .env (same convention as executor.py).
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(_env_path)

XAI_API_BASE = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4-fast"

# Rough chars-per-token estimate for budgeting context size before sending.
# Not exact (no tokenizer dependency), but close enough to keep prompts
# from silently ballooning as the corpus and --k grow.
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_CONTEXT_TOKEN_BUDGET = 3000

# Friendly section labels, grouped in the order a quant would reason:
# theory/grounding first, your own validated notes next, narrative/sentiment last.
SECTION_LABELS = {
    "arxiv_paper": "Quant research (peer-reviewed / pre-print)",
    "onchain_reference": "On-chain & quant reference material",
    "strategy_notes": "Your own strategy notes",
    "blog_or_news": "Blog / news / sentiment (lower trust — treat as narrative, not fact)",
}
SECTION_ORDER = ["arxiv_paper", "onchain_reference", "strategy_notes", "blog_or_news"]

SYSTEM_PROMPT = """You are a quant research assistant embedded in a personal Bitcoin \
algorithmic trading project. Your job is to give accurate, actionable answers that \
help the user build and validate a profitable trading system.

## Answering rule — always give a substantive answer

**You must always give a real answer. Refusing or saying "the context doesn't cover \
this" is not acceptable on its own.**

Use this priority order:

1. **If the retrieved context directly addresses the question** — answer from it \
and cite inline as (source_type: "Title"). Prefer tier-3 sources (peer-reviewed / \
curated reference) over tier-1 (blog/news).

2. **If the retrieved context is sparse or off-topic** — answer from your own \
quantitative finance knowledge. Clearly flag this with a one-line prefix: \
*[From general quant knowledge — not in the retrieved sources for this query]*. \
Then give the full answer. This is the normal fallback — use it freely.

3. **Only after giving the answer**, note any specific empirical data points \
(e.g. exact current on-chain readings, a specific paper's backtested numbers) \
that would sharpen the answer further, and where to get them \
(e.g. "current MVRV reading: Glassnode or CryptoQuant"). \
This note is an addition to the answer, not a replacement for it.

## Context trust tiers
- trust=3: peer-reviewed papers or curated on-chain reference docs — treat as fact
- trust=2: user's own strategy notes — useful but unvalidated
- trust=1: blog/news/sentiment — color only, not a basis for claims

## Other rules
- If sources disagree, surface the disagreement — don't silently pick one.
- Never invent specific figures or paper citations not in the context or your knowledge.
- Be concrete. "It depends on many factors" without naming them is not useful.
- User context: $100 proof-of-concept capital, 1-4 hour swing timeframe, \
  hard 1-2% capital-at-risk-per-trade rule, paper-trading / validation phase. \
  Calibrate all advice to this scale."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def retrieve(query, db_path, collection_name, embedder, k, min_trust, source_type,
             no_hybrid=False):
    """
    Retrieve → re-rank → dedup → attention-order pipeline.

    By default runs hybrid search (BM25 + dense, fused with RRF) when a
    BM25 index exists alongside the ChromaDB store.  Pass no_hybrid=True
    to force dense-only retrieval (e.g. for ablation / debugging).
    """
    collection = get_collection(db_path, collection_name, embedder=embedder)

    where = {}
    if min_trust:
        where["trust_tier"] = {"$gte": min_trust}
    if source_type:
        where["source_type"] = source_type

    fetch_n = max(k * FETCH_MULTIPLIER, k)

    # ── Dense retrieval ───────────────────────────────────────────────────────
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

    # ── Hybrid fusion (BM25 + RRF) ────────────────────────────────────────────
    docs, metas, dists = dense_docs, dense_metas, dense_dists
    if not no_hybrid:
        bm25_data = load_bm25_index(db_path)
        if bm25_data is not None:
            dense_ranked  = list(zip(dense_ids, [1 - d for d in dense_dists]))
            sparse_ranked = bm25_retrieve(query, bm25_data, fetch_n)
            rrf_scores    = rrf_merge(dense_ranked, sparse_ranked)

            # Unified lookup from dense + BM25-only hits
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


def build_context_block(rows, token_budget):
    """
    Assemble retrieved rows into a structured, labeled context block —
    grouped by source type (so Grok can reason about *kinds* of evidence,
    not just a flat snippet pile), within each group ordered for attention,
    and trimmed to a token budget so prompts don't silently balloon.

    Returns (context_text, included_count, dropped_count).
    """
    grouped = {key: [] for key in SECTION_ORDER}
    for score, sim, doc, meta, _dist in rows:
        source_type = meta.get("source_type", "blog_or_news")
        grouped.setdefault(source_type, []).append((score, sim, doc, meta))

    sections = []
    used_chars = 0
    budget_chars = token_budget * CHARS_PER_TOKEN_ESTIMATE
    included, dropped = 0, 0

    for source_type in SECTION_ORDER:
        rows_for_type = grouped.get(source_type) or []
        if not rows_for_type:
            continue

        lines = [f"## {SECTION_LABELS.get(source_type, source_type)}"]
        for score, sim, doc, meta in rows_for_type:
            title = meta.get("title", "untitled")
            trust = meta.get("trust_tier", "?")
            extra = ""
            if source_type == "arxiv_paper":
                extra = f" (arXiv:{meta.get('arxiv_id','')}, {meta.get('published','')[:10]})"
            snippet = trim_to_sentences(doc, max_chars=600)
            entry = f'- [trust={trust}, sim={sim:.2f}] "{title}"{extra}\n  {snippet}'

            if used_chars + len(entry) > budget_chars and included > 0:
                dropped += 1
                continue
            lines.append(entry)
            used_chars += len(entry)
            included += 1

        if len(lines) > 1:
            sections.append("\n".join(lines))

    return "\n\n".join(sections), included, dropped


def stream_grok(api_key, model, system_prompt, user_prompt):
    """
    Stream a chat completion from Grok via xAI's OpenAI-compatible API,
    printing tokens as they arrive (SSE: lines prefixed `data: {...}`,
    terminated by `data: [DONE]`). Falls back to a clear error message
    rather than a stack trace if the request fails.
    """
    url = f"{XAI_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            print(f"\n[Grok API error {resp.status_code}]: {resp.text[:500]}", file=sys.stderr)
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
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content")
            if token:
                print(token, end="", flush=True)
    print()  # final newline after the stream ends


def main():
    parser = argparse.ArgumentParser(description="Ask the Quant RAG + Grok pipeline a question")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--db-path", default="./rag_db", help="Path to persistent ChromaDB store")
    parser.add_argument("--collection", default="quant_rag", help="ChromaDB collection name")
    parser.add_argument("--model-embed", default="default", help="Embedding function (matches rag_query.py)")
    parser.add_argument("--model-llm", default=DEFAULT_MODEL, help="Grok model name")
    parser.add_argument("--k", type=int, default=8, help="Number of context chunks to retrieve")
    parser.add_argument("--min-trust", type=int, default=0, help="Minimum trust tier (1-3)")
    parser.add_argument("--source-type", default=None, help="Filter to one source_type")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_CONTEXT_TOKEN_BUDGET,
                        help="Approx. max tokens of retrieved context to send")
    parser.add_argument("--show-context", action="store_true",
                        help="Print the assembled context block before the answer (for debugging)")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Disable BM25 hybrid search — use dense-only retrieval")
    args = parser.parse_args()

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("XAI_API_KEY not found. Add it to your .env file at the project root:\n"
              "    XAI_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    embedder = get_embedder(args.model_embed)
    rows = retrieve(args.query, args.db_path, args.collection, embedder,
                    args.k, args.min_trust, args.source_type,
                    no_hybrid=args.no_hybrid)

    if not rows:
        print("No context retrieved — has the index been built? Run rag_build.py first.")
        sys.exit(1)

    context_block, included, dropped = build_context_block(rows, args.token_budget)

    if args.show_context:
        print("--- Assembled context block ---")
        print(context_block)
        print(f"--- ({included} chunks included, {dropped} dropped to fit "
              f"~{args.token_budget}-token budget) ---\n")

    user_prompt = (
        f"## Grounding context (retrieved from local Quant RAG index)\n\n{context_block}\n\n"
        f"## Question\n{args.query}\n\n"
        f"Answer the question above using the rules in your system prompt."
    )

    present_types = {row[3].get("source_type") for row in rows}
    section_count = sum(1 for s in SECTION_ORDER if s in present_types)

    print(f"Query: {args.query}")
    print(f"(grounded on {included} chunks across {section_count} "
          f"source-type sections — streaming from {args.model_llm}...)\n")

    stream_grok(api_key, args.model_llm, SYSTEM_PROMPT, user_prompt)


if __name__ == "__main__":
    main()
