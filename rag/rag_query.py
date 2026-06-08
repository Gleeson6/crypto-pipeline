"""
RAG Query / Retrieval Script — Quant RAG pipeline

Embeds a query, retrieves chunks from the local ChromaDB collection
(built by rag_build.py), re-ranks them by a blend of similarity + source
trust, deduplicates near-duplicates from the same document, and prints
them as a clean, LLM-ready context block (ready to paste into your Grok
prompt, or to pipe programmatically).

Usage:
    python3 rag_query.py "why does bitcoin spike sharply after large exchange outflows?"
    python3 rag_query.py "funding rate squeeze patterns" --k 8 --min-trust 2

Install deps (if not already):
    pip install chromadb sentence-transformers --break-system-packages
"""

import argparse
import re

from rag_build import get_collection, get_embedder  # reuse the same model/DB setup

# How much to over-fetch before re-ranking/deduping down to --k.
# Casting a wider net first means dedup/trust-weighting has real choices
# to work with instead of just reordering an already-narrow top-k.
FETCH_MULTIPLIER = 4

# Blend weight for trust tier vs raw similarity in the combined score.
# combined = (1 - TRUST_WEIGHT) * similarity + TRUST_WEIGHT * (trust_tier / 3)
# At 0.25, a trust=3 source needs only a modest similarity edge to beat a
# trust=1 source — enough to matter, not so much that weak matches win.
TRUST_WEIGHT = 0.25

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def trim_to_sentences(text: str, max_chars: int = 500) -> str:
    """
    Trim `text` to roughly `max_chars`, but cut on a sentence boundary
    rather than mid-word/mid-sentence. Falls back to the raw cut only if
    no sentence boundary exists within range — an LLM gets more value
    from one clean sentence than a longer fragment ending mid-thought.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= max_chars:
        return text

    sentences = _SENTENCE_END.split(text)
    out = ""
    for sentence in sentences:
        candidate = (out + " " + sentence).strip() if out else sentence
        if len(candidate) > max_chars:
            break
        out = candidate

    if out:
        return out + " […]"
    # No full sentence fit — hard cut as last resort.
    return text[:max_chars].rstrip() + " […]"


def doc_key(meta: dict) -> str:
    """
    Identity key for deduplication — chunks sharing this key come from
    the same source document. arXiv papers use their arXiv ID; everything
    else falls back to title (source markdown files, strategy notes).
    """
    return meta.get("arxiv_id") or meta.get("title", "untitled")


def combined_score(similarity: float, meta: dict) -> float:
    """
    Blend similarity with trust tier so that, e.g., a peer-reviewed paper
    at sim=0.70 can outrank a low-trust blog snippet at sim=0.74. Trust
    tiers run 1 (low) to 3 (high); normalize to a 0-1 scale to combine.
    """
    trust = meta.get("trust_tier", 1) or 1
    trust_norm = trust / 3.0
    return (1 - TRUST_WEIGHT) * similarity + TRUST_WEIGHT * trust_norm


def select_diverse_top_k(docs, metas, dists, k: int):
    """
    Re-rank by combined (similarity + trust) score, then walk the ranked
    list keeping at most one chunk per source document until we have k —
    this is what stops near-duplicate chunks from the same paper crowding
    out genuinely different perspectives (seen in testing: two chunks of
    the same gradient-boosting paper took two of five slots).

    If fewer than k distinct documents are available, backfill with the
    next-best remaining chunks (including repeats) so --k is still honored.
    """
    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1 - dist
        scored.append((combined_score(sim, meta), sim, doc, meta, dist))
    scored.sort(key=lambda row: row[0], reverse=True)

    selected, seen_keys, leftovers = [], set(), []
    for row in scored:
        key = doc_key(row[3])
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(row)
        else:
            leftovers.append(row)
        if len(selected) == k:
            return selected

    for row in leftovers:
        if len(selected) == k:
            break
        selected.append(row)
    return selected


def order_for_attention(rows):
    """
    LLMs attend most reliably to the start and end of a context window
    (the "lost in the middle" effect) and least to the center. Place the
    single best result first, the second-best last, and fill inward from
    there so the strongest evidence brackets the weaker middle entries.
    """
    if len(rows) <= 2:
        return rows
    ordered = [None] * len(rows)
    lo, hi = 0, len(rows) - 1
    for i, row in enumerate(rows):
        if i % 2 == 0:
            ordered[lo] = row
            lo += 1
        else:
            ordered[hi] = row
            hi -= 1
    return ordered


def format_result(rank: int, doc: str, meta: dict, sim: float, score: float) -> str:
    title = meta.get("title", "untitled")
    source_type = meta.get("source_type", "unknown")
    trust = meta.get("trust_tier", "?")
    extra = ""
    if source_type == "arxiv_paper":
        extra = f" | arXiv:{meta.get('arxiv_id','')} | {meta.get('published','')[:10]}"
    snippet = trim_to_sentences(doc, max_chars=500)
    return (
        f"[{rank}] ({source_type}, trust={trust}, sim={sim:.3f}, score={score:.3f}) {title}{extra}\n"
        f"    {snippet}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Query the local Quant RAG index")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--db-path", default="./rag_db", help="Path to persistent ChromaDB store")
    parser.add_argument("--collection", default="quant_rag", help="ChromaDB collection name")
    parser.add_argument("--model", default="default", help="embedding function to use (default = bundled ONNX MiniLM)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to keep after re-ranking")
    parser.add_argument("--min-trust", type=int, default=0,
                        help="Filter out chunks below this trust tier (1=low ... 3=high)")
    parser.add_argument("--source-type", default=None,
                        help="Optionally filter to one source_type (arxiv_paper, onchain_reference, strategy_notes)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable per-document deduplication (allow multiple chunks from the same source)")
    parser.add_argument("--no-reorder", action="store_true",
                        help="Disable start/end attention ordering (keep pure rank order)")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw chunk text only, in final context order (for piping into another tool/LLM)")
    args = parser.parse_args()

    embedder = get_embedder(args.model)
    collection = get_collection(args.db_path, args.collection, embedder=embedder)

    where = {}
    if args.min_trust:
        where["trust_tier"] = {"$gte": args.min_trust}
    if args.source_type:
        where["source_type"] = args.source_type

    # Over-fetch so re-ranking/dedup has real candidates to choose from,
    # not just a re-shuffle of an already-narrow top-k.
    fetch_n = max(args.k * FETCH_MULTIPLIER, args.k)
    results = collection.query(
        query_texts=[args.query],
        n_results=fetch_n,
        where=where or None,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        print("No results. Has the index been built? Run rag_build.py first.")
        return

    if args.no_dedup:
        scored = sorted(
            ((combined_score(1 - d, m), 1 - d, doc, m, d) for doc, m, d in zip(docs, metas, dists)),
            key=lambda row: row[0],
            reverse=True,
        )[: args.k]
    else:
        scored = select_diverse_top_k(docs, metas, dists, args.k)

    final = scored if args.no_reorder else order_for_attention(scored)

    if args.raw:
        for _, _, doc, _, _ in final:
            print(doc)
            print("---")
        return

    print(f"Query: {args.query}\n")
    print(f"Top {len(final)} chunks — re-ranked by similarity + trust tier, "
          f"deduped per source, ordered for LLM attention:\n")
    for i, (score, sim, doc, meta, _dist) in enumerate(final, start=1):
        print(format_result(i, doc, meta, sim, score))

    print("\n--- Suggested LLM context block ---")
    print("Use the snippets above as grounding context, in the order given (highest-")
    print("confidence evidence is placed first and last). Cite source_type + title when")
    print("reasoning, and weight higher trust_tier sources more heavily in your answer.")


if __name__ == "__main__":
    main()
