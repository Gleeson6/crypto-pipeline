"""
RAG Query / Retrieval Script — Quant RAG pipeline

Embeds a query, retrieves top-k chunks from the local ChromaDB collection
(built by rag_build.py), and prints them as formatted context — ready to
paste into your LLM (Grok) prompt, or to pipe programmatically.

Usage:
    python3 rag_query.py "why does bitcoin spike sharply after large exchange outflows?"
    python3 rag_query.py "funding rate squeeze patterns" --k 8 --min-trust 2

Install deps (if not already):
    pip install chromadb sentence-transformers --break-system-packages
"""

import argparse
import textwrap

from rag_build import get_collection, get_embedder  # reuse the same model/DB setup


def format_result(rank: int, doc: str, meta: dict, distance: float) -> str:
    title = meta.get("title", "untitled")
    source_type = meta.get("source_type", "unknown")
    trust = meta.get("trust_tier", "?")
    extra = ""
    if source_type == "arxiv_paper":
        extra = f" | arXiv:{meta.get('arxiv_id','')} | {meta.get('published','')[:10]}"
    snippet = textwrap.shorten(doc, width=500, placeholder=" […]")
    return (
        f"[{rank}] ({source_type}, trust={trust}, sim={1 - distance:.3f}) {title}{extra}\n"
        f"    {snippet}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Query the local Quant RAG index")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--db-path", default="./rag_db", help="Path to persistent ChromaDB store")
    parser.add_argument("--collection", default="quant_rag", help="ChromaDB collection name")
    parser.add_argument("--model", default="default", help="embedding function to use (default = bundled ONNX MiniLM)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--min-trust", type=int, default=0,
                        help="Filter out chunks below this trust tier (1=low ... 3=high)")
    parser.add_argument("--source-type", default=None,
                        help="Optionally filter to one source_type (arxiv_paper, onchain_reference, strategy_notes)")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw chunk text only (for piping into another tool/LLM)")
    args = parser.parse_args()

    embedder = get_embedder(args.model)
    collection = get_collection(args.db_path, args.collection, embedder=embedder)

    where = {}
    if args.min_trust:
        where["trust_tier"] = {"$gte": args.min_trust}
    if args.source_type:
        where["source_type"] = args.source_type

    # Collection has the embedding_function attached, so plain query_texts works.
    results = collection.query(
        query_texts=[args.query],
        n_results=args.k,
        where=where or None,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        print("No results. Has the index been built? Run rag_build.py first.")
        return

    if args.raw:
        for doc in docs:
            print(doc)
            print("---")
        return

    print(f"Query: {args.query}\n")
    print(f"Top {len(docs)} retrieved chunks (sorted by similarity):\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(format_result(i, doc, meta, dist))

    print("\n--- Suggested LLM context block ---")
    print("Use the snippets above as grounding context. Cite source_type + title when")
    print("reasoning, and weight higher trust_tier sources more heavily in your answer.")


if __name__ == "__main__":
    main()
