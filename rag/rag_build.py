"""
RAG Index Builder — Quant RAG pipeline (local stack: sentence-transformers + ChromaDB)

Loads source documents (arXiv JSON records from arxiv_ingest.py, on-chain
reference markdown, your own strategy notes), chunks them, embeds them
locally, and stores them in a persistent ChromaDB collection with rich
metadata (source type, trust tier, title, date) so retrieval can filter
or weight by reliability.

Usage:
    python3 rag_build.py --arxiv-dir ./arxiv_papers/records \
                         --onchain-dir ./onchain_docs \
                         --notes-dir ./strategy_notes \
                         --db-path ./rag_db

Install deps:
    pip install chromadb sentence-transformers --break-system-packages
"""

import argparse
import glob
import json
import os
import pickle
import re

CHUNK_SIZE = 1000       # target characters per chunk (soft cap — see chunk_text)
CHUNK_OVERLAP = 150     # characters carried forward into the next chunk for context

# Trust tiers — used downstream to weight or filter retrieval results.
# Higher tier = more empirically grounded / rigorous.
TRUST_TIERS = {
    "arxiv_paper": 3,        # peer-reviewed / pre-print research
    "onchain_reference": 3,  # curated metric definitions, factual
    "strategy_notes": 2,     # your own findings — useful but unvalidated
    "blog_or_news": 1,       # narrative/sentiment — lowest trust
}


# Separator ladder for recursive splitting, most-meaningful first.
# We try to keep whole paragraphs/sentences together and only fall back
# to a harder split when a piece is still too large for one chunk.
_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on_separator(text: str, separators):
    """
    Recursively split `text` using the first separator in `separators`
    that actually breaks it into more than one piece, falling through to
    the next separator (and finally to a raw character cut) if needed.
    Returns a list of pieces, each as small as the chosen separator allows.
    """
    if not separators:
        # Last resort: no separator left — return as a single piece.
        # The caller's size-based packer will hard-cut it if necessary.
        return [text]

    sep, rest = separators[0], separators[1:]
    if sep == " ":
        parts = text.split(" ")
        # Re-attach the separator so re-joining preserves spacing.
        pieces = [p + " " for p in parts[:-1]] + [parts[-1]]
    else:
        parts = text.split(sep)
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]

    pieces = [p for p in pieces if p.strip()]
    if len(pieces) <= 1:
        # This separator didn't help — try the next one down the ladder.
        return _split_on_separator(text, rest)
    return pieces


def _hard_cut(piece: str, size: int):
    """Absolute last resort: fixed-size character slices of an oversized piece."""
    return [piece[i:i + size] for i in range(0, len(piece), size)]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Recursive / structure-aware chunker.

    Rather than blindly slicing every `size` characters, this tries to keep
    natural units of text (paragraphs, then lines, then sentences, then
    words) intact, and packs them together up to `size` characters. Only an
    unusually long unit gets force-cut, and even then at the lowest-priority
    boundary available. Adjacent chunks overlap by `overlap` characters so
    context isn't lost across a boundary.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    # Recursively break the text into small, semantically-coherent pieces.
    def expand(piece, seps):
        if len(piece) <= size:
            return [piece]
        sub_pieces = _split_on_separator(piece, seps)
        if len(sub_pieces) == 1:
            # Nothing left to split on — hard-cut as a last resort.
            return _hard_cut(piece, size)
        out = []
        for sp in sub_pieces:
            if sp == piece:
                out.extend(_hard_cut(sp, size))
            else:
                out.extend(expand(sp, seps))
        return out

    units = expand(text, _SPLIT_SEPARATORS)
    units = [u for u in units if u.strip()]
    if not units:
        return []

    # Pack consecutive small units together up to `size`, carrying a
    # character-based overlap from the tail of one chunk into the next
    # so retrieval doesn't lose context right at a boundary.
    chunks = []
    current = ""
    for unit in units:
        candidate = current + unit
        if len(candidate) <= size or not current:
            current = candidate
        else:
            chunk = current.strip()
            if chunk:
                chunks.append(chunk)
            # Carry the tail of the finished chunk forward as overlap.
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + unit

    final = current.strip()
    if final:
        chunks.append(final)

    return chunks


def load_arxiv_records(arxiv_dir: str):
    """Yield (text, metadata) for each arXiv JSON record."""
    for path in glob.glob(os.path.join(arxiv_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  ! Skipping {path}: {e}")
            continue

        body = rec.get("text") or rec.get("summary", "")
        if not body:
            continue

        meta = {
            "source_type": "arxiv_paper",
            "trust_tier": TRUST_TIERS["arxiv_paper"],
            "title": rec.get("title", ""),
            "arxiv_id": rec.get("arxiv_id", ""),
            "authors": ", ".join(rec.get("authors", [])[:5]),
            "categories": ", ".join(rec.get("categories", [])),
            "published": rec.get("published", ""),
        }
        yield body, meta


def load_markdown_dir(dir_path: str, source_type: str):
    """Yield (text, metadata) for each .md file in a directory."""
    if not os.path.isdir(dir_path):
        return
    for path in glob.glob(os.path.join(dir_path, "**", "*.md"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                body = f.read()
        except Exception as e:
            print(f"  ! Skipping {path}: {e}")
            continue
        if not body.strip():
            continue
        meta = {
            "source_type": source_type,
            "trust_tier": TRUST_TIERS.get(source_type, 1),
            "title": os.path.splitext(os.path.basename(path))[0],
            "file_path": path,
        }
        yield body, meta


def get_collection(db_path: str, collection_name: str = "quant_rag", embedder=None):
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_embedder(model_name: str = "default"):
    """
    Lightweight ONNX-based embedding function bundled with ChromaDB
    (all-MiniLM-L6-v2 served via onnxruntime — no PyTorch required).
    This keeps the stack small and avoids multi-GB torch downloads.
    Swap to sentence-transformers later if you want, by passing a
    SentenceTransformerEmbeddingFunction here instead.
    """
    from chromadb.utils import embedding_functions
    print("Loading bundled ONNX MiniLM embedding function (first run downloads ~80MB)...")
    return embedding_functions.DefaultEmbeddingFunction()


def build_bm25_index(docs: list, ids: list, metas: list, db_path: str):
    """
    Build a BM25 (Okapi BM25) keyword index over the same chunks stored in
    ChromaDB and pickle it to {db_path}/bm25_index.pkl.

    This is the sparse half of hybrid search.  At query time, rag_query.py
    loads this file, runs a BM25 keyword search alongside the dense vector
    search, and fuses both result sets with Reciprocal Rank Fusion (RRF)
    before re-ranking by trust tier.

    Why BM25 + dense?
      Dense embeddings excel at semantic/conceptual similarity ("how do
      funding rates predict squeeze bottoms?") but miss exact keyword hits
      (ticker symbols, metric names like "MVRV", author names).  BM25 covers
      those gaps.  RRF fusion gets you the best of both without any tuning.

    Install dep (once):
        pip install rank-bm25 --break-system-packages
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("\n[BM25] rank_bm25 not installed — skipping BM25 index build.")
        print("  Run: pip install rank-bm25 --break-system-packages")
        print("  Then re-run rag_build.py to get hybrid search.\n")
        return

    print(f"\nBuilding BM25 index over {len(docs)} chunks...")

    _token_re = re.compile(r'\b\w+\b')

    def tokenize(text: str):
        return _token_re.findall(text.lower())

    tokenized_corpus = [tokenize(doc) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25": bm25,
        "ids": ids,        # parallel to ChromaDB chunk IDs — used for RRF join
        "docs": docs,      # raw text (needed when BM25 surfaces chunks not in dense top-k)
        "metas": metas,    # metadata (trust_tier, source_type, title, ...)
    }

    out_path = os.path.join(db_path, "bm25_index.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"BM25 index saved -> {out_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build local RAG index for Quant RAG")
    parser.add_argument("--arxiv-dir", default="./arxiv_papers/records",
                        help="Directory of arXiv JSON records (from arxiv_ingest.py)")
    parser.add_argument("--onchain-dir", default="./onchain_docs",
                        help="Directory of on-chain reference markdown files")
    parser.add_argument("--notes-dir", default="./strategy_notes",
                        help="Directory of your own strategy notes (markdown)")
    parser.add_argument("--db-path", default="./rag_db",
                        help="Path for persistent ChromaDB store")
    parser.add_argument("--collection", default="quant_rag",
                        help="ChromaDB collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Embedding batch size")
    args = parser.parse_args()

    print("=== Quant RAG index builder ===")
    embedder = get_embedder(args.model)
    collection = get_collection(args.db_path, args.collection, embedder=embedder)

    sources = []
    sources += list(load_arxiv_records(args.arxiv_dir))
    sources += list(load_markdown_dir(args.onchain_dir, "onchain_reference"))
    sources += list(load_markdown_dir(args.notes_dir, "strategy_notes"))

    print(f"Loaded {len(sources)} source documents")

    docs, metadatas, ids = [], [], []
    for doc_idx, (text, meta) in enumerate(sources):
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            uid = f"{meta.get('source_type')}_{meta.get('arxiv_id') or meta.get('title','doc')}_{doc_idx}_{chunk_idx}"
            uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:200]
            docs.append(chunk)
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = chunk_idx
            metadatas.append(chunk_meta)
            ids.append(uid)

    print(f"Produced {len(docs)} chunks total")

    if not docs:
        print("Nothing to index. Add source files and re-run.")
        return

    # Embed and upsert in batches
    for i in range(0, len(docs), args.batch_size):
        batch_docs = docs[i:i + args.batch_size]
        batch_meta = metadatas[i:i + args.batch_size]
        batch_ids = ids[i:i + args.batch_size]

        # Collection has an embedding_function attached, so passing documents
        # is enough — Chroma embeds them automatically on upsert/query.
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        print(f"  Indexed {min(i + args.batch_size, len(docs))}/{len(docs)} chunks")

    print(f"\nDone. Collection '{args.collection}' now has {collection.count()} chunks.")
    print(f"DB path: {args.db_path}")

    # Build BM25 sparse index alongside ChromaDB for hybrid search.
    build_bm25_index(docs, ids, metadatas, args.db_path)


if __name__ == "__main__":
    main()
