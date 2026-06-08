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
import re

CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# Trust tiers — used downstream to weight or filter retrieval results.
# Higher tier = more empirically grounded / rigorous.
TRUST_TIERS = {
    "arxiv_paper": 3,        # peer-reviewed / pre-print research
    "onchain_reference": 3,  # curated metric definitions, factual
    "strategy_notes": 2,     # your own findings — useful but unvalidated
    "blog_or_news": 1,       # narrative/sentiment — lowest trust
}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
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


if __name__ == "__main__":
    main()
