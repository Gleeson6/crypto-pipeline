"""
Build BM25 index from existing ChromaDB collection — no re-embedding needed.

Run this once to enable hybrid search without waiting for a full rag_build.py run.
After this, rag_query.py and rag_generate.py will automatically use hybrid mode.

Usage:
    python3 build_bm25_only.py

Install dep (once):
    pip install rank-bm25 --break-system-packages
"""

import os
import pickle
import re
import sys

DB_PATH = "./rag_db"
COLLECTION = "quant_rag"
BATCH_SIZE = 1000   # fetch from ChromaDB in pages (avoids loading 36k chunks at once)


def main():
    try:
        import chromadb
    except ImportError:
        print("chromadb not installed. Run: pip install chromadb --break-system-packages")
        sys.exit(1)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed. Run: pip install rank-bm25 --break-system-packages")
        sys.exit(1)

    print(f"Connecting to ChromaDB at {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collection = client.get_collection(COLLECTION)
    except Exception as e:
        print(f"Collection '{COLLECTION}' not found: {e}")
        print("Run rag_build.py first to create the index.")
        sys.exit(1)

    total = collection.count()
    print(f"Collection '{COLLECTION}' has {total} chunks. Fetching all ...")

    all_ids, all_docs, all_metas = [], [], []
    offset = 0

    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids   = batch.get("ids", [])
        docs  = batch.get("documents", [])
        metas = batch.get("metadatas", [])

        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(metas)

        offset += len(ids)
        print(f"  Fetched {offset}/{total} chunks ...")

        if not ids:
            break   # safety: stop if collection.get returns empty before offset

    print(f"\nTokenising {len(all_docs)} chunks for BM25 ...")
    _token_re = re.compile(r'\b\w+\b')

    def tokenize(text: str):
        return _token_re.findall(text.lower())

    tokenized_corpus = [tokenize(doc) for doc in all_docs]

    print("Fitting BM25Okapi ...")
    bm25 = BM25Okapi(tokenized_corpus)

    payload = {
        "bm25":  bm25,
        "ids":   all_ids,
        "docs":  all_docs,
        "metas": all_metas,
    }

    out_path = os.path.join(DB_PATH, "bm25_index.pkl")
    print(f"Saving to {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"\nDone. BM25 index saved ({size_mb:.1f} MB).")
    print("rag_query.py and rag_generate.py will now use hybrid search automatically.")


if __name__ == "__main__":
    main()
