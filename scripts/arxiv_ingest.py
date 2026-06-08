"""
arXiv Research Paper Ingestion Script — for Quant RAG pipeline

Queries the arXiv API for papers matching keyword sets relevant to crypto
price prediction / quant finance, downloads PDFs, extracts text, and writes
structured JSON records suitable for downstream RAG ingestion (chunking +
embedding).

Usage:
    python3 arxiv_ingest.py --max-results 50 --out ./arxiv_papers

Notes:
- Uses arXiv's public API (export.arxiv.org) — no API key required.
- Be polite to the API: this script sleeps between requests per arXiv's
  usage guidelines (no more than 1 request / 3 seconds).
- PDFs are saved to <out>/pdfs/, extracted text + metadata to <out>/records/
  as one JSON file per paper, and a combined manifest at <out>/manifest.jsonl
"""

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Curated keyword sets — tune these to your project's focus areas.
DEFAULT_QUERIES = [
    "cryptocurrency price prediction",
    "bitcoin volatility forecasting",
    "market microstructure cryptocurrency",
    "deep learning financial time series",
    "on-chain analysis blockchain",
    "algorithmic trading reinforcement learning",
]

# Restrict to relevant arXiv categories (quant finance + ML).
CATEGORIES = ["q-fin.ST", "q-fin.TR", "q-fin.CP", "cs.LG", "cs.AI", "stat.ML"]


def build_query(keyword: str) -> str:
    cat_filter = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    kw = keyword.replace(" ", "+")
    return f"(all:{kw}) AND ({cat_filter})"


def fetch_arxiv(query: str, max_results: int = 25, start: int = 0) -> bytes:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "quant-rag-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_entries(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_full = entry.findtext(f"{ATOM_NS}id", default="").strip()
        arxiv_id = arxiv_id_full.rstrip("/").split("/")[-1]
        title = " ".join(entry.findtext(f"{ATOM_NS}title", default="").split())
        summary = " ".join(entry.findtext(f"{ATOM_NS}summary", default="").split())
        published = entry.findtext(f"{ATOM_NS}published", default="").strip()
        authors = [
            a.findtext(f"{ATOM_NS}name", default="").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall(f"{ATOM_NS}category")
        ]
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "published": published,
            "summary": summary,
            "pdf_url": pdf_url,
        })
    return entries


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:120]


def download_pdf(url: str, dest_path: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quant-rag-ingest/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"  ! PDF download failed: {e}")
        return False


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using pypdf (falls back gracefully if missing)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # fallback
        except ImportError:
            print("  ! No PDF text library available (pip install pypdf). Skipping extraction.")
            return ""
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"  ! Text extraction failed: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers for Quant RAG")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES,
                        help="Keyword phrases to search for")
    parser.add_argument("--max-results", type=int, default=20,
                        help="Max results per query")
    parser.add_argument("--out", default="./arxiv_papers", help="Output directory")
    parser.add_argument("--no-download", action="store_true",
                        help="Fetch metadata only; skip PDF download/extraction")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="Seconds to sleep between API requests (be polite to arXiv)")
    args = parser.parse_args()

    pdf_dir = os.path.join(args.out, "pdfs")
    rec_dir = os.path.join(args.out, "records")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    seen_ids = set()
    total = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for q in args.queries:
            query_str = build_query(q)
            print(f"\n=== Query: '{q}' ===")
            try:
                xml_bytes = fetch_arxiv(query_str, max_results=args.max_results)
            except Exception as e:
                print(f"  ! Fetch failed: {e}")
                continue

            entries = parse_entries(xml_bytes)
            print(f"  Found {len(entries)} candidate papers")

            for entry in entries:
                aid = entry["arxiv_id"]
                if aid in seen_ids:
                    continue
                seen_ids.add(aid)

                print(f"  -> {aid}: {entry['title'][:90]}")
                record = dict(entry)
                record["source_query"] = q
                record["pdf_path"] = None
                record["text"] = ""

                if not args.no_download:
                    fname = f"{safe_filename(aid)}.pdf"
                    pdf_path = os.path.join(pdf_dir, fname)
                    if download_pdf(entry["pdf_url"], pdf_path):
                        record["pdf_path"] = pdf_path
                        record["text"] = extract_text(pdf_path)
                    time.sleep(args.sleep)

                rec_path = os.path.join(rec_dir, f"{safe_filename(aid)}.json")
                with open(rec_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)

                manifest.write(json.dumps({
                    "arxiv_id": aid,
                    "title": entry["title"],
                    "categories": entry["categories"],
                    "published": entry["published"],
                    "pdf_path": record["pdf_path"],
                    "record_path": rec_path,
                    "has_text": bool(record["text"]),
                }, ensure_ascii=False) + "\n")
                manifest.flush()
                total += 1

            time.sleep(args.sleep)

    print(f"\nDone. Ingested {total} unique papers.")
    print(f"Manifest: {manifest_path}")
    print(f"Records:  {rec_dir}")
    print(f"PDFs:     {pdf_dir}")


if __name__ == "__main__":
    main()
