"""
Full Data Ingestion Orchestrator
=================================
Runs all ingestion scripts in the correct order:
  1. setup_db     — create DuckDB schema (idempotent)
  2. fetch_klines — 1H OHLCV + taker buy/sell
  3. fetch_funding_oi   — funding rates + open interest
  4. fetch_liquidations — liquidation snapshots
  5. fetch_footprint    — aggTrades → footprint features (SLOW: ~2-5 min/month)

Each script is resumable — already-ingested months are skipped.

Usage:
    python3 ingest_all.py                 # last 24 months, all data
    python3 ingest_all.py --months 12     # last 12 months
    python3 ingest_all.py --skip-footprint  # skip slow aggTrades step
    python3 ingest_all.py --start 2023-01 --end 2024-12
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script: str, extra_args: list = None):
    cmd = [sys.executable, os.path.join(HERE, script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    t0     = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status  = "✅ OK" if result.returncode == 0 else f"❌ FAILED (code {result.returncode})"
    print(f"\n{status} — {elapsed:.1f}s\n")
    return result.returncode == 0


def summary(db_path: str):
    import duckdb
    con = duckdb.connect(db_path)
    tables = ["klines", "funding_rates", "open_interest", "liquidations", "footprint"]
    print(f"\n{'='*60}")
    print("DuckDB Feature Store Summary")
    print(f"{'='*60}")
    for t in tables:
        try:
            count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<20} {count:>8,} rows")
        except Exception:
            print(f"  {t:<20}    (table missing)")

    # Date range from klines
    try:
        first, last = con.execute(
            "SELECT MIN(open_time), MAX(open_time) FROM klines"
        ).fetchone()
        if first and last:
            def fmt(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"\n  Klines range: {fmt(first)} → {fmt(last)}")
    except Exception:
        pass

    # Features view
    try:
        feat_count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        print(f"\n  features view: {feat_count:,} rows (joined on open_time)")
    except Exception:
        pass

    con.close()
    print(f"\n  DB path: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Run full Bitcoin data ingestion pipeline")
    parser.add_argument("--months",         type=int,  default=24)
    parser.add_argument("--start",          type=str,  default=None)
    parser.add_argument("--end",            type=str,  default=None)
    parser.add_argument("--skip-footprint", action="store_true",
                        help="Skip aggTrades footprint step (fastest run without it)")
    parser.add_argument("--only-footprint", action="store_true",
                        help="Only run footprint step")
    parser.add_argument("--db-path",        type=str,  default=None)
    args = parser.parse_args()

    # Build shared extra args
    extra = [f"--months={args.months}"]
    if args.start:   extra.append(f"--start={args.start}")
    if args.end:     extra.append(f"--end={args.end}")
    if args.db_path: extra.append(f"--db-path={args.db_path}")

    db_path = args.db_path or os.path.join(HERE, "feature_store.duckdb")

    print("Bitcoin ML Data Ingestion Pipeline")
    print(f"Target: {args.months} months of history")
    print(f"DB:     {db_path}")
    print()

    if args.only_footprint:
        run("fetch_footprint.py", extra)
    else:
        run("setup_db.py",         [f"--db-path={db_path}"] if args.db_path else [])
        run("fetch_klines.py",       extra)
        run("fetch_funding_oi.py",   extra)
        run("fetch_liquidations.py", extra)
        if not args.skip_footprint:
            print("\n⚠️  Starting aggTrades footprint download (~1-3 GB/month, ~2-5 min/month)")
            print("   You can Ctrl+C and re-run later — already-processed months are skipped.\n")
            run("fetch_footprint.py", extra)
        else:
            print("\n⏭  Skipping footprint (--skip-footprint flag set)")

    summary(db_path)


if __name__ == "__main__":
    main()
