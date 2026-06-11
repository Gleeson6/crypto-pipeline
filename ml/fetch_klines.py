"""
Fetch 1H OHLCV klines from Binance Vision
==========================================
Downloads monthly kline zip files from data.binance.vision and loads them
into the DuckDB feature store. Skips months already present in the DB.

Source:  https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/
Format:  open_time, open, high, low, close, volume, close_time,
         quote_volume, count, taker_buy_base_vol, taker_buy_quote_vol, ignore

Usage:
    python3 fetch_klines.py               # last 24 months
    python3 fetch_klines.py --months 36   # last 36 months
    python3 fetch_klines.py --start 2022-01 --end 2024-12
"""

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import duckdb
import requests

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH, setup

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
SYMBOL   = "BTCUSDT"
INTERVAL = "1h"


def month_range(start: datetime, end: datetime):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur += relativedelta(months=1)


def already_ingested(con, year: int, month: int) -> bool:
    """True if any kline rows exist for this month."""
    t_start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    if month == 12:
        t_end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    else:
        t_end = int(datetime(year, month + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    count = con.execute(
        "SELECT COUNT(*) FROM klines WHERE open_time >= ? AND open_time < ?",
        [t_start, t_end]
    ).fetchone()[0]
    return count > 0


def fetch_month(con, year: int, month: int, retries: int = 3):
    fname = f"BTCUSDT-1h-{year:04d}-{month:02d}.zip"
    url   = f"{BASE_URL}/{fname}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {fname} ...", end=" ", flush=True)
            resp = requests.get(url, timeout=120, stream=True)
            if resp.status_code == 404:
                print("not found (month may not exist yet) — skipping.")
                return 0
            resp.raise_for_status()

            raw = io.BytesIO(resp.content)
            rows_inserted = 0
            batch = []

            with zipfile.ZipFile(raw) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                    for row in reader:
                        if not row or row[0].strip().startswith("#"):
                            continue
                        try:
                            open_time          = int(row[0])
                            open_              = float(row[1])
                            high               = float(row[2])
                            low                = float(row[3])
                            close              = float(row[4])
                            volume             = float(row[5])
                            close_time         = int(row[6])
                            quote_volume       = float(row[7])
                            trade_count        = int(row[8])
                            taker_buy_base     = float(row[9])
                            taker_sell_base    = volume - taker_buy_base
                            taker_buy_ratio    = taker_buy_base / volume if volume > 0 else 0.5
                            batch.append((
                                open_time, open_, high, low, close, volume,
                                close_time, quote_volume, trade_count,
                                taker_buy_base, taker_sell_base, taker_buy_ratio
                            ))
                        except (ValueError, IndexError):
                            continue  # skip header or malformed rows

            if batch:
                con.executemany("""
                    INSERT OR IGNORE INTO klines VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, batch)
                rows_inserted = len(batch)

            print(f"{rows_inserted} candles inserted.")
            return rows_inserted

        except Exception as e:
            print(f"error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC 1H klines into DuckDB")
    parser.add_argument("--months",  type=int, default=24,    help="How many months back (default 24)")
    parser.add_argument("--start",   type=str, default=None,  help="Start month YYYY-MM (overrides --months)")
    parser.add_argument("--end",     type=str, default=None,  help="End month YYYY-MM (default: last complete month)")
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    args = parser.parse_args()

    setup(args.db_path)
    con = duckdb.connect(args.db_path)

    now = datetime.now(timezone.utc)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m").replace(tzinfo=timezone.utc)
    else:
        # Last fully completed month
        end_dt = (now.replace(day=1) - relativedelta(months=1))

    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m").replace(tzinfo=timezone.utc)
    else:
        start_dt = end_dt - relativedelta(months=args.months - 1)

    months = list(month_range(start_dt, end_dt))
    print(f"Fetching BTCUSDT 1H klines: {start_dt.strftime('%Y-%m')} → {end_dt.strftime('%Y-%m')} ({len(months)} months)")

    total = 0
    skipped = 0
    for dt in months:
        if already_ingested(con, dt.year, dt.month):
            print(f"  {dt.strftime('%Y-%m')} already in DB — skipping.")
            skipped += 1
            continue
        total += fetch_month(con, dt.year, dt.month)

    con.close()
    total_in_db = duckdb.connect(args.db_path).execute("SELECT COUNT(*) FROM klines").fetchone()[0]
    print(f"\n✅ Done. Inserted {total} new candles. Skipped {skipped} months already ingested.")
    print(f"   Total klines in DB: {total_in_db}")


if __name__ == "__main__":
    main()
