"""
Fetch Liquidation Snapshots from Binance Vision
================================================
Downloads monthly liquidationSnapshot zip files and aggregates them to 1H
buckets (liq_buy_vol, liq_sell_vol, liq_count) in DuckDB.

BUY  liquidations = shorts being force-closed  (bullish pressure)
SELL liquidations = longs being force-closed   (bearish pressure)

Source:
  data/futures/um/monthly/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-YYYY-MM.zip

CSV cols: symbol, side, orderType, timeInForce, origQty, price,
          avgPrice, orderStatus, lastFilledQty, totalFilledQty, time

Usage:
    python3 fetch_liquidations.py             # last 24 months
    python3 fetch_liquidations.py --months 36
"""

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import duckdb
import requests

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH, setup

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/liquidationSnapshot/BTCUSDT"


def month_range(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur += relativedelta(months=1)


def already_has(con, year: int, month: int) -> bool:
    t_start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end   = int((datetime(year, month, 1, tzinfo=timezone.utc) + relativedelta(months=1)).timestamp() * 1000)
    return con.execute(
        "SELECT COUNT(*) FROM liquidations WHERE open_time >= ? AND open_time < ?",
        [t_start, t_end]
    ).fetchone()[0] > 0


def floor_to_hour_ms(ts_ms: int) -> int:
    """Floor a Unix-ms timestamp to the start of its 1H candle."""
    return (ts_ms // 3_600_000) * 3_600_000


def fetch_liq_month(con, year: int, month: int, retries: int = 3):
    fname = f"BTCUSDT-liquidationSnapshot-{year:04d}-{month:02d}.zip"
    url   = f"{VISION_BASE}/{fname}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  {fname} ...", end=" ", flush=True)
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                print("not found — skipping.")
                return 0
            resp.raise_for_status()

            # Aggregate to 1H buckets
            buckets: dict = defaultdict(lambda: {"buy": 0.0, "sell": 0.0, "count": 0})

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    reader = csv.reader(io.TextIOWrapper(f))
                    for row in reader:
                        if not row or row[0].strip() == "symbol":
                            continue
                        try:
                            side         = row[1].upper()   # BUY or SELL
                            filled_qty   = float(row[9])    # totalFilledQty (BTC)
                            ts_ms        = int(row[10])      # time
                            bucket       = floor_to_hour_ms(ts_ms)
                            if side == "BUY":
                                buckets[bucket]["buy"] += filled_qty
                            else:
                                buckets[bucket]["sell"] += filled_qty
                            buckets[bucket]["count"] += 1
                        except (ValueError, IndexError):
                            continue

            batch = [
                (
                    open_time,
                    v["buy"],
                    v["sell"],
                    v["count"],
                    v["buy"] - v["sell"],
                )
                for open_time, v in sorted(buckets.items())
            ]

            if batch:
                con.executemany(
                    "INSERT OR IGNORE INTO liquidations VALUES (?,?,?,?,?)", batch
                )
            print(f"{len(batch)} hourly buckets inserted.")
            return len(batch)

        except Exception as e:
            print(f"error (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch BTC liquidation data into DuckDB")
    parser.add_argument("--months",  type=int, default=24)
    parser.add_argument("--start",   type=str, default=None)
    parser.add_argument("--end",     type=str, default=None)
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    args = parser.parse_args()

    setup(args.db_path)
    con = duckdb.connect(args.db_path)

    now      = datetime.now(timezone.utc)
    end_dt   = datetime.strptime(args.end, "%Y-%m").replace(tzinfo=timezone.utc) if args.end \
               else now.replace(day=1) - relativedelta(months=1)
    start_dt = datetime.strptime(args.start, "%Y-%m").replace(tzinfo=timezone.utc) if args.start \
               else end_dt - relativedelta(months=args.months - 1)

    months = list(month_range(start_dt, end_dt))
    print(f"Fetching liquidations: {start_dt.strftime('%Y-%m')} → {end_dt.strftime('%Y-%m')} ({len(months)} months)")

    total = 0
    for dt in months:
        if already_has(con, dt.year, dt.month):
            print(f"  {dt.strftime('%Y-%m')} already in DB — skipping.")
            continue
        total += fetch_liq_month(con, dt.year, dt.month)

    con.close()
    total_in_db = duckdb.connect(args.db_path).execute("SELECT COUNT(*) FROM liquidations").fetchone()[0]
    print(f"\n✅ Done. Inserted {total} new hourly buckets. Total in DB: {total_in_db}")


if __name__ == "__main__":
    main()
