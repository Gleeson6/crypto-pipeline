"""
Fetch Funding Rates + Open Interest
=====================================
- Funding rates:  Binance Vision monthly zips (full history)
  CSV format: calc_time, funding_interval_hours, last_funding_rate

- Open Interest:  Binance Futures REST API (paginated, full history)
  Endpoint: GET https://fapi.binance.com/futures/data/openInterestHist
  Returns 1H OI snapshots; paginate backwards from now

Both are free, no API key required.

Usage:
    python3 fetch_funding_oi.py             # last 24 months
    python3 fetch_funding_oi.py --months 36
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

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT"
FAPI_BASE   = "https://fapi.binance.com"


def month_range(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur += relativedelta(months=1)


def already_has(con, table, ts_col, year, month):
    t_start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end   = int((datetime(year, month, 1, tzinfo=timezone.utc) + relativedelta(months=1)).timestamp() * 1000)
    return con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {ts_col} >= ? AND {ts_col} < ?",
        [t_start, t_end]
    ).fetchone()[0] > 0


# ── Funding Rates ─────────────────────────────────────────────────────────────

def fetch_funding_month(con, year, month, retries=3):
    """
    CSV format: calc_time, funding_interval_hours, last_funding_rate
    (no mark_price column in current Binance Vision files)
    """
    fname = f"BTCUSDT-fundingRate-{year:04d}-{month:02d}.zip"
    url   = f"{VISION_BASE}/{fname}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  [funding] {fname} ...", end=" ", flush=True)
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                print("not found — skipping.")
                return 0
            resp.raise_for_status()

            batch = []
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    reader = csv.reader(io.TextIOWrapper(f))
                    for row in reader:
                        if not row:
                            continue
                        # Skip header row (first col is 'calc_time')
                        try:
                            int(row[0])
                        except ValueError:
                            continue
                        try:
                            # col 0: calc_time (Unix ms)
                            # col 1: funding_interval_hours
                            # col 2: last_funding_rate
                            # col 3: mark_price (optional — not always present)
                            funding_time = int(row[0])
                            funding_rate = float(row[2])
                            mark_price   = float(row[3]) if len(row) >= 4 and row[3] else None
                            batch.append((funding_time, funding_rate, mark_price))
                        except (ValueError, IndexError):
                            continue

            if batch:
                con.executemany(
                    "INSERT OR IGNORE INTO funding_rates VALUES (?,?,?)", batch
                )
            print(f"{len(batch)} rows inserted.")
            return len(batch)

        except Exception as e:
            print(f"error (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return 0


# ── Open Interest (Binance Futures REST API) ──────────────────────────────────
# NOTE: Binance limits openInterestHist to ~30 days of history per request.
# Strategy: paginate backwards from NOW in 30-day windows until we reach
# start_ms or the API returns no data (hard limit reached).

def fetch_oi_rest(con, start_ms, end_ms, retries=3):
    """
    Paginate Binance Futures OI history walking backwards using endTime only.
    Binance does not support arbitrary startTime ranges — use endTime + limit
    and walk backwards until history limit or start_ms is reached.
    """
    url   = f"{FAPI_BASE}/futures/data/openInterestHist"
    total = 0

    # Collect all batches first (newest→oldest), then insert oldest→newest
    # so oi_delta is computed in correct chronological order
    all_rows = []
    cursor_end = end_ms

    print(f"  [OI] Fetching backwards from "
          f"{datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')}...",
          flush=True)

    while True:
        params = {
            "symbol":  "BTCUSDT",
            "period":  "1h",
            "limit":   500,
            "endTime": cursor_end,
        }

        data = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == retries:
                    print(f"    REST error: {e}")
                else:
                    time.sleep(2 ** attempt)

        if not data:
            break

        batch_rows = [(int(r["timestamp"]),
                       float(r["sumOpenInterest"]),
                       float(r["sumOpenInterestValue"])) for r in data]

        # Filter to rows within our target range
        batch_rows = [r for r in batch_rows if r[0] >= start_ms]
        all_rows = batch_rows + all_rows  # prepend (older data goes first)

        oldest_ts = int(data[0]["timestamp"])
        print(f"    fetched {len(data)} rows back to "
              f"{datetime.fromtimestamp(oldest_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')}",
              flush=True)

        if oldest_ts <= start_ms or len(data) < 500:
            break  # reached start or no more data

        cursor_end = oldest_ts - 1  # step back 1ms
        time.sleep(0.2)

    if not all_rows:
        print("  [OI] No data returned.")
        return 0

    # Insert chronologically, computing oi_delta
    prev_oi = None
    insert_batch = []
    for open_time, sum_oi, sum_oi_v in all_rows:
        oi_delta = (sum_oi - prev_oi) if prev_oi is not None else None
        prev_oi  = sum_oi
        insert_batch.append((open_time, sum_oi, sum_oi_v, oi_delta))

    con.executemany(
        "INSERT OR IGNORE INTO open_interest VALUES (?,?,?,?)", insert_batch
    )
    total = len(insert_batch)
    print(f"  [OI] {total} rows inserted "
          f"({datetime.fromtimestamp(all_rows[0][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
          f"→ {datetime.fromtimestamp(all_rows[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')})")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",  type=int, default=24)
    parser.add_argument("--start",   type=str, default=None)
    parser.add_argument("--end",     type=str, default=None)
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    args = parser.parse_args()

    setup(args.db_path)
    con = duckdb.connect(args.db_path)

    # Clear bad funding_rates row (funding_time=8 from previous buggy run)
    bad = con.execute("SELECT COUNT(*) FROM funding_rates WHERE funding_time < 1000000000000").fetchone()[0]
    if bad > 0:
        con.execute("DELETE FROM funding_rates WHERE funding_time < 1000000000000")
        print(f"  Cleaned {bad} bad funding_rate rows from previous run.")

    now      = datetime.now(timezone.utc)
    end_dt   = datetime.strptime(args.end, "%Y-%m").replace(tzinfo=timezone.utc) if args.end \
               else now.replace(day=1) - relativedelta(months=1)
    start_dt = datetime.strptime(args.start, "%Y-%m").replace(tzinfo=timezone.utc) if args.start \
               else end_dt - relativedelta(months=args.months - 1)

    months = list(month_range(start_dt, end_dt))
    print(f"Fetching funding rates + OI: {start_dt.strftime('%Y-%m')} → {end_dt.strftime('%Y-%m')} ({len(months)} months)\n")

    # ── Funding rates ─────────────────────────────────────────────────────────
    total_fr = 0
    for dt in months:
        if already_has(con, "funding_rates", "funding_time", dt.year, dt.month):
            print(f"  [funding] {dt.strftime('%Y-%m')} already in DB — skipping.")
            continue
        total_fr += fetch_funding_month(con, dt.year, dt.month)

    # ── Open Interest via REST ────────────────────────────────────────────────
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int((end_dt + relativedelta(months=1)).timestamp() * 1000)

    # Binance OI history is limited to ~3 months for 1H period.
    # Fetch what's available; older rows will be NaN in the feature table.
    oi_start_ms = end_ms - (90 * 24 * 3_600_000)  # 90 days back from end

    existing_oi = con.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
    if existing_oi > 0:
        last_oi_ts = con.execute("SELECT MAX(open_time) FROM open_interest").fetchone()[0]
        print(f"  [OI] {existing_oi} rows already in DB, resuming from last timestamp...")
        oi_start_ms = last_oi_ts + 3_600_000

    print(f"  [OI] Note: Binance limits 1H OI history to ~3 months. Fetching from "
          f"{datetime.fromtimestamp(oi_start_ms/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → now.")
    total_oi = fetch_oi_rest(con, oi_start_ms, end_ms)

    con.close()

    # Summary
    fr_count = duckdb.connect(args.db_path).execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
    oi_count = duckdb.connect(args.db_path).execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
    print(f"\n✅ Done.")
    print(f"   Funding rates: {fr_count:,} total rows in DB")
    print(f"   Open Interest: {oi_count:,} total rows in DB")


if __name__ == "__main__":
    main()
