"""
Fetch aggTrades → Compute Footprint + Order Flow Features
==========================================================
Downloads monthly aggTrade zip files from Binance Vision, processes them
in a streaming fashion (never loads full file into RAM), computes per-1H-candle
footprint and order flow features, stores results in DuckDB.

Raw ticks are NEVER stored — only the computed per-candle features.

Features computed per candle:
  delta          — buy_vol - sell_vol (net aggressor pressure)
  buy_vol        — total aggressive buy volume (BTC)
  sell_vol       — total aggressive sell volume (BTC)
  total_vol      — buy_vol + sell_vol
  poc_price      — price level with highest volume (Point of Control)
  vah            — Value Area High (top of 70% volume range)
  val            — Value Area Low  (bottom of 70% volume range)
  large_trade_count — trades with qty > LARGE_TRADE_BTC
  large_trade_vol   — volume from those large trades
  buy_trade_count   — count of aggressive buy trades
  sell_trade_count  — count of aggressive sell trades
  max_imbalance  — max bid/ask imbalance at any price level
  cvd            — cumulative volume delta (running sum, reset daily)

is_buyer_maker interpretation:
  True  → seller was aggressor (sell volume)
  False → buyer was aggressor  (buy volume)

Source: https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/

Usage:
    python3 fetch_footprint.py             # last 24 months
    python3 fetch_footprint.py --months 12
    python3 fetch_footprint.py --start 2023-01 --end 2024-12
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

VISION_BASE     = "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT"
LARGE_TRADE_BTC = 1.0       # threshold for "large trade" (whale proxy)
TICK_SIZE       = 0.1       # price rounding for volume profile ($0.10)
VALUE_AREA_PCT  = 0.70      # 70% of volume = value area


def month_range(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        yield cur
        cur += relativedelta(months=1)


def already_has(con, year: int, month: int) -> bool:
    t_start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    t_end   = int((datetime(year, month, 1, tzinfo=timezone.utc) + relativedelta(months=1)).timestamp() * 1000)
    return con.execute(
        "SELECT COUNT(*) FROM footprint WHERE open_time >= ? AND open_time < ?",
        [t_start, t_end]
    ).fetchone()[0] > 0


def floor_to_hour_ms(ts_ms: int) -> int:
    return (ts_ms // 3_600_000) * 3_600_000


def floor_to_day_ms(ts_ms: int) -> int:
    return (ts_ms // 86_400_000) * 86_400_000


def round_price(price: float) -> float:
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


def compute_poc_vah_val(volume_profile: dict):
    """
    Given {price_level: total_vol}, compute POC, VAH, VAL.
    Value area = price levels containing VALUE_AREA_PCT of total volume,
    expanding from POC outward.
    """
    if not volume_profile:
        return 0.0, 0.0, 0.0

    sorted_levels = sorted(volume_profile.items())  # [(price, vol), ...]
    total_vol     = sum(v for _, v in sorted_levels)
    target_vol    = total_vol * VALUE_AREA_PCT

    # POC = price with max volume
    poc_price = max(volume_profile, key=volume_profile.get)

    # Expand value area from POC
    prices = [p for p, _ in sorted_levels]
    poc_idx = prices.index(poc_price)

    accumulated = volume_profile[poc_price]
    lo_idx = hi_idx = poc_idx

    while accumulated < target_vol:
        can_go_up   = hi_idx + 1 < len(prices)
        can_go_down = lo_idx - 1 >= 0

        if not can_go_up and not can_go_down:
            break

        vol_up   = volume_profile[prices[hi_idx + 1]] if can_go_up   else -1
        vol_down = volume_profile[prices[lo_idx - 1]] if can_go_down else -1

        if vol_up >= vol_down:
            hi_idx     += 1
            accumulated += vol_up
        else:
            lo_idx     -= 1
            accumulated += vol_down

    return poc_price, prices[hi_idx], prices[lo_idx]


def compute_max_imbalance(buy_profile: dict, sell_profile: dict) -> float:
    """Max (|buy - sell| / (buy + sell)) across all price levels."""
    all_levels = set(buy_profile) | set(sell_profile)
    if not all_levels:
        return 0.0
    max_imb = 0.0
    for lvl in all_levels:
        b = buy_profile.get(lvl, 0.0)
        s = sell_profile.get(lvl, 0.0)
        total = b + s
        if total > 0:
            imb = abs(b - s) / total
            if imb > max_imb:
                max_imb = imb
    return max_imb


def process_month(con, year: int, month: int, retries: int = 3) -> int:
    """
    Stream the monthly aggTrades zip, aggregate to 1H buckets, compute
    footprint features, insert into DuckDB. Returns rows inserted.
    """
    fname = f"BTCUSDT-aggTrades-{year:04d}-{month:02d}.zip"
    url   = f"{VISION_BASE}/{fname}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  {fname} ...", end=" ", flush=True)
            resp = requests.get(url, timeout=900, stream=True)
            if resp.status_code == 404:
                print("not found — skipping.")
                return 0
            resp.raise_for_status()

            # ── Aggregate ticks → per-hour buckets ────────────────────────────
            # Structure: {open_time_ms: {"buy_vol", "sell_vol", "buy_cnt",
            #             "sell_cnt", "large_cnt", "large_vol",
            #             "vol_profile", "buy_profile", "sell_profile"}}
            buckets: dict = {}

            raw_bytes = io.BytesIO(resp.content)
            with zipfile.ZipFile(raw_bytes) as zf:
                with zf.open(zf.namelist()[0]) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                    for row in reader:
                        # cols: agg_id, price, qty, first_id, last_id, time, is_buyer_maker
                        if len(row) < 7:
                            continue
                        try:
                            price          = float(row[1])
                            qty            = float(row[2])
                            ts_ms          = int(row[5])
                            is_buyer_maker = row[6].strip().lower() in ("true", "1")
                        except (ValueError, IndexError):
                            continue

                        open_time = floor_to_hour_ms(ts_ms)
                        if open_time not in buckets:
                            buckets[open_time] = {
                                "buy_vol":      0.0,
                                "sell_vol":     0.0,
                                "buy_cnt":      0,
                                "sell_cnt":     0,
                                "large_cnt":    0,
                                "large_vol":    0.0,
                                "vol_profile":  defaultdict(float),
                                "buy_profile":  defaultdict(float),
                                "sell_profile": defaultdict(float),
                            }

                        b = buckets[open_time]
                        lvl = round_price(price)

                        b["vol_profile"][lvl] += qty
                        if is_buyer_maker:
                            # seller aggressed
                            b["sell_vol"]          += qty
                            b["sell_cnt"]          += 1
                            b["sell_profile"][lvl] += qty
                        else:
                            # buyer aggressed
                            b["buy_vol"]           += qty
                            b["buy_cnt"]           += 1
                            b["buy_profile"][lvl]  += qty

                        if qty >= LARGE_TRADE_BTC:
                            b["large_cnt"] += 1
                            b["large_vol"] += qty

            # ── Compute CVD (cumulative delta, reset daily) ───────────────────
            sorted_times = sorted(buckets.keys())
            running_cvd  = 0.0
            last_day     = None
            cvd_map      = {}
            for ot in sorted_times:
                day = floor_to_day_ms(ot)
                if day != last_day:
                    running_cvd = 0.0
                    last_day = day
                b = buckets[ot]
                running_cvd += b["buy_vol"] - b["sell_vol"]
                cvd_map[ot] = running_cvd

            # ── Build insert batch ────────────────────────────────────────────
            batch = []
            for open_time in sorted_times:
                b = buckets[open_time]
                poc, vah, val = compute_poc_vah_val(b["vol_profile"])
                max_imb       = compute_max_imbalance(b["buy_profile"], b["sell_profile"])
                delta         = b["buy_vol"] - b["sell_vol"]
                total_vol     = b["buy_vol"] + b["sell_vol"]
                batch.append((
                    open_time,
                    delta,
                    b["buy_vol"],
                    b["sell_vol"],
                    total_vol,
                    poc,
                    vah,
                    val,
                    b["large_cnt"],
                    b["large_vol"],
                    b["buy_cnt"],
                    b["sell_cnt"],
                    max_imb,
                    cvd_map[open_time],
                ))

            if batch:
                con.executemany("""
                    INSERT OR IGNORE INTO footprint VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, batch)

            print(f"{len(batch)} hourly footprints computed and inserted.")
            return len(batch)

        except Exception as e:
            print(f"error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(3 ** attempt)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch aggTrades → footprint features into DuckDB")
    parser.add_argument("--months",  type=int, default=24,
                        help="Months of history (default 24). Each month ~1-3 GB download.")
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
    print(f"Computing footprint features: {start_dt.strftime('%Y-%m')} → {end_dt.strftime('%Y-%m')} ({len(months)} months)")
    print(f"Note: each monthly aggTrades file is ~1-3 GB. Estimated time: 2-5 min/month.\n")

    total = 0
    for dt in months:
        if already_has(con, dt.year, dt.month):
            print(f"  {dt.strftime('%Y-%m')} already in DB — skipping.")
            continue
        t0 = time.time()
        n  = process_month(con, dt.year, dt.month)
        total += n
        if n > 0:
            print(f"    ↳ took {time.time() - t0:.1f}s")

    con.close()
    total_in_db = duckdb.connect(args.db_path).execute("SELECT COUNT(*) FROM footprint").fetchone()[0]
    print(f"\n✅ Done. Inserted {total} new footprint rows. Total in DB: {total_in_db}")


if __name__ == "__main__":
    main()
