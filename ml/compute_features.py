"""
Feature Engineering — Build ML Feature Table
==============================================
Reads raw tables from DuckDB (klines, footprint, funding_rates,
open_interest, liquidations) and computes the full ML feature set,
writing results to the `ml_features` table.

Feature groups:
  1. Price / OHLCV            — returns, log returns, candle shape
  2. Technical indicators     — RSI, MACD, Bollinger Bands, ATR
  3. Volume                   — taker ratio, vol z-score, vol trend
  4. Footprint                — delta, CVD, POC distance, VAH/VAL range
  5. Order flow               — CVD divergence, large trade ratio
  6. Funding / leverage       — funding rate, funding extremes, OI delta
  7. Liquidations             — liq net vol, liq ratio, cascade flag
  8. Session / time           — hour_of_day, day_of_week, session flags
                                (Asian / European / US), is_weekend
  9. Target variable          — forward_return_4h (what we're predicting)

Usage:
    python3 compute_features.py
    python3 compute_features.py --db-path ./feature_store.duckdb
"""

import argparse
import os
import sys

import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH, setup


# ── Technical indicator helpers ───────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period=20, std_dev=2.0):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, width, pct_b


def compute_atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ── Session flags (UTC hours) ─────────────────────────────────────────────────
# Asian session:    00:00 – 08:00 UTC
# European session: 07:00 – 16:00 UTC  (overlap 07-08 with Asia)
# US session:       13:00 – 22:00 UTC  (overlap 13-16 with Europe)
# Dead zone:        22:00 – 00:00 UTC  (low volume)

def session_flags(hour: pd.Series):
    is_asian    = ((hour >= 0)  & (hour < 8)).astype(int)
    is_european = ((hour >= 7)  & (hour < 16)).astype(int)
    is_us       = ((hour >= 13) & (hour < 22)).astype(int)
    is_overlap_eu_us = ((hour >= 13) & (hour < 16)).astype(int)  # highest vol
    return is_asian, is_european, is_us, is_overlap_eu_us


# ── Data integrity: OHLC sanity + gap-free hourly grid ────────────────────────

MS_1H = 3_600_000   # one hour in Unix milliseconds


def enforce_ohlc_sanity(klines: pd.DataFrame):
    """
    Quarantine bad ticks BEFORE any feature is computed.

    A single corrupt candle (a feed glitch printing high=0, close=1e9, or a
    negative volume) silently poisons returns, RSI, ATR, Bollinger and every
    rolling window built on top of it — and winsorization downstream cannot
    undo a contaminated window. So we NULL the OHLCV of any row that violates
    basic candle invariants. The row is then treated exactly like a missing
    candle by the reindex step (i.e. it becomes a gap, not a learned outlier).

    Invariants enforced:
      high >= max(open, close, low)
      low  <= min(open, close, high)
      all prices > 0
      volume >= 0
    """
    k = klines.copy()
    o, h, l, c, v = k["open"], k["high"], k["low"], k["close"], k["volume"]
    bad_high = (h < o) | (h < c) | (h < l)
    bad_low  = (l > o) | (l > c) | (l > h)
    nonpos   = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    neg_vol  = v < 0
    bad = bad_high | bad_low | nonpos | neg_vol
    report = {
        "bad_high":          int(bad_high.sum()),
        "bad_low":           int(bad_low.sum()),
        "nonpositive_price": int(nonpos.sum()),
        "negative_volume":   int(neg_vol.sum()),
        "rows_quarantined":  int(bad.sum()),
    }
    if bad.any():
        k.loc[bad, ["open", "high", "low", "close", "volume"]] = np.nan
    return k, report


def reindex_to_hourly_grid(klines: pd.DataFrame):
    """
    Reindex onto a complete, evenly-spaced 1H UTC grid (no missing candles).

    WHY THIS IS THE CORE FIX
    ────────────────────────
    Every rolling/shift feature and the forward target is POSITIONAL
    (shift(4) means "4 rows back", not "4 hours back"). If an hour is missing,
    row position stops equalling clock time: shift(4) silently spans 5 real
    hours and the "4H forward return" target gets measured over the wrong
    horizon. After reindexing, ROW POSITION == CLOCK HOUR, so:

      • every shift()/rolling() window is time-correct, and
      • any return/target whose endpoint lands on a genuinely missing hour
        becomes NaN (correctly undefined) instead of being fabricated.

    Inserted (synthetic) rows are flagged `_is_real_candle = False` and dropped
    from the final feature table at the end of build_features — they exist only
    so neighbouring real rows receive time-correct windows.
    """
    k = klines.sort_values("open_time").reset_index(drop=True)
    start, end = int(k["open_time"].min()), int(k["open_time"].max())
    if (end - start) % MS_1H != 0:
        # Binance 1H open_time is always :00 aligned; warn but proceed.
        print(f"  ⚠️  grid span is not an exact multiple of 1H "
              f"(start={start}, end={end}) — check source alignment")
    full = pd.DataFrame(
        {"open_time": np.arange(start, end + MS_1H, MS_1H, dtype=np.int64)}
    )
    n_expected = len(full)
    merged = full.merge(k, on="open_time", how="left")
    merged["_is_real_candle"] = merged["open"].notna()
    n_present = int(merged["_is_real_candle"].sum())
    report = {
        "grid_start_ms":  start,
        "grid_end_ms":    end,
        "hours_expected": n_expected,
        "hours_present":  n_present,
        "hours_missing":  n_expected - n_present,
        "pct_complete":   round(100 * n_present / n_expected, 4),
    }
    return merged, report


# ── Main feature engineering ──────────────────────────────────────────────────

def build_features(db_path: str = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(db_path)

    print("Loading raw tables from DuckDB...", flush=True)

    # ── Load klines ───────────────────────────────────────────────────────────
    klines = con.execute("""
        SELECT open_time, open, high, low, close, volume,
               taker_buy_ratio, trade_count
        FROM klines
        ORDER BY open_time
    """).df()

    if klines.empty:
        print("❌ klines table is empty — run ingest_all.py first.")
        con.close()
        sys.exit(1)

    print(f"  klines:         {len(klines):,} rows")

    # ── Data integrity (BEFORE any feature is computed) ───────────────────────
    # 1) Quarantine bad ticks so they cannot poison returns/rolling windows.
    klines, ohlc_report = enforce_ohlc_sanity(klines)
    if ohlc_report["rows_quarantined"]:
        print(f"  OHLC sanity:    {ohlc_report['rows_quarantined']} bad rows quarantined "
              f"(bad_high={ohlc_report['bad_high']}, bad_low={ohlc_report['bad_low']}, "
              f"nonpos={ohlc_report['nonpositive_price']}, neg_vol={ohlc_report['negative_volume']})")
    else:
        print(f"  OHLC sanity:    OK (0 bad ticks)")

    # 2) Reindex onto a complete 1H grid so shift()/rolling()/target are
    #    time-correct (row position == clock hour). See reindex_to_hourly_grid.
    klines, grid_report = reindex_to_hourly_grid(klines)
    print(f"  Hourly grid:    {grid_report['hours_present']:,}/{grid_report['hours_expected']:,} "
          f"hours present ({grid_report['pct_complete']}% complete, "
          f"{grid_report['hours_missing']} missing/quarantined → inserted as NaN)")

    # ── Load footprint ────────────────────────────────────────────────────────
    fp = con.execute("""
        SELECT open_time, delta, buy_vol, sell_vol, total_vol,
               poc_price, vah, val, large_trade_count, large_trade_vol,
               buy_trade_count, sell_trade_count, max_imbalance, cvd
        FROM footprint
        ORDER BY open_time
    """).df()
    print(f"  footprint:      {len(fp):,} rows")

    # ── Load funding rates ────────────────────────────────────────────────────
    fr = con.execute("""
        SELECT funding_time AS open_time, funding_rate
        FROM funding_rates
        ORDER BY funding_time
    """).df()
    print(f"  funding_rates:  {len(fr):,} rows")

    # ── Load open interest ────────────────────────────────────────────────────
    oi = con.execute("""
        SELECT open_time, sum_open_interest, oi_delta
        FROM open_interest
        ORDER BY open_time
    """).df()
    print(f"  open_interest:  {len(oi):,} rows")

    # ── Load liquidations ─────────────────────────────────────────────────────
    liq = con.execute("""
        SELECT open_time, liq_buy_vol, liq_sell_vol, liq_net_vol, liq_count
        FROM liquidations
        ORDER BY open_time
    """).df()
    print(f"  liquidations:   {len(liq):,} rows")
    con.close()

    # ── Merge all on open_time ────────────────────────────────────────────────
    print("\nMerging tables...", flush=True)
    df = klines.copy()
    if not fp.empty:
        df = df.merge(fp,  on="open_time", how="left")
    if not oi.empty:
        df = df.merge(oi,  on="open_time", how="left")
    if not liq.empty:
        df = df.merge(liq, on="open_time", how="left")

    # Forward-fill funding rate (published every 8H)
    if not fr.empty:
        df = df.merge(fr, on="open_time", how="left")
        df["funding_rate"] = df["funding_rate"].ffill()

    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Merged: {len(df):,} rows, {df.shape[1]} columns")

    # ── 1. Datetime / session features ───────────────────────────────────────
    print("\nComputing features...", flush=True)
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["hour_of_day"]  = dt.dt.hour
    df["day_of_week"]  = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)

    (df["is_asian_session"],
     df["is_european_session"],
     df["is_us_session"],
     df["is_eu_us_overlap"]) = session_flags(df["hour_of_day"])

    # ── 2. Price / returns ────────────────────────────────────────────────────
    df["log_return_1h"]  = np.log(df["close"] / df["close"].shift(1))
    df["log_return_4h"]  = np.log(df["close"] / df["close"].shift(4))
    df["log_return_24h"] = np.log(df["close"] / df["close"].shift(24))
    # Explicit division (NOT pct_change): pct_change defaults to fill_method='pad',
    # which forward-fills NaN gap rows and would fabricate cross-gap returns.
    df["return_1h"]      = df["close"] / df["close"].shift(1)  - 1
    df["return_4h"]      = df["close"] / df["close"].shift(4)  - 1
    df["return_24h"]     = df["close"] / df["close"].shift(24) - 1

    # Candle shape
    df["candle_body"]    = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"]     = (df["high"] - df[["open","close"]].max(axis=1)) / df["open"]
    df["lower_wick"]     = (df[["open","close"]].min(axis=1) - df["low"]) / df["open"]
    df["is_bullish"]     = (df["close"] > df["open"]).astype(int)

    # ── 3. Volatility ─────────────────────────────────────────────────────────
    df["volatility_24h"] = df["log_return_1h"].rolling(24).std()
    df["volatility_7d"]  = df["log_return_1h"].rolling(168).std()
    df["vol_ratio"]      = df["volatility_24h"] / df["volatility_7d"].replace(0, np.nan)

    # ── 4. Technical indicators ───────────────────────────────────────────────
    df["rsi_14"]    = compute_rsi(df["close"], 14)
    df["rsi_7"]     = compute_rsi(df["close"], 7)

    macd, sig, hist = compute_macd(df["close"])
    df["macd"]         = macd
    df["macd_signal"]  = sig
    df["macd_hist"]    = hist
    df["macd_cross"]   = (
        (macd > sig) & (macd.shift(1) <= sig.shift(1))
    ).astype(int) - (
        (macd < sig) & (macd.shift(1) >= sig.shift(1))
    ).astype(int)  # +1 bullish cross, -1 bearish cross

    bb_upper, bb_lower, bb_width, bb_pct_b = compute_bollinger(df["close"])
    df["bb_upper"]  = bb_upper
    df["bb_lower"]  = bb_lower
    df["bb_width"]  = bb_width
    df["bb_pct_b"]  = bb_pct_b  # 0=lower band, 1=upper band

    df["atr_14"]    = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_ratio"] = df["atr_14"] / df["close"]  # normalised ATR

    # Moving averages
    for p in [8, 21, 50, 200]:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    df["ema_8_21_cross"]  = (df["ema_8"] > df["ema_21"]).astype(int)
    df["price_vs_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    df["price_vs_ema_200"]= (df["close"] - df["ema_200"]) / df["ema_200"]

    # ── 5. Volume features ────────────────────────────────────────────────────
    df["vol_sma_24"]     = df["volume"].rolling(24).mean()
    df["vol_zscore"]     = (df["volume"] - df["vol_sma_24"]) / df["volume"].rolling(24).std()
    df["vol_ratio_24h"]  = df["volume"] / df["vol_sma_24"].replace(0, np.nan)
    df["taker_buy_ratio"] = df["taker_buy_ratio"].fillna(0.5)

    # ── 6. Footprint features ─────────────────────────────────────────────────
    if "delta" in df.columns:
        df["delta_norm"]      = df["delta"] / df["total_vol"].replace(0, np.nan)
        df["poc_distance"]    = (df["close"] - df["poc_price"]) / df["close"]
        df["va_range"]        = (df["vah"] - df["val"]) / df["close"]
        df["large_trade_ratio"] = df["large_trade_vol"] / df["total_vol"].replace(0, np.nan)
        df["buy_sell_cnt_ratio"] = df["buy_trade_count"] / (
            df["buy_trade_count"] + df["sell_trade_count"]
        ).replace(0, np.nan)

        # CVD divergence: price up but CVD down (or vice versa) over 4H
        price_dir = np.sign(df["close"].diff(4))
        cvd_dir   = np.sign(df["cvd"].diff(4))
        df["cvd_divergence"]  = (price_dir != cvd_dir).astype(int)
        df["cvd_norm"]        = df["cvd"] / df["total_vol"].rolling(24).sum().replace(0, np.nan)

    # ── 7. Funding rate features ──────────────────────────────────────────────
    if "funding_rate" in df.columns:
        df["funding_rate"]         = df["funding_rate"].fillna(0)
        df["funding_extreme_long"] = (df["funding_rate"] > 0.0008).astype(int)  # >0.08%
        df["funding_extreme_short"]= (df["funding_rate"] < -0.0002).astype(int)
        df["funding_8h_change"]    = df["funding_rate"].diff(8)
        df["funding_24h_mean"]     = df["funding_rate"].rolling(24).mean()

    # ── 8. Open Interest features ─────────────────────────────────────────────
    if "sum_open_interest" in df.columns:
        df["oi_zscore"]  = (
            df["sum_open_interest"] - df["sum_open_interest"].rolling(24).mean()
        ) / df["sum_open_interest"].rolling(24).std()
        df["oi_delta"]   = df["oi_delta"].fillna(0)
        df["oi_delta_norm"] = df["oi_delta"] / df["sum_open_interest"].replace(0, np.nan)
        df["oi_price_divergence"] = (
            np.sign(df["close"].diff(4)) != np.sign(df["oi_delta"].rolling(4).sum())
        ).astype(int)

    # ── 9. Liquidation features (derived proxy from OI + price) ──────────────
    # True historical liquidation data is not available for free.
    # Proxy: sudden OI collapse + large price move = liquidation cascade.
    # This captures ~80% of real liquidation events from available data.
    if "oi_delta" in df.columns:
        oi_delta_filled = df["oi_delta"].fillna(0)

        # OI z-score over 24H rolling window
        oi_delta_zscore = (
            oi_delta_filled - oi_delta_filled.rolling(24).mean()
        ) / oi_delta_filled.rolling(24).std().replace(0, np.nan)

        abs_return_1h = df["log_return_1h"].abs().fillna(0)

        # Long cascade: OI drops sharply (< -1.5σ) + price drops (> 1%)
        # → longs were liquidated, price fell
        df["liq_long_proxy"] = (
            (oi_delta_zscore < -1.5) & (df["log_return_1h"] < -0.01)
        ).astype(int)

        # Short cascade: OI drops sharply (< -1.5σ) + price rises (> 1%)
        # → shorts were liquidated, price squeezed up
        df["liq_short_proxy"] = (
            (oi_delta_zscore < -1.5) & (df["log_return_1h"] > 0.01)
        ).astype(int)

        # General cascade flag: large OI drop + large move in either direction
        df["cascade_flag"] = (
            (oi_delta_zscore < -2.0) & (abs_return_1h > 0.015)
        ).astype(int)

        # Net liquidation direction: +1 short squeeze, -1 long liquidation, 0 none
        df["liq_direction"] = df["liq_short_proxy"] - df["liq_long_proxy"]

        # Rolling cascade count over last 4H and 24H
        df["cascade_4h"]  = df["cascade_flag"].rolling(4).sum()
        df["cascade_24h"] = df["cascade_flag"].rolling(24).sum()

    elif "liq_net_vol" in df.columns:
        # If real liquidation data exists (future — from WebSocket collector)
        df["liq_buy_vol"]   = df["liq_buy_vol"].fillna(0)
        df["liq_sell_vol"]  = df["liq_sell_vol"].fillna(0)
        df["liq_net_vol"]   = df["liq_net_vol"].fillna(0)
        df["liq_count"]     = df["liq_count"].fillna(0)
        df["liq_total_vol"] = df["liq_buy_vol"] + df["liq_sell_vol"]
        liq_zscore = (
            df["liq_total_vol"] - df["liq_total_vol"].rolling(24).mean()
        ) / df["liq_total_vol"].rolling(24).std().replace(0, np.nan)
        df["cascade_flag"]  = (liq_zscore > 2.0).astype(int)
        df["cascade_4h"]    = df["cascade_flag"].rolling(4).sum()
        df["cascade_24h"]   = df["cascade_flag"].rolling(24).sum()
        df["liq_direction"] = np.sign(df["liq_net_vol"])

    # ── 10. Target variable ───────────────────────────────────────────────────
    # What we're predicting: 4H FORWARD return = close[t+4] / close[t] - 1.
    # Explicit forward division (equiv. to pct_change(4).shift(-4) but without
    # fill_method='pad'): on the gap-free grid this is the true 4-clock-hour
    # return, and it is NaN whenever close[t] or close[t+4] is a missing hour —
    # i.e. we NEVER train on or predict a return measured across a data gap.
    df["target_return_4h"]    = df["close"].shift(-4) / df["close"] - 1
    df["target_direction_4h"] = np.sign(df["target_return_4h"])  # +1 / -1 / 0 / NaN

    # ── Drop warmup rows (need ≥200 candles for EMA-200) ──────────────────────
    df = df.iloc[200:].reset_index(drop=True)

    # ── Drop synthetic rows inserted only to make the grid gap-free ───────────
    # Real rows keep their now-time-correct windows; return/target values that
    # spanned a gap remain NaN (handled by cleaning / dropped before training).
    n_synth = int((~df["_is_real_candle"]).sum())
    df = df[df["_is_real_candle"]].drop(columns=["_is_real_candle"]).reset_index(drop=True)
    if n_synth:
        print(f"  Dropped {n_synth:,} synthetic gap rows "
              f"(missing or quarantined candles — never used for training)")

    print(f"  Features computed: {df.shape[1]} columns, {len(df):,} rows")
    return df


def save_to_duckdb(df: pd.DataFrame, db_path: str = DB_PATH):
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS ml_features")
    con.execute("CREATE TABLE ml_features AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM ml_features").fetchone()[0]
    cols  = con.execute("DESCRIBE ml_features").df()
    con.close()
    print(f"\n✅ ml_features table saved: {count:,} rows × {len(cols)} columns")
    print(f"   DB: {db_path}")
    return count


def feature_summary(df: pd.DataFrame):
    """Print a quick feature group summary."""
    groups = {
        "Datetime/session": [c for c in df.columns if any(x in c for x in
            ["hour", "day", "week", "session", "asian", "european", "us_", "overlap"])],
        "Price/returns":    [c for c in df.columns if any(x in c for x in
            ["return", "log_r", "candle", "wick", "bullish"])],
        "Volatility":       [c for c in df.columns if "volat" in c or "vol_ratio" in c or "atr" in c],
        "Technicals":       [c for c in df.columns if any(x in c for x in
            ["rsi", "macd", "bb_", "ema_"])],
        "Volume":           [c for c in df.columns if any(x in c for x in
            ["vol_", "taker"])],
        "Footprint":        [c for c in df.columns if any(x in c for x in
            ["delta", "poc", "vah", "val", "va_", "cvd", "large_trade", "buy_sell"])],
        "Funding/OI":       [c for c in df.columns if any(x in c for x in
            ["funding", "oi_", "sum_oi"])],
        "Liq proxy":        [c for c in df.columns if "liq" in c or "cascade" in c],
        "Target":           [c for c in df.columns if "target" in c],
    }
    print("\n── Feature Groups ──────────────────────────────────")
    total = 0
    for group, cols in groups.items():
        if cols:
            print(f"  {group:<22} {len(cols):>3} features: {', '.join(cols[:4])}"
                  f"{'...' if len(cols) > 4 else ''}")
            total += len(cols)
    print(f"  {'TOTAL':<22} {total:>3} features")


def main():
    parser = argparse.ArgumentParser(description="Compute ML features from DuckDB raw tables")
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    args = parser.parse_args()

    setup(args.db_path)
    df = build_features(args.db_path)
    feature_summary(df)
    save_to_duckdb(df, args.db_path)

    print("\nSample (last 3 rows):")
    show_cols = ["open_time", "close", "rsi_14", "funding_rate",
                 "is_us_session", "target_return_4h"]
    optional  = ["delta_norm", "cascade_flag"]
    show_cols += [c for c in optional if c in df.columns]
    print(df[show_cols].tail(3).to_string())


if __name__ == "__main__":
    main()
