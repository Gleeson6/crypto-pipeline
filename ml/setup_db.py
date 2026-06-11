"""
DuckDB Feature Store — Schema Setup
=====================================
Creates all tables for the Bitcoin ML feature pipeline.

Tables:
  klines            — 1H OHLCV + taker buy/sell (from Binance Vision)
  funding_rates     — 8H funding rate history (Binance futures)
  open_interest     — 1H OI history (Binance Vision metrics)
  liquidations      — 1H aggregated liquidation vol (Binance Vision)
  footprint         — 1H footprint features (derived from aggTrades)
  features          — Final merged feature table (view over all above)

Usage:
    python3 setup_db.py
"""

import os
import duckdb

DB_PATH = os.path.join(os.path.dirname(__file__), "feature_store.duckdb")


def setup(db_path: str = DB_PATH):
    con = duckdb.connect(db_path)

    # ── klines ────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            open_time           BIGINT PRIMARY KEY,  -- Unix ms (candle open)
            open                DOUBLE NOT NULL,
            high                DOUBLE NOT NULL,
            low                 DOUBLE NOT NULL,
            close               DOUBLE NOT NULL,
            volume              DOUBLE NOT NULL,     -- base asset vol (BTC)
            close_time          BIGINT NOT NULL,
            quote_volume        DOUBLE NOT NULL,     -- USDT vol
            trade_count         INTEGER NOT NULL,
            taker_buy_base_vol  DOUBLE NOT NULL,     -- aggressive buy vol (BTC)
            taker_sell_base_vol DOUBLE NOT NULL,     -- aggressive sell vol (BTC)
            taker_buy_ratio     DOUBLE NOT NULL      -- taker_buy / volume
        )
    """)

    # ── funding_rates ─────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            funding_time  BIGINT PRIMARY KEY,  -- Unix ms
            funding_rate  DOUBLE NOT NULL,     -- e.g. 0.0001 = 0.01%
            mark_price    DOUBLE              -- mark price at funding time
        )
    """)

    # ── open_interest ─────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            open_time          BIGINT PRIMARY KEY,  -- Unix ms, 1H aligned
            sum_open_interest  DOUBLE NOT NULL,     -- contracts
            sum_oi_value       DOUBLE NOT NULL,     -- USDT notional
            oi_delta           DOUBLE              -- change from prev candle
        )
    """)

    # ── liquidations ──────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            open_time     BIGINT PRIMARY KEY,  -- Unix ms, 1H aligned
            liq_buy_vol   DOUBLE NOT NULL,     -- forced BUY liq (shorts liquidated)
            liq_sell_vol  DOUBLE NOT NULL,     -- forced SELL liq (longs liquidated)
            liq_count     INTEGER NOT NULL,
            liq_net_vol   DOUBLE NOT NULL      -- buy - sell (positive = short squeeze)
        )
    """)

    # ── footprint ─────────────────────────────────────────────────────────────
    con.execute("""
        CREATE TABLE IF NOT EXISTS footprint (
            open_time           BIGINT PRIMARY KEY,  -- Unix ms, 1H aligned
            delta               DOUBLE NOT NULL,     -- buy_vol - sell_vol
            buy_vol             DOUBLE NOT NULL,     -- aggressor buy volume
            sell_vol            DOUBLE NOT NULL,     -- aggressor sell volume
            total_vol           DOUBLE NOT NULL,
            poc_price           DOUBLE NOT NULL,     -- price with max volume
            vah                 DOUBLE NOT NULL,     -- value area high (70% vol)
            val                 DOUBLE NOT NULL,     -- value area low  (70% vol)
            large_trade_count   INTEGER NOT NULL,    -- trades > 1 BTC
            large_trade_vol     DOUBLE NOT NULL,     -- vol from large trades
            buy_trade_count     INTEGER NOT NULL,
            sell_trade_count    INTEGER NOT NULL,
            max_imbalance       DOUBLE NOT NULL,     -- max level bid/ask imbalance
            cvd                 DOUBLE              -- cumulative delta (running)
        )
    """)

    # ── features (VIEW joining everything) ───────────────────────────────────
    # Align all sources to the 1H kline open_time
    con.execute("DROP VIEW IF EXISTS features")
    con.execute("""
        CREATE VIEW features AS
        SELECT
            k.open_time,
            -- OHLCV
            k.open, k.high, k.low, k.close, k.volume,
            k.taker_buy_ratio,
            k.trade_count,
            -- Footprint
            f.delta,
            f.buy_vol,
            f.sell_vol,
            f.poc_price,
            f.vah,
            f.val,
            f.large_trade_count,
            f.large_trade_vol,
            f.cvd,
            f.max_imbalance,
            -- Funding (forward-fill to 1H — funding is every 8H)
            fr.funding_rate,
            -- Open Interest
            oi.sum_open_interest,
            oi.oi_delta,
            -- Liquidations
            COALESCE(liq.liq_buy_vol,  0) AS liq_buy_vol,
            COALESCE(liq.liq_sell_vol, 0) AS liq_sell_vol,
            COALESCE(liq.liq_net_vol,  0) AS liq_net_vol,
            COALESCE(liq.liq_count,    0) AS liq_count
        FROM klines k
        LEFT JOIN footprint      f   ON f.open_time   = k.open_time
        LEFT JOIN open_interest  oi  ON oi.open_time  = k.open_time
        LEFT JOIN liquidations   liq ON liq.open_time = k.open_time
        LEFT JOIN (
            -- Latest funding rate at or before each candle
            SELECT fr1.funding_time,
                   fr1.funding_rate
            FROM   funding_rates fr1
        ) fr ON fr.funding_time = (
            SELECT MAX(fr2.funding_time)
            FROM   funding_rates fr2
            WHERE  fr2.funding_time <= k.open_time
        )
        ORDER BY k.open_time
    """)

    count = con.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
    con.close()

    print(f"✅ DuckDB feature store ready: {db_path}")
    print(f"   Tables: klines, funding_rates, open_interest, liquidations, footprint")
    print(f"   View:   features  (joins all tables on open_time)")
    print(f"   Klines rows: {count}")


if __name__ == "__main__":
    setup()
