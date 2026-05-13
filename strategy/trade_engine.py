"""
trade_engine.py — Multi-Coin Trading Loop
------------------------------------------
Phase 8: Runs parallel RSI strategy loops for BTC, ETH, SOL, BNB.

How parallelism works here:
  Python's threading module lets us run multiple functions simultaneously.
  Each coin gets its own thread — its own independent loop, its own
  executor, its own RSI calculation, its own position tracking.

  Thread 1: BTC loop (runs every POLL_SEC seconds)
  Thread 2: ETH loop (runs every POLL_SEC seconds)
  Thread 3: SOL loop (runs every POLL_SEC seconds)
  Thread 4: BNB loop (runs every POLL_SEC seconds)

Why threads and not separate scripts?
  One script is easier to manage and start. Airflow triggers one task
  instead of four. All coin logs appear in one terminal window.
  If you want to add a new coin, just add it to COINS list.

Loop order every POLL_SEC seconds (same as before, per coin):
  1. monitor_position() — risk checks FIRST
  2. Build candles from TimescaleDB for this coin
  3. Calculate RSI
  4. Execute buy/sell based on signal

Usage:
    python3 strategy/trade_engine.py
"""

import os
import psycopg2
import pandas as pd
import time
import threading
from datetime import datetime
from dotenv import load_dotenv

from rsi_engine import calc_rsi, signal, RSI_PERIOD, CANDLE_MIN, OVERSOLD, OVERBOUGHT
from executor import Executor

# Load .env
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)

# --- Phase 8: All coins to trade ---
# Each coin gets its own independent executor and RSI loop.
# To add a new coin: add it here, make sure Binance testnet supports it.
COINS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
POLL_SEC = 10


def get_timescale_connection():
    return psycopg2.connect(
        host="localhost", port=5432,
        dbname="cryptodb", user="gleezon", password="crypto123"
    )


def build_candles(con, symbol, candle_minutes=1):
    """
    Build OHLCV candles from TimescaleDB for a specific symbol.
    Filters by symbol so each coin's loop only sees its own data.
    """
    query = f"""
        SELECT
            time_bucket('{candle_minutes} minutes', time) AS candle_time,
            FIRST(price, time)  AS open,
            MAX(price)          AS high,
            MIN(price)          AS low,
            LAST(price, time)   AS close,
            SUM(quantity)       AS volume
        FROM ticks
        WHERE symbol = '{symbol}'
        GROUP BY candle_time
        ORDER BY candle_time ASC
    """
    return pd.read_sql(query, con)


def run_coin(symbol):
    """
    Full trading loop for one coin.
    This function runs inside its own thread — completely independent
    from the loops running for other coins.

    Each coin has:
      - Its own TimescaleDB connection
      - Its own Executor instance (its own position state, its own API calls)
      - Its own RSI calculation
    """
    # Override TRADE_SYMBOL for this executor instance
    os.environ["TRADE_SYMBOL"] = symbol

    con      = get_timescale_connection()
    executor = Executor()

    print(f"\n{'='*60}")
    print(f"  TRADE ENGINE started — {symbol}")
    print(f"  RSI Period: {RSI_PERIOD} | Candle: {CANDLE_MIN}m")
    print(f"  BUY < {OVERSOLD} | SELL > {OVERBOUGHT}")
    print(f"{'='*60}")
    executor.status()

    while True:
        try:
            # Step 1: Risk management always runs first
            executor.monitor_position()

            # Step 2: Build candles for this specific coin
            candles = build_candles(con, symbol, CANDLE_MIN)

            if len(candles) < 2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{symbol}] Waiting for data...")
                time.sleep(POLL_SEC)
                continue

            closes       = candles['close'].tolist()
            latest       = candles.iloc[-1]
            rsi          = calc_rsi(closes, RSI_PERIOD)
            action, icon = signal(rsi)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{symbol} | "
                f"Candles: {len(candles):>3} | "
                f"Close: ${latest['close']:>10,.2f} | "
                f"RSI: {str(rsi) if rsi else 'N/A':>6} | "
                f"{icon} {action}"
            )

            if action.strip() == "BUY":
                executor.buy(rsi_value=rsi)
            elif action.strip() == "SELL":
                executor.sell(rsi_value=rsi, reason="RSI_SIGNAL")

        except Exception as e:
            print(f"[ERROR] [{symbol}] {e}")

        time.sleep(POLL_SEC)


def run():
    """
    Start one thread per coin — all running simultaneously.
    daemon=True means threads stop automatically when the main
    script exits (Ctrl+C or Airflow stops the task).
    """
    print("=" * 60)
    print("  MULTI-COIN TRADE ENGINE — Binance Testnet")
    print(f"  Coins: {', '.join(COINS)}")
    print(f"  RSI Period: {RSI_PERIOD} | Candle: {CANDLE_MIN}m")
    print("=" * 60)

    threads = []
    for coin in COINS:
        t = threading.Thread(target=run_coin, args=(coin,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)  # stagger starts by 1s to avoid simultaneous API calls

    # Keep main thread alive — if it exits, all daemon threads stop
    for t in threads:
        t.join()


if __name__ == "__main__":
    run()
