"""
trade_engine.py — Full Trading Loop
-------------------------------------
Combines:
  - RSI signal calculation (from rsi_engine.py)
  - Live order execution (from executor.py)

Run this instead of rsi_engine.py when you want real (testnet) trades.

Usage:
    python3 strategy/trade_engine.py
"""

import duckdb
import time
from datetime import datetime

from rsi_engine import build_candles, calc_rsi, signal, RSI_PERIOD, CANDLE_MIN, OVERSOLD, OVERBOUGHT
from executor import Executor

# How often to check for new signals (seconds)
POLL_SEC = 10

DB_PATH = "storage/crypto.db"


def run():
    con      = duckdb.connect(DB_PATH, read_only=True)
    executor = Executor()

    print("=" * 60)
    print("  LIVE TRADE ENGINE — BTC/USDT (Binance Testnet)")
    print(f"  RSI Period : {RSI_PERIOD} candles | Candle: {CANDLE_MIN}m")
    print(f"  BUY  when RSI < {OVERSOLD}")
    print(f"  SELL when RSI > {OVERBOUGHT}")
    print("=" * 60)
    executor.status()
    print()

    while True:
        try:
            candles = build_candles(con, CANDLE_MIN)

            if len(candles) < 2:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Waiting for data... ({len(candles)}/{RSI_PERIOD + 1} candles)"
                )
                time.sleep(POLL_SEC)
                continue

            closes       = candles['close'].tolist()
            latest       = candles.iloc[-1]
            rsi          = calc_rsi(closes, RSI_PERIOD)
            action, icon = signal(rsi)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Candles: {len(candles):>3} | "
                f"Close: ${latest['close']:>10,.2f} | "
                f"RSI: {str(rsi) if rsi else 'N/A':>6} | "
                f"{icon} {action}"
            )

            # --- Execute trades based on signal ---
            if action.strip() == "BUY":
                executor.buy(rsi_value=rsi)

            elif action.strip() == "SELL":
                executor.sell(rsi_value=rsi)

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    run()
