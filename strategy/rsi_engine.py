"""
rsi_engine.py — RSI Signal Engine (Multi-Coin)
-----------------------------------------------
Phase 8: Now accepts a symbol parameter so the same engine
works for BTC, ETH, SOL, BNB — any coin in the ticks table.

The RSI calculation itself doesn't change — the only difference
is which symbol we filter on when building candles.
"""

import duckdb
import time
from datetime import datetime

# --- Configuration ---
DB_PATH    = 'storage/crypto.db'
RSI_PERIOD = 14   # number of candles to calculate RSI over
CANDLE_MIN = 1    # candle size in minutes
OVERSOLD   = 30   # RSI below this → BUY signal
OVERBOUGHT = 70   # RSI above this → SELL signal
POLL_SEC   = 10   # how often to re-run the engine (seconds)


def build_candles(con, symbol, candle_minutes=1):
    """
    Group raw ticks into OHLCV candles for a specific symbol.

    Phase 8 change: added WHERE symbol = ? filter so the same
    function works for any coin. Previously it queried btc_ticks
    with no filter — now it queries the unified ticks table
    and filters by symbol.
    """
    query = f"""
        SELECT
            time_bucket(INTERVAL '{candle_minutes} minutes', received_at) AS candle_time,
            FIRST(price, received_at)  AS open,
            MAX(price)                 AS high,
            MIN(price)                 AS low,
            LAST(price, received_at)   AS close,
            SUM(quantity)              AS volume
        FROM ticks
        WHERE symbol = '{symbol}'
        GROUP BY candle_time
        ORDER BY candle_time ASC
    """
    return con.execute(query).df()


def calc_rsi(closes, period=14):
    """
    Calculate RSI from a list of closing prices.
    Returns the latest RSI value, or None if not enough data.
    """
    if len(closes) < period + 1:
        return None

    gains  = []
    losses = []

    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def signal(rsi):
    """Translate RSI value into a trading signal."""
    if rsi is None:
        return "WAITING", "⏳"
    elif rsi < OVERSOLD:
        return "BUY ", "🟢"
    elif rsi > OVERBOUGHT:
        return "SELL", "🔴"
    else:
        return "HOLD", "⚪"


def run(symbol="BTCUSDT"):
    """Run the RSI engine for a single symbol."""
    con = duckdb.connect(DB_PATH, read_only=True)
    print("=" * 55)
    print(f"  RSI Strategy Engine — {symbol}")
    print(f"  Period: {RSI_PERIOD} candles | Candle: {CANDLE_MIN}m")
    print(f"  BUY < {OVERSOLD}  |  SELL > {OVERBOUGHT}")
    print("=" * 55)

    while True:
        try:
            candles = build_candles(con, symbol, CANDLE_MIN)

            if len(candles) < 2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{symbol}] Waiting for data...")
                time.sleep(POLL_SEC)
                continue

            closes        = candles['close'].tolist()
            latest_candle = candles.iloc[-1]
            rsi           = calc_rsi(closes, RSI_PERIOD)
            action, icon  = signal(rsi)

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{symbol} | "
                f"Candles: {len(candles):>3} | "
                f"Close: ${latest_candle['close']:>10,.2f} | "
                f"RSI: {str(rsi) if rsi else 'N/A':>6} | "
                f"{icon} {action}"
            )

        except Exception as e:
            print(f"[ERROR] [{symbol}] {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    run()
