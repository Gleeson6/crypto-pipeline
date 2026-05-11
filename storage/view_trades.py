"""
view_trades.py — Trade History Viewer
---------------------------------------
Shows all executed trades from DuckDB trade_log table.

Usage:
    python3 storage/view_trades.py
"""

import duckdb

DB_PATH = "storage/crypto.db"

con = duckdb.connect(DB_PATH, read_only=True)

try:
    trades = con.execute("""
        SELECT
            id,
            strftime(time, '%Y-%m-%d %H:%M:%S') AS time,
            side,
            quantity,
            ROUND(price, 2)  AS price_usd,
            ROUND(rsi, 2)    AS rsi,
            ROUND(pnl, 4)    AS pnl_usd,
            status
        FROM trade_log
        ORDER BY time DESC
        LIMIT 50
    """).df()

    if trades.empty:
        print("No trades yet. Run the trade engine and wait for RSI signals.")
    else:
        total_pnl = con.execute("SELECT ROUND(SUM(pnl), 4) FROM trade_log WHERE side = 'SELL'").fetchone()[0] or 0
        buy_count  = con.execute("SELECT COUNT(*) FROM trade_log WHERE side = 'BUY'").fetchone()[0]
        sell_count = con.execute("SELECT COUNT(*) FROM trade_log WHERE side = 'SELL'").fetchone()[0]

        print("=" * 75)
        print("  TRADE HISTORY (last 50)")
        print("=" * 75)
        print(trades.to_string(index=False))
        print("=" * 75)
        print(f"  Total trades : {buy_count + sell_count}  ({buy_count} buys / {sell_count} sells)")
        pnl_str = f"+${total_pnl}" if total_pnl >= 0 else f"-${abs(total_pnl)}"
        print(f"  Realised PnL : {pnl_str} USDT")
        print("=" * 75)

except Exception as e:
    print(f"Error reading trade log: {e}")

con.close()
