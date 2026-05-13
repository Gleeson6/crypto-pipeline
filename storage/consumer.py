"""
consumer.py — Kafka Consumer (dual-write: DuckDB + TimescaleDB)
---------------------------------------------------------------
Phase 8: Multi-coin support — unified ticks table for all coins

Reads every tick from the Kafka topic `crypto_ticks` and writes it to:
  - DuckDB       (storage/crypto.db)   — for RSI / backtesting queries
  - TimescaleDB  (PostgreSQL)           — for live concurrent access

Phase 8 change: btc_ticks table → ticks table
  Previously we had a table called btc_ticks hardcoded for Bitcoin.
  Now all coins (BTC, ETH, SOL, BNB) write into a single 'ticks' table.
  The 'symbol' column tells you which coin each row belongs to.

  Why one table instead of one table per coin?
  - Simpler schema — one place to query all market data
  - RSI engine filters by symbol: WHERE symbol = 'ETHUSDT'
  - Grafana can show all coins on one dashboard with a symbol filter
  - Adding a new coin requires zero schema changes

Auto-reconnect: if TimescaleDB goes down and comes back up,
this consumer detects the broken connection and reconnects automatically.

Usage:
    python3 storage/consumer.py
"""

import duckdb
import psycopg2
import json
import time
from kafka import KafkaConsumer
from datetime import datetime, timezone

TS_RETRY_DELAY = 3
TS_CONFIG = dict(
    host="localhost", port=5432,
    dbname="cryptodb", user="gleezon", password="crypto123"
)

# ---------------------------------------------------------------------------
# DuckDB — unified ticks table for all coins
# ---------------------------------------------------------------------------
duck = duckdb.connect('storage/crypto.db')
duck.execute("""
    CREATE TABLE IF NOT EXISTS ticks (
        symbol      VARCHAR,
        price       DOUBLE,
        quantity    DOUBLE,
        timestamp   BIGINT,
        received_at TIMESTAMP
    )
""")
print("✓ DuckDB connected → storage/crypto.db (ticks table)")


# ---------------------------------------------------------------------------
# TimescaleDB — with auto-reconnect
# ---------------------------------------------------------------------------
def connect_timescale():
    """Connect to TimescaleDB, retrying until successful."""
    while True:
        try:
            con = psycopg2.connect(**TS_CONFIG)
            con.autocommit = True
            cur = con.cursor()

            # Create unified ticks table if it doesn't exist
            # TimescaleDB hypertable partitioned by time — handles millions
            # of rows across all coins efficiently
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    time        TIMESTAMPTZ      NOT NULL,
                    symbol      TEXT             NOT NULL,
                    price       DOUBLE PRECISION NOT NULL,
                    quantity    DOUBLE PRECISION NOT NULL,
                    trade_time  BIGINT
                )
            """)
            cur.execute("""
                SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol
                ON ticks (symbol, time DESC)
            """)

            print("✓ TimescaleDB connected → cryptodb@localhost:5432 (ticks table)")
            return con, cur
        except psycopg2.OperationalError as e:
            print(f"[WARN] TimescaleDB unavailable: {e}")
            print(f"       Retrying in {TS_RETRY_DELAY}s...")
            time.sleep(TS_RETRY_DELAY)


def is_ts_alive(cur):
    """Ping TimescaleDB — returns True if connection is healthy."""
    try:
        cur.execute("SELECT 1")
        return True
    except Exception:
        return False


ts_con, ts_cur = connect_timescale()

# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------
consumer = KafkaConsumer(
    'crypto_ticks',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='dual-write-consumer'
)

print("✓ Kafka consumer connected → topic: crypto_ticks")
print("\nDual-writing ticks: DuckDB ✓  TimescaleDB ✓")
print("Coins: BTC, ETH, SOL, BNB\n")

tick_count = 0

for message in consumer:
    tick = message.value
    now  = datetime.now(timezone.utc)

    # --- Write to DuckDB ---
    duck_ok = False
    try:
        duck.execute("""
            INSERT INTO ticks VALUES (?, ?, ?, ?, ?)
        """, [tick['symbol'], tick['price'], tick['quantity'], tick['timestamp'], now])
        duck_ok = True
    except Exception as e:
        print(f"[ERROR] DuckDB write failed: {e}")

    # --- Write to TimescaleDB (with reconnect on broken connection) ---
    ts_ok = False
    for attempt in range(3):
        try:
            if not is_ts_alive(ts_cur):
                raise psycopg2.OperationalError("Connection lost")

            ts_cur.execute("""
                INSERT INTO ticks (time, symbol, price, quantity, trade_time)
                VALUES (%s, %s, %s, %s, %s)
            """, [now, tick['symbol'], tick['price'], tick['quantity'], tick['timestamp']])
            ts_ok = True
            break

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"[WARN] TimescaleDB connection lost — reconnecting (attempt {attempt + 1}/3)...")
            try:
                ts_con.close()
            except Exception:
                pass
            ts_con, ts_cur = connect_timescale()

        except Exception as e:
            print(f"[ERROR] TimescaleDB write failed: {e}")
            break

    tick_count += 1
    duck_icon = "✓" if duck_ok else "✗"
    ts_icon   = "✓" if ts_ok   else "✗"
    print(
        f"[{now.strftime('%H:%M:%S')}] "
        f"{tick['symbol']}: ${tick['price']:,.2f} | "
        f"DuckDB {duck_icon}  TimescaleDB {ts_icon} | "
        f"Total: {tick_count:,}"
    )
