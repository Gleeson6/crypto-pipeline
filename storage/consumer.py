"""
consumer.py — Kafka Consumer (dual-write: DuckDB + TimescaleDB)
---------------------------------------------------------------
Reads every tick from the Kafka topic `crypto_ticks` and writes it to:
  - DuckDB       (storage/crypto.db)   — for RSI / backtesting queries
  - TimescaleDB  (PostgreSQL)           — for live concurrent access

Auto-reconnect: if TimescaleDB goes down and comes back up (e.g. docker
restart), this consumer detects the broken connection and reconnects
automatically instead of failing silently.

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
# DuckDB — write lock held for lifetime of this process (intentional)
# ---------------------------------------------------------------------------
duck = duckdb.connect('storage/crypto.db')
duck.execute("""
    CREATE TABLE IF NOT EXISTS btc_ticks (
        symbol      VARCHAR,
        price       DOUBLE,
        quantity    DOUBLE,
        timestamp   BIGINT,
        received_at TIMESTAMP
    )
""")
print("✓ DuckDB connected → storage/crypto.db")


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
            print("✓ TimescaleDB connected → cryptodb@localhost:5432")
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
print("\nDual-writing ticks: DuckDB ✓  TimescaleDB ✓\n")

tick_count = 0

for message in consumer:
    tick = message.value
    now  = datetime.now(timezone.utc)

    # --- Write to DuckDB ---
    duck_ok = False
    try:
        duck.execute("""
            INSERT INTO btc_ticks VALUES (?, ?, ?, ?, ?)
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
                INSERT INTO btc_ticks (time, symbol, price, quantity, trade_time)
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
