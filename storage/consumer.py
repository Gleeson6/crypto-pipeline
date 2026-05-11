import duckdb
import psycopg2
import json
from kafka import KafkaConsumer
from datetime import datetime, timezone

# --- DuckDB connection ---
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

# --- TimescaleDB connection ---
ts = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="cryptodb",
    user="gleezon",
    password="crypto123"
)
ts.autocommit = True
ts_cur = ts.cursor()

# --- Kafka consumer ---
consumer = KafkaConsumer(
    'crypto_ticks',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='dual-write-consumer'
)

print("Consumer started — dual-writing to DuckDB + TimescaleDB...\n")

for message in consumer:
    tick = message.value
    now = datetime.now(timezone.utc)

    # Write to DuckDB
    duck.execute("""
        INSERT INTO btc_ticks VALUES (?, ?, ?, ?, ?)
    """, [
        tick['symbol'],
        tick['price'],
        tick['quantity'],
        tick['timestamp'],
        now
    ])

    # Write to TimescaleDB
    ts_cur.execute("""
        INSERT INTO btc_ticks (time, symbol, price, quantity, trade_time)
        VALUES (%s, %s, %s, %s, %s)
    """, [
        now,
        tick['symbol'],
        tick['price'],
        tick['quantity'],
        tick['timestamp']
    ])

    print(f"[{now.strftime('%H:%M:%S')}] {tick['symbol']}: ${tick['price']:,.2f} → DuckDB ✓  TimescaleDB ✓")
