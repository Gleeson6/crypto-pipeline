import duckdb
import json
from kafka import KafkaConsumer
from datetime import datetime

# Connect to DuckDB (creates the file if it doesn't exist)
con = duckdb.connect('storage/crypto.db')

# Create table if it doesn't exist
con.execute("""
    CREATE TABLE IF NOT EXISTS btc_ticks (
        symbol      VARCHAR,
        price       DOUBLE,
        quantity    DOUBLE,
        timestamp   BIGINT,
        received_at TIMESTAMP
    )
""")

# Connect to Kafka
consumer = KafkaConsumer(
    'crypto_ticks',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='duckdb-consumer'
)

print("Consumer started — reading from Kafka and storing in DuckDB...\n")

for message in consumer:
    tick = message.value
    con.execute("""
        INSERT INTO btc_ticks VALUES (?, ?, ?, ?, ?)
    """, [
        tick['symbol'],
        tick['price'],
        tick['quantity'],
        tick['timestamp'],
        datetime.now()
    ])
    print(f"Stored → {tick['symbol']}: ${tick['price']:,.2f}")
