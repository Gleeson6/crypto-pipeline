import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="cryptodb",
    user="gleezon",
    password="crypto123"
)
conn.autocommit = True
cur = conn.cursor()

# Enable TimescaleDB extension
cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

# Create the raw ticks table
cur.execute("""
    CREATE TABLE IF NOT EXISTS btc_ticks (
        time        TIMESTAMPTZ     NOT NULL,
        symbol      TEXT            NOT NULL,
        price       DOUBLE PRECISION NOT NULL,
        quantity    DOUBLE PRECISION NOT NULL,
        trade_time  BIGINT
    );
""")

# Convert to hypertable (partitioned by time automatically)
cur.execute("""
    SELECT create_hypertable('btc_ticks', 'time', if_not_exists => TRUE);
""")

# Index for fast symbol lookups
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_btc_ticks_symbol
    ON btc_ticks (symbol, time DESC);
""")

print("✓ TimescaleDB extension enabled")
print("✓ btc_ticks hypertable created")
print("✓ Index on (symbol, time) created")
print("\nTimescaleDB is ready to receive ticks.")

cur.close()
conn.close()
