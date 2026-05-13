"""
binance_stream.py — Binance Multi-Coin WebSocket → Kafka
---------------------------------------------------------
Phase 8: Multi-coin support — BTC, ETH, SOL, BNB

Connects to Binance's combined stream endpoint which lets us subscribe
to multiple coin streams over a SINGLE WebSocket connection.

Why one connection instead of four?
  - Fewer network connections = less overhead
  - Binance rate limits connections — one combined stream is cleaner
  - Simpler reconnect logic — one handler manages everything

URL format for combined streams:
  wss://stream.binance.com:9443/stream?streams=btcusdt@trade/ethusdt@trade/...

Message format (different from single stream):
  Single stream: {"p": price, "q": qty, "T": timestamp}
  Combined stream: {"stream": "btcusdt@trade", "data": {"p": price, ...}}

All ticks published to the same Kafka topic: crypto_ticks
Symbol field differentiates which coin each tick belongs to.

Usage:
    python3 ingestion/binance_stream.py
"""

import websocket
import json
import time
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

RECONNECT_DELAY = 5
KAFKA_TOPIC     = "crypto_ticks"

# --- Phase 8: All coins we're trading ---
# To add a new coin, just add it to this list — nothing else needs changing.
# Format must match Binance's stream name: lowercase symbol + @trade
COINS = ["btcusdt", "ethusdt", "solusdt", "bnbusdt"]

# Build the combined stream URL from the coins list
STREAMS = "/".join([f"{coin}@trade" for coin in COINS])
WS_URL  = f"wss://stream.binance.com:9443/stream?streams={STREAMS}"


def get_producer():
    """Create Kafka producer, retrying until Kafka is available."""
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("✓ Kafka producer connected")
            return producer
        except NoBrokersAvailable:
            print(f"[WARN] Kafka not ready — retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


producer = get_producer()


def on_message(ws, message):
    """
    Handle incoming messages from the combined stream.

    Combined stream wraps each tick in a 'stream' + 'data' envelope.
    We extract the symbol from 'stream' field and the tick from 'data'.
    Then publish a normalised tick to Kafka with the symbol included.
    """
    msg = json.loads(message)

    # Extract which coin this tick is for
    # "btcusdt@trade" → split on "@" → "btcusdt" → uppercase → "BTCUSDT"
    stream_name = msg.get('stream', '')
    symbol      = stream_name.split('@')[0].upper()

    # The actual tick data is nested inside 'data'
    data = msg.get('data', {})

    tick = {
        'symbol':    symbol,
        'price':     float(data['p']),
        'quantity':  float(data['q']),
        'timestamp': data['T']
    }

    producer.send(KAFKA_TOPIC, tick)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: ${float(data['p']):,.2f}")


def on_error(ws, error):
    print(f"[ERROR] WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"[WARN] Connection closed — reconnecting in {RECONNECT_DELAY}s...")


def on_open(ws):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Connected to Binance combined stream")
    print(f"  → Streaming: {', '.join([c.upper() for c in COINS])}")
    print(f"  → Publishing to Kafka topic: {KAFKA_TOPIC}\n")


def run():
    print("=" * 60)
    print("  Binance Multi-Coin WebSocket Ingestion")
    print(f"  Coins    : {', '.join([c.upper() for c in COINS])}")
    print(f"  Topic    : {KAFKA_TOPIC}")
    print(f"  Reconnect: auto, {RECONNECT_DELAY}s delay on drop")
    print("=" * 60)

    while True:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)
        print(f"Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()
