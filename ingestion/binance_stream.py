"""
binance_stream.py — Binance WebSocket → Kafka
----------------------------------------------
Connects to Binance's live trade stream for BTC/USDT.
Publishes every tick to the Kafka topic: crypto_ticks

Auto-reconnect: if the WebSocket drops (network blip, timeout,
Binance server-side close), it waits RECONNECT_DELAY seconds
and tries again — forever. This is the production pattern for
any always-on data ingestion process.

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
WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
KAFKA_TOPIC = "crypto_ticks"


def get_producer():
    """Create a Kafka producer, retrying until Kafka is available."""
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
    data = json.loads(message)
    tick = {
        'symbol':    'BTCUSDT',
        'price':     float(data['p']),
        'quantity':  float(data['q']),
        'timestamp': data['T']
    }
    producer.send(KAFKA_TOPIC, tick)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Published → BTC/USDT: ${tick['price']:,.2f}")


def on_error(ws, error):
    print(f"[ERROR] WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print(f"[WARN] Connection closed (code={close_status_code}) — reconnecting in {RECONNECT_DELAY}s...")


def on_open(ws):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Connected to Binance WebSocket")
    print(f"  → Publishing live BTC/USDT ticks to Kafka topic: {KAFKA_TOPIC}\n")


def run():
    print("=" * 55)
    print("  Binance WebSocket Ingestion — BTC/USDT")
    print(f"  Kafka topic : {KAFKA_TOPIC}")
    print(f"  Reconnect  : auto, {RECONNECT_DELAY}s delay on drop")
    print("=" * 55)

    while True:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        # ping_interval keeps connection alive (Binance drops idle after 24h)
        ws.run_forever(ping_interval=30, ping_timeout=10)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reconnecting in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    run()
