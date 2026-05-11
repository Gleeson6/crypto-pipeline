import websocket
import json
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def on_message(ws, message):
    data = json.loads(message)
    tick = {
        'symbol': 'BTCUSDT',
        'price': float(data['p']),
        'quantity': float(data['q']),
        'timestamp': data['T']
    }
    producer.send('crypto_ticks', tick)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Published → BTC/USDT: ${tick['price']:,.2f}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connected — publishing live ticks to Kafka topic: crypto_ticks\n")

if __name__ == "__main__":
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ws = websocket.WebSocketApp(
        url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()
