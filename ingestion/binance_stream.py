import websocket
import json
from datetime import datetime

def on_message(ws, message):
    data = json.loads(message)
    price = data['p']
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] BTC/USDT: ${float(price):,.2f}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connected to Binance WebSocket — streaming live BTC/USDT prices...\n")

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
