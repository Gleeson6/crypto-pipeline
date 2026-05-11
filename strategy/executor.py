"""
executor.py — Binance Testnet Order Execution
----------------------------------------------
Handles:
  - Connecting to Binance Testnet
  - Tracking open positions (in / out)
  - Placing BUY / SELL market orders
  - Logging every trade to DuckDB
"""

import os
import duckdb
from datetime import datetime, timezone
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
SYMBOL     = os.getenv("TRADE_SYMBOL", "BTCUSDT")
QUANTITY   = float(os.getenv("TRADE_QUANTITY", "0.001"))
DB_PATH    = "storage/crypto.db"


class Executor:
    def __init__(self):
        # Connect to Binance Testnet
        self.client = Client(API_KEY, API_SECRET, testnet=True)
        self.in_position = False   # Are we currently holding BTC?
        self.entry_price = None    # Price we bought at

        # Set up trade log table in DuckDB
        self.duck = duckdb.connect(DB_PATH)
        self.duck.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id          INTEGER PRIMARY KEY,
                time        TIMESTAMP,
                symbol      VARCHAR,
                side        VARCHAR,
                quantity    DOUBLE,
                price       DOUBLE,
                rsi         DOUBLE,
                pnl         DOUBLE,
                order_id    VARCHAR,
                status      VARCHAR
            )
        """)
        self._next_id = self._get_next_id()
        print("✓ Executor connected to Binance Testnet")
        print(f"✓ Trading {SYMBOL} | Quantity per trade: {QUANTITY} BTC\n")

    def _get_next_id(self):
        result = self.duck.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM trade_log").fetchone()
        return result[0]

    def get_balance(self):
        """Return current USDT and BTC balances on testnet."""
        try:
            account = self.client.get_account()
            balances = {b['asset']: float(b['free']) for b in account['balances']}
            usdt = balances.get('USDT', 0)
            btc  = balances.get('BTC', 0)
            return usdt, btc
        except BinanceAPIException as e:
            print(f"[EXECUTOR] Balance check failed: {e}")
            return 0, 0

    def get_current_price(self):
        """Get latest BTC/USDT price from Binance."""
        ticker = self.client.get_symbol_ticker(symbol=SYMBOL)
        return float(ticker['price'])

    def buy(self, rsi_value):
        """Place a market BUY order if not already in position."""
        if self.in_position:
            print(f"  [SKIP BUY] Already in position — waiting for SELL signal")
            return

        try:
            order = self.client.order_market_buy(
                symbol=SYMBOL,
                quantity=QUANTITY
            )
            price = float(order['fills'][0]['price']) if order['fills'] else self.get_current_price()
            self.in_position = True
            self.entry_price = price

            self._log_trade(
                side="BUY",
                quantity=QUANTITY,
                price=price,
                rsi=rsi_value,
                pnl=0.0,
                order_id=str(order['orderId']),
                status="FILLED"
            )

            print(f"  🟢 BUY  executed | {QUANTITY} BTC @ ${price:,.2f} | RSI: {rsi_value}")

        except BinanceAPIException as e:
            print(f"  [ERROR] BUY failed: {e}")

    def sell(self, rsi_value):
        """Place a market SELL order if currently in position."""
        if not self.in_position:
            print(f"  [SKIP SELL] No open position to sell")
            return

        try:
            order = self.client.order_market_sell(
                symbol=SYMBOL,
                quantity=QUANTITY
            )
            price = float(order['fills'][0]['price']) if order['fills'] else self.get_current_price()
            pnl   = (price - self.entry_price) * QUANTITY if self.entry_price else 0

            self.in_position = False
            self.entry_price = None

            self._log_trade(
                side="SELL",
                quantity=QUANTITY,
                price=price,
                rsi=rsi_value,
                pnl=round(pnl, 4),
                order_id=str(order['orderId']),
                status="FILLED"
            )

            pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
            print(f"  🔴 SELL executed | {QUANTITY} BTC @ ${price:,.2f} | RSI: {rsi_value} | PnL: {pnl_str}")

        except BinanceAPIException as e:
            print(f"  [ERROR] SELL failed: {e}")

    def _log_trade(self, side, quantity, price, rsi, pnl, order_id, status):
        """Write trade record to DuckDB trade_log table."""
        now = datetime.now(timezone.utc)
        self.duck.execute("""
            INSERT INTO trade_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            self._next_id, now, SYMBOL, side,
            quantity, price, rsi, pnl, order_id, status
        ])
        self._next_id += 1

    def status(self):
        """Print current position status and balances."""
        usdt, btc = self.get_balance()
        position  = f"IN  (entry: ${self.entry_price:,.2f})" if self.in_position else "OUT"
        print(f"  [STATUS] Position: {position} | Balance: ${usdt:,.2f} USDT | {btc:.6f} BTC")
