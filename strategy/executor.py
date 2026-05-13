"""
executor.py — Binance Testnet Order Execution + Risk Management
---------------------------------------------------------------
Handles:
  - Connecting to Binance Testnet
  - Tracking open positions (in / out)
  - Placing BUY / SELL market orders
  - Logging every trade to TimescaleDB (PostgreSQL)
  - PHASE 6: Stop-loss, take-profit, max daily trade limits

Risk Management:
  STOP_LOSS_PCT    → if price drops this % below entry, sell immediately
  TAKE_PROFIT_PCT  → if price rises this % above entry, sell and lock profit
  MAX_DAILY_TRADES → bot stops opening new positions after this many trades/day

Why TimescaleDB instead of DuckDB for trade logging?
  DuckDB allows only one writer at a time (file lock).
  TimescaleDB is a server-based database — handles multiple
  concurrent connections safely. The consumer already holds
  the DuckDB write lock, so trade logging goes to TimescaleDB.
"""

import os
import psycopg2
from datetime import datetime, timezone, date
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load .env from project root (one level up from strategy/)
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)

API_KEY          = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET       = os.getenv("BINANCE_TESTNET_SECRET")
SYMBOL           = os.getenv("TRADE_SYMBOL", "BTCUSDT")
QUANTITY         = float(os.getenv("TRADE_QUANTITY", "0.001"))

# --- Risk management config (read from .env, safe defaults if missing) ---
STOP_LOSS_PCT    = float(os.getenv("STOP_LOSS_PCT", "0.02"))    # 2% drop  → emergency sell
TAKE_PROFIT_PCT  = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))  # 3% rise  → lock profit
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "5"))       # max buys per calendar day


def get_timescale_connection():
    return psycopg2.connect(
        host="localhost", port=5432,
        dbname="cryptodb", user="gleezon", password="crypto123"
    )


class Executor:
    def __init__(self):
        # Connect to Binance Testnet
        self.client = Client(API_KEY, API_SECRET, testnet=True)
        self.in_position = False   # Are we currently holding BTC?
        self.entry_price = None    # Price we bought at

        # --- Phase 6: Daily trade counter ---
        # Tracks how many BUYs have happened today.
        # Resets automatically when the calendar date changes (midnight).
        self.daily_trades = 0
        self.trade_date   = date.today()

        # Connect to TimescaleDB for trade logging
        self.ts = get_timescale_connection()
        self.ts.autocommit = True
        self.cur = self.ts.cursor()

        # Create trade log table — added 'reason' column in Phase 6
        # 'reason' tells us WHY a sell happened: RSI_SIGNAL, STOP_LOSS, TAKE_PROFIT
        # This is essential for later analysis — if stop-loss fires too often,
        # your strategy thresholds need tuning
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id          SERIAL PRIMARY KEY,
                time        TIMESTAMPTZ,
                symbol      TEXT,
                side        TEXT,
                quantity    DOUBLE PRECISION,
                price       DOUBLE PRECISION,
                rsi         DOUBLE PRECISION,
                pnl         DOUBLE PRECISION,
                order_id    TEXT,
                status      TEXT,
                reason      TEXT
            )
        """)

        # Add reason column if it doesn't exist (for existing databases)
        self.cur.execute("""
            ALTER TABLE trade_log ADD COLUMN IF NOT EXISTS reason TEXT
        """)

        print("✓ Executor connected to Binance Testnet")
        print("✓ Trade logging → TimescaleDB")
        print(f"✓ Trading {SYMBOL} | Quantity per trade: {QUANTITY} BTC")
        print(f"✓ Stop-loss: {STOP_LOSS_PCT*100}% | Take-profit: {TAKE_PROFIT_PCT*100}% | Max daily trades: {MAX_DAILY_TRADES}\n")

    def _reset_daily_counter_if_needed(self):
        """
        Resets the daily trade counter if the calendar date has changed.
        Called before every buy attempt — ensures the limit applies per day,
        not per session. The bot can run for days without restarting.
        """
        today = date.today()
        if today != self.trade_date:
            self.daily_trades = 0
            self.trade_date   = today
            print(f"  [RISK] New trading day — daily trade counter reset")

    def get_balance(self):
        """Return current USDT and BTC balances on testnet."""
        try:
            account  = self.client.get_account()
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

    def monitor_position(self):
        """
        Phase 6 — Risk monitor. Called every loop in trade_engine.py.

        If we're in a position, this checks the current price against
        our entry price and fires an emergency sell if:
          - Price dropped >= STOP_LOSS_PCT  below entry (protect from crash)
          - Price rose  >= TAKE_PROFIT_PCT above entry (lock in gains)

        This runs independently of RSI — it doesn't care what RSI says.
        Risk management always takes priority over strategy signals.
        """
        if not self.in_position or self.entry_price is None:
            return  # nothing to monitor

        try:
            current_price = self.get_current_price()
            change_pct    = (current_price - self.entry_price) / self.entry_price

            # Stop-loss: price fell too far — cut the loss immediately
            if change_pct <= -STOP_LOSS_PCT:
                print(f"  🛑 STOP-LOSS triggered | Entry: ${self.entry_price:,.2f} | "
                      f"Current: ${current_price:,.2f} | Drop: {change_pct*100:.2f}%")
                self.sell(rsi_value=None, reason="STOP_LOSS")

            # Take-profit: price rose enough — lock in the gain
            elif change_pct >= TAKE_PROFIT_PCT:
                print(f"  💰 TAKE-PROFIT triggered | Entry: ${self.entry_price:,.2f} | "
                      f"Current: ${current_price:,.2f} | Rise: {change_pct*100:.2f}%")
                self.sell(rsi_value=None, reason="TAKE_PROFIT")

        except BinanceAPIException as e:
            print(f"  [ERROR] monitor_position failed: {e}")

    def buy(self, rsi_value):
        """Place a market BUY order with daily trade limit check."""
        if self.in_position:
            print(f"  [SKIP BUY] Already in position — waiting for signal")
            return

        # Phase 6: check and reset daily counter, then enforce limit
        self._reset_daily_counter_if_needed()
        if self.daily_trades >= MAX_DAILY_TRADES:
            print(f"  [RISK] Daily trade limit reached ({MAX_DAILY_TRADES}) — no more buys today")
            return

        try:
            order = self.client.order_market_buy(
                symbol=SYMBOL,
                quantity=QUANTITY
            )
            price = float(order['fills'][0]['price']) if order['fills'] else self.get_current_price()
            self.in_position  = True
            self.entry_price  = price
            self.daily_trades += 1   # increment daily counter

            self._log_trade(
                side="BUY", quantity=QUANTITY, price=price,
                rsi=rsi_value, pnl=0.0,
                order_id=str(order['orderId']),
                status="FILLED", reason="RSI_SIGNAL"
            )

            print(f"  🟢 BUY  executed | {QUANTITY} BTC @ ${price:,.2f} | "
                  f"RSI: {rsi_value} | Trades today: {self.daily_trades}/{MAX_DAILY_TRADES}")

        except BinanceAPIException as e:
            print(f"  [ERROR] BUY failed: {e}")

    def sell(self, rsi_value, reason="RSI_SIGNAL"):
        """
        Place a market SELL order.
        'reason' tells us why we're selling — used for analysis later.
        Defaults to RSI_SIGNAL when called by the strategy engine.
        Set to STOP_LOSS or TAKE_PROFIT when called by monitor_position().
        """
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
                side="SELL", quantity=QUANTITY, price=price,
                rsi=rsi_value, pnl=round(pnl, 4),
                order_id=str(order['orderId']),
                status="FILLED", reason=reason
            )

            pnl_str = f"+${pnl:.4f}" if pnl >= 0 else f"-${abs(pnl):.4f}"
            print(f"  🔴 SELL executed | {QUANTITY} BTC @ ${price:,.2f} | "
                  f"RSI: {rsi_value} | PnL: {pnl_str} | Reason: {reason}")

        except BinanceAPIException as e:
            print(f"  [ERROR] SELL failed: {e}")

    def _log_trade(self, side, quantity, price, rsi, pnl, order_id, status, reason):
        """Write trade record to TimescaleDB trade_log table."""
        now = datetime.now(timezone.utc)
        self.cur.execute("""
            INSERT INTO trade_log (time, symbol, side, quantity, price, rsi, pnl, order_id, status, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, [now, SYMBOL, side, quantity, price, rsi, pnl, order_id, status, reason])

    def status(self):
        """Print current position status, balances, and risk settings."""
        usdt, btc = self.get_balance()
        position  = f"IN  (entry: ${self.entry_price:,.2f})" if self.in_position else "OUT"
        print(f"  [STATUS] Position: {position} | Balance: ${usdt:,.2f} USDT | {btc:.6f} BTC")
        print(f"  [RISK]   Stop-loss: {STOP_LOSS_PCT*100}% | Take-profit: {TAKE_PROFIT_PCT*100}% | "
              f"Trades today: {self.daily_trades}/{MAX_DAILY_TRADES}")
