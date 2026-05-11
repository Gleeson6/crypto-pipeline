# Crypto Trading Pipeline

## Project Overview
A production-grade algorithmic trading system built on real-time data engineering infrastructure. The dual goal is to **learn core data engineering deeply** (Kafka, DuckDB, TimescaleDB, Docker) while **building a system that eventually generates profit** from automated crypto trading.

**Owner:** Gleezon (gleesonminoy7@gmail.com)  
**Environment:** WSL (Ubuntu) on Windows — project lives at `~/crypto-pipeline/`  
**GitHub:** https://github.com/Gleeson6/crypto-pipeline  
**Timeline to live trading:** 3–6 months  
**Current phase:** Phase 4 complete — running end-to-end on Binance Testnet

---

## Architecture

```
Binance WebSocket (live prices)
        ↓
    Kafka (message queue)
    Topic: crypto_ticks
        ↓
  Python Consumer (dual-write)
        ↓
   ┌────┴────────────┐
DuckDB           TimescaleDB
(analysis/RSI)   (live storage)
        ↓
  RSI Strategy Engine
  (signals: BUY / SELL / HOLD)
        ↓
  Binance Testnet Executor
  (places real paper trades)
        ↓
  Grafana Dashboard (Phase 5)
        ↓
  Airflow Orchestration (Phase 7)
```

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Ingestion | Binance WebSocket | Live BTC/USDT price stream |
| Queue | Apache Kafka | Decouples ingestion from processing |
| Coordinator | Zookeeper | Manages Kafka internals |
| Analytical DB | DuckDB | Fast local queries, RSI calculation |
| Time-series DB | TimescaleDB (PostgreSQL) | Permanent tick storage |
| Strategy | Python (RSI engine) | Generates BUY/SELL/HOLD signals |
| Execution | python-binance (Testnet) | Places paper trades automatically |
| Infrastructure | Docker Compose | Runs Kafka + Zookeeper + TimescaleDB |
| Environment | Python venv | Isolated package management |

---

## Trading Configuration

**Current coins:** BTC/USDT  
**Planned coins:** ETH/USDT, SOL/USDT, BNB/USDT  
**Current strategy:** RSI (14-period, 1-minute candles)  
**Planned strategies:** ML-based price direction prediction  
**Trade size:** 0.001 BTC per trade  
**Exchange:** Binance (Testnet now → Live in 3–6 months)

**RSI thresholds:**
- RSI < 30 → BUY (oversold)
- RSI > 70 → SELL (overbought)
- 30–70 → HOLD

---

## Project Structure

```
crypto-pipeline/
├── ingestion/
│   └── binance_stream.py      # Connects to Binance WebSocket, publishes to Kafka
├── storage/
│   ├── consumer.py            # Kafka consumer, dual-writes to DuckDB + TimescaleDB
│   ├── timescale_setup.py     # One-time DB schema setup (hypertable, indexes)
│   ├── view_trades.py         # CLI trade history viewer
│   └── crypto.db              # DuckDB database file (gitignored)
├── strategy/
│   ├── rsi_engine.py          # RSI calculation + signal generation
│   ├── executor.py            # Binance Testnet order placement + trade logging
│   └── trade_engine.py        # Main loop: RSI signals → execute trades
├── docker-compose.yml         # Kafka + Zookeeper + TimescaleDB containers
├── .env                       # API keys (gitignored — NEVER commit this)
├── .env.example               # Template for API keys
└── .gitignore
```

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Binance WebSocket → live BTC ticks | ✅ Complete |
| 2 | Kafka message queue | ✅ Complete |
| 3 | DuckDB + TimescaleDB storage | ✅ Complete |
| 4 | RSI strategy + Binance Testnet execution | ✅ Complete |
| 5 | Grafana monitoring dashboard | 🔄 Next |
| 6 | Risk management (stop-loss, position sizing) | ⬜ Pending |
| 7 | Airflow orchestration | ⬜ Pending |
| 8 | Multi-coin support (ETH, SOL, BNB) | ⬜ Pending |
| 9 | ML-based prediction strategy | ⬜ Pending |
| 10 | Live trading with real money | ⬜ 3–6 months |

---

## How to Run (Every Session)

**Step 1 — Start Docker infrastructure:**
```bash
cd ~/crypto-pipeline
sudo service docker start
docker compose up -d
```

**Step 2 — Activate virtual environment:**
```bash
source venv/bin/activate
```

**Step 3 — Open 3 terminal tabs and run:**
```bash
# Tab 1 — Live data ingestion from Binance
python3 ingestion/binance_stream.py

# Tab 2 — Store ticks to DuckDB + TimescaleDB
python3 storage/consumer.py

# Tab 3 — RSI strategy + auto trading on Testnet
python3 strategy/trade_engine.py
```

**View trade history anytime:**
```bash
python3 storage/view_trades.py
```

---

## Important Notes

- **Never commit `.env`** — it contains real API keys
- **Testnet only** — all trades use fake money until Phase 10
- **venv must be active** — always run `source venv/bin/activate` before any Python script
- **WSL path:** `~/crypto-pipeline/` — always `cd` here before running commands
- **Windows path:** `C:\Users\User\Documents\Claude\Projects\crypto data pipeline\`
- **Git reminder:** Push code to GitHub after every new feature or phase

---

## Key Decisions Made

- **Kafka over Redis Streams** — chosen for industry-standard message queue patterns
- **DuckDB for analysis** — zero setup, blazing fast for RSI/backtesting queries
- **TimescaleDB for production** — handles millions of tick rows with time-series superpowers
- **RSI first, ML later** — understand rule-based logic before adding ML complexity
- **Testnet before live** — validate all strategies with paper money first
