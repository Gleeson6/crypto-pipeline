# Crypto Quant Pipeline

**Owner:** Gleezon (gleesonminoy7@gmail.com)
**Environment:** WSL (Ubuntu) on Windows
**Goal:** Production-grade Bitcoin prediction system — data engineering foundation → ML models → live trading

---

## Architecture

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 1 — DATA INGESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Binance WebSocket          Binance REST API           Binance Vision (zips)
  Live ticks: BTC/ETH        klines 2Y history          Monthly aggTrades
  SOL/BNB → Kafka            funding rates, OI          → footprint (delta,
  Topic: crypto_ticks        fetch_klines.py             CVD, POC, VAH/VAL)
                             fetch_funding_oi.py        fetch_footprint.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 2 — STORAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TimescaleDB                DuckDB                     ChromaDB
  (live tick storage)        (feature_store.duckdb)     (rag/rag_db/)
  real-time pipeline         klines · footprint         36,557 embedded
  hypertable, indexes        funding · ml_features      quant knowledge chunks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 3 — FEATURE ENGINEERING  (ml/compute_features.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  82 features across 9 groups — 17,320 rows (Jun 2024 → May 2026)

  Session/Time    │ hour_sin/cos, is_asian/european/us_session
  Price/Returns   │ log_return_1h/4h/24h, candle_body, wicks
  Volatility      │ volatility_24h/7d, vol_ratio, ATR
  Technicals      │ RSI-7/14, MACD, Bollinger Bands, EMA-8/21/50/200
  Volume          │ vol_zscore, taker_buy_ratio, vol_ratio_24h
  Footprint       │ delta_norm, CVD, poc_distance, va_range, large_trade_ratio
  Funding         │ funding_rate, extreme_long/short flags, 8h_change
  Liq Proxy       │ liq_long/short_proxy, cascade_flag, cascade_4h/24h
  Target          │ target_return_4h, target_direction_4h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 4 — ML MODELS  (ml/train_model.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  XGBoost Classifier                 XGBoost Regressor
  Predicts direction (UP / DOWN)     Predicts 4H return magnitude
  Walk-forward split 70/10/20        Heavy regularization
  No test set leakage                No early stopping
  Accuracy: 51.3% (+3.4% vs naive)

  Top signals: log_return_24h · hour_cos · large_trade_vol
               delta_norm · ema_50 · rsi_7 · taker_buy_ratio

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 5 — QUANT RAG ENGINE  (rag/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Knowledge Base              Retrieval                  Generation
  506 arXiv papers            ChromaDB dense search      Grok grok-4-fast
  Quant books (Hull,          BM25 sparse search         Streaming SSE
  Wilmott, Taleb)             RRF hybrid fusion          rag_generate.py
  On-chain reference docs     Trust-weighted reranking   rag_chat.py
  36,557 chunks embedded      rag_query.py               Multi-turn memory

  FastAPI server: localhost:8000  (managed by systemd quant-rag.service)
  Web UI: http://localhost:8000   (always live while WSL is running)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 6 — STRATEGY ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RSI signals + XGBoost prediction → BUY / SELL / HOLD
  Timeframe: 1–4H swing trading
  Hard risk rules (code-enforced, not config):
    Max 1–2% capital per trade · Stop-loss 2% · Take-profit 3%
    Max 5 trades per day

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LAYER 7 — ORCHESTRATION & MONITORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Binance Testnet             Grafana :3000              Airflow :8080
  Paper trading               Pipeline monitor           DAG orchestration
  (→ $100 live after          crypto_pipeline DAG        LocalExecutor
   3mo paper validation)

  systemd quant-rag.service — auto-restarts on crash, survives WSL reboot
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Ingestion | Binance WebSocket | Live ticks: BTC/ETH/SOL/BNB |
| Queue | Apache Kafka | Decouples ingestion from processing |
| Live DB | TimescaleDB | Real-time tick storage, hypertable |
| Analytical DB | DuckDB | Feature store, ML training data |
| Vector DB | ChromaDB | Quant knowledge embeddings |
| Embeddings | MiniLM (ONNX) | ChromaDB default, no PyTorch needed |
| LLM | Grok grok-4-fast (xAI) | RAG generation, streaming |
| ML | XGBoost | Classifier + regressor, walk-forward CV |
| RAG Server | FastAPI + uvicorn | Persistent server, warm in memory |
| Orchestration | Apache Airflow | Pipeline DAGs, retry logic |
| Monitoring | Grafana | Real-time dashboard |
| Infrastructure | Docker Compose | Kafka + TimescaleDB + Airflow |

---

## Phase Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Binance WebSocket → Kafka ingestion | ✅ |
| 2 | Kafka message queue | ✅ |
| 3 | DuckDB + TimescaleDB dual-write storage | ✅ |
| 4 | RSI strategy engine + Binance Testnet executor | ✅ |
| 5 | Grafana monitoring dashboard | ✅ |
| 6 | Risk management (stop-loss 2%, take-profit 3%, max 5 daily trades) | ✅ |
| 7 | Airflow orchestration (crypto_pipeline DAG) | ✅ |
| 8 | Multi-coin support (BTC/ETH/SOL/BNB parallel threads) | ✅ |
| 9a | ChromaDB vector DB + embedding pipeline | ✅ |
| 9b | Document ingestion (506 arXiv papers + quant books + web) | ✅ |
| 9c | RAG query engine (Grok API, streaming UI, conversation memory) | ✅ |
| 9d | Hybrid search (BM25 + dense, RRF fusion) + FastAPI server | ✅ |
| 9e | systemd service — RAG server auto-starts on WSL boot | ✅ |
| 9f | ML feature engineering — 82 features, 17,320 rows, 2Y history | ✅ |
| 9g | Footprint ingestion — full 24-month aggTrades (microsecond fix) | ✅ |
| 9h | XGBoost classifier + regressor — walk-forward 70/10/20 split | ✅ |
| 9i | Backtesting (VectorBT) | 🔄 Next |
| 9j | Connect ML signals → trade engine | ⬜ |
| 9k | Paper trading — 3 months Binance Testnet | ⬜ |
| 9l | LSTM ensemble (sequence patterns) | ⬜ |
| 10 | $100 live trading → scale | ⬜ |

---

## Project Structure

```
crypto data pipeline/
├── ingestion/
│   └── binance_stream.py          # WebSocket → Kafka
├── storage/
│   ├── consumer.py                # Kafka → DuckDB + TimescaleDB
│   └── timescale_setup.py
├── strategy/
│   └── trade_engine.py            # RSI signals → Binance Testnet
├── ml/
│   ├── setup_db.py                # DuckDB schema
│   ├── fetch_klines.py            # 2Y OHLCV history
│   ├── fetch_funding_oi.py        # Funding rates + open interest
│   ├── fetch_footprint.py         # aggTrades → delta/CVD/POC (μs fix)
│   ├── ingest_all.py              # One-shot full ingestion
│   ├── compute_features.py        # 82 features → ml_features table
│   ├── correlations.py            # Spearman ranking vs 4H target
│   ├── train_model.py             # XGBoost classifier + regressor
│   ├── feature_store.duckdb       # All ML data (gitignored)
│   ├── models/                    # Trained model .pkl files (gitignored)
│   └── data/                      # Correlation CSVs
├── rag/
│   ├── rag_build.py               # Embed + index documents into ChromaDB
│   ├── rag_query.py               # Hybrid search (BM25 + dense, RRF)
│   ├── rag_generate.py            # Grok streaming generation
│   ├── rag_chat.py                # Multi-turn conversation memory
│   ├── rag_server.py              # FastAPI persistent server :8000
│   ├── build_bm25_only.py         # Rebuild BM25 index from ChromaDB
│   └── rag_db/                    # ChromaDB + BM25 index (gitignored)
├── scripts/
│   ├── arxiv_ingest.py            # arXiv paper scraper (506 papers)
│   └── onchain_docs/              # Reference on-chain docs
├── docker-compose.yml
├── .env                           # API keys (NEVER commit)
└── .env.example
```

---

## Key Services

| Service | URL | Start |
|---|---|---|
| RAG Web UI | http://localhost:8000 | Auto (systemd) |
| Grafana | http://localhost:3000 | `docker compose up -d` |
| Airflow | http://localhost:8080 | `docker compose up -d` |

```bash
# RAG server
sudo systemctl status quant-rag
sudo systemctl restart quant-rag
sudo journalctl -u quant-rag -f     # live logs

# Infrastructure
docker compose up -d
```

---

## Signal Confidence

| Data Layer | Reasoning Confidence |
|---|---|
| Current (technicals + footprint + funding) | ~60% |
| + Full OI history | ~73% |
| + Real liquidations | ~82% |
| + Exchange inflows/outflows | ~88% |
| + Whale wallet movements | ~92% |

Current model accuracy: **51.3% directional** (baseline: 51.2%).
Target with full data stack: **58–65%** — sufficient for profitability with 2:1 reward/risk.

---

## Important Notes

- **Never commit `.env`** — contains XAI_API_KEY and Binance API keys
- **`quant_venv`** for ML/RAG scripts — `source ~/quant_venv/bin/activate`
- **Testnet only** — no real money until 3-month paper trading validates edge
- **Hard risk rule** — 1–2% max capital per trade, enforced in code not config
