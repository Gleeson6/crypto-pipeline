# Exchange Flow Signals — Bitcoin Trading Reference

Curated signal definitions, thresholds, and interpretation patterns for
exchange inflow/outflow metrics. Written for 1–4h swing trading on Bitcoin
with a 1–2% capital-at-risk-per-trade rule.

---

## Exchange Netflow

**Definition:** Net BTC moving onto exchanges minus BTC moving off exchanges
over a rolling window (typically 24h).

```
Netflow = Exchange Inflow − Exchange Outflow
Positive netflow → more BTC arriving on exchanges (bearish pressure)
Negative netflow → more BTC leaving exchanges (bullish pressure)
```

**Why it matters:** Coins on exchanges are available for immediate sale.
Coins moving to cold wallets / self-custody are being taken off the market.
Supply reduction on exchanges with steady or rising demand = price appreciation.

---

## Exchange Inflow Signals

**Large inflow spike (>10,000 BTC in 24h):**
- Strong bearish signal — large holders (whales) depositing coins to sell
- Historically precedes sell-offs within 12–48 hours
- Most reliable when combined with: funding rate already positive (overleveraged longs) + RSI overbought on 4H

**Sustained elevated inflow (3–5 days of above-average inflow):**
- Distribution phase — whales selling into strength
- Often seen at local tops
- Watch for: price holding up despite inflow (buy-side absorption) vs price starting to drop

**Inflow spike during a downtrend:**
- Panic selling / capitulation
- Can be a contrarian buy signal if: inflow spike is extreme (>2 standard deviations above 30-day average) + funding rate flips negative + SOPR < 1

---

## Exchange Outflow Signals

**Large outflow (>10,000 BTC in 24h):**
- Bullish signal — holders moving coins to cold storage, not planning to sell
- Supply shock in progress: fewer coins available to meet demand
- Most reliable when: price already in uptrend + funding rate neutral (not over-leveraged) + MVRV > 1

**Sustained outflow (3–7 days of consistent negative netflow):**
- Accumulation phase — strong hands absorbing supply
- Often precedes parabolic moves when combined with low exchange reserves

**Outflow during a rally:**
- "Buying and holding" behavior — very bullish
- Coins being pulled from exchanges immediately after purchase

---

## Exchange Reserve

**Definition:** Total BTC held across all tracked exchanges at a given moment.

**Trend interpretation:**
- Declining reserves (multi-month trend) → structural supply reduction → long-term bullish
- Rising reserves → supply increasing on exchanges → long-term bearish
- Exchange reserves at multi-year lows historically precede major bull runs

**Thresholds (approximate, based on historical data):**
- <2.3M BTC on exchanges → historically low, supply shock territory
- >3.0M BTC on exchanges → historically high, distribution risk elevated

---

## Signal Combination Patterns for 1–4H Swings

### Bullish setup (exchange flow component):
1. 24h netflow turns negative (outflows > inflows)
2. Exchange reserve declining over 7-day trend
3. No large inflow spikes in prior 48h
4. Funding rate neutral or negative (not overheated)
→ Supports long entry on RSI pullback to oversold on 1H

### Bearish setup (exchange flow component):
1. Large inflow spike (>2σ above 30-day average)
2. Exchange reserve rising over 3-day trend
3. Funding rate elevated positive (leveraged longs vulnerable)
4. RSI overbought on 4H
→ Supports short entry or exit of longs

---

## Timing Notes

- Exchange flow data is on-chain; it updates in near-real-time but has a
  latency of ~10–30 minutes depending on data provider (Glassnode, CryptoQuant)
- For 1–4H swings: use 24H netflow as the trend filter, not a precise entry trigger
- Entry timing should come from order book / CVD / RSI — exchange flows set the context

---

## Data Sources

- **Glassnode** — industry standard for on-chain metrics, paid API
- **CryptoQuant** — exchange-specific flows, often more granular
- **Coinglass** — exchange reserve aggregation, free tier available
- **CryptoQuant Exchange Reserve** — tracks per-exchange balances (Binance, Coinbase, Kraken etc.)
