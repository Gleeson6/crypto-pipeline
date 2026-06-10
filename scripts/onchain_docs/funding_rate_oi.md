# Funding Rate & Open Interest — Bitcoin Derivatives Signals

Derivatives market signals that reveal how leveraged the market is and
which direction traders are positioned. Among the most actionable short-term
signals for 1–4H swing trading.

---

## Funding Rate

**Definition:**
Perpetual futures contracts (the dominant Bitcoin trading instrument) have no
expiry. To keep their price anchored to the spot price, exchanges use a funding
mechanism: longs pay shorts (or shorts pay longs) every 8 hours.

```
Funding Rate > 0 → longs paying shorts → market is net long / bullish bias
Funding Rate < 0 → shorts paying longs → market is net short / bearish bias
Funding Rate = 0 → balanced positioning
```

Standard funding interval: every 8 hours (3× per day).
Typical neutral rate: 0.01% per 8h (= ~10.95% annualized).

**Interpretation:**

| Funding Rate (8h) | Condition | Signal |
|-------------------|-----------|--------|
| > +0.10% | Extreme long bias | Short squeeze risk RESOLVED, now top signal |
| +0.03% to +0.10% | Elevated long bias | Longs overextended, vulnerable to flush |
| +0.01% to +0.03% | Neutral-to-bullish | Healthy trend continuation |
| -0.01% to +0.01% | Balanced | No directional signal |
| -0.03% to -0.01% | Slightly short | Mild bearish bias |
| < -0.03% | Extreme short bias | Short squeeze setup — contrarian long |

**Key patterns:**

### Funding Rate Squeeze (most powerful pattern)
1. Funding rate goes highly negative (< -0.05%) — market aggressively short
2. Price has been falling, sentiment is bearish
3. Trigger: any positive catalyst or just time
4. Result: shorts forced to close (buy back) → price spikes rapidly
5. This is the "short squeeze" — can produce 5–15% moves in 1–4H

**Setup for long entry on short squeeze:**
- Funding rate < -0.03% for 12+ hours
- Exchange outflows (coins leaving exchanges — holders not selling)
- SOPR near 1.0 (not in capitulation yet)
- RSI oversold on 1H or 4H
→ Enter long, target: funding rate returning to neutral (0.01%)

### Overleveraged Long Flush
1. Funding rate goes highly positive (> +0.05%) — market aggressively long
2. Price has been rising, sentiment euphoric
3. Trigger: any negative catalyst or large sell order
4. Result: longs liquidated in cascade → price drops rapidly
5. Can produce -5% to -15% moves in 1–4H

**Setup for short entry / exit longs:**
- Funding rate > +0.05% for 12+ hours
- Large exchange inflows detected
- RSI overbought on 4H
- SOPR > 1.03 (profit-taking in progress)
→ Exit longs / consider short, tight stop above recent high

---

## Open Interest (OI)

**Definition:**
Total value of all outstanding futures/perpetuals contracts that have not
been settled. Represents the total amount of money currently at risk in
derivative positions.

**Interpretation:**

| OI Trend | Price Trend | Signal |
|----------|-------------|--------|
| Rising | Rising | Strong bull trend — new money entering longs |
| Rising | Falling | Strong bear trend — new money entering shorts |
| Falling | Rising | Short squeeze — shorts closing (forced or voluntary) |
| Falling | Falling | Long liquidation cascade — longs being wiped out |
| Flat | Any | Consolidation, no strong directional conviction |

**OI spike patterns:**
- **OI spike up + price spike up:** momentum continuation but watch for reversal
  when OI reaches local highs (everyone already in — who's left to buy?)
- **OI spike down (rapid):** mass liquidation event — often the end of a move
  in either direction, potential reversal point

**OI as a leverage indicator:**
```
High OI relative to market cap → market is highly leveraged → volatile
Low OI relative to market cap → market is unleveraged → trending moves more sustainable
```

---

## Liquidation Levels

**Definition:** Price levels where leveraged positions will be automatically
closed by the exchange if reached.

**Why they matter:** Liquidation clusters act as price magnets — market makers
and large players know where liquidations are clustered and will often push
price toward those levels to collect liquidity.

**Sources:** Coinglass liquidation heatmap (free) shows the density of
liquidations at each price level.

**Signal patterns:**
- Large liquidation cluster just above current price → likely target for a
  short-term pump to "hunt" those longs
- Large liquidation cluster just below current price → likely target for a
  short-term dip to "hunt" those shorts
- After a major liquidation sweep → often reversal point (liquidation fuel
  is exhausted)

---

## Combining Funding Rate + OI + Exchange Flows

**Strongest long setup:**
1. Funding rate < -0.03% (shorts overextended)
2. OI elevated (lots of shorts to squeeze)
3. Exchange outflows (not in panic selling)
4. RSI oversold on 1H
→ High-probability long, tight stop below recent low

**Strongest short setup:**
1. Funding rate > +0.05% (longs overextended)
2. OI at local high (lots of longs to flush)
3. Exchange inflows spiking
4. RSI overbought on 4H, bearish divergence
→ High-probability short or exit of longs

---

## Data Sources

- **Binance Futures** — funding rate visible in the trading interface and via API
- **Coinglass** — best free aggregator for funding rates, OI, and liquidation heatmaps
- **CryptoQuant** — funding rate + OI with historical data
- **Glassnode** — longer-term OI trends
