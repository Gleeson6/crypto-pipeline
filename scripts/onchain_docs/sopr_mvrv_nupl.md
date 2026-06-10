# SOPR, MVRV, NUPL — Bitcoin On-Chain Valuation Signals

Three of the most powerful on-chain metrics for identifying market cycle
position and short-to-medium term price direction. All three are derived
from the Bitcoin UTXO (Unspent Transaction Output) set.

---

## SOPR — Spent Output Profit Ratio

**Definition:**
```
SOPR = Price at which coin was spent / Price at which coin was received
SOPR > 1 → coins being spent at a profit
SOPR < 1 → coins being spent at a loss
SOPR = 1 → coins being spent at break-even
```

**Interpretation:**

| SOPR Value | Market Condition | Signal |
|------------|-----------------|--------|
| > 1.05 | Holders taking significant profit | Bearish — distribution |
| ~1.0 (from above) | Break-even resistance level | Watch for rejection |
| ~1.0 (from below) | Break-even reclaim | Bullish confirmation |
| < 1.0 | Holders selling at a loss | Capitulation / potential bottom |
| < 0.95 | Severe loss-selling | Extreme fear, contrarian buy zone |

**Key signal patterns:**

1. **SOPR reset to 1.0:** During bull markets, pullbacks that bring SOPR back
   to 1.0 (break-even for recent movers) are historically strong buy zones —
   holders refuse to sell at a loss, creating support.

2. **SOPR > 1 consistently:** Healthy bull market — coins being moved at a
   profit, but not extreme. Flip to sustained SOPR > 1.05 signals distribution.

3. **SOPR < 1 during downtrend:** Capitulation — people panic-selling at a
   loss. When SOPR reaches multi-month lows AND volume is extreme, it often
   marks local/cycle bottoms.

**Short-Term Holder SOPR (STH-SOPR):**
- Tracks only coins moved within the last 155 days (recent buyers)
- More sensitive and faster-moving than aggregate SOPR
- STH-SOPR < 1 during a rally = recent buyers in profit, likely to hold
- STH-SOPR > 1.02 = recent buyers taking profits, potential resistance

**For 1–4H swings:** Use STH-SOPR as a daily context filter:
- STH-SOPR near 1.0 from above = potential support for long entries
- STH-SOPR > 1.03 = profit-taking environment, tighten stops on longs

---

## MVRV — Market Value to Realized Value

**Definition:**
```
Market Value = Current price × circulating supply (= market cap)
Realized Value = Average price each coin last moved × circulating supply
MVRV = Market Value / Realized Value
```

Realized Value represents the aggregate cost basis of all Bitcoin holders.
MVRV tells you how much profit (or loss) the entire market is sitting on.

**Interpretation:**

| MVRV | Market Position | Implication |
|------|----------------|-------------|
| > 3.5 | Extreme unrealized profit | Historical top zone |
| 2.0–3.5 | Strong bull market | Healthy uptrend, watch for distribution |
| 1.0–2.0 | Moderate profit / fair value | Accumulation or early bull |
| ~1.0 | At cost basis | Strong historical support |
| < 1.0 | Below cost basis | Bear market / capitulation zone |
| < 0.8 | Severe underwater | Extreme fear, historically strong buy |

**Key MVRV levels:**
- **MVRV = 1.0** → Bitcoin trades at the aggregate cost basis of all holders.
  Historically one of the strongest support levels in any market cycle.
- **MVRV > 3.5** → Each cycle top has occurred near or above this level
  (2013: ~5.8, 2017: ~4.8, 2021: ~4.0)
- **MVRV < 0.8** → Every cycle bottom has occurred at or below this level

**MVRV Z-Score:**
- Normalized version: (Market Cap − Realized Cap) / std dev of market cap
- Z-Score > 7 → historically extreme top territory
- Z-Score < 0 → historically extreme bottom territory

**For 1–4H swings:** MVRV is a slow-moving metric (changes over days/weeks).
Use it as a cycle position indicator, not an entry trigger:
- MVRV 1.0–2.0 → favorable for longs, market not overextended
- MVRV > 3.0 → tighten profit targets, reduce position size, higher reversal risk

---

## NUPL — Net Unrealized Profit/Loss

**Definition:**
```
NUPL = (Market Cap − Realized Cap) / Market Cap
     = fraction of market cap representing unrealized profit
```

Ranges from −1 to +1. Positive = aggregate profit; negative = aggregate loss.

**Interpretation bands:**

| NUPL | Zone | Color (Glassnode) | Signal |
|------|------|-------------------|--------|
| > 0.75 | Euphoria / Greed | Red | Top zone — extreme caution |
| 0.5–0.75 | Belief / Denial | Orange | Late bull market |
| 0.25–0.5 | Optimism / Anxiety | Yellow | Mid bull market |
| 0–0.25 | Hope / Fear | Green | Early bull / recovery |
| < 0 | Capitulation | Blue | Bottom zone — accumulation |

**Key signal:**
- NUPL crossing from negative to positive → market recovering from capitulation,
  historically a strong long-term buy signal
- NUPL > 0.75 → distribution territory, reduce exposure
- NUPL negative for extended period → bear market bottom, accumulate

**For 1–4H swings:** Like MVRV, use NUPL as a cycle position filter:
- NUPL in "Hope/Fear" or "Optimism" bands → favorable for longs
- NUPL in "Euphoria" → take profits aggressively, don't add to longs

---

## Using All Three Together

The three metrics are complementary and confirm each other:

**Bottom signal (all three agree):**
- SOPR < 0.95 (capitulation selling)
- MVRV < 0.9 (below aggregate cost basis)
- NUPL < 0 (aggregate loss territory)
→ Strongest possible accumulation signal. Rare but historically very reliable.

**Top signal (all three agree):**
- SOPR consistently > 1.05 (heavy profit-taking)
- MVRV > 3.5 (extreme unrealized profit)
- NUPL > 0.75 (euphoria)
→ Historical cycle top territory. Reduce exposure significantly.

**Healthy bull market (mid-cycle):**
- SOPR oscillating around 1.0 with brief dips as buying signals
- MVRV 1.5–2.5
- NUPL 0.25–0.5
→ Continue trend-following strategy, normal stop-losses apply.
