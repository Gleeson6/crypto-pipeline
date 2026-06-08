# Risk Management for Trading Systems — Reference

Curated overview of risk management concepts, written for ingestion into the Quant RAG
knowledge base. If [[quant_trading]] and [[ml_in_finance]] are about finding an edge, this
document is about the layer that determines whether that edge survives contact with real
markets and real capital. Many technically sound strategies fail not because the signal was
wrong, but because risk wasn't managed.

## Position Sizing
How much capital to allocate to a single trade — arguably more important to long-run results
than the entry signal itself. Sizing too large means a string of losses (which *will* happen,
even with a good strategy) can wipe out the account before the edge has a chance to play out;
sizing too small means even a strong edge produces meaningless returns. Common frameworks
include fixed-fractional sizing (risking a constant percentage of capital per trade) and
volatility-based sizing (allocating less to more volatile assets/periods, more to calmer
ones) — the latter is particularly relevant for an asset like Bitcoin where volatility itself
fluctuates substantially.

## Stop-Losses and Defined Risk
A stop-loss defines in advance the point at which a losing position will be closed, capping
the damage from any single trade. Without this, a single adverse move — especially the sharp,
fast moves common on 1-4hr BTC charts — can produce losses far larger than intended. The
specific placement matters: too tight, and normal volatility triggers exits before the trade
has a chance to work (a problem connected to [[onchain_metrics]]'s discussion of liquidation
heatmaps — price often "hunts" obvious stop levels); too loose, and the defined risk becomes
meaningless.

## Drawdown
The peak-to-trough decline in account value over some period — a measure of "how bad did it
get" rather than just "what was the average return." Two strategies with identical average
returns can have very different drawdown profiles, and large drawdowns are far harder to
recover from than they are to prevent (a 50% loss requires a 100% gain just to break even).
Evaluating a strategy by its worst historical drawdown — not just its average performance —
is essential to understanding whether it's survivable.

## Risk-Adjusted Return Measures
Raw return numbers can be misleading — a strategy that returns 40% by taking wild, erratic
risks isn't obviously "better" than one that returns 20% smoothly and predictably. Metrics
like the **Sharpe ratio** (return relative to volatility) and **Sortino ratio** (return
relative to downside volatility specifically) attempt to capture "return per unit of risk
taken," giving a fairer basis for comparing strategies that behave very differently. A
strategy's raw return number, viewed alone, tells you much less than it appears to.

## Value at Risk (VaR) and Tail Risk
VaR estimates the maximum loss expected over a given period at a given confidence level
(e.g., "95% confident the loss won't exceed X over the next day"). It's a useful framing
tool, but has a well-known weakness: it says little about what happens in the remaining 5%
of cases — exactly the fat-tailed, extreme scenarios discussed in [[stats_probability]] that
matter most for survival. Relying on VaR alone, without separately considering worst-case
"what if everything goes wrong at once" scenarios, is a common and dangerous oversight.

## Leverage
Borrowing capital to amplify position size — and, symmetrically, amplifying both gains and
losses. Leverage is one of the most common reasons technically sound strategies blow up:
it compresses the time available to react, turns ordinary volatility into existential threats,
and interacts dangerously with the kind of sharp, fast moves and liquidation cascades
described in [[onchain_metrics]]. Conservative use of leverage — or none at all while still
learning — is one of the most consistently repeated pieces of advice from experienced
practitioners, often learned the hard way.

## Diversification and Correlation Risk
Spreading risk across multiple uncorrelated positions, strategies, or assets reduces the
chance that a single bad event causes catastrophic damage. The key word is *uncorrelated* —
many assets that seem diversified (e.g., different cryptocurrencies) tend to move together
strongly during stress events, when diversification benefits are needed most. True
diversification requires understanding how correlations behave specifically during volatile,
high-stress periods — not just in calm markets.

## Operational and System Risk
Beyond market risk, automated trading systems face risks from bugs, connectivity failures,
exchange outages, API changes, and data feed errors — any of which can produce unintended
trades or missed exits at the worst possible moment. Building in safeguards (sanity checks
on order sizes, automatic kill-switches, monitoring/alerting, redundant data feeds) is not
optional polish — it's a core part of risk management for any live automated system,
arguably as important as the strategy logic itself.

## Psychological / Behavioral Risk
Even fully automated systems are built, monitored, and (often) overridden by humans —
and the temptation to intervene during a drawdown (turning off the system, manually closing
positions, "tweaking" parameters mid-stream) is one of the most common ways a sound strategy
gets undermined. Defining rules in advance for *when* and *how* a human may override the
system — and sticking to them — is itself a risk management practice, not a side issue.

---

**Usage notes for the RAG:**
- Tag this document as `risk_management_reference` / trust tier 3 (curated, conceptual
  foundation).
- This is the layer that should inform position sizing and safety logic in your actual
  trading bot — retrieve alongside [[quant_trading]] and [[ml_in_finance]] whenever
  reasoning about whether to act on a signal, not just whether the signal looks promising.
