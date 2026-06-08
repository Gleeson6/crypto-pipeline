# Quantitative Trading — Core Concepts Reference

A curated overview of quantitative trading fundamentals, written for ingestion into the
Quant RAG knowledge base. Each section is a self-contained concept with a relevance note
for short-to-mid-term (1-4hr) crypto swing strategies.

## What Quantitative Trading Is
Quantitative trading uses mathematical models, statistical analysis, and historical data
to identify and execute trading opportunities — replacing discretionary "gut feel" decisions
with rules that can be tested, measured, and refined. The core loop is: form a hypothesis →
encode it as a rule or model → test it on historical data (backtest) → validate on unseen
data (out-of-sample / paper trading) → deploy with risk controls → monitor and iterate.

## Strategy Families
**Trend-following / momentum** — bets that assets moving in a direction will continue
(e.g., moving average crossovers, breakout systems). Tends to perform well in sustained
directional markets, poorly in choppy/range-bound conditions.
**Mean-reversion** — bets that prices that have moved far from a statistical "average" will
revert toward it (e.g., RSI extremes, Bollinger Band touches). Tends to perform well in
range-bound markets, poorly during strong trends or regime breaks.
**Arbitrage / relative value** — exploits price discrepancies between related instruments
or venues (e.g., cross-exchange spreads, spot-futures basis). Requires speed and low costs;
less reliant on directional forecasting.
**Market making** — profits from the bid-ask spread by continuously quoting both sides.
Requires infrastructure and tight risk controls; not typically a retail starting point.

## Signal Generation
A "signal" is any measurable input that suggests a trading opportunity — a technical
indicator crossing a threshold, an on-chain metric spiking, a sentiment shift, etc. Good
signals share three traits: they have a plausible causal or structural reason to work (not
just a historical coincidence), they remain stable across different time periods and market
regimes, and they're cheap enough to act on that transaction costs don't erase the edge.
Combining multiple weak, loosely-correlated signals ("signal stacking") tends to produce
more robust systems than relying on one strong signal alone.

## Backtesting — and Its Traps
A backtest simulates how a strategy would have performed on historical data. It's essential
for validating ideas before risking capital, but it's also where most strategies quietly
fail before they ever go live. Key traps: **overfitting** (tuning a strategy until it fits
historical noise rather than real structure — it will look great in the backtest and fail
live), **look-ahead bias** (accidentally using information that wouldn't have been available
at the time, e.g., using a daily close price to make an intraday decision), **survivorship
bias** (testing only on assets that still exist today, ignoring those that failed or were
delisted), and **ignoring costs** (fees, slippage, and spread can turn a "profitable" backtest
into a losing live strategy). A strategy that looks mediocre after realistic costs and
walk-forward testing is far more trustworthy than one that looks spectacular in a naive
backtest.

## Walk-Forward / Out-of-Sample Testing
Rather than testing a strategy once on all available history, walk-forward testing splits
data into sequential windows: optimize on one window, test on the next unseen window, then
roll forward. This better simulates how a strategy behaves in real deployment — where it
must perform on data it has never "seen" — and is one of the strongest defenses against
overfitting.

## Execution Considerations
Even a statistically sound signal can lose money if execution is poor. Relevant factors:
**slippage** (the difference between expected and actual fill price, worse in fast-moving or
thin markets — directly relevant to the sharp wick-like moves on 1-4hr BTC charts),
**latency** (the delay between signal generation and order placement — matters more for
high-frequency strategies than swing strategies, but still shapes entry/exit quality), and
**order types** (market orders guarantee execution but not price; limit orders guarantee
price but not execution — the choice affects both cost and the chance of missing a move).

## Market Regimes
Markets cycle through different "regimes" — trending, ranging, high-volatility,
low-volatility — and a strategy that thrives in one regime often struggles in another.
Recognizing which regime is currently active (e.g., via volatility measures or trend-strength
indicators) and adapting strategy choice or position sizing accordingly is a hallmark of
mature quant systems, as opposed to a single static rule applied blindly at all times.

## Edge, Expectancy, and Why Win Rate Isn't Everything
A strategy's "edge" is its statistical advantage over random chance — it doesn't require a
high win rate to be profitable. **Expectancy** (average win size × win rate − average loss
size × loss rate) is the more meaningful measure: a strategy that wins only 35% of the time
but captures large moves while cutting losses quickly can vastly outperform one that wins 70%
of the time but lets losses run. This reframes "being right" as far less important than
"managing the consequences of being wrong."

---

**Usage notes for the RAG:**
- Tag this document as `quant_trading_reference` / trust tier 3 (curated, conceptual
  foundation — not a specific empirical claim).
- Pairs naturally with [[onchain_metrics]] (signal sources), [[risk_management]] (translating
  edge into safe position sizing), and [[ml_in_finance]] (when signals are learned rather than
  hand-specified).
