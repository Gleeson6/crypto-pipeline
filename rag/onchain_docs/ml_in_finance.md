# Machine Learning in Quantitative Trading & Finance — Reference

Curated overview of how ML is applied (and misapplied) in quant finance, written for
ingestion into the Quant RAG knowledge base. Each section connects the concept to
practical relevance for a Bitcoin price-prediction system.

## Why ML in Finance Is Different from Typical ML
Most ML success stories (image recognition, language modeling) involve stable, abundant
patterns: a cat looks like a cat regardless of when the photo was taken. Financial markets
are different — they're **adversarial** (other participants react to and counteract
discovered patterns), **non-stationary** (relationships shift as the market evolves), and
**low signal-to-noise** (most price movement is effectively random from a prediction
standpoint). This means techniques that work brilliantly elsewhere often underperform naive
baselines in finance — a humbling but important starting expectation.

## Feature Engineering for Price Prediction
Raw price data is rarely fed directly into models. Useful feature categories include:
**price-derived** (returns, volatility measures, technical indicators like RSI/MACD),
**volume-derived** (volume trends, volume-price relationships), **on-chain** (exchange
netflow, MVRV, funding rates — see [[onchain_metrics]]), and **cross-asset / macro**
(correlations with other assets, broader market sentiment). Thoughtful feature engineering
— grounded in a plausible reason a feature *should* matter — generally outperforms throwing
raw data at a large model and hoping it finds structure.

## Model Families Commonly Used
**Tree-based models** (random forests, gradient boosting / XGBoost, LightGBM) — popular for
tabular financial features; handle non-linear relationships well, are relatively robust to
noisy features, and are easier to interpret than deep networks.
**Recurrent networks / LSTMs** — designed for sequential data; historically popular for
price-series prediction, though their advantage over simpler methods on noisy financial data
is debated in the literature.
**Transformer-based models** — increasingly applied to financial time series, leveraging
attention mechanisms; powerful but data-hungry and prone to overfitting on relatively short
financial histories.
**Simpler statistical/linear models** — often underestimated; in low signal-to-noise
environments, simpler models that don't overfit can match or beat complex ones, especially
out-of-sample.

## The Overfitting & Data Leakage Problem (Amplified in Finance)
ML models can achieve excellent-looking results by learning patterns that don't generalize
— a problem discussed generally in [[quant_trading]] and [[stats_probability]], but
especially severe in finance because historical data is limited (compared to, say, internet
text) and markets evolve. **Data leakage** — accidentally letting the model "see" information
from the future during training (e.g., normalizing features using statistics computed over
the entire dataset, including future periods) — is one of the most common and damaging
mistakes, producing models that look excellent in testing and fail immediately live.

## Evaluation: Why Standard ML Metrics Can Mislead
Accuracy or R² scores that look strong on a held-out test set can still represent a losing
trading strategy once realistic costs, slippage, and position sizing are considered. The
right evaluation translates predictions into simulated trading outcomes — incorporating
[[risk_management]] principles — rather than stopping at a abstract statistical score. A
model that's "right" 52% of the time but captures larger moves on its correct calls can be
far more valuable than one that's "right" 65% of the time on small, noisy moves.

## Reinforcement Learning for Trading
Reinforcement learning (RL) frames trading as a sequential decision problem: an agent takes
actions (buy/sell/hold), observes outcomes (P&L, market state changes), and learns a policy
that maximizes long-run reward. It's conceptually appealing for trading, but in practice
faces serious challenges: defining a reward function that captures real-world trading
objectives (including risk, not just raw return) is hard, training requires either large
amounts of historical data or risky live experimentation, and RL agents are notoriously
prone to finding degenerate strategies that exploit quirks of the simulation rather than
real market structure.

## Ensembles and Combining Signals
Rather than relying on one model, combining multiple models or signal sources — especially
ones that are individually weak but only loosely correlated with each other — often produces
more robust predictions than any single strong model. This mirrors the "signal stacking"
idea in [[quant_trading]] and reflects a broader truth in noisy domains: diversification of
information sources tends to reduce the impact of any one source's blind spots.

## Explainability and Trust
Especially when real money is involved, understanding *why* a model makes a given prediction
matters — not just whether it's statistically accurate. Simpler, more interpretable models
(or techniques that explain complex model outputs) make it easier to catch when a model has
learned something spurious, to build justified confidence before deploying capital, and to
diagnose what went wrong when performance degrades — which it eventually will, as market
regimes shift.

---

**Usage notes for the RAG:**
- Tag this document as `ml_finance_reference` / trust tier 3 (curated, conceptual
  foundation).
- Pairs with [[stats_probability]] (the statistical traps ML amplifies), [[quant_trading]]
  (translating predictions into strategies), and [[risk_management]] (the layer that
  determines whether a "good" model produces a safe trading system).
