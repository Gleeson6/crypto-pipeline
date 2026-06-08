# Statistics & Probability for Quantitative Finance — Reference

Curated overview of the statistical and probabilistic concepts that underpin quant
finance, written for ingestion into the Quant RAG knowledge base. Each section connects
the concept to why it matters when reasoning about Bitcoin price behavior.

## Distributions and "Fat Tails"
A probability distribution describes how likely different outcomes are. Many introductory
models assume returns follow a "normal" (bell-curve) distribution, but real financial — and
especially crypto — returns exhibit **fat tails**: extreme moves (large dips and spikes)
happen far more often than a normal distribution would predict. This is directly relevant
to the "huge dips and ups" pattern: the market is structurally prone to producing outsized
moves, and models that assume mild, normal-ish behavior will systematically underestimate
real risk.

## Stationarity
A time series is "stationary" if its statistical properties (mean, variance, autocorrelation)
stay stable over time. Raw price series are typically non-stationary (their average level
drifts), which can make naive statistical comparisons misleading. Returns (percentage changes)
are usually closer to stationary and are the more common basis for modeling. Checking whether
a relationship holds across different time periods — not just in one favorable window — is a
direct application of this idea.

## Correlation vs. Causation
Two variables can move together (correlate) without one causing the other — both might be
driven by a third factor, or the relationship might be coincidental and likely to break down.
This is one of the most important guardrails in quant work: a metric that "predicted" past
moves might simply have been correlated with the true driver, and will fail when conditions
change. Always ask *why* a relationship should exist, not just *whether* it appeared in the
data.

## Autocorrelation and Volatility Clustering
Autocorrelation measures whether a series is related to its own past values. In crypto
markets, returns themselves show little autocorrelation (past price changes don't reliably
predict the direction of the next change), but **volatility** shows strong autocorrelation —
large moves tend to be followed by more large moves, and calm periods by more calm periods.
This "volatility clustering" is one of the most robust empirical patterns in financial data
and is directly relevant to anticipating when a 1-4hr swing window is more likely to produce
a significant move.

## Mean and Variance — and Their Limits
The mean (average) and variance (spread) summarize a distribution's center and dispersion,
and underlie many risk metrics (e.g., volatility is the standard deviation of returns).
However, in fat-tailed, regime-shifting markets like crypto, these summary statistics can be
unstable — a "typical" volatility measured over the last month may say little about what's
coming next. Treat them as useful but time-bound snapshots, not fixed truths.

## Hypothesis Testing and Statistical Significance
A hypothesis test asks: "if there were truly no real effect, how likely is it that we'd see
a pattern this strong just by chance?" A small probability (commonly framed via a "p-value")
suggests the pattern is unlikely to be pure coincidence — but it does *not* prove the pattern
is large, stable, or tradeable after costs. With enough data and enough strategies tested,
some will appear "significant" purely by chance (multiple-testing / data-dredging risk) —
a major reason backtested strategies often disappoint in live trading.

## Sampling, Sample Size, and Survivorship
Conclusions drawn from small samples (e.g., "this pattern worked the last 5 times") are
fragile — randomness alone can produce short streaks. Larger, more diverse samples — across
different time periods, volatility regimes, and market conditions — produce more trustworthy
conclusions. Relatedly, only studying assets, strategies, or time periods that "survived" or
looked good in hindsight (survivorship bias) systematically inflates how good a pattern looks.

## Expected Value and Probabilistic Thinking
Rather than asking "will this trade win?", probabilistic thinking asks "across many similar
situations, what's the average outcome, weighted by how likely each result is?" This
reframing — from individual outcomes to long-run distributions of outcomes — is the
foundation of expectancy-based strategy evaluation (see [[quant_trading]]) and of rational
position sizing (see [[risk_management]]). It also helps emotionally: a probabilistic
framing treats any single loss as an expected, normal part of a process rather than a
personal failure.

## Regression and Overfitting
Regression-style techniques fit a relationship between inputs and outcomes. With enough
free parameters, any model can be made to fit historical data almost perfectly — but a model
that has "memorized" noise rather than learned real structure will perform poorly on new
data. This is the statistical root of the overfitting problem discussed in [[quant_trading]]
and [[ml_in_finance]] — simpler models that generalize tend to beat complex models that
merely fit the past.

---

**Usage notes for the RAG:**
- Tag this document as `stats_probability_reference` / trust tier 3 (curated, conceptual
  foundation).
- This document explains *why* many of the empirical patterns in [[onchain_metrics]] and
  [[quant_trading]] behave the way they do — retrieve alongside those when reasoning about
  *why* a pattern might or might not be trustworthy.
