# On-Chain & Market Metric Reference (Bitcoin)

Curated definitions of on-chain and market-structure metrics relevant to short-to-mid-term
(1-4hr swing) Bitcoin price prediction. Written for ingestion into the Quant RAG knowledge
base — each section is a self-contained definition + relevance note.

## Exchange Netflow
The net amount of BTC moving onto vs. off of exchanges (inflow minus outflow) over a window.
Large inflows often precede selling pressure (holders moving coins to sell); large outflows
often suggest accumulation (coins moving to cold storage / long-term holding). Spikes in
netflow — positive or negative — frequently cluster around volatile price moves, making this
one of the more directly actionable on-chain signals for swing timeframes.

## MVRV (Market Value to Realized Value) Ratio
Compares the current market cap to the "realized cap" (the aggregate value of all coins priced
at the time they last moved). A high MVRV suggests the market is trading well above the
aggregate cost basis of holders (potential overvaluation / profit-taking risk); a low or
negative MVRV suggests the market is trading near or below aggregate cost basis (potential
undervaluation). MVRV is more of a macro/regime indicator than a short-term timing tool, but
extreme readings can mark turning points that ripple into shorter-timeframe volatility.

## Realized Cap
The sum of the value of every coin in circulation, each valued at the price when it last moved
on-chain (rather than current market price). Often used as a "cost basis" proxy for the
network and as the denominator for MVRV. Rising realized cap suggests new capital entering at
higher prices; a flattening or falling realized cap can indicate capital exiting.

## Funding Rates (Perpetual Futures)
The periodic payment exchanged between long and short positions on perpetual futures contracts,
designed to keep the contract price anchored to spot. Positive funding means longs pay shorts
(market is leaning bullish/over-leveraged long); negative funding means shorts pay longs.
Extreme funding rates — in either direction — often precede "squeezes" (rapid liquidation
cascades that produce sharp, fast price moves), which is directly relevant to spotting
the "huge dips and ups" pattern on short timeframes.

## Open Interest
The total number of outstanding derivative contracts (futures/perpetuals) that have not been
settled. Rising open interest alongside rising price can confirm a trend (new money entering);
rising open interest alongside falling price can signal building short pressure. Sharp drops in
open interest often coincide with liquidation events — another marker of the volatility
clusters worth studying.

## Active Addresses
The count of unique wallet addresses that transacted on-chain in a given period. A rising trend
suggests growing network usage/adoption; sudden spikes can indicate unusual activity (e.g.,
exchange-related batch transactions, airdrops, or coordinated movement). Used more as a
medium-term health gauge than a short-term trading trigger, but sudden anomalies are worth
flagging.

## Hash Rate
The total computational power securing the Bitcoin network (miners). Hash rate trends reflect
miner confidence and infrastructure investment. Sharp hash rate drops (e.g., regional mining
bans, energy crises) can correlate with miner capitulation — periods where miners sell holdings
to cover costs, adding sell-side pressure. A slower-moving signal, but useful context for why a
larger move might be underway.

## Miner Reserves / Miner Outflows
The amount of BTC held in known miner wallets, and the rate at which it moves out (typically to
exchanges to be sold). Large miner outflows can add identifiable sell pressure, particularly
around halving events or periods of compressed mining margins. Useful as a corroborating signal
when combined with exchange netflow data.

## Stablecoin Supply (USDT/USDC etc.)
The total circulating supply of major stablecoins, and the rate of new issuance. Rising supply
is often interpreted as "dry powder" entering the crypto ecosystem, frequently preceding buying
pressure into BTC and other majors; large redemptions can signal capital leaving the ecosystem.
Issuance/redemption events are discrete, news-driven, and can produce outsized short-term moves
— directly relevant to the "huge dips and ups" the swing strategy is trying to anticipate.

## Realized Volatility vs. Implied Volatility
Realized volatility measures how much price actually moved over a recent window; implied
volatility (derived from options pricing) reflects what the market expects going forward. A
large gap between the two — especially implied running far above realized — often signals the
market is pricing in an expected large move, which can itself become a self-fulfilling
contributor to volatility clustering on short timeframes.

## Liquidation Levels / Liquidation Heatmaps
Estimated price levels where large clusters of leveraged positions would be forcibly closed.
Price often gravitates toward zones with dense liquidation clusters ("liquidity hunts"),
producing the sharp wick-like dips and spikes commonly observed on 1-4hr charts. This is one of
the more directly relevant concepts for explaining sudden, sharp moves that don't appear to be
driven by news.

---

**Usage notes for the RAG:**
- These are *definitions and mechanisms*, not live values — pair this reference layer with a
  live data feed (Glassnode, CryptoQuant, exchange APIs) that supplies current readings.
- Tag this document as `onchain_reference` / trust tier 3 (curated, factual) — distinct from
  blog/news content which should be tagged at a lower trust tier.
- When reasoning about a specific price move, the model should retrieve both (a) the relevant
  metric definition from this doc and (b) the corresponding live metric value from your data
  pipeline, then connect the two.
