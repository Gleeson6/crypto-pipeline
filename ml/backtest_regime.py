"""
Regime Backtest — tests whether the model's edge holds across different
market conditions: bull, bear, and sideways.

Regime classification (on 1H close prices from klines table):
  Bull     — close > SMA50  AND  SMA50 > SMA200  (trending up)
  Bear     — close < SMA50  AND  SMA50 < SMA200  (trending down)
  Sideways — everything in between (choppy / ranging)

  SMA50  = 50-bar rolling mean  ≈ 2 days on 1H data
  SMA200 = 200-bar rolling mean ≈ 8 days on 1H data

Why this matters:
  A model that profits only in bull markets is not an edge — it's
  just buying and holding. A real edge works across all regimes,
  or at minimum doesn't blow up in any single one.

Usage:
    python3 backtest_regime.py
    python3 backtest_regime.py --threshold 0.57
"""

import argparse
import glob
import os
import pickle
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH
from train_model import get_feature_cols
from compute_features import UPPER_BARRIER, LOWER_BARRIER

MODELS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEE              = 0.001
DEFAULT_SLIPPAGE = 0.0003
BE_LONG          = (LOWER_BARRIER + FEE + DEFAULT_SLIPPAGE) / (UPPER_BARRIER + LOWER_BARRIER)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_latest(pattern):
    files = sorted(glob.glob(os.path.join(MODELS_DIR, pattern)))
    if not files:
        print(f"No model found: {pattern}. Run train_model.py first.")
        sys.exit(1)
    with open(files[-1], "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded: {os.path.basename(files[-1])}")
    return model


def load_klines_close(db_path):
    """Load open_time + close from klines for regime classification."""
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute(
        "SELECT open_time, close FROM klines ORDER BY open_time"
    ).df()
    con.close()
    return df


def load_test_data(db_path, train_ratio=0.70, val_ratio=0.10):
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute("SELECT * FROM ml_features_clean ORDER BY open_time").df()
    con.close()
    n          = len(df)
    test_start = int(n * (train_ratio + val_ratio))
    test_df    = df.iloc[test_start:].copy().reset_index(drop=True)
    t0 = pd.to_datetime(test_df["open_time"].iloc[0],  unit="ms")
    t1 = pd.to_datetime(test_df["open_time"].iloc[-1], unit="ms")
    print(f"  Test window: {t0.date()} → {t1.date()}  ({len(test_df):,} rows)")
    return df, test_df


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regimes(klines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SMA50 and SMA200 on close prices, then classify each bar.

    Returns klines_df with added columns: sma50, sma200, regime.
    """
    df = klines_df.copy().sort_values("open_time").reset_index(drop=True)
    df["sma50"]  = df["close"].rolling(50,  min_periods=50).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=200).mean()

    def _regime(row):
        if pd.isna(row["sma200"]):
            return "unknown"
        if row["close"] > row["sma50"] and row["sma50"] > row["sma200"]:
            return "bull"
        elif row["close"] < row["sma50"] and row["sma50"] < row["sma200"]:
            return "bear"
        else:
            return "sideways"

    df["regime"] = df.apply(_regime, axis=1)
    return df[["open_time", "close", "sma50", "sma200", "regime"]]


def print_regime_distribution(regime_col: pd.Series, label="Full dataset"):
    counts = regime_col.value_counts()
    total  = len(regime_col)
    print(f"\n  Regime distribution ({label}):")
    for r in ["bull", "bear", "sideways", "unknown"]:
        n   = counts.get(r, 0)
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {r:<10} {n:>5,}  ({pct:>5.1f}%)  {bar}")


# ── Predictions ───────────────────────────────────────────────────────────────

def get_predictions(data, clf, meta_model, feature_cols):
    X        = data[feature_cols]
    m1_pred  = clf.predict(X)
    m1_proba = clf.predict_proba(X)[:, 1]
    meta_X                   = X.copy()
    meta_X["model1_prob_up"] = m1_proba
    meta_cols  = feature_cols + ["model1_prob_up"]
    meta_proba = meta_model.predict_proba(meta_X[meta_cols])[:, 1]
    return m1_pred, m1_proba, meta_proba


# ── Per-regime backtest ───────────────────────────────────────────────────────

def run_regime(data_slice, m1_pred_slice, meta_proba_slice, threshold,
               slippage=DEFAULT_SLIPPAGE):
    """Run the long-only backtest on a single regime slice."""
    actual  = data_slice["target_tb_direction"].values
    take    = (m1_pred_slice == 1) & (meta_proba_slice >= threshold)
    returns = []

    for i, t in enumerate(take):
        if not t:
            continue
        gross = UPPER_BARRIER if actual[i] == 1 else -LOWER_BARRIER
        returns.append(gross - FEE - slippage)

    return np.array(returns)


def regime_metrics(returns: np.ndarray, n_decisive: int, n_signals: int) -> dict:
    if len(returns) == 0:
        return {
            "trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
            "total": 0.0, "pf": 0.0, "max_dd": 0.0,
            "n_decisive": n_decisive, "n_signals": n_signals,
        }
    wins   = (returns > 0).sum()
    equity = np.cumprod(1 + returns)
    max_dd = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    gw     = returns[returns > 0].sum() if (returns > 0).any() else 0.0
    gl     = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    return {
        "trades":      len(returns),
        "win_rate":    wins / len(returns),
        "avg_ret":     float(returns.mean()),
        "total":       float(returns.sum()),
        "pf":          gw / gl,
        "max_dd":      max_dd,
        "n_decisive":  n_decisive,
        "n_signals":   n_signals,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_regime_results(regime_data: dict, threshold: float):
    print("\n" + "=" * 88)
    print(f"  REGIME BACKTEST  (threshold={threshold}  |  fee={FEE*100:.2f}%  |  slip={DEFAULT_SLIPPAGE*100:.2f}%)")
    print(f"  Break-even win rate (long, after all costs): {BE_LONG:.1%}")
    print("=" * 88)
    print(f"  {'Regime':<12}  {'Bars':>6}  {'Decisive':>9}  {'Signals':>8}  "
          f"{'Trades':>7}  {'Win Rate':>10}  {'Avg Ret':>9}  {'Total':>9}  {'PF':>5}  {'MaxDD':>7}  ")
    print(f"  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*5}  {'-'*7}")

    regime_order = ["bull", "bear", "sideways"]
    totals = {"trades": 0, "wins": 0, "total": 0.0}

    for regime in regime_order:
        if regime not in regime_data:
            continue
        m  = regime_data[regime]["metrics"]
        nb = regime_data[regime]["n_bars"]

        if m["trades"] == 0:
            flag = "⚠️ "
            print(f"  {regime:<12}  {nb:>6,}  {m['n_decisive']:>9,}  {m['n_signals']:>8,}  "
                  f"{'—':>7}  {'—':>10}  {'—':>9}  {'—':>9}  {'—':>5}  {'—':>7}  {flag} no trades")
            continue

        flag = "✅" if m["win_rate"] >= BE_LONG else "❌"
        pf_s = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "∞"

        print(f"  {regime:<12}  {nb:>6,}  {m['n_decisive']:>9,}  {m['n_signals']:>8,}  "
              f"{m['trades']:>7,}  {m['win_rate']:>10.1%}  {m['avg_ret']*100:>+9.3f}%  "
              f"{m['total']*100:>+9.2f}%  {pf_s:>5}  {m['max_dd']*100:>6.2f}%  {flag}")

        totals["trades"] += m["trades"]
        totals["wins"]   += int(m["win_rate"] * m["trades"])
        totals["total"]  += m["total"]

    overall_wr = totals["wins"] / totals["trades"] if totals["trades"] else 0
    print(f"  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*5}  {'-'*7}")
    print(f"  {'ALL':<12}  {'':>6}  {'':>9}  {'':>8}  "
          f"{totals['trades']:>7,}  {overall_wr:>10.1%}  {'':>9}  "
          f"{totals['total']*100:>+9.2f}%")

    print(f"""
  How to read:
    ✅  Regime beat {BE_LONG:.1%} break-even — edge present in this market condition
    ❌  Regime below break-even — model struggles here
    ⚠️   Too few trades to judge (< 10 trades)

  Ideal: consistent ✅ across all regimes.
  If only bull ✅ → strategy is just buy-and-hold dressed up.
  If only bear  ✅ → model is a short bias, not a real ML edge.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",     default=DB_PATH)
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    args = parser.parse_args()

    print("Regime Backtest — Bull / Bear / Sideways  [fee + slippage adjusted]")
    print(f"Threshold: {args.threshold}  |  Fee: {FEE*100:.2f}%  |  Slippage: {DEFAULT_SLIPPAGE*100:.2f}%\n")

    print("Loading models...")
    clf        = load_latest("clf_direction_*.pkl")
    meta_model = load_latest("meta_model_*.pkl")

    print("\nLoading data...")
    full_df, test_df = load_test_data(args.db_path, args.train_ratio, args.val_ratio)
    feature_cols     = [c for c in get_feature_cols(full_df) if c in test_df.columns]

    print("\nLoading klines for regime classification...")
    klines_df = load_klines_close(args.db_path)
    print(f"  Klines loaded: {len(klines_df):,} bars")

    print("\nClassifying market regimes...")
    regime_df = classify_regimes(klines_df)

    # Merge regime labels into test_df
    test_df = test_df.merge(
        regime_df[["open_time", "regime"]],
        on="open_time", how="left"
    )
    test_df["regime"] = test_df["regime"].fillna("unknown")

    # Filter to decisive rows (same as all other backtests)
    data = test_df.dropna(subset=["target_tb_direction"]).copy()
    data = data[data["target_tb_direction"] != 0].reset_index(drop=True)

    print_regime_distribution(data["regime"], label="test decisive rows")

    print("\nComputing predictions...")
    m1_pred, m1_proba, meta_proba = get_predictions(data, clf, meta_model, feature_cols)
    data["_m1_pred"]    = m1_pred
    data["_meta_proba"] = meta_proba

    # Run backtest per regime
    print("\nRunning per-regime simulation...")
    regime_data = {}
    for regime in ["bull", "bear", "sideways", "unknown"]:
        mask = data["regime"] == regime
        if mask.sum() == 0:
            continue

        slice_df   = data[mask].reset_index(drop=True)
        mp_slice   = data["_m1_pred"].values[mask]
        meta_slice = data["_meta_proba"].values[mask]

        n_signals  = int(((mp_slice == 1) & (meta_slice >= args.threshold)).sum())
        returns    = run_regime(slice_df, mp_slice, meta_slice, args.threshold)
        metrics    = regime_metrics(returns, int(mask.sum()), n_signals)

        n_test_bars = int((test_df["regime"] == regime).sum())
        regime_data[regime] = {"metrics": metrics, "n_bars": n_test_bars}

        status = f"{metrics['trades']} trades, {metrics['win_rate']:.1%} WR" \
                 if metrics["trades"] > 0 else "no trades"
        print(f"  {regime:<10}: {mask.sum():,} decisive rows → {status}")

    print_regime_results(regime_data, args.threshold)


if __name__ == "__main__":
    main()
