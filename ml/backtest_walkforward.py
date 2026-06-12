"""
Walk-Forward Backtest — month-by-month breakdown on held-out test window.

Loads the ALREADY TRAINED model and tests it on each calendar month
of the test window separately. Shows whether the edge is consistent
across different market conditions or concentrated in a lucky period.

Usage:
    python3 backtest_walkforward.py
    python3 backtest_walkforward.py --threshold 0.57
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

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEE        = 0.001   # 0.10% round trip


def load_latest(pattern):
    files = sorted(glob.glob(os.path.join(MODELS_DIR, pattern)))
    if not files:
        print(f"No model found: {pattern}. Run train_model.py first.")
        sys.exit(1)
    path = files[-1]
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Loaded: {os.path.basename(path)}")
    return model


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


def run_month(month_df, clf, meta_model, feature_cols, threshold):
    """Run backtest on a single month slice. Returns dict of metrics."""
    data = month_df.dropna(subset=["target_tb_direction"]).copy()
    data = data[data["target_tb_direction"] != 0].reset_index(drop=True)

    if len(data) < 5:
        return None

    X        = data[feature_cols]
    m1_pred  = clf.predict(X)
    m1_proba = clf.predict_proba(X)[:, 1]

    meta_X                   = X.copy()
    meta_X["model1_prob_up"] = m1_proba
    meta_cols  = feature_cols + ["model1_prob_up"]
    meta_proba = meta_model.predict_proba(meta_X[meta_cols])[:, 1]

    take   = (m1_pred == 1) & (meta_proba >= threshold)
    actual = data["target_tb_direction"].values

    returns = []
    for i, trade in enumerate(take):
        if not trade:
            continue
        ret = (UPPER_BARRIER if actual[i] == 1 else -LOWER_BARRIER) - FEE
        returns.append(ret)

    if len(returns) == 0:
        return {
            "trades": 0, "win_rate": 0.0, "avg_ret": 0.0,
            "total": 0.0, "profit_factor": 0.0, "max_dd": 0.0,
        }

    rets     = np.array(returns)
    wins     = (rets > 0).sum()
    win_rate = wins / len(rets)
    equity   = np.cumprod(1 + rets)
    max_dd   = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    gross_w  = rets[rets > 0].sum()
    gross_l  = abs(rets[rets < 0].sum())
    pf       = gross_w / gross_l if gross_l > 0 else float("inf")

    return {
        "trades":        len(rets),
        "win_rate":      win_rate,
        "avg_ret":       float(rets.mean()),
        "total":         float(rets.sum()),
        "profit_factor": pf,
        "max_dd":        max_dd,
    }


def print_monthly(results: list, threshold: float):
    net_win  = UPPER_BARRIER - FEE
    net_loss = LOWER_BARRIER + FEE
    be_rate  = net_loss / (net_win + net_loss)

    print("\n" + "=" * 85)
    print(f"  WALK-FORWARD RESULTS  (threshold={threshold}  |  fee={FEE*100:.2f}%)")
    print(f"  Break-even win rate after fees: {be_rate:.1%}")
    print("=" * 85)
    print(f"  {'Month':<12}  {'Trades':>7}  {'Win Rate':>10}  {'Avg Ret':>10}  {'Total':>10}  {'PF':>6}  {'MaxDD':>8}  {'':>3}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*3}")

    totals = {"trades": 0, "total": 0.0, "wins": 0, "gross_w": 0.0, "gross_l": 0.0}

    for r in results:
        month   = r["month"]
        m       = r["metrics"]
        if m is None:
            print(f"  {month:<12}  {'—':>7}  {'—':>10}  {'—':>10}  {'—':>10}  {'—':>6}  {'—':>8}  (no data)")
            continue

        flag = "✅" if m["win_rate"] >= be_rate and m["trades"] > 0 else ("⚠️ " if m["trades"] == 0 else "❌")
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "∞"

        print(
            f"  {month:<12}  {m['trades']:>7,}  {m['win_rate']:>10.1%}  "
            f"{m['avg_ret']*100:>+10.3f}%  {m['total']*100:>+10.2f}%  "
            f"{pf_str:>6}  {m['max_dd']*100:>7.2f}%  {flag}"
        )

        totals["trades"] += m["trades"]
        totals["total"]  += m["total"]
        if m["win_rate"] >= be_rate and m["trades"] > 0:
            totals["wins"] += 1

    print(f"  {'-'*12}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*3}")
    n_months = sum(1 for r in results if r["metrics"] is not None and r["metrics"]["trades"] > 0)
    avg_tr   = totals["trades"] / n_months if n_months else 0
    print(
        f"  {'TOTAL':<12}  {totals['trades']:>7,}  {'':>10}  {'':>10}  "
        f"{totals['total']*100:>+10.2f}%  {'':>6}  {'':>8}  "
        f"{totals['wins']}/{n_months} months profitable"
    )
    print(f"  Avg trades/month: {avg_tr:.0f}")


def equity_curve_combined(results: list):
    """Print one equity curve across all months."""
    all_rets = []
    for r in results:
        m = r["metrics"]
        if m and m["trades"] > 0:
            # Reconstruct trade-level returns for equity curve
            # (we stored only aggregate — use avg_ret × n as approximation)
            pass
    # Skip detailed equity here — monthly table is more informative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",     default=DB_PATH)
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    args = parser.parse_args()

    print("Walk-Forward Backtest — Month-by-Month  [fee-adjusted]")
    print(f"Threshold: {args.threshold}  |  Fee: {FEE*100:.2f}% per trade\n")

    print("Loading models...")
    clf        = load_latest("clf_direction_*.pkl")
    meta_model = load_latest("meta_model_*.pkl")

    print("\nLoading data...")
    df, test_df = load_test_data(args.db_path, args.train_ratio, args.val_ratio)
    feature_cols = [c for c in get_feature_cols(df) if c in test_df.columns]

    # Attach datetime index for month slicing
    test_df["_dt"] = pd.to_datetime(test_df["open_time"], unit="ms")
    test_df        = test_df.set_index("_dt")

    months  = test_df.index.to_period("M").unique()
    results = []

    print(f"\nRunning month-by-month simulation ({len(months)} months)...")
    for period in months:
        month_df = test_df[test_df.index.to_period("M") == period].copy().reset_index(drop=True)
        metrics  = run_month(month_df, clf, meta_model, feature_cols, args.threshold)
        results.append({"month": str(period), "metrics": metrics})
        status = f"{metrics['trades']} trades, {metrics['win_rate']:.1%} WR" if metrics and metrics["trades"] > 0 else "no trades"
        print(f"  {period}: {status}")

    print_monthly(results, args.threshold)

    print(f"""
─────────────────────────────────────────────────────────
How to read this:
  ✅  Month beat the {(LOWER_BARRIER+FEE)/(UPPER_BARRIER+LOWER_BARRIER):.0%} break-even — edge present
  ❌  Month below break-even — model struggled
  ⚠️   No trades taken (threshold too high for that month)

Consistent ✅ across months = robust edge.
Mixed results = regime-dependent edge (requires regime filter).
─────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
