"""
Position Sizing Backtest — compares four sizing strategies on the long-only signals.

Strategies:
  1. flat       — same capital every trade (baseline)
  2. kelly      — full Kelly Criterion (mathematically optimal but aggressive)
  3. half_kelly — Kelly × 0.5 (industry standard, conservative)
  4. meta_prob  — position size = meta_proba score (model's own confidence)

Kelly Criterion:
  f* = (p × R - (1-p)) / R
  where p = win rate estimate, R = win/loss ratio (UPPER / LOWER = 2.0)

  With p=0.60, R=2.0:  f* = (0.60×2 - 0.40) / 2 = 0.40  → bet 40% of capital
  Half Kelly           = 0.40 × 0.5 = 0.20                → bet 20% of capital

Meta-prob sizing:
  position_size_i = meta_proba_i  (between 0.55 and 1.0 given threshold=0.55)
  Naturally scales up size on high-confidence signals.

Usage:
    python3 backtest_sizing.py
    python3 backtest_sizing.py --threshold 0.57
    python3 backtest_sizing.py --kelly-p 0.63   # use a different win rate estimate
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH
from train_model import get_feature_cols
from compute_features import UPPER_BARRIER, LOWER_BARRIER

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEE              = 0.001   # 0.10% round trip
DEFAULT_SLIPPAGE = 0.0003  # 0.03% per trade
RR_RATIO         = UPPER_BARRIER / LOWER_BARRIER  # 2.0


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


def get_predictions(data, clf, meta_model, feature_cols):
    X        = data[feature_cols]
    m1_pred  = clf.predict(X)
    m1_proba = clf.predict_proba(X)[:, 1]
    meta_X                   = X.copy()
    meta_X["model1_prob_up"] = m1_proba
    meta_cols  = feature_cols + ["model1_prob_up"]
    meta_proba = meta_model.predict_proba(meta_X[meta_cols])[:, 1]
    return m1_pred, m1_proba, meta_proba


# ── Kelly ─────────────────────────────────────────────────────────────────────

def kelly_fraction(p: float, rr: float = RR_RATIO) -> float:
    """
    Full Kelly fraction.
    f* = (p × R - (1-p)) / R
    Clamped to [0, 1].
    """
    f = (p * rr - (1 - p)) / rr
    return float(np.clip(f, 0.0, 1.0))


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(data, m1_pred, meta_proba, threshold,
             sizing="flat", kelly_p=0.60, slippage=DEFAULT_SLIPPAGE):
    """
    Run the long-only simulation with a given sizing strategy.
    Returns a list of (position_size, gross_return) tuples for taken trades.
    """
    actual = data["target_tb_direction"].values
    take   = (m1_pred == 1) & (meta_proba >= threshold)

    kf = kelly_fraction(kelly_p)

    raw_wins  = []
    raw_loss  = []
    records   = []

    for i, t in enumerate(take):
        if not t:
            continue

        win   = actual[i] == 1
        gross = UPPER_BARRIER if win else -LOWER_BARRIER
        net   = gross - FEE - slippage   # cost-adjusted return at 100% position

        mp = float(meta_proba[i])

        if sizing == "flat":
            f = 1.0
        elif sizing == "kelly":
            f = kf
        elif sizing == "half_kelly":
            f = kf * 0.5
        elif sizing == "meta_prob":
            # Scale meta_proba to a position fraction.
            # At threshold=0.55 → min size; at 1.0 → max size.
            # Normalize: f = (meta_prob - threshold) / (1 - threshold)
            # Then scale into [0.25, 1.0] range so we never go to zero.
            norm = (mp - threshold) / (1.0 - threshold) if threshold < 1.0 else 0.0
            f = 0.25 + 0.75 * float(np.clip(norm, 0, 1))
        else:
            f = 1.0

        records.append({
            "net_return":    net,
            "position_size": f,
            "sized_return":  f * net,   # P&L on total capital = fraction × per-unit P&L
            "win":           win,
            "meta_proba":    mp,
        })

        if win:
            raw_wins.append(net)
        else:
            raw_loss.append(net)

    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, label: str, slippage: float):
    if len(df) == 0:
        print(f"  {label}: no trades")
        return

    rets   = df["sized_return"].values
    equity = np.cumprod(1 + rets)
    wins   = df["win"].sum()
    wr     = wins / len(df)
    total  = rets.sum()
    avg    = rets.mean()
    max_dd = float(np.min(equity / np.maximum.accumulate(equity) - 1))
    gross_w = rets[rets > 0].sum() if (rets > 0).any() else 0
    gross_l = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-9
    pf      = gross_w / gross_l

    avg_size = df["position_size"].mean()
    max_size = df["position_size"].max()
    min_size = df["position_size"].min()

    print(f"\n  ── {label} ──")
    print(f"  Trades:           {len(df):,}")
    print(f"  Win rate:         {wr:.1%}")
    print(f"  Avg position:     {avg_size:.1%}  (min={min_size:.1%}, max={max_size:.1%})")
    print(f"  Avg return/trade: {avg*100:+.3f}%  (on total capital)")
    print(f"  Total return:     {total*100:+.2f}%")
    print(f"  Max drawdown:     {max_dd*100:.2f}%")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Final equity:     {equity[-1]:.4f}")

    vals = equity
    lo, hi = vals.min(), vals.max()
    rng  = hi - lo if hi > lo else 1
    pts  = vals[::max(1, len(vals)//40)]
    print(f"  Equity  (1.00→{vals[-1]:.4f})  "
          + "".join("█" if (v-lo)/rng > 0.6 else
                    "▄" if (v-lo)/rng > 0.3 else
                    "▂" for v in pts))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",     default=DB_PATH)
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--slippage",    type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--kelly-p",     type=float, default=0.60,
                        help="Win rate estimate for Kelly (default 0.60 — conservative)")
    args = parser.parse_args()

    kf = kelly_fraction(args.kelly_p)
    print("Position Sizing Backtest  [LONG only, fee + slippage adjusted]")
    print(f"Threshold: {args.threshold}  |  Fee: {FEE*100:.2f}%  |  Slippage: {args.slippage*100:.2f}%")
    print(f"Kelly p={args.kelly_p:.2f}  →  full Kelly={kf:.1%}  |  half Kelly={kf*0.5:.1%}\n")

    print("Loading models...")
    clf        = load_latest("clf_direction_*.pkl")
    meta_model = load_latest("meta_model_*.pkl")

    print("\nLoading data...")
    df, test_df = load_test_data(args.db_path, args.train_ratio, args.val_ratio)
    feature_cols = [c for c in get_feature_cols(df) if c in test_df.columns]

    data = test_df.dropna(subset=["target_tb_direction"]).copy()
    data = data[data["target_tb_direction"] != 0].reset_index(drop=True)
    print(f"  Decisive moves in test: {len(data):,}")

    print("\nComputing predictions...")
    m1_pred, m1_proba, meta_proba = get_predictions(data, clf, meta_model, feature_cols)
    taken = (m1_pred == 1) & (meta_proba >= args.threshold)
    print(f"  Long signals taken: {taken.sum():,} / {len(data):,}  ({taken.mean():.1%})")

    print("\n" + "=" * 65)
    print("  SIZING STRATEGY COMPARISON")
    print("=" * 65)

    strategies = ["flat", "kelly", "half_kelly", "meta_prob"]
    labels = {
        "flat":       f"FLAT BET         (100% position every trade)",
        "kelly":      f"FULL KELLY       ({kf:.0%} position — aggressive)",
        "half_kelly": f"HALF KELLY       ({kf*0.5:.0%} position — conservative)",
        "meta_prob":  f"META-PROB SIZING (25–100% scaled by model confidence)",
    }

    results = {}
    for s in strategies:
        sim_df = simulate(data, m1_pred, meta_proba, args.threshold,
                          sizing=s, kelly_p=args.kelly_p, slippage=args.slippage)
        results[s] = sim_df
        compute_metrics(sim_df, labels[s], args.slippage)

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Strategy':<18}  {'Total Ret':>10}  {'Max DD':>8}  {'Profit Factor':>14}  {'Final Equity':>13}")
    for s in strategies:
        sim_df = results[s]
        rets   = sim_df["sized_return"].values
        equity = np.cumprod(1 + rets)
        total  = rets.sum()
        max_dd = float(np.min(equity / np.maximum.accumulate(equity) - 1))
        gw     = rets[rets > 0].sum() if (rets > 0).any() else 0
        gl     = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-9
        pf     = gw / gl
        print(f"  {s:<18}  {total*100:>+10.2f}%  {max_dd*100:>7.2f}%  {pf:>14.2f}  {equity[-1]:>13.4f}")

    print(f"""
─────────────────────────────────────────────────────────────────
How to read this:
  flat       → baseline, same risk every trade
  kelly      → maximises long-run growth but high drawdowns
  half_kelly → best risk-adjusted returns in practice
  meta_prob  → uses the model's own confidence as position size
               (naturally reduces size on weaker signals)

Best total return ≠ best strategy. Watch drawdown — a strategy
that makes 30% but can draw down 25% is dangerous with real money.
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
