"""
Backtest — Model 1 + Meta-Filter on held-out test window.

Supports LONG, SHORT, and BOTH directions. Includes slippage modeling.

Important R:R note:
  LONG  — win = +UPPER (0.4%), loss = -LOWER (0.2%)  → break-even 50.0% after fees
  SHORT — win = +LOWER (0.2%), loss = -UPPER (0.4%)  → break-even 83.3% after fees

Slippage = additional cost beyond fees (price impact, bid-ask spread).
  Normal BTC:     1–3 bps (0.01–0.03%)
  Volatile:       5–15 bps
  Flash crash:    20–50 bps

Usage:
    python3 backtest.py                               # long only, 3 bps slippage
    python3 backtest.py --slippage 0.0005             # 5 bps slippage
    python3 backtest.py --slippage-mode random        # randomized per trade
    python3 backtest.py --direction short
    python3 backtest.py --direction both
    python3 backtest.py --threshold 0.60
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

# Binance taker fee: 0.05% per side = 0.10% round trip
FEE = 0.001

# Default slippage: 3 bps (0.03%) — realistic for BTC perps with small orders
DEFAULT_SLIPPAGE = 0.0003


def be_long(slippage=0.0):
    """Break-even win rate for LONG after fees + slippage."""
    return (LOWER_BARRIER + FEE + slippage) / (UPPER_BARRIER + LOWER_BARRIER)


def be_short(slippage=0.0):
    """Break-even win rate for SHORT after fees + slippage."""
    return (UPPER_BARRIER + FEE + slippage) / (UPPER_BARRIER + LOWER_BARRIER)


# ── Model + data loading ──────────────────────────────────────────────────────

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


# ── Prediction engine ─────────────────────────────────────────────────────────

def get_predictions(data, clf, meta_model, feature_cols):
    """Compute M1 predictions + meta probabilities once for all rows."""
    X        = data[feature_cols]
    m1_pred  = clf.predict(X)
    m1_proba = clf.predict_proba(X)[:, 1]

    meta_X                   = X.copy()
    meta_X["model1_prob_up"] = m1_proba
    meta_cols  = feature_cols + ["model1_prob_up"]
    meta_proba = meta_model.predict_proba(meta_X[meta_cols])[:, 1]

    return m1_pred, m1_proba, meta_proba


# ── Return computation ────────────────────────────────────────────────────────

def compute_returns(data, m1_pred, meta_proba, threshold, direction,
                    slippage=DEFAULT_SLIPPAGE, slippage_mode="fixed"):
    """
    Compute trade-level returns for a given direction.

    LONG:  signal = m1_pred==1 AND meta_proba>=threshold
           win  (+1 target) → +(UPPER_BARRIER - FEE - slippage)
           loss (-1 target) → -(LOWER_BARRIER + FEE + slippage)

    SHORT: signal = m1_pred==0 AND meta_proba>=threshold
           win  (-1 target) → +(LOWER_BARRIER - FEE - slippage)
           loss (+1 target) → -(UPPER_BARRIER + FEE + slippage)

    slippage_mode:
      'fixed'  — same slippage on every trade
      'random' — uniform between 0 and 2× slippage (same expected value, more realistic)
    """
    actual = data["target_tb_direction"].values

    if direction == "long":
        take   = (m1_pred == 1) & (meta_proba >= threshold)
        is_win = lambda a: a == 1
        win_gross  =  UPPER_BARRIER
        loss_gross = -LOWER_BARRIER
    else:  # short
        take   = (m1_pred == 0) & (meta_proba >= threshold)
        is_win = lambda a: a == -1
        win_gross  =  LOWER_BARRIER
        loss_gross = -UPPER_BARRIER

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    returns     = []
    trade_times = []
    for i, t in enumerate(take):
        if not t:
            continue
        # Slippage: always hurts (reduces win, increases loss)
        slip = rng.uniform(0, 2 * slippage) if slippage_mode == "random" else slippage
        gross = win_gross if is_win(actual[i]) else loss_gross
        ret   = gross - FEE - slip   # slip always subtracts (costs money on entry AND exit)
        returns.append(ret)
        trade_times.append(data["open_time"].iloc[i])

    ts_index = pd.to_datetime(trade_times, unit="ms")
    return pd.Series(returns, index=ts_index, name="trade_return")


# ── Metrics printing ──────────────────────────────────────────────────────────

def print_metrics(returns, threshold, direction, slippage=DEFAULT_SLIPPAGE):
    be_rate   = be_long(slippage) if direction == "long" else be_short(slippage)
    dir_label = direction.upper()

    print("\n" + "=" * 62)
    print(f"  BACKTEST RESULTS  [{dir_label}]  (threshold={threshold})")
    print("=" * 62)

    if len(returns) == 0:
        print("  No trades taken — lower the threshold.")
        return

    wins          = (returns > 0).sum()
    win_rate      = wins / len(returns)
    avg_ret       = returns.mean()
    total         = returns.sum()
    equity        = (1 + returns).cumprod()
    max_dd        = (equity / equity.cummax() - 1).min()
    gross_wins    = returns[returns > 0].sum()
    gross_losses  = returns[returns < 0].abs().sum()
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    print(f"  Trades taken:     {len(returns):,}")
    print(f"  Wins:             {wins:,}  ({win_rate:.1%})")
    print(f"  Losses:           {len(returns)-wins:,}  ({1-win_rate:.1%})")
    print(f"  Avg return/trade: {avg_ret*100:+.3f}%")
    print(f"  Total return:     {total*100:+.2f}%")
    print(f"  Max drawdown:     {max_dd*100:.2f}%")
    print(f"  Profit factor:    {profit_factor:.2f}  (>1.5 = good, >2.5 = strong)")
    print(f"  Fee per trade:    {FEE*100:.2f}%  (0.05% entry + 0.05% exit)")
    print(f"  Slippage/trade:   {slippage*100:.2f}%")
    print(f"  Break-even rate:  {be_rate:.1%}  [{dir_label}] (after fees + slippage)")

    if win_rate > be_rate:
        print(f"\n  ✅ Profitable after fees — {win_rate:.1%} beats {be_rate:.1%} break-even")
        print(f"     Expected value per trade: {avg_ret*100:+.3f}%")
    else:
        print(f"\n  ❌ Not profitable after fees — {win_rate:.1%} below {be_rate:.1%} break-even")

    vals  = equity.values
    lo, hi = vals.min(), vals.max()
    rng   = hi - lo if hi > lo else 1
    pts   = vals[::max(1, len(vals)//50)]
    print(f"\n  Equity curve  (start=1.00, end={vals[-1]:.4f})")
    print(f"  {hi:.4f} |" + "".join("█" if (v-lo)/rng > 0.6 else
                                     "▄" if (v-lo)/rng > 0.3 else
                                     "▂" for v in pts) + "|")
    print(f"  {lo:.4f} |{'─'*len(pts)}|")


# ── Threshold sweep ───────────────────────────────────────────────────────────

def threshold_sweep(data, m1_pred, meta_proba, direction):
    actual  = data["target_tb_direction"].values
    be_rate = be_long() if direction == "long" else be_short()

    if direction == "long":
        base_mask = m1_pred == 1
        win_cond  = actual == 1
        win_ret   = UPPER_BARRIER - FEE
        loss_ret  = -(LOWER_BARRIER + FEE)
    else:
        base_mask = m1_pred == 0
        win_cond  = actual == -1
        win_ret   = LOWER_BARRIER - FEE
        loss_ret  = -(UPPER_BARRIER + FEE)

    print(f"\n  ── Threshold Sweep [{direction.upper()}]  (break-even: {be_rate:.1%}) ──")
    print(f"  {'Threshold':>10}  {'Trades':>8}  {'Win Rate':>10}  {'Avg Ret(net)':>14}  {'Total(net)':>12}  {'':>3}")
    for thr in [0.48, 0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65]:
        mask = base_mask & (meta_proba >= thr)
        if mask.sum() < 5:
            break
        rets = np.where(win_cond[mask], win_ret, loss_ret)
        wr   = (rets > 0).mean()
        ar   = rets.mean()
        flag = "✅" if wr > be_rate else "❌"
        print(f"  {thr:>10.2f}  {mask.sum():>8,}  {wr:>10.1%}  {ar*100:>+14.3f}%  {rets.sum()*100:>+12.2f}%  {flag}")


# ── Slippage sweep ────────────────────────────────────────────────────────────

def slippage_sweep(data, m1_pred, meta_proba, threshold, direction="long"):
    """Show how edge degrades as slippage increases. Finds the breakeven slippage."""
    print(f"\n  ── Slippage Sweep [{direction.upper()}]  (threshold={threshold}) ──")
    print(f"  {'Slippage':>10}  {'Total Cost':>12}  {'Avg Ret/trade':>15}  {'Total Ret':>11}  {'Profitable':>12}")
    print(f"  {'(per trade)':>10}  {'fee+slip':>12}  {'':>15}  {'':>11}  {'':>12}")

    slip_levels = [0.0000, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0010, 0.0015, 0.0020]
    edge_gone_at = None

    for slip in slip_levels:
        rets = compute_returns(data, m1_pred, meta_proba, threshold, direction, slippage=slip)
        if len(rets) == 0:
            continue
        be   = be_long(slip) if direction == "long" else be_short(slip)
        wr   = (rets > 0).mean()
        ar   = rets.mean()
        tot  = rets.sum()
        cost = FEE + slip
        flag = "✅" if wr > be else "❌"
        if flag == "❌" and edge_gone_at is None:
            edge_gone_at = slip
        print(f"  {slip*100:>10.2f}%  {cost*100:>11.2f}%  {ar*100:>+15.3f}%  {tot*100:>+10.2f}%  {flag} ({wr:.1%} vs {be:.1%})")

    if edge_gone_at is not None:
        print(f"\n  ⚠️  Edge disappears at {edge_gone_at*100:.2f}% slippage per trade")
    else:
        print(f"\n  ✅  Edge survives all tested slippage levels")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",     default=DB_PATH)
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--direction",     default="long",
                        choices=["long", "short", "both"],
                        help="Trade direction: long, short, or both (default: long)")
    parser.add_argument("--slippage",      type=float, default=DEFAULT_SLIPPAGE,
                        help=f"Slippage per trade as decimal (default: {DEFAULT_SLIPPAGE} = {DEFAULT_SLIPPAGE*100:.2f}%%)")
    parser.add_argument("--slippage-mode", default="fixed",
                        choices=["fixed", "random"],
                        help="fixed = same each trade, random = uniform 0 to 2× slippage (default: fixed)")
    args = parser.parse_args()

    slip = args.slippage

    print("Backtest — Model 1 + Meta-Filter  [fee + slippage adjusted]")
    print(f"Threshold: {args.threshold}  |  Fee: {FEE*100:.2f}%  |  Slippage: {slip*100:.2f}% ({args.slippage_mode})  |  Direction: {args.direction.upper()}")
    print(f"Break-even → LONG: {be_long(slip):.1%}  |  SHORT: {be_short(slip):.1%}\n")

    print("Loading models...")
    clf        = load_latest("clf_direction_*.pkl")
    meta_model = load_latest("meta_model_*.pkl")

    print("\nLoading data...")
    df, test_df = load_test_data(args.db_path, args.train_ratio, args.val_ratio)
    feature_cols = [c for c in get_feature_cols(df) if c in test_df.columns]

    # Filter to decisive moves only
    data = test_df.dropna(subset=["target_tb_direction"]).copy()
    data = data[data["target_tb_direction"] != 0].reset_index(drop=True)
    print(f"  Decisive moves in test: {len(data):,}")

    print("\nComputing predictions...")
    m1_pred, m1_proba, meta_proba = get_predictions(data, clf, meta_model, feature_cols)

    directions = ["long", "short"] if args.direction == "both" else [args.direction]

    for direction in directions:
        returns = compute_returns(data, m1_pred, meta_proba, args.threshold, direction,
                                  slippage=slip, slippage_mode=args.slippage_mode)
        signals = (m1_pred == 1 if direction == "long" else m1_pred == 0)
        taken   = signals & (meta_proba >= args.threshold)
        print(f"\n  [{direction.upper()}] Signals taken: {taken.sum():,} / {len(data):,}  ({taken.mean():.1%})")
        print_metrics(returns, args.threshold, direction, slippage=slip)
        threshold_sweep(data, m1_pred, meta_proba, direction)
        slippage_sweep(data, m1_pred, meta_proba, args.threshold, direction)

    if args.direction == "both":
        # Combined equity curve
        long_ret  = compute_returns(data, m1_pred, meta_proba, args.threshold, "long",  slippage=slip, slippage_mode=args.slippage_mode)
        short_ret = compute_returns(data, m1_pred, meta_proba, args.threshold, "short", slippage=slip, slippage_mode=args.slippage_mode)
        combined  = pd.concat([long_ret, short_ret]).sort_index()

        print("\n" + "=" * 62)
        print(f"  COMBINED  (LONG + SHORT)")
        print("=" * 62)
        total   = combined.sum()
        equity  = (1 + combined).cumprod()
        max_dd  = (equity / equity.cummax() - 1).min()
        print(f"  Total trades:   {len(combined):,}  (long={len(long_ret):,}, short={len(short_ret):,})")
        print(f"  Total return:   {total*100:+.2f}%")
        print(f"  Max drawdown:   {max_dd*100:.2f}%")
        vals = equity.values
        lo, hi = vals.min(), vals.max()
        rng  = hi - lo if hi > lo else 1
        pts  = vals[::max(1, len(vals)//50)]
        print(f"\n  Combined equity curve  (start=1.00, end={vals[-1]:.4f})")
        print(f"  {hi:.4f} |" + "".join("█" if (v-lo)/rng > 0.6 else
                                         "▄" if (v-lo)/rng > 0.3 else
                                         "▂" for v in pts) + "|")
        print(f"  {lo:.4f} |{'─'*len(pts)}|")


if __name__ == "__main__":
    main()
