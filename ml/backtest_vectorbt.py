"""
VectorBT Backtest — professional portfolio simulation.

Unlike the custom backtest scripts, VectorBT:
  - Applies SL/TP on real price bars (high/low intrabar detection)
  - Runs proper compound portfolio accounting
  - Gives standard financial metrics (Sortino, Calmar, trade stats)
  - Catches timing errors the manual scripts can miss

Architecture:
  - Entry: close of signal bar (m1_pred==1 AND meta_proba >= threshold)
  - Stop loss:    -LOWER_BARRIER (0.2%) from entry price
  - Take profit:  +UPPER_BARRIER (0.4%) from entry price
  - Time barrier: close position after MAX_BARS if neither SL/TP hit
  - Long only

Usage:
    python3 backtest_vectorbt.py
    python3 backtest_vectorbt.py --threshold 0.57
    python3 backtest_vectorbt.py --threshold 0.55 --init-cash 1000
"""

import argparse
import glob
import os
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError:
    raise SystemExit(
        "vectorbt is required.\n"
        "  Install: pip install vectorbt --break-system-packages"
    )

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH
from train_model import get_feature_cols
from compute_features import UPPER_BARRIER, LOWER_BARRIER, MAX_BARS

MODELS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FEE              = 0.001    # 0.10% round trip
DEFAULT_SLIPPAGE = 0.0003   # 0.03%


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


def load_price_data(db_path):
    """Load full OHLC from klines with a DatetimeIndex."""
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute(
        "SELECT open_time, open, high, low, close FROM klines ORDER BY open_time"
    ).df()
    con.close()
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df.drop(columns=["open_time"])
    return df


def load_feature_data(db_path, train_ratio=0.70, val_ratio=0.10):
    con = duckdb.connect(db_path, read_only=True)
    df  = con.execute("SELECT * FROM ml_features_clean ORDER BY open_time").df()
    con.close()
    n          = len(df)
    test_start = int(n * (train_ratio + val_ratio))
    test_df    = df.iloc[test_start:].copy().reset_index(drop=True)
    t0 = pd.to_datetime(test_df["open_time"].iloc[0],  unit="ms", utc=True)
    t1 = pd.to_datetime(test_df["open_time"].iloc[-1], unit="ms", utc=True)
    print(f"  Test window: {t0.date()} → {t1.date()}  ({len(test_df):,} rows)")
    return df, test_df, t0, t1


def get_predictions(data, clf, meta_model, feature_cols):
    X        = data[feature_cols]
    m1_pred  = clf.predict(X)
    m1_proba = clf.predict_proba(X)[:, 1]
    meta_X                   = X.copy()
    meta_X["model1_prob_up"] = m1_proba
    meta_cols  = feature_cols + ["model1_prob_up"]
    meta_proba = meta_model.predict_proba(meta_X[meta_cols])[:, 1]
    return m1_pred, m1_proba, meta_proba


# ── Signal construction ───────────────────────────────────────────────────────

def build_signal_series(data, m1_pred, meta_proba, threshold, price_index):
    """
    Build a boolean entry Series aligned to the full price DatetimeIndex.
    Entry fires on decisive rows where m1_pred==1 AND meta_proba >= threshold.
    """
    decisive = data[data["target_tb_direction"] != 0].copy().reset_index(drop=True)
    decisive_m1    = m1_pred[(data["target_tb_direction"] != 0).values]
    decisive_meta  = meta_proba[(data["target_tb_direction"] != 0).values]

    take_mask = (decisive_m1 == 1) & (decisive_meta >= threshold)
    signal_times = pd.to_datetime(
        decisive.loc[take_mask, "open_time"].values, unit="ms", utc=True
    )

    entries = pd.Series(False, index=price_index, name="entries")
    valid   = signal_times[signal_times.isin(price_index)]
    entries[valid] = True

    print(f"  Signals mapped to price index: {valid.shape[0]:,}  "
          f"({len(signal_times) - valid.shape[0]:,} not matched)")
    return entries


# ── VectorBT portfolio ────────────────────────────────────────────────────────

def run_vbt(price_df, entries, init_cash=10_000.0, slippage=DEFAULT_SLIPPAGE):
    """
    Run the VectorBT portfolio simulation.

    Entry:  close of signal bar
    SL:     -LOWER_BARRIER below entry price (detected intrabar via low)
    TP:     +UPPER_BARRIER above entry price (detected intrabar via high)
    Time:   explicit exit at T + MAX_BARS if neither SL/TP triggered

    IMPORTANT: high + low must be passed so VBT detects SL/TP when the
    intrabar price crosses the level — not when the bar *close* crosses it.
    Without high/low, a bar that opens at entry and crashes -3% before
    closing -3% would exit at -3% instead of the -0.2% SL price.
    """
    close = price_df["close"]
    high  = price_df["high"]
    low   = price_df["low"]

    # Build time-barrier exits: for each signal, fire an exit at T+MAX_BARS.
    # VBT will use whichever comes first — SL, TP, or this explicit exit.
    time_exits = pd.Series(False, index=price_df.index, name="exits")
    signal_locs = np.where(entries.values)[0]
    for loc in signal_locs:
        exit_loc = min(loc + MAX_BARS, len(price_df) - 1)
        time_exits.iloc[exit_loc] = True

    portfolio = vbt.Portfolio.from_signals(
        close    = close,
        high     = high,            # intrabar TP detection (high >= entry × 1.004)
        low      = low,             # intrabar SL detection (low  <= entry × 0.998)
        entries  = entries,
        exits    = time_exits,      # time-barrier fallback at T + MAX_BARS
        sl_stop  = LOWER_BARRIER,   # 0.2% below entry → stop loss
        tp_stop  = UPPER_BARRIER,   # 0.4% above entry → take profit
        fees     = FEE / 2,         # VBT applies fee per side (entry + exit)
        slippage = slippage,
        init_cash= init_cash,
        freq     = "1h",
    )
    return portfolio


# ── Printing ──────────────────────────────────────────────────────────────────

def print_vbt_results(portfolio, threshold, init_cash):
    stats = portfolio.stats()

    print("\n" + "=" * 65)
    print(f"  VECTORBT RESULTS  (threshold={threshold})")
    print(f"  Initial capital: ${init_cash:,.0f}")
    print("=" * 65)

    # Core performance
    total_ret  = portfolio.total_return()
    max_dd     = portfolio.max_drawdown()
    final_val  = init_cash * (1 + total_ret)

    print(f"\n  ── Portfolio ──")
    print(f"  Initial cash:      ${init_cash:>10,.2f}")
    print(f"  Final value:       ${final_val:>10,.2f}")
    print(f"  Total return:      {total_ret*100:>+10.2f}%")
    print(f"  Max drawdown:      {max_dd*100:>10.2f}%")

    # Risk-adjusted metrics (VBT computes these properly)
    try:
        sharpe  = portfolio.sharpe_ratio()
        sortino = portfolio.sortino_ratio()
        calmar  = portfolio.calmar_ratio()
        print(f"\n  ── Risk-Adjusted Metrics ──")
        print(f"  Sharpe ratio:      {sharpe:>10.3f}  (daily returns, annualized)")
        print(f"  Sortino ratio:     {sortino:>10.3f}  (only penalises downside)")
        print(f"  Calmar ratio:      {calmar:>10.3f}  (return / max drawdown)")
        if sortino > 1.0:
            print(f"  ✅ Sortino > 1.0 — good risk-adjusted performance")
        else:
            print(f"  ⚠️  Sortino < 1.0 — weak risk-adjusted performance")
    except Exception as e:
        print(f"  (Risk metrics unavailable: {e})")

    # Trade statistics
    try:
        trades = portfolio.trades
        n      = trades.count()
        if n > 0:
            wr     = trades.win_rate()
            avg_pnl = trades.pnl.mean()
            avg_ret_tr = trades.returns.mean()
            best  = trades.returns.max()
            worst = trades.returns.min()
            avg_dur = trades.duration.mean()

            print(f"\n  ── Trade Statistics ──")
            print(f"  Total trades:      {n:>10,}")
            print(f"  Win rate:          {wr*100:>10.1f}%")
            print(f"  Avg P&L/trade:     ${avg_pnl:>+10.2f}")
            print(f"  Avg return/trade:  {avg_ret_tr*100:>+10.3f}%")
            print(f"  Best trade:        {best*100:>+10.2f}%")
            print(f"  Worst trade:       {worst*100:>+10.2f}%")
            print(f"  Avg duration:      {avg_dur}")
    except Exception as e:
        print(f"  (Trade stats unavailable: {e})")

    # Equity curve ASCII
    try:
        val = portfolio.value()
        vals = val.values
        lo, hi = vals.min(), vals.max()
        rng = hi - lo if hi > lo else 1
        pts = vals[::max(1, len(vals)//55)]
        print(f"\n  Equity curve  (${lo:,.0f} → ${vals[-1]:,.2f})")
        print(f"  ${hi:>8,.0f} |" + "".join(
            "█" if (v-lo)/rng > 0.6 else
            "▄" if (v-lo)/rng > 0.3 else
            "▂" for v in pts) + "|")
        print(f"  ${lo:>8,.0f} |{'─'*len(pts)}|")
    except Exception as e:
        print(f"  (Equity curve unavailable: {e})")

    print(f"""
─────────────────────────────────────────────────────────────────
Key differences from custom backtest:
  - SL/TP checked on real intrabar high/low (not assumed binary)
  - Fees applied per side (entry AND exit separately)
  - Sortino and Calmar computed on actual daily P&L
  - Duration shows how long each trade actually stays open
─────────────────────────────────────────────────────────────────
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path",     default=DB_PATH)
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--slippage",    type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument("--init-cash",   type=float, default=10_000.0,
                        help="Starting portfolio value in USD (default: 10000)")
    args = parser.parse_args()

    print("VectorBT Backtest — Professional Portfolio Simulation")
    print(f"Threshold: {args.threshold}  |  Fee: {FEE*100:.2f}%  |  "
          f"Slippage: {args.slippage*100:.2f}%  |  "
          f"Barriers: +{UPPER_BARRIER*100:.1f}% / -{LOWER_BARRIER*100:.1f}% / {MAX_BARS}H\n")

    print("Loading models...")
    clf        = load_latest("clf_direction_*.pkl")
    meta_model = load_latest("meta_model_*.pkl")

    print("\nLoading price data...")
    price_df = load_price_data(args.db_path)
    print(f"  Price bars: {len(price_df):,}  ({price_df.index[0].date()} → {price_df.index[-1].date()})")

    print("\nLoading feature data...")
    full_df, test_df, t0, t1 = load_feature_data(
        args.db_path, args.train_ratio, args.val_ratio
    )
    feature_cols = [c for c in get_feature_cols(full_df) if c in test_df.columns]

    # Test window price slice
    price_test = price_df.loc[t0:t1]
    print(f"  Price bars in test window: {len(price_test):,}")

    # Get predictions on full test set (including non-decisive rows for alignment)
    data_decisive = test_df.dropna(subset=["target_tb_direction"]).copy()
    data_decisive = data_decisive[data_decisive["target_tb_direction"] != 0].reset_index(drop=True)

    print("\nComputing predictions...")
    m1_pred, m1_proba, meta_proba = get_predictions(
        data_decisive, clf, meta_model, feature_cols
    )
    taken = (m1_pred == 1) & (meta_proba >= args.threshold)
    print(f"  Long signals: {taken.sum():,} / {len(data_decisive):,}  ({taken.mean():.1%})")

    print("\nBuilding signal series...")
    entries = build_signal_series(
        data_decisive, m1_pred, meta_proba, args.threshold, price_test.index
    )

    print("\nRunning VectorBT simulation...")
    try:
        portfolio = run_vbt(price_test, entries, args.init_cash, args.slippage)
        print_vbt_results(portfolio, args.threshold, args.init_cash)
    except Exception as e:
        print(f"\n❌ VectorBT error: {e}")
        print("   Try: pip install vectorbt --break-system-packages --upgrade")
        raise


if __name__ == "__main__":
    main()
